"""Finite individual configuration worker; queue and release are separate."""
import copy
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from src.pointwise_benchmark_v1.models import build_model, random_state
from .model import load_b_model, CenteredProcess
from .data import Bank
from .objective import objective, native_quantile_objective
from .evaluate import predict, selection, evaluation_top_masks, score_cells, supplemental_masks, phase_brier
from src.pointwise_benchmark_v1.dataset import sha256
from src.pointwise_benchmark_v1.dataset import hierarchy_weights
from src.d500_process_adapter_v1.joint_scores import row_metrics
from src.d500_process_adapter_v1.runtime import fingerprint
from .report import report
from .paths import checked_output

GROUPS = {
    'M_N_E': ('natural', False, True, False),
    'M_X_E': ('balanced', False, True, False),
    'M_N_ME': ('natural', True, True, False),
    'M_X_ME': ('balanced', True, True, False),
    'H_X_MU_ONLY': ('balanced', True, False, False),
    'NB_N': ('natural', False, False, True),
    'NB_X': ('balanced', False, False, True),
}


def save_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def construct(group, artifacts, device):
    _, center, adapter_trainable, nb = GROUPS[group]
    if nb:
        return build_model('nbeatsx', 'pq', 24, 24, 2, device), None
    model = load_b_model(artifacts, device)
    teacher = copy.deepcopy(model.process).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with random_state(torch.device(device), seed=2):
        model.process = CenteredProcess(model.process, center, adapter_trainable).to(device)
    return model, teacher


def make_optimizer(model, nb):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return (torch.optim.Adam(parameters, lr=.001) if nb else
            torch.optim.AdamW(parameters, lr=.0003, weight_decay=.0001))


def checkpoint(model, optimizer, step, group, device, identity):
    return dict(group=group, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(), identity=identity)


def restore(checkpoint_value, model, optimizer, group, device, identity):
    if checkpoint_value['group'] != group or checkpoint_value.get('identity') != identity:
        raise ValueError('checkpoint configuration mismatch')
    model.load_state_dict(checkpoint_value['model'])
    optimizer.load_state_dict(checkpoint_value['optimizer'])
    torch.set_rng_state(checkpoint_value['cpu_rng'].cpu())
    torch.cuda.set_rng_state(checkpoint_value['cuda_rng'].cpu(), device)
    np.random.set_state(checkpoint_value['numpy_rng'])
    random.setstate(checkpoint_value['python_rng'])
    return int(checkpoint_value['step'])


@torch.no_grad()
def validation_loss(model, teacher, bank, nb, device):
    model.eval()
    total = 0.; n = 0
    for start in range(0, len(bank.select_rows), 8):
        ids = np.arange(start, min(start+8, len(bank.select_rows)))
        x, y = bank.batch('select', ids, device)
        weight = torch.ones(len(ids), device=device)
        ordinary = (y.amax(1) <= 1.).to(x.dtype)
        if nb:
            value = native_quantile_objective(model, x, y, weight, 910000+start)
        else:
            value, _ = objective(model.process, teacher, x, y, weight, ordinary, 910000+start)
        total += float(value)*len(ids); n += len(ids)
    return total/n


def train_step(model, teacher, optimizer, bank, ids, ratio, mode, nb, device, step):
    x, y = bank.batch('fit', ids, device)
    weight = torch.as_tensor(np.ones(len(ids)) if mode == 'natural' else ratio[ids],
                             dtype=torch.float32, device=device)
    ordinary = torch.as_tensor(bank.severity[ids] == 0, dtype=torch.float32, device=device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    if nb:
        loss = native_quantile_objective(model, x, y, weight, 2100000+step)
        parts = {'mqloss': loss}
    else:
        loss, parts = objective(model.process, teacher, x, y, weight, ordinary, 2100000+step)
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    for name,p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError('nonfinite gradient '+name)
    if step < 2 or (step+1)%50 == 0:
        model._last_gradient_health = {name:(float(p.grad.detach().norm()) if p.grad is not None else None)
                                       for name,p in model.named_parameters() if p.requires_grad}
    torch.nn.utils.clip_grad_norm_(parameters, 10. if nb else 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('nonfinite parameters')
    return float(loss.detach()), {k:float(v.detach()) for k,v in parts.items()}


def run(group, bank, schedules, ratio, artifacts, device, folder, steps=None,
        candidate_updates=None, evaluation_draws=32, resume=None, interrupt_after=None):
    """Run one configuration; supplied shortened steps are engineering-only."""
    folder, _, host = checked_output(folder)
    if host!='configured-compute':raise ValueError('model execution requires the public checkout compute role')
    if resume is None:
        if folder.exists():
            raise FileExistsError('preserve prior worker output')
        folder.mkdir(parents=True)
    elif not folder.is_dir():
        raise FileNotFoundError('resume requires original output directory')
    elif (folder/'final_scores.json').exists():
        raise ValueError('final artifacts already exist; do not overwrite during recovery')
    mode, _, _, nb = GROUPS[group]
    final = (3000 if nb else 1200) if steps is None else steps
    candidates = ([0, 1000, 2000, 3000] if nb else [0, 400, 800, 1200]) if candidate_updates is None else candidate_updates
    config_path=Path(__file__).resolve().parents[2]/'configs/extreme_process_transfer_pilot_20261101.json'
    config=json.loads(config_path.read_text())
    if (config['seed']!=2 or config['batch_size']!=64 or config['validation_draws']!=32 or
        config['final_draws']!=64 or config['train_groups'][group]!=(3000 if nb else 1200) or
        config['m_candidate_updates']!=[0,400,800,1200] or config['nb_candidate_updates']!=[0,1000,2000,3000] or
        config['loss_coefficients']!={'crps_u':.1,'energy_u':.1,'balance':.01,'ordinary_keep':.1}):
        raise ValueError('config differs from implemented approved condition')
    if candidates[-1] != final:
        raise ValueError('final update must be a candidate')
    model, teacher = construct(group, artifacts, device)
    optimizer = make_optimizer(model, nb)
    frozen_base = fingerprint(model.process.base) if not nb else None
    frozen_teacher = fingerprint(teacher) if not nb else None
    frozen_adapter = fingerprint(model.process.adapter) if not nb and not GROUPS[group][2] else None
    sequence = schedules[mode]; started = time.monotonic(); cpu = time.process_time()
    schedule_hash = hashlib.sha256(sequence.tobytes()+ratio.tobytes()).hexdigest()
    identity = dict(data_manifest=sha256(bank.root/'manifest.json'),schedule_sha256=schedule_hash,
                    code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    config_sha=sha256(config_path),
                    final_steps=final, candidates=candidates, k_validation=evaluation_draws,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(device))
    if resume is None:
        packages={}
        for package in ('torch','numpy','neuralforecast','torch-timeseries','matplotlib'):
            try:packages[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:packages[package]=None
        save_json(folder/'manifest.json',dict(group=group,identity=identity,packages=packages,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            frozen_base_sha256=frozen_base,frozen_teacher_sha256=frozen_teacher,
            frozen_adapter_sha256=frozen_adapter,baseline='B-M' if not nb else 'seed2 fresh N-BEATSx',
            model_config=model.config,seed=2,training_mode=mode,
            batch_size=64,learning_rate=.001 if nb else .0003,
            optimizer='Adam' if nb else 'AdamW weight_decay=0.0001',
            gradient_clip=10. if nb else 1.,engineering_only=steps is not None))
    records = []; best = None; last_train = None; start_step = 0
    if resume is not None:
        saved = torch.load(resume,map_location=device,weights_only=False)
        start_step = restore(saved,model,optimizer,group,device,identity)
        existing = json.loads((folder/'selection.json').read_text())
        records = existing['candidates']; best = existing['best']
        if not records or records[-1]['step'] != start_step or start_step not in candidates:
            raise ValueError('resume only from latest persisted candidate')
        if Path(resume).resolve() != (folder/records[-1]['checkpoint']).resolve():
            raise ValueError('resume checkpoint identity mismatch')
        last_train = records[-1]['training_loss']
    for step in range(start_step, final+1):
        if step in candidates and not (resume is not None and step == start_step):
            if not nb and (fingerprint(model.process.base)!=frozen_base or fingerprint(teacher)!=frozen_teacher):
                raise AssertionError('frozen D500 or teacher changed')
            if frozen_adapter is not None and fingerprint(model.process.adapter)!=frozen_adapter:
                raise AssertionError('frozen simple-head M changed')
            model.eval()
            predictions = predict(model, bank, evaluation_draws, device, nb)
            score = selection(predictions, bank)
            valid_loss = validation_loss(model, teacher, bank, nb, device)
            path = folder / ('step%04d.pt' % step)
            torch.save(checkpoint(model, optimizer, step, group, device,identity), path)
            record = dict(step=step, score=score, validation_loss=valid_loss,
                          training_loss=last_train, checkpoint=path.name,
                          updated_at_utc=datetime.now(timezone.utc).isoformat())
            records.append(record)
            if best is None or score < best['score']:
                best = record.copy()
            save_json(folder/'selection.json', dict(candidates=records, best=best,
                      rule='earliest minimum station/event/origin weighted hourly Q95/Q99 Brier', draws=evaluation_draws))
            if interrupt_after == step:
                raise RuntimeError('intentional engineering interruption after persisted candidate')
        if step == final:
            break
        ids = sequence[step]
        last_train, parts = train_step(model,teacher,optimizer,bank,ids,ratio,mode,nb,device,step)
        if step < 2 or (step+1) % 50 == 0:
            save_json(folder/'progress.json', dict(state='running', group=group, step=step+1,
                      total=final, loss=last_train, components=parts,
                      wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                      updated_at_utc=datetime.now(timezone.utc).isoformat(),
                      gradient_norms=model._last_gradient_health,
                      latest_validation=records[-1] if records else None))
    final_records = {}
    for label, record in (('fixed_last', records[-1]), ('best_brier', best)):
        key = str(record['step'])
        if key not in final_records:
            saved = torch.load(folder/record['checkpoint'], map_location=device, weights_only=False)
            model.load_state_dict(saved['model']); model.eval()
            noise_receipts = []
            samples = predict(model, bank, 64, device, nb, plan_receipts=noise_receipts)
            masks, thresholds, support = evaluation_top_masks(bank.select_rows, bank.arrays['select']['future_q'])
            more_masks, phase_support = supplemental_masks(bank)
            masks.update(more_masks)
            scores = score_cells(samples, bank, masks)
            phase_scores, _ = phase_brier(samples,bank)
            if nb:
                joint = dict(state='NOT_EVALUABLE', reason='deterministic marginal quantile grid is not a joint scenario')
            else:
                a = bank.arrays['select']; scale = a['flow_scale']
                physical = samples*scale[:,None,None]
                th = np.stack((a['q95_fit'],a['q99_fit']),axis=-1)
                joint_rows, _ = row_metrics(physical, np.full(samples.shape[:2], 1/samples.shape[1]), a['future_q'], th)
                hweights = hierarchy_weights(bank.select_rows)
                joint = dict(state='EVALUABLE', values={name:float(hweights@value) for name,value in joint_rows.items()})
            destination = folder/('final_step%04d.npz' % record['step'])
            np.savez_compressed(destination, draws_q95=samples,
                target_m3_s=bank.arrays['select']['future_q'],
                sample_id=np.asarray([r['sample_id'] for r in bank.select_rows]))
            final_records[key] = dict(checkpoint=record['checkpoint'], checkpoint_sha256=sha256(folder/record['checkpoint']),
               predictions=destination.name, predictions_sha256=sha256(destination), noise_plan=noise_receipts,
               top_thresholds=thresholds, unique_hours=support, phase_support=phase_support,
               phase_brier=phase_scores, scores=scores, joint=joint)
        final_records[key].setdefault('roles', []).append(label)
    save_json(folder/'final_scores.json', dict(checkpoints=final_records,
              status='scored', scientific_state='exploratory_single_seed'))
    chosen = final_records[str(best['step'])]
    with np.load(folder/chosen['predictions'], allow_pickle=False) as values:
        report(group, records, best['step'], values['draws_q95'], bank, folder, engineering=steps is not None)
    save_json(folder/'completion.json', dict(state='complete', group=group, updates=final, best=best,
              last=records[-1], engineering_only=steps is not None,
              scientific_state='engineering_only' if steps is not None else 'exploratory_single_seed'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--exposure', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--engineering-interrupt-after', type=int)
    args = parser.parse_args()
    bank = Bank(args.data)
    with np.load(args.exposure, allow_pickle=False) as z:
        schedules = {name:z[name].copy() for name in ('natural','balanced')}
        ratio = z['ratio'].copy()
    if schedules['natural'].shape != (3000,64) or schedules['balanced'].shape != (3000,64):
        raise ValueError('unexpected exposure shape')
    if not torch.cuda.is_available():
        raise RuntimeError('this worker requires its authorized GPU')
    if args.engineering_interrupt_after is not None and not args.engineering:
        raise ValueError('interruption is restricted to engineering checks')
    run(args.group, bank, schedules, ratio, args.artifacts, torch.device('cuda:0'), args.out,
        steps=1 if args.engineering else None,
        candidate_updates=[0,1] if args.engineering else None, resume=args.resume,
        interrupt_after=args.engineering_interrupt_after)


if __name__ == '__main__':
    main()
