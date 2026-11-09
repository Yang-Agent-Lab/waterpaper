"""D-only CN-fit adaptation. Separate zero-update preflight and authorized formal run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from . import b2_gate_formal_runner as f
from . import b2_gate_runner as p1
from .b2_gate_models import build_gate_model
from .models import random_state
from .dataset import sha256

CHANGE_ID = 'WP-20261018-B2-CHINA-ADAPT-V1'
GROUP = 'C_state'  # architecture key only; output/name identify D's CN training condition


def cn_loss(model, x, y, components, weights, generator=None):
    """Identical native random draws and forward; only row reduction changes."""
    model.check(x, y)
    weights = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
    if weights.shape != (len(x),) or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('nonnegative nonempty row weights required')
    raw_weight_sum = float(weights.sum().detach())
    weights = weights / weights.sum()
    with model.use_cached_components(components), random_state(x.device, generator=generator):
        fx, gx = model.conditions(x)
        target = y.unsqueeze(-1)
        n, m = len(x), model.diffusion
        t = torch.randint(0, m.num_timesteps, (n // 2 + 1,), device=x.device)
        t = torch.cat([t, m.num_timesteps - 1 - t])[:n]
        row_nll = (.5 * (torch.log(gx) + (fx - target).square() / gx)).flatten(1).mean(1)
        e = torch.randn_like(target)
        forward_variance = model.author_functions['cal_forward_noise'](m.betas_bar, gx, t)
        noise = e * torch.sqrt(forward_variance)
        yt = model.author_functions['q_sample'](target, fx, m.alphas_bar_sqrt, m.one_minus_alphas_bar_sqrt, t, noise=noise)
        pred, _ = m(x, None, yt, fx, gx, t)
        row_denoise = (e - pred).square().flatten(1).mean(1)
        nll, denoise = (weights * row_nll).sum(), (weights * row_denoise).sum()
    return nll + denoise, dict(gaussian_nll=float(nll.detach()), denoise_mse=float(denoise.detach()),
        loss_weight_sum=raw_weight_sum, active_parameter_scalars=30,
        loss_definition='native row NLL+denoising MSE; CN fit rows equal normalized weight; nonCN zero')


def mask_for(inputs, ids):
    return np.asarray([inputs['bank'].rows['fit'][int(i)]['country'] == 'CN' for i in ids], dtype=bool)


class Budget:
    """One new cumulative receipt, no restart of a running/failed formal phase."""
    def __init__(self, config, args):
        self.path = Path(config['budget_receipt'])
        wrapper = f.read_json(args.run_record)
        if wrapper['state'] != 'running' or Path(wrapper['cwd']).resolve() != Path.cwd().resolve():
            raise ValueError('current run_versioned record required')
        if str(Path(args.output).resolve()) not in wrapper['argv']:
            raise ValueError('run record must name exact output')
        if self.path.exists():
            self.receipt = f.read_json(self.path)
            if self.receipt['change_id'] != CHANGE_ID or self.receipt['active_run'] is not None:
                raise ValueError('receipt identity/active run mismatch; do not reset')
        else:
            if args.mode != 'preflight':
                raise ValueError('preflight receipt must exist before formal')
            self.receipt = dict(change_id=CHANGE_ID, total_budget_seconds=180., cumulative_compute_seconds=0.,
                attempted_updates_total=0, completed_updates_total=0, active_run=None, runs=[])
        if args.mode == 'formal' and (not self.receipt.get('preflight_passed') or any(r['mode'] == 'formal' for r in self.receipt['runs'])):
            raise ValueError('passed preflight and no prior formal attempt required')
        self.before = float(self.receipt['cumulative_compute_seconds'])
        self.start = time.monotonic()
        self.import_seconds = max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(wrapper['started_at_utc'])).total_seconds())
        self.attempts = Counter({GROUP: 0})
        self.run = dict(mode=args.mode, output=str(Path(args.output).resolve()), run_record=str(Path(args.run_record).resolve()),
            state='running', started_at_utc=wrapper['started_at_utc'], wrapper_head=wrapper['head'])
        self.receipt['runs'].append(self.run)
        self.receipt['active_run'] = dict(process=p1.runtime_identity(), output=self.run['output'])
        self.sync()

    def elapsed(self):
        return self.import_seconds + time.monotonic() - self.start

    def sync(self):
        self.run['elapsed_compute_seconds'] = self.elapsed()
        self.receipt['cumulative_compute_seconds'] = self.before + self.elapsed()
        p1.write_json(self.path, self.receipt)

    def checkpoint(self, phase):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.run['last_phase'] = phase
        self.sync()
        if self.before + self.elapsed() >= 160.:
            raise TimeoutError('new 180s budget engineering share160 reached; preserve20 parent reserve')

    def begin_update(self, group):
        self.checkpoint('before_update')
        if self.receipt['attempted_updates_total'] >= 1000:
            raise RuntimeError('1000 formal attempt cap reached')
        self.attempts[group] += 1
        self.receipt['attempted_updates_total'] += 1
        self.sync()

    def complete_update(self, group):
        self.receipt['completed_updates_total'] += 1
        self.sync()

    def finish(self, state, error=None):
        self.run.update(state=state, error=None if error is None else dict(type=type(error).__name__, message=str(error)))
        self.receipt['active_run'] = None
        if state == 'preflight_passed':
            self.receipt['preflight_passed'] = True
        self.sync()


def fresh(inputs, cache, device):
    model = build_gate_model('C_state', inputs['c_checkpoint'], inputs['a_checkpoint'], cache['z_fit_mean'], cache['z_fit_std'], device=device)
    model.name = 'b2_gate_d_china_state'
    model.config.update(name=model.name, training_gradient_population='CN accepted fit rows only',
        training_loss_reduction='CN-fit row weighted; divide by CN count; nonCN zero',
        select_loss_definition='unchanged native full CN select NLL+MSE', change_id=CHANGE_ID)
    active = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in active) != 30 or any(p.requires_grad and not n.startswith('gate_adapter.') for n, p in model.named_parameters()):
        raise ValueError('only30 gate scalars may train')
    a_state = torch.load(inputs['a_checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    contract = p1.all_a_key_contract(model, a_state)
    if contract['a_state_dict_keys'] != 82:
        raise ValueError('expected original82 A tensors')
    return model, active, a_state


def preflight(inputs, cache, device, ledger, output):
    model, active, a_state = fresh(inputs, cache, device)
    ids = np.asarray(inputs['exposure'][0, 0, :64], dtype=np.int64)
    lookup = p1.cache_row_lookup(cache['fit'])
    local = np.asarray([lookup[int(i)] for i in ids])
    x, y, components = f.batch_from_cache(inputs, cache, 'fit', local, device)
    mask = mask_for(inputs, ids)
    if not mask.any() or mask.all():
        raise ValueError('preflight batch must have CN and nonCN')
    def evaluate(target, weights=None):
        ledger.checkpoint('preflight_loss_gradient')
        model.zero_grad(set_to_none=True)
        gen = torch.Generator(device=device).manual_seed(2)
        loss, detail = (model.loss_from_cache(x, target, components, generator=gen) if weights is None
            else cn_loss(model, x, target, components, weights, generator=gen))
        loss.backward()
        grad = torch.cat([p.grad.detach().flatten() for p in active])
        if not torch.isfinite(loss) or not torch.isfinite(grad).all():
            raise FloatingPointError('preflight nonfinite')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('frozen A gradient')
        return loss.detach(), grad, detail
    native, ng, _ = evaluate(y)
    ones, og, _ = evaluate(y, np.ones(64))
    torch.testing.assert_close(ones, native, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(og, ng, atol=1e-6, rtol=1e-5)
    masked, mg, detail = evaluate(y, mask)
    changed = y.clone()
    changed[torch.as_tensor(~mask, device=device)] += 7.
    altered, ag, _ = evaluate(changed, mask)
    torch.testing.assert_close(altered, masked, atol=0, rtol=0)
    torch.testing.assert_close(ag, mg, atol=0, rtol=0)
    p1.write_json(output / 'preflight.json', dict(state='passed', optimizer_attempts=0, active_parameter_scalars=30,
        cn_count=int(mask.sum()), non_cn_count=int((~mask).sum()), loss_weight_sum=detail['loss_weight_sum'],
        native_loss=float(native), allones_loss=float(ones), allones_loss_abs_diff=float((ones-native).abs()),
        allones_gradient_max_abs_diff=float((og-ng).abs().max()), nonCN_label_loss_abs_diff=float((altered-masked).abs()),
        nonCN_label_gradient_max_abs_diff=float((ag-mg).abs().max()), frozen_A=p1.all_a_key_contract(model, a_state),
        noise_seed=2, full_batch_shape=list(x.shape), cache_rebuilt=False))
def train_group(group, config, inputs, cache, phase1, output, device, ledger, progress):
    output.mkdir(parents=True, exist_ok=False)
    progress[group]['state'] = 'preparing_fresh_model'
    # Always reconstruct from immutable original C and A, never phase1 best/last.
    model, _, a_state = fresh(inputs, cache, device)
    p1.all_a_key_contract(model, a_state)
    active = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in active) != config['gate_parameters'][group]:
        raise ValueError('formal optimizer must contain exactly B2 or C30 scalars')
    if any(p.requires_grad and not n.startswith('gate_adapter.') for n,p in model.named_parameters()):
        raise ValueError('formal optimizer has non-gate parameters')
    initial = {n: p.detach().cpu().double().clone() for n,p in model.gate_adapter.named_parameters()}
    optimizer = torch.optim.Adam(active, lr=.001)
    p1.write_json(output / 'model_config.json', model.config)
    fit_lookup = p1.cache_row_lookup(cache['fit'])
    exposure = inputs['exposure'][0, :1000, :64]
    pairing = dict(global_exposure_sha256=p1.hashlib.sha256(exposure.astype(np.int64).tobytes()).hexdigest(),
        noise_seeds=list(range(2, 1002)), batch_size=64, rows=64000,
        source='original C accepted_bank_indices[0,:1000,:64]')
    p1.write_json(output / 'pairing.json', pairing)
    selected = dict(updates=0, score=config['baseline_a_selection_score_128'], draws=128, seed=600002,
        provenance='reused original selected A128 score; fresh initial a1/tau6', epoch=0)
    candidates = [dict(selected)]
    history, health, records, block = [], [], [], []
    completed = 0
    training_complete = False
    monitor = None

    def checkpoint(path, step):
        # Full model state; all original A keys are checked immediately before saving.
        freeze = p1.all_a_key_contract(model, a_state)
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), updates=step,
            condition_id=config['condition_id'], bank_identity=inputs['bank'].identity,
            a_checkpoint_sha256=inputs['a_sha'], c_checkpoint_sha256=inputs['c_sha'],
            model_config=model.config, frozen_A_contract=freeze), path)
        return freeze

    try:
        checkpoint(output / 'last.pt', 0)
        checkpoint(output / 'best.pt', 0)
        health.extend(health_snapshot(model, inputs, cache, initial, 0, device, ledger, group))
        p1.write_json(output / 'health.json', health)
        p1.write_json(output / 'selection.json', selected)
        # No sender process or external monitoring for this experiment.
        for step in range(1, 1001):
            ledger.begin_update(group)
            ids = np.asarray(exposure[step - 1], dtype=np.int64)
            local = np.asarray([fit_lookup[int(i)] for i in ids], dtype=np.int64)
            x, y, components = batch_from_cache(inputs, cache, 'fit', local, device)
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            cn_mask = mask_for(inputs, ids)
            loss, details = cn_loss(model, x, y, components, cn_mask,
                generator=torch.Generator(device=device).manual_seed(2 + step - 1))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite native training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
                raise FloatingPointError('missing/nonfinite active gate gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(active, 10.).detach().cpu())
            optimizer.step()
            completed = step  # Save a just-completed update even if the following budget check fails.
            ledger.complete_update(group)
            row = dict(step=step, noise_seed=step + 1, train_loss=float(loss.detach().cpu()),
                gaussian_nll=details['gaussian_nll'], denoise_mse=details['denoise_mse'], preclip_gradient_l2=norm,
                global_indices_sha256=p1.hashlib.sha256(ids.tobytes()).hexdigest())
            records.append(row); block.append(row)
            row.update(cn_indices_sha256=hashlib.sha256(ids[cn_mask].tobytes()).hexdigest(),
                cn_count=int(cn_mask.sum()), non_cn_count=int((~cn_mask).sum()),
                loss_weight_sum=details['loss_weight_sum'], active_parameter_scalars=30,
                cn_weighted_gaussian_nll=details['gaussian_nll'], cn_weighted_denoise_mse=details['denoise_mse'])
            p1.write_json(output / 'updates.json', records)
            progress[group].update(state='running', attempted_updates=ledger.attempts[group], completed_updates=completed,
                training_ended=completed == 1000, latest_native_loss=row['train_loss'],
                loss_unit='normalized-Q native objective', loss_direction='lower_is_better')
            p1.write_json(output.parent / 'progress.json', dict(groups=progress,
                cumulative_compute_seconds=ledger.before + ledger.elapsed(), formal_attempts=sum(ledger.attempts.values()), smoke_attempts=0))
            if step in config['health_updates']:
                health.extend(health_snapshot(model, inputs, cache, initial, step, device, ledger, group))
                p1.write_json(output / 'health.json', health)
                checkpoint(output / 'last.pt', completed)
                p1.write_json(output / 'updates.json', records)
            if step in (500, 1000):
                model.eval()
                candidate = output / 'candidate_checkpoints' / f'updates_{step:06d}'
                candidate.mkdir(parents=True)
                checkpoint(candidate / 'model.pt', completed)
                raw, mu, g = full_prediction(model, inputs, cache, config, device, 128, 600002, ledger)
                score = p1.score_probe(raw, inputs['canonical']['truth_normalized'],
                    inputs['bank'].arrays['select']['thresholds_normalized'][inputs['china_indices']],
                    inputs['china_rows'], inputs['canonical']['observed'])
                p1.save_npz(candidate / 'select_predictions.npz', **prediction_payload(raw, mu, g, score, inputs))
                p1.write_json(candidate / 'scores.json', score['summary'])
                choice = dict(updates=step, score=score['selection_score'], draws=128, seed=600002,
                    checkpoint=str(candidate / 'model.pt'), checkpoint_sha256=sha256(candidate / 'model.pt'), epoch=step // 500)
                candidates.append(choice)
                if choice['score'] < selected['score']:  # Strict < retains earliest on ties, including zero.
                    selected = choice
                    shutil.copy2(candidate / 'model.pt', output / 'best.pt')
                fit_loss, fit_detail = native_loss_probe(model, inputs, cache, 'fit', np.arange(16), 700002, device, ledger)
                select_loss = selection_loss(model, inputs, cache, device, ledger)
                epoch = dict(model=model.name, role='fit_select', country='CN', epoch=step // 500, step=step,
                    train_loss=float(np.mean([r['train_loss'] for r in block])), fit_loss=fit_loss, select_loss=select_loss,
                    selection_metric=choice['score'], train_gaussian_nll=float(np.mean([r['gaussian_nll'] for r in block])),
                    train_denoise_mse=float(np.mean([r['denoise_mse'] for r in block])),
                    train_loss_definition=model.config['loss_definition'], fit_loss_definition=model.config['loss_definition'],
                    select_loss_definition=model.config['loss_definition'], selected=selected['updates'] == step,
                    train_loss_population='CN rows in accepted64 batch; per-step CN count normalized',
                    fit_loss_population='first16 mixed fit cache rows', select_loss_population='full1706 CN select',
                    state='measured', reason='CN-weighted training and mixed-fit16 diagnostic have different populations; select loss remains native')
                history.append(epoch); block = []
                p1.write_json(output / 'epochs.json', history)
                p1.write_json(output / 'selection.json', selected)
                p1.write_json(output / 'candidate_checkpoints.json', candidates)
            ledger.checkpoint(group + '_update_complete')
        progress[group]['selection_ended'] = True
        # last retains the final1000 fit state; best is the selected fresh gate state.
        checkpoint(output / 'last.pt', 1000)
        training_complete = True
        model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=False)['state_dict'])
        frozen = p1.all_a_key_contract(model, a_state)
        model.eval()
        if selected['updates'] == 0:
            a_folder = inputs['a_checkpoint'].parent
            shutil.copy2(inputs['a_predictions_path'], output / 'select_predictions.npz')
            with np.load(inputs['a_predictions_path'], allow_pickle=False) as archive:
                payload = {k:np.asarray(archive[k]) for k in archive.files}
            stored_scores = f.read_json(a_folder / 'scores.json')
            score = dict(summary=stored_scores, probability=payload['probability'], observed=inputs['canonical']['observed'],
                selection_score=float(np.mean([r['brier'] for r in stored_scores if r['country'] == 'CN'])))
            if not math.isclose(score['selection_score'], config['baseline_a_final_score_256'], abs_tol=1e-12, rel_tol=0):
                raise ValueError('reused A256 score differs')
            # Preserve exact accepted A conditions and their provenance.
            shutil.copy2(a_folder / 'select_china_conditions.npz', output / 'select_china_conditions.npz')
            reuse = dict(reused=True, source_predictions=str(inputs['a_predictions_path']),
                source_predictions_sha256=sha256(inputs['a_predictions_path']),
                source_conditions=str(a_folder / 'select_china_conditions.npz'),
                source_conditions_sha256=sha256(a_folder / 'select_china_conditions.npz'),
                source_scores=str(a_folder / 'scores.json'), source_scores_sha256=sha256(a_folder / 'scores.json'),
                source_checkpoint=str(inputs['a_checkpoint']), source_checkpoint_sha256=inputs['a_sha'],
                selection_score_128=config['baseline_a_selection_score_128'], final_score_256=score['selection_score'],
                reason='update0 selected; exact original A prediction/condition/score files reused; no full-A resampling or retained gate improvement')
            p1.write_json(output / 'reused_A_provenance.json', reuse)
        else:
            raw, mu, g = full_prediction(model, inputs, cache, config, device, 256, 900002, ledger)
            score = p1.score_probe(raw, inputs['canonical']['truth_normalized'],
                inputs['bank'].arrays['select']['thresholds_normalized'][inputs['china_indices']],
                inputs['china_rows'], inputs['canonical']['observed'])
            payload = prediction_payload(raw, mu, g, score, inputs)
            p1.save_npz(output / 'select_predictions.npz', **payload)
            p1.save_npz(output / 'select_china_conditions.npz', mu_normalized=mu, g_variance_normalized=g)
        p1.write_json(output / 'select_rows.json', inputs['china_rows'])
        p1.write_json(output / 'select_china_condition_rows.json', inputs['china_rows'])
        p1.write_json(output / 'scores.json', score['summary'])
        p1.write_json(output / 'phase_scores.json', subgroup_scores(score, inputs['china_rows'], inputs['canonical']['truth_q_m3_s']))
        p1.write_json(output / 'health.json', health)
        p1.write_json(output / 'epochs.json', history)
        progress[group]['final_scoring_ended'] = True
        interventions = selected_actions(model, selected, group, inputs, cache, phase1, output, device, ledger)
        report_group(model, inputs, output, history, health, interventions, payload, score, selected, config, ledger)
        progress[group]['report_ended'] = True
        p1.write_json(output / 'epochs.json', history)
        completion = dict(state='complete', group='D_china_state', gate_architecture=group,
            attempted_updates=ledger.attempts[group], completed_updates=completed,
            training_ended=True, selection_ended=True, final_scoring_ended=True, report_ended=True,
            selected_updates=selected['updates'], selection_score_128=selected['score'],
            final_brier=score['selection_score'], final_draws=256, final_seed=900002,
            final_predictions_sha256=sha256(output / 'select_predictions.npz'),
            best_checkpoint_sha256=sha256(output / 'best.pt'), last_checkpoint_sha256=sha256(output / 'last.pt'),
            all_A_state_dict_equal=True, frozen_contract=frozen, active_parameters=sum(p.numel() for p in active),
            active_parameter_tensors=len(active),
            training_monitor={'state':'disabled_no_external_sender'}, paired_training=pairing)
        p1.write_json(output / 'completion.json', completion)
        progress[group].update(state='complete', selected_updates=selected['updates'], final_brier=score['selection_score'])
        return completion
    except BaseException as error:
        if not training_complete:
            checkpoint(output / 'last.pt', completed)
        p1.write_json(output / 'updates.json', records)
        p1.write_json(output / 'epochs.json', history)
        p1.write_json(output / 'health.json', health)
        p1.write_json(output / 'candidate_checkpoints.json', candidates)
        p1.write_json(output / 'selection.json', selected)
        progress[group].update(state='partial' if isinstance(error, TimeoutError) else 'failed',
            attempted_updates=ledger.attempts[group], completed_updates=completed,
            selected_updates=selected['updates'])
        p1.write_json(output / 'completion.json', dict(**progress[group],
            error=dict(type=type(error).__name__, message=str(error)),
            training_monitor=monitor.finish('FAILED') if monitor else None))
        raise




def validate_config(c):
    expected = dict(change_id=CHANGE_ID, state='APPROVED_D_ONLY', seed=2, batch_size=64,
        formal_updates_per_group=1000, formal_attempt_cap_total=1000, learning_rate=.001,
        gradient_clip_norm=10., evaluation_batch_size=32, select_draws=128, final_draws=256,
        selection_seed=600002, final_seed=900002, gpu_budget_total_seconds=180.)
    if (any(c.get(k) != v for k, v in expected.items()) or c['groups'] != ['D_china_state']
            or c['condition_id'] != CHANGE_ID + '-FORMAL'):
        raise ValueError('D-only fixed schedule/budget/config required')
    if c['candidate_updates'] != [0,500,1000] or c['health_updates'] != [0,250,500,1000]:
        raise ValueError('fixed selection/health schedule required')
    # Reuse the original input contract without inheriting its old budget/group permissions.
    p1.validate_config({**c, 'change_id':p1.CHANGE_ID, 'condition_id':p1.CHANGE_ID+'-PHASE1',
        'state':'APPROVED_SMOKE', 'groups':list(p1.GROUPS), 'gpu_budget_total_seconds':1200.})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['preflight','formal'], required=True)
    parser.add_argument('--run-record', required=True)
    parser.add_argument('--preflight-receipt', help='required passed preflight JSON for formal')
    args = parser.parse_args()
    config = f.read_json(args.config)
    validate_config(config)
    output = Path(args.output).resolve()
    root = Path(config['output_root']).resolve()
    if output.exists() or root not in output.parents or Path(config['budget_receipt']).resolve() != root/'budget_receipt.json':
        raise ValueError('new output below the unique approved root and single budget receipt required')
    if args.mode == 'preflight' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preflight requires explicit CUDA_VISIBLE_DEVICES empty')
    if args.mode == 'formal':
        if not args.preflight_receipt:
            raise ValueError('formal requires --preflight-receipt after parent review')
        passed = f.read_json(args.preflight_receipt)
        if passed['state'] != 'passed' or passed['optimizer_attempts'] != 0:
            raise ValueError('zero-update passed preflight required')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not torch.cuda.is_available():
            raise ValueError('formal requires explicit physical CUDA0')
    fd = os.open(config['gpu_lock'], os.O_CREAT | os.O_RDWR, 0o600)
    ledger = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = Budget(config, args)
        output.mkdir(parents=True, exist_ok=False)
        device = torch.device('cpu' if args.mode == 'preflight' else 'cuda:0')
        torch.set_num_threads(config['cpu_threads'])
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        ledger.checkpoint('input_validation')
        p1.write_json(output/'condition.json', config)
        p1.write_json(output/'runtime.json', dict(**p1.runtime_identity(), current_run_versioned_record=args.run_record,
            config_sha256=sha256(args.config), mode=args.mode))
        receipt_path = Path(config['accepted_cache_receipt'])
        receipt = f.read_json(receipt_path)
        if (receipt['state'] != 'complete' or receipt['bank_manifest_sha256'] != config['expected_bank_sha256']
                or receipt['a_checkpoint_sha256'] != config['baseline_a_checkpoint_sha256']
                or receipt['c_checkpoint_sha256'] != config['baseline_c_checkpoint_sha256']
                or receipt['denoiser_outputs_cached'] or receipt['inverse_mixed_variance_used']
                or not receipt['gs_computed_directly_from_frozen_state_delta']):
            raise ValueError('accepted cache provenance differs')
        inputs = p1.load_inputs(config)
        cache = f.load_cache(receipt, inputs, ledger)
        p1.write_json(output/'cache_reuse_receipt.json', dict(cache_receipt=str(receipt_path),
            sha256=sha256(receipt_path), rebuilt=False, smoke_weights_loaded=False,
            full_select_rows=len(inputs['china_rows']), global_exposure_source=config['baseline_c_exposure']))
        if args.mode == 'preflight':
            preflight(inputs, cache, device, ledger, output)
            ledger.checkpoint('preflight_finalize')
            ledger.finish('preflight_passed')
        else:
            expected_preflight = [r for r in ledger.receipt['runs'] if r['mode']=='preflight' and r['state']=='preflight_passed']
            if len(expected_preflight) != 1 or Path(args.preflight_receipt).resolve() != Path(expected_preflight[0]['output'])/'preflight.json':
                raise ValueError('preflight must belong to this exact cumulative budget')
            progress = {GROUP:dict(state='not_started', attempted_updates=0, completed_updates=0,
                training_ended=False, selection_ended=False, final_scoring_ended=False, report_ended=False)}
            completion = train_group(GROUP, config, inputs, cache, receipt_path.parent, output/'D_china_state', device, ledger, progress)
            p1.write_json(output/'completion.json', completion)
            ledger.checkpoint('formal_finalize')
            ledger.finish('formal_complete_pending_parent_audit')
    except BaseException as error:
        if ledger is not None:
            p1.write_json(output/'failure.json', dict(state='partial' if isinstance(error,TimeoutError) else 'failed',
                error_type=type(error).__name__, message=str(error)))
            ledger.finish('partial' if isinstance(error,TimeoutError) else 'failed', error)
        raise
    finally:
        os.close(fd)


# Reuse accepted implementations, with unchanged native select loss and reporting schema.
batch_from_cache = f.batch_from_cache
health_snapshot = f.health_snapshot
full_prediction = f.full_prediction
prediction_payload = f.prediction_payload
native_loss_probe = f.native_loss_probe
selection_loss = f.selection_loss
subgroup_scores = f.subgroup_scores
selected_actions = f.selected_actions
def report_group(model, inputs, output, history, health, interventions, payload, score, selected, config, ledger):
    ledger.checkpoint(model.gate_variant + '_report')
    from src.training_report_v1.plot import render_report
    bank = inputs['bank']
    wanted = bank.manifest['report_case_ids']
    cases = []
    for index, row in enumerate(inputs['china_rows']):
        if row['sample_id'] not in wanted:
            continue
        physical = payload['draws_normalized'][index] * float(row['q_scale'])
        lo, median, hi = np.quantile(physical, [.05, .5, .95], axis=0)
        cases.append(dict(model=model.name, role='select', country='CN', station=row['station'],
            case_kind=bank.manifest['report_case_kinds'][row['sample_id']], case_id=row['sample_id'],
            x_kind='lead_hour_pointwise', x=list(range(1, 25)), origin_time=row['origin_time'], unit='m3/s',
            truth=inputs['canonical']['truth_q_m3_s'][index].tolist(), median=median.tolist(), lower=lo.tolist(), upper=hi.tolist()))
    if not cases or not health or not interventions:
        raise ValueError('formal report requires actual fixed cases, health and intervention rows')
    for row in history:
        row['selected'] = row['step'] == selected['updates']
    p1.write_json(output / 'epochs.json', history)
    bundle = dict(schema_version=1, target='pointwise_exceedance', run_id=output.parent.name + '_' + model.gate_variant,
        models=[dict(id=model.name, label='D中国适配', config=model.config)], epochs=history, cases=cases,
        risk=[dict(r, model=model.name) for r in score['summary']], health=health, interventions=interventions,
        metadata=dict(input_group='pq', forecast_hours=24, history_hours=24,
            loss_definition=model.config['loss_definition'], selection_country='CN',
            selection_label='完整1706中国select逐时Q95/Q99 Brier（越低越好）',
            training_countries=['CN'], loss_label='CN加权NLL+去噪MSE',
            train_loss_population='CN rows in accepted64 batch; each step divided by CN count',
            fit_loss_population='first16 mixed fit cache rows; native diagnostic mean',
            select_loss_population='full1706 CN select; unchanged native mean',
            loss_components=[('train_gaussian_nll', '--', '训练Gaussian NLL'), ('train_denoise_mse', ':', '训练去噪MSE')],
            preferred_cases=[dict(station=r['station'], role='select', case_kind=r['case_kind'], case_id=r['case_id']) for r in cases],
            reliability_horizon=6, reliability_target='pointwise_exceedance_Q95',
            scientific_state='single_seed_select_exploration', actual_model_forecasts=True,
            selected_updates=selected['updates'], selected0_reuses_A=selected['updates'] == 0,
            learning_curve_note='D CN-weighted fit training; native full CN select validation; distinct populations',
            probability_semantics='state at each future hour; no temporal cummax',
            physical_projection='normalized Q draws clipped at zero before scoring',
            data_manifest_sha256=bank.identity, fit_scale_identity=bank.manifest['fit_scale_identity'],
            sources={str(output / name): sha256(output / name) for name in
                ('select_predictions.npz', 'epochs.json', 'health.json', 'interventions.json')}))
    manifest = render_report(bundle, output / 'report')
    p1.write_json(output / 'report_receipt.json', dict(report_bundle=str(output / 'report/report_bundle.json'),
        pages=len(manifest['pages']), health_rows=len(health), intervention_rows=len(interventions)))




if __name__ == '__main__':
    main()
