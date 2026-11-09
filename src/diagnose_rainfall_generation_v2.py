"""CPU-only descriptive audit of already saved joint windows; never samples."""
import argparse
import csv
import gzip
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def dump(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def finite(value):
    return float(value) if np.isfinite(value) else None


def quantiles(values):
    a = np.asarray(values); a = a[np.isfinite(a)]
    return dict(n=len(a), mean=finite(a.mean()) if len(a) else None,
                **{f'p{q}': finite(np.percentile(a, q)) if len(a) else None for q in [5, 25, 50, 75, 95]})


def weighted_median(a, weights):
    ok = np.isfinite(a) & (weights > 0)
    a, weights = a[ok], weights[ok]
    if not len(a): return None
    order = np.argsort(a, kind='stable'); a, weights = a[order], weights[order]
    return finite(a[np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def metrics(raw, q95, scale):
    p, q = raw[:, 0], raw[:, 1]; dq = np.diff(q, axis=1) / scale
    qp, pp = q.argmax(1), p.argmax(1); rainy = p.sum(1) > 1e-6
    high = q[:, 12:] > q95; longest = np.zeros(len(q)); running = np.zeros(len(q))
    for col in high.T:
        running = np.where(col, running + 1, 0); longest = np.maximum(longest, running)
    d2 = np.abs(np.diff(dq, axis=1))
    rise, fall = [], []
    for i, peak in enumerate(qp):
        rise.append(d2[i, :max(0, peak - 1)].mean() if peak >= 2 else np.nan)
        fall.append(d2[i, peak:].mean() if peak <= 21 else np.nan)
    centroid = (p * np.arange(24)).sum(1) / np.maximum(p.sum(1), 1e-12)
    result = dict(p_total=p.sum(1), p_history=p[:, :12].sum(1), q_initial=q[:, 0],
        q_future_peak=q[:, 12:].max(1), q_future_peak_hour=q[:, 12:].argmax(1),
        q_global_peak=q.max(1), high95_hours=high.sum(1), high95_longest_run=longest,
        q_total_variation=np.abs(dq).sum(1), q_max_step=np.abs(dq).max(1),
        q_roughness=d2.mean(1), q_rise_roughness=np.array(rise), q_recession_roughness=np.array(fall),
        q_turns=((dq[:, :-1] * dq[:, 1:] < 0) & (np.abs(dq[:, :-1]) > .01) & (np.abs(dq[:, 1:]) > .01)).sum(1),
        pq_argmax_lag=np.where(rainy, qp - pp, np.nan), pq_centroid_lag=np.where(rainy, qp - centroid, np.nan),
        no_rain=(~rainy).astype(float), p_peak_boundary=((pp == 0) | (pp == 23)).astype(float),
        q_peak_boundary=((qp == 0) | (qp == 23)).astype(float),
        pq_interior_argmax_lag=np.where(rainy & (pp > 0) & (pp < 23) & (qp > 0) & (qp < 23), qp - pp, np.nan))
    for phase, sl in [('history', slice(0, 12)), ('future', slice(12, 24))]:
        d = np.diff(q[:, sl], axis=1) / scale
        result[f'q_{phase}_roughness'] = np.abs(np.diff(d, axis=1)).mean(1)
        result[f'q_{phase}_max_step'] = np.abs(d).max(1)
        result[f'p_{phase}_total'] = p[:, sl].sum(1)
        result[f'p_{phase}_roughness'] = np.abs(np.diff(p[:, sl], n=2, axis=1)).mean(1)
    return result


def nearest(a, b, events_a=None, events_b=None, self_distance=False):
    """Bounded batches; real references optionally exclude the entire same event."""
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    distances, indices = [], []
    for start in range(0, len(a), 64):
        # ponytail: exact quadratic distances are fine for these saved small pools.
        d = np.maximum((a[start:start+64] ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a[start:start+64] @ b.T, 0)
        if events_a is not None: d[np.asarray(events_a[start:start+64])[:, None] == np.asarray(events_b)[None]] = np.inf
        if self_distance:
            rows = np.arange(min(64, len(a)-start)); d[rows, start + rows] = np.inf
        idx = d.argmin(1); distances.extend(np.sqrt(d[np.arange(len(idx)), idx] / a.shape[1])); indices.extend(idx)
    return np.array(distances), np.array(indices)


def strata(m, thresholds):
    return (m['q_future_peak_hour'] // 4) * 3 + np.searchsorted(thresholds, m['q_future_peak'], side='right')


def self_check():
    raw = np.zeros((1, 2, 24)); raw[0, 0, 3] = 2; raw[0, 1, 14:17] = [2, 3, 2]
    m = metrics(raw, 1, 1)
    assert m['pq_argmax_lag'][0] == 12 and m['high95_longest_run'][0] == 3
    assert m['high95_hours'][0] == 3 and m['q_future_peak_hour'][0] == 3
    d, i = nearest(np.array([[[0., 1.]]]), np.array([[[0., 1.]], [[2., 3.]]]))
    assert d[0] == 0 and i[0] == 0
    assert weighted_median(np.array([1., 2., 3.]), np.array([1., 2., 1.])) == 2


def diagnose(args):
    import torch
    torch.set_num_threads(1)
    source_commit = subprocess.check_output(['git', '-C', str(Path(args.source).parents[3]), 'rev-parse', 'HEAD'], text=True).strip()
    assert source_commit == '374fdabee6a0fb757d72162a36b9033e862572fe'
    started = time.monotonic(); out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    targets = {}; curve_candidates = {}
    for folder in sorted(Path(args.data_root).iterdir()):
        if not (folder / 'training_transforms.json').is_file(): continue
        t = folder.name; meta = json.loads((folder / 'training_transforms.json').read_text())
        with gzip.open(folder / 'train.jsonl.gz', 'rt') as handle: rows = [json.loads(line) for line in handle]
        joint = np.array([[r['x_p'] + r['offline_generation_future_p'], r['x_q'] + r['y_q']] for r in rows])
        highidx = np.array([i for i, r in enumerate(rows) if r['generation_high95']])
        real = joint[highidx]; highrows = [rows[i] for i in highidx]
        source = Path(args.source) / t / 'sample'; tensor = torch.load(source / 'synthetic.pt', map_location='cpu', weights_only=True)
        assert tensor.ndim == 3 and tuple(tensor.shape[1:]) == (2, 24) and len(tensor) > 0
        means = [meta['scales'][k]['mean'] for k in ['p_mm_h', 'q_m3_s']]
        scales = [meta['scales'][k]['scale'] for k in ['p_mm_h', 'q_m3_s']]
        synth_raw32 = tensor * tensor.new_tensor(scales)[None, :, None] + tensor.new_tensor(means)[None, :, None]
        assert torch.isfinite(synth_raw32).all() and (synth_raw32 >= 0).all()
        assert (synth_raw32[:, 1, 12:] > meta['q95_m3_s']).any(1).all()
        synthetic = tensor.numpy().astype(np.float64); sr = synth_raw32.numpy().astype(np.float64)
        rr = real * np.array(scales)[None, :, None] + np.array(means)[None, :, None]
        rm, sm = [metrics(a, meta['q95_m3_s'], scales[1]) for a in [rr, sr]]
        thresholds = np.quantile(rm['q_future_peak'], [1/3, 2/3]); rc, sc = [strata(m, thresholds) for m in [rm, sm]]
        common = sorted(set(rc) & set(sc)); rw, sw = np.zeros(len(real)), np.zeros(len(synthetic)); cells = []
        for cell in range(9):
            ri, si = np.flatnonzero(rc == cell), np.flatnonzero(sc == cell)
            cells.append(dict(cell=cell, real=len(ri), synthetic=len(si)))
            if cell in common:
                rw[ri] = 1.; sw[si] = len(ri) / len(si)
        ns, ni = nearest(synthetic, real); na, ai = nearest(synthetic, joint)
        ev = [r['event_uid'] for r in highrows]; nr, _ = nearest(real, real, ev, ev)
        nd, _ = nearest(synthetic, synthetic, self_distance=True)
        copy_max = np.abs(synthetic - real[ni]).reshape(len(synthetic), -1).max(1)
        assignments = {event: int(sum(ev[i] == event for i in ni)) for event in sorted(set(ev))}
        shares = np.array(list(assignments.values())) / len(synthetic); nonzero = shares[shares > 0]
        status = json.loads((source / 'status.json').read_text())
        assert len(synthetic) == status['accepted']
        assert json.loads((source / 'training_transforms.json').read_text()) == meta
        record = dict(target=t, n_train=len(rows), n_real_high95=len(real), n_synthetic=len(synthetic),
            n_real_events=len(set(ev)), n_train_events=len(set(r['event_uid'] for r in rows)),
            n_real_unique_source_rows=len(set((r['event_uid'], x) for r in highrows for x in r['source_rows'])),
            synthetic_independent_events='unknown', effective_independent_n='unknown', original_status=status,
            transform=meta, inputs=[str(folder / 'train.jsonl.gz'), str(folder / 'training_transforms.json'), str(source / 'synthetic.pt'), str(source / 'status.json'), str(source / 'training_transforms.json')],
            shared_support_real_fraction=float((rw > 0).mean()), shared_support_synthetic_fraction=float((sw > 0).mean()),
            peak_tertile_thresholds=thresholds.tolist(), cells=cells,
            metrics={k: dict(real=quantiles(rm[k]), synthetic=quantiles(sm[k]),
                adjusted_real_median=weighted_median(rm[k], rw), adjusted_synthetic_median=weighted_median(sm[k], sw)) for k in rm},
            real_event_medians={event:{k:quantiles(rm[k][np.array(ev) == event])['p50'] for k in rm} for event in sorted(set(ev))},
            distance=dict(synthetic_real_high95=quantiles(ns), synthetic_all_train=quantiles(na),
                nearest_all_train_high95_fraction=float(np.mean([rows[i]['generation_high95'] for i in ai])),
                real_leave_event_out=quantiles(nr), synthetic_other_synthetic=quantiles(nd),
                numerical_near_copy_count=int(((ns <= 1e-6) & (copy_max <= 1e-5)).sum()),
                exact_unique_synthetic=len(np.unique(synthetic.reshape(len(synthetic), -1), axis=0)),
                nearest_event_assignments=assignments, nearest_event_coverage=sum(x > 0 for x in assignments.values()),
                nearest_event_entropy_effective=float(np.exp(-np.sum(nonzero * np.log(nonzero))))) )
        targets[t] = record
        examples = []
        for quantile in [.25, .5, .75]:
            wanted = np.quantile(sm['q_future_peak'], quantile); si = int(np.argmin(np.abs(sm['q_future_peak']-wanted)))
            candidates = np.flatnonzero(rc == sc[si]); fallback = not len(candidates)
            if fallback: candidates = np.arange(len(real))
            _, idx = nearest(synthetic[si:si+1], real[candidates]); ri = int(candidates[idx[0]])
            examples.append(dict(peak_quantile=quantile, synthetic_index=si, real_train_index=int(highidx[ri]),
                event_uid=highrows[ri]['event_uid'], origin_source_time=highrows[ri]['origin_source_time'],
                source_rows=highrows[ri]['source_rows'], stratum=int(sc[si]), real_stratum=int(rc[ri]), fallback=fallback,
                real_raw=rr[ri].tolist(), synthetic_raw=sr[si].tolist(),
                metric_excerpt={k:dict(real=finite(rm[k][ri]), synthetic=finite(sm[k][si])) for k in rm}))
        curve_candidates[t] = examples
        print(t, len(real), len(synthetic), 'shape medians', rm['q_roughness'].mean(), sm['q_roughness'].mean(), flush=True)
        if time.monotonic() - started > 540: raise TimeoutError('CPU diagnostic budget exceeded')
    ordered = sorted(targets, key=lambda t:(targets[t]['n_synthetic'], t)); chosen = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    curves = dict(selection='minimum, median, maximum saved accepted pool size; within-target future peak quartile examples', targets={t:curve_candidates[t] for t in chosen})
    dump(out / 'curve_examples.json', curves)
    report = dict(state='complete', execution_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        source_generation_commit='374fdabee6a0fb757d72162a36b9033e862572fe', elapsed_seconds=time.monotonic()-started, targets=targets)
    dump(out / 'diagnosis.json', report)
    columns = ['target','n_train','n_real_high95','n_real_events','n_synthetic','shared_support_real_fraction','shared_support_synthetic_fraction']
    selected_metrics = ['p_total','q_future_peak','high95_hours','q_roughness','q_max_step','pq_interior_argmax_lag','no_rain','q_peak_boundary']
    columns += [f'{m}_{s}' for m in selected_metrics for s in ['real_median','synthetic_median','adjusted_real_median','adjusted_synthetic_median']]
    columns += ['nn_real_median','real_leave_event_out_median','nn_synthetic_median','near_copy_count','nearest_event_coverage']
    with (out / 'summary.csv').open('x', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader()
        for record in targets.values():
            row={k:record[k] for k in columns if k in record}
            for m in selected_metrics:
                x=record['metrics'][m]; row.update({f'{m}_real_median':x['real']['p50'],f'{m}_synthetic_median':x['synthetic']['p50'],f'{m}_adjusted_real_median':x['adjusted_real_median'],f'{m}_adjusted_synthetic_median':x['adjusted_synthetic_median']})
            d=record['distance']; row.update(nn_real_median=d['synthetic_real_high95']['p50'],real_leave_event_out_median=d['real_leave_event_out']['p50'],nn_synthetic_median=d['synthetic_other_synthetic']['p50'],near_copy_count=d['numerical_near_copy_count'],nearest_event_coverage=d['nearest_event_coverage']); writer.writerow(row)
    dump(out / 'acceptance.json', dict(state='execution_valid_descriptive_only', execution_commit=report['execution_commit'],
        source_generation_commit=report['source_generation_commit'], train_only=True, no_sampling=True, no_training=True,
        cpu_threads=1, definitions_spec=str(Path('configs/rainfall_generation_diagnosis_v2.md').resolve()),
        metric_definitions=Path('configs/rainfall_generation_diagnosis_v2.md').read_text(),
        targets={t:{k:r[k] for k in ['inputs','n_train','n_real_high95','n_real_events','n_synthetic','transform','synthetic_independent_events','effective_independent_n']} for t,r in targets.items()},
        excerpts=curves, limits=['24h windows, overlapping real windows, unknown synthetic event independence','argmax/centroid lag is not a violation test','only future peak position/strength adjusted; historical state remains unadjusted','nearest event means resemblance, not origin or proof of copying','no downstream predictive efficacy measured']))


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(args.render).read_text()); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    for target, examples in data['targets'].items():
        fig, axes=plt.subplots(3,2,figsize=(7.1,7),sharex=True,layout='constrained')
        for i,e in enumerate(examples):
            for j, label in enumerate(['P (mm/h)','Q (m³/s)']):
                ax=axes[i,j]
                ax.plot(range(24),e['real_raw'][j],label='Real train',color='#275D80',lw=1.4)
                ax.plot(range(24),e['synthetic_raw'][j],label='Saved synthetic',color='#B24C32',lw=1.4)
                ax.axvline(11.5,color='grey',ls=':',lw=.8); ax.set_ylabel(label)
                if j==0: ax.set_title(f"Synthetic peak quantile {e['peak_quantile']:.2f}",fontsize=9)
                else: ax.set_title(f"Nearest real event {e['event_uid'].split('_')[-1]} / row {e['source_rows'][0]}",fontsize=9)
                if i==2: ax.set_xlabel('Hour within saved 24h window')
                ax.spines[['top','right']].set_visible(False)
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle(target+' — representative pairs, not independent events',fontsize=11)
        for ext in ['pdf','png']:
            path=out/(target+'_curves.'+ext)
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path,dpi=300)
        plt.close(fig)
    dump(out/'render_receipt.json',dict(state='complete',targets=list(data['targets']),source=str(Path(args.render).resolve()),matplotlib_version=matplotlib.__version__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root'); parser.add_argument('--source'); parser.add_argument('--out',required=True); parser.add_argument('--render')
    args=parser.parse_args(); self_check()
    if args.render: render(args)
    else: diagnose(args)


if __name__=='__main__': main()
