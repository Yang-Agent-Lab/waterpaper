"""Post-run comparison from saved select outputs only; no models, sampling or optimization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.d500_process_adapter_v1.scoring import score
from src.d500_process_adapter_v1.reference_scoring import phases, weights


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def phase_probability(p, rows, truth, observed):
    keys, masks, support = phases(rows, truth, observed)
    result = {}
    for qi, q in enumerate((95, 99)):
        for phase in ('entry_first3', 'positive', 'negative'):
            means = []
            for lead in range(24):
                ix = np.flatnonzero(masks[f'q{q}_{phase}'][:, lead])
                if len(ix):
                    means.append(float(weights([keys[i] for i in ix]) @ p[ix, lead, qi]))
            result[f'q{q}_{phase}_mean_probability'] = float(np.mean(means)) if means else None
    return result, support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal-repo', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    began = time.monotonic()
    repo, out = Path(args.formal_repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'figures').mkdir()
    cfg = json.loads((repo / 'configs/d500_process_adapter_v1.json').read_text())
    run = repo / 'results/experiments/d500_process_adapter_v1_20261022_G1'
    final = run / 'finalize/artifacts'
    done = json.loads((final / 'completion.json').read_text())
    queue = json.loads((run / 'queue_run/artifacts/queue_status.json').read_text())
    assert queue['state'] == 'complete'
    rows = json.loads((Path(cfg['data_root']) / 'select_rows.json').read_text())
    old_dir = Path(cfg['checkpoint']).parent
    with np.load(old_dir / 'select_predictions.npz', allow_pickle=False) as z:
        assert z['sample_id'].tolist() == [r['sample_id'] for r in rows]
        truth = z['truth_q_m3_s'].copy()
        threshold = z['thresholds_q95_q99_m3_s'].copy()
        observed = z['observed'].copy()
        scale = np.asarray([r['q_scale'] for r in rows])
        draws = z['draws_normalized'].astype(np.float64) * scale[:, None, None] / threshold[:, 0, None, None]
        old_p = z['probability'].copy()
    common_d0 = score(draws, truth / threshold[:, 0, None], threshold / threshold[:, :1], rows,
                      truth, q95_physical=threshold[:, 0])
    old_means, support = phase_probability(old_p, rows, truth, observed)
    entries = {'D500': dict(summary=common_d0['summary'], constraints=common_d0['constraints'],
                           phase=common_d0['phase'], phase_mean_probability=old_means)}
    for selection in done['selections']:
        v = selection['variant']
        m = json.loads((final / v / 'metrics.json').read_text())
        with np.load(final / v / 'select_predictions.npz', allow_pickle=False) as z:
            assert z['sample_id'].tolist() == [r['sample_id'] for r in rows]
            np.testing.assert_array_equal(z['truth_q_m3_s'], truth)
            p = z['probability'].copy()
        means, _ = phase_probability(p, rows, truth, observed)
        warning = json.loads((final / v / 'fixed_warning.json').read_text())['summary']
        warning['mean_lead_hours_q95_q99'] = [float(np.mean(x)) if x else None for x in warning['lead_hours']]
        entries[v] = dict(summary=m['summary'], constraints=m['constraints'], phase=m['phase'],
                          phase_mean_probability=means, fixed_warning=warning,
                          selected=selection['selected'], decision=selection['decision'],
                          checkpoint_sha256=m['checkpoint_sha256'])
    fields = ['brier', 'crps_q95', 'q95_entry_first3_brier', 'q99_entry_first3_brier']
    improvements = {v: {k: (entries['D500']['constraints'][k] - entries[v]['constraints'][k]) /
                              entries['D500']['constraints'][k] * 100 for k in fields} for v in ('T', 'M', 'A')}
    histories, updates, failed = [], 0, {}
    prep = Path('./results/experiments/d500_adapter_prepare_20261022/artifacts')
    reference = json.loads((prep / 'D0_K128_constraints.json').read_text())['constraints']
    tolerance = json.loads((prep / 'mc_tolerance.json').read_text())['delta']
    for job in queue['jobs']:
        folder = run / job['id'] / 'artifacts'
        completion = json.loads((folder / 'completion.json').read_text())
        updates += completion['completed_updates']
        h = json.loads((folder / 'validation_history.json').read_text())
        for item in h:
            failures = [k for k in reference if item['constraints'][k] > reference[k] + tolerance[k]]
            for k in failures:
                failed[k] = failed.get(k, 0) + 1
            histories.append(dict(job=job['id'], update=item['update'], eligible=item['eligible'], failed=failures))
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(4)
    colors = ['#4477AA', '#EE9933', '#228866']
    for i, (v, label) in enumerate(zip(('T', 'M', 'A'), ('T: TCN adapter', 'M: homogeneous MoE', 'A: heterogeneous experts'))):
        values = [improvements[v][k] for k in fields]
        bars = ax.bar(x + (i - 1) * .25, values, width=.23, color=colors[i], label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + (.12 if value >= 0 else -.12),
                    f'{value:+.1f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks(x, ['Overall Brier', 'CRPS / Q95', 'Q95 onset Brier', 'Q99 onset Brier'])
    ax.set_ylabel('Error reduction vs D500 (%)')
    ax.set_ylim(-6.2, 8.5)
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / 'figures/main_tradeoff.png', dpi=220)
    fig.savefig(out / 'figures/main_tradeoff.pdf')
    plt.close(fig)
    shutil.copy2(final / 'comparison.csv', out / 'comparison_original_full.csv')
    comparison = []
    for v in ('D500', 'T', 'M', 'A'):
        for k in fields:
            comparison.append(dict(model=v, metric=k, value=entries[v]['constraints'][k],
                                   improvement_percent_vs_D500=0 if v == 'D500' else improvements[v][k]))
    with (out / 'core_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    result = dict(state='SAVED_RESULTS_ANALYSIS_COMPLETE', source_commit=queue['source_commit'],
                  queue_started_utc=queue['started_at_utc'], queue_finished_utc=queue['finished_at_utc'],
                  wall_seconds=queue['wall_seconds'], formal_updates=updates, completed_trials=len(queue['jobs']),
                  checked_validation_points=len(histories), eligible_validation_points=sum(x['eligible'] for x in histories),
                  failed_constraint_counts=failed, histories=histories, entries=entries,
                  improvements_percent=improvements, onset_support=support,
                  scope='1706 exposed CN select origins,9 stations,17 source units,K256,single seed2; descriptive rejected fallback',
                  no_new_model_forward=True, no_new_forecast_samples=True, no_new_optimizer_updates=True,
                  analysis_wall_seconds=time.monotonic()-began,
                  inputs={'formal_completion':sha(final/'completion.json'),'D500_predictions':sha(old_dir/'select_predictions.npz')})
    save(out / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ['state','wall_seconds','formal_updates','completed_trials','checked_validation_points','eligible_validation_points','failed_constraint_counts','improvements_percent','analysis_wall_seconds']}, indent=2))
    print(json.dumps({v: {'joint': {k: e['summary'].get(k) for k in ['energy_score_q95','peak6_coverage90','total_variation_q95','truth_total_variation_q95']}, 'phase_mean_probability':e['phase_mean_probability']} for v,e in entries.items()}, indent=2))


if __name__ == '__main__':
    main()
