"""Read-only formal-result report; run on the server, never in the local workspace."""
import argparse
import gzip
import json
import math
import re
import statistics
from pathlib import Path

EXECUTION_COMMIT = 'bc9464995ac3fa3806f6e03004304c0d12f87ec9'
ARMS = ('A', 'B', 'C', 'D', 'persistence')


def read_json(path):
    return json.loads(path.read_text())


def matched_predictions(path, wanted):
    """Scan identifiers only; deserialize complete rows only for selected windows."""
    identifier = re.compile(r'"window_id"\s*:\s*("(?:\\.|[^"\\])*")')
    found = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            match = identifier.search(line)
            if match and json.loads(match.group(1)) in wanted:
                row = json.loads(line)
                found[row['window_id']] = row
                if len(found) == len(wanted):
                    break
    if set(found) != wanted:
        raise ValueError(f'Missing selected windows in {path}: {wanted-set(found)}')
    return found


def collect(run_dir):
    record = read_json(run_dir/'run_record.json')
    if record['head'] != EXECUTION_COMMIT or record['exit_code'] != 0:
        raise ValueError('Require the completed, accepted bc94649 formal execution')
    artifacts = run_dir/'artifacts'
    config = read_json(artifacts/'phase_config.json')
    seeds = config['seeds']
    if set(seeds) != {2, 17, 42} or config['smoke_run']:
        raise ValueError('Require the approved three-seed formal run')
    china = [r for r in read_json(artifacts/'country_seed_summary.json') if r['country'] == 'China']
    for arm in ARMS:
        if sorted(r['seed'] for r in china if r['arm'] == arm) != sorted(seeds):
            raise ValueError(f'Incomplete China country summary for {arm}')
    stations, events = {}, {}
    country_leads, china_lead6 = [], []
    for seed in seeds:
        for arm in ARMS:
            folder = artifacts/f'seed_{seed}'/arm
            country_leads.extend(dict(r, seed=seed, arm=arm) for r in read_json(folder/'country_lead_metrics.json'))
            rows = [r for r in read_json(folder/'event_peak_scores.json') if r['country'] == 'China' and r['lead'] == 6]
            china_lead6.extend(dict(r, seed=seed, arm=arm) for r in rows)
            if arm not in ('C', 'D'):
                continue
            stations[seed, arm] = {r['station_uid']:r for r in read_json(folder/'primary_metrics.json')['stations'] if r['country'] == 'China'}
            events[seed, arm] = {(r['station_uid'], r['event_group']):r for r in rows}
            if len(events[seed, arm]) != len(rows):
                raise ValueError('Expected one lead6 record per station/event')
    station_ids = set(stations[seeds[0], 'C'])
    if any(set(rows) != station_ids for rows in stations.values()):
        raise ValueError('China station pairing differs across seeds/arms')
    station_differences = [dict(station_uid=uid,
                                seed_differences={str(s):stations[s, 'D'][uid]['window_peak_error']-stations[s, 'C'][uid]['window_peak_error'] for s in seeds},
                                mean_difference=statistics.mean(stations[s, 'D'][uid]['window_peak_error']-stations[s, 'C'][uid]['window_peak_error'] for s in seeds))
                           for uid in sorted(station_ids)]
    event_keys = set.intersection(*(set(rows) for rows in events.values()))
    paired = []
    for key in sorted(event_keys):
        windows = {events[s, arm][key]['window_id'] for s in seeds for arm in ('C', 'D')}
        if len(windows) != 1:
            raise ValueError('Lead6 event window differs across seeds/arms')
        differences = {str(s):events[s, 'D'][key]['ape']-events[s, 'C'][key]['ape'] for s in seeds}
        paired.append(dict(station_uid=key[0], event_group=key[1], window_id=windows.pop(),
                           seed_ape_differences_pp=differences, mean_ape_difference_pp=statistics.mean(differences.values())))
    paired.sort(key=lambda row: (row['mean_ape_difference_pp'], row['station_uid'], row['event_group']))
    if len(paired) < 3:
        raise ValueError('Need at least three distinct fully paired lead6 events')
    cases = [dict(paired[0], case='最大改善' if paired[0]['mean_ape_difference_pp'] < 0 else '最小退步'),
             dict(paired[len(paired)//2], case='居中'),
             dict(paired[-1], case='最大退步' if paired[-1]['mean_ape_difference_pp'] > 0 else '最小改善')]
    wanted = {r['window_id'] for r in cases}
    if len(wanted) != 3:
        raise ValueError('Selected cases must have three distinct windows')
    predictions = {arm:matched_predictions(artifacts/f'seed_2/{arm}/validation_predictions.jsonl.gz', wanted) for arm in ('B', 'C', 'D')}
    for case in cases:
        rows = {arm:predictions[arm][case['window_id']] for arm in predictions}
        base = rows['B']
        if any(row['observed_q'] != base['observed_q'] or row['qscale'] != base['qscale'] or row['q_now_normalized'] != base['q_now_normalized'] for row in rows.values()):
            raise ValueError('Case truth/current Q differs across arms')
        series = dict(observed_q=base['observed_q'], predicted_q={arm:row['predicted_q'] for arm,row in rows.items()},
                      q_now=base['q_now_normalized']*base['qscale'], qscale=base['qscale'], curve_seed=2)
        for values in [series['observed_q'], *series['predicted_q'].values()]:
            if len(values) != 12 or not all(math.isfinite(v) for v in values):
                raise ValueError('Case requires 12 finite, unclipped physical-Q values')
        case['series'] = series
    folder = artifacts/'seed_2/D'
    epochs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
    selection = read_json(folder/'selection.json')
    if selection['epoch'] not in {r['epoch'] for r in epochs}:
        raise ValueError('Selected D epoch is absent from training history')
    access = read_json(artifacts/'model_input_access.json')
    return dict(execution_commit=EXECUTION_COMMIT, source_run=str(run_dir), seeds=seeds,
                counts={k:access[k] for k in ('train_windows', 'validation_windows')}, china_stations=len(station_ids),
                country_metrics=china, country_lead_metrics=country_leads, china_old_lead6_scores=china_lead6,
                station_D_minus_C=station_differences,
                seed2_D_training=[dict(epoch=r['epoch'], train_sequence_loss=r['train_sequence_loss'], validation_sequence_mae=r['validation']['sequence_mae']) for r in epochs],
                seed2_D_selected_epoch=selection['epoch'], paired_lead6_events=paired, cases=cases,
                case_selection='China lead6 station/event paired mean APE(D-C) over all three seeds; minimum, median rank, maximum',
                curve_scope='fixed seed2; retrospective peak-minus-6h diagnostic, not operational trigger or ensemble',
                physical_Q_unit='m³/s', normalized_unit='dimensionless', APE_difference_unit='percentage points',
                primary_weighting='source event mean then station equal; China saved validation origins',
                persistence_timing='earliest argmax for a flat persistence forecast; computational convention only',
                source_files=[str(artifacts/'country_seed_summary.json'), str(artifacts/'model_input_access.json'),
                              *[str(artifacts/f'seed_{s}/{a}'/name) for s in seeds for a in ARMS for name in ('country_lead_metrics.json', 'event_peak_scores.json')],
                              *[str(artifacts/f'seed_{s}/{a}'/name) for s in seeds for a in ('C', 'D') for name in ('primary_metrics.json', 'event_peak_scores.json')],
                              *[str(artifacts/f'seed_2/{a}/validation_predictions.jsonl.gz') for a in ('B', 'C', 'D')],
                              str(folder/'epochs.jsonl'), str(folder/'selection.json')])


def draw(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    candidates = [f for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if not candidates:
        raise RuntimeError('Noto Sans CJK font is required in the server plotting environment')
    font_manager.fontManager.addfont(candidates[0].fname)
    plt.rcParams.update({'font.family': candidates[0].name, 'font.size': 10, 'axes.unicode_minus': False})
    colors = {'A':'#777777', 'B':'#0072B2', 'C':'#E69F00', 'D':'#009E73', 'persistence':'#999999'}
    figures = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.12, right=.97, bottom=.17, top=.87, wspace=.35, hspace=.48)
    for ax, metric, title, unit in ((axes[0, 0], 'window_peak_error', '中国验证窗口峰值误差', '归一化绝对误差（无量纲，越低越好）'),
                                     (axes[0, 1], 'timing_abs_hours', '中国验证窗口峰时误差', '绝对峰时误差（小时，越低越好）')):
        for i, arm in enumerate(ARMS):
            values = [next(r[metric] for r in data['country_metrics'] if r['arm'] == arm and r['seed'] == s) for s in data['seeds']]
            offsets = [-.12, 0, .12]
            ax.scatter([i+j for j in offsets], values, color=colors[arm], s=32, zorder=3)
            ax.plot([i-.24, i+.24], [statistics.mean(values)]*2, color=colors[arm], linewidth=2.5)
        ax.set_xticks(range(len(ARMS)), ['A', 'B', 'C', 'D', '持续性'])
        ax.set(title=title, ylabel=unit)
        ax.grid(axis='y', alpha=.2)
    station_rows = sorted(data['station_D_minus_C'], key=lambda r:r['mean_difference'])
    ax = axes[1, 0]
    ax.barh([r['station_uid'].split(':', 1)[-1] for r in station_rows], [r['mean_difference'] for r in station_rows],
            color=['#009E73' if r['mean_difference'] < 0 else '#D55E00' for r in station_rows])
    ax.axvline(0, color='black', linewidth=.8)
    ax.invert_yaxis()
    ax.set(title='各中国站：D−C 三种子平均差', xlabel='归一化窗口峰值误差差值（负值为D更好）')
    ax.grid(axis='x', alpha=.2)
    ax = axes[1, 1]
    rows = data['seed2_D_training']
    ax.plot([r['epoch'] for r in rows], [r['train_sequence_loss'] for r in rows], label='多国加权训练', color='#0072B2')
    ax.plot([r['epoch'] for r in rows], [r['validation_sequence_mae'] for r in rows], label='中国验证', color='#D55E00')
    ax.axvline(data['seed2_D_selected_epoch'], color='#555555', linestyle='--', label=f"选中轮次 {data['seed2_D_selected_epoch']}")
    ax.set(title='D / seed2 训练过程', xlabel='训练轮次', ylabel='序列绝对误差（无量纲）')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.suptitle('三支路正式验证：峰值、峰时与过程的取舍', fontsize=16, y=.97)
    fig.text(.5, .92, f"{len(data['seeds'])} 个种子；{data['china_stations']} 个中国站；事件内平均后站点等权；点为单种子，横线为均值", ha='center', fontsize=10)
    fig.text(.5, .055, '持续性的平峰时刻采用最早最大值，仅为计算约定，不代表它能预测峰时。', ha='center', fontsize=10)
    fig.text(.5, .023, f"来源：已保存真实验证窗口；执行 {EXECUTION_COMMIT[:7]}；验证集用于开发与选模", ha='center', fontsize=9)
    figures.append(('page_1_summary', fig))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.subplots_adjust(left=.10, right=.97, bottom=.16, top=.82, hspace=.52)
    for ax, case in zip(axes, data['cases']):
        series = case['series']
        observed = [series['q_now'], *series['observed_q']]
        ax.plot(range(13), observed, color='black', marker='o', markersize=3, label='实测')
        for arm in ('B', 'C', 'D'):
            ax.plot(range(13), [series['q_now'], *series['predicted_q'][arm]], color=colors[arm], label=arm)
        true_peak = max(range(12), key=lambda i:series['observed_q'][i])
        d_peak = max(range(12), key=lambda i:series['predicted_q']['D'][i])
        ax.scatter(true_peak+1, series['observed_q'][true_peak], color='black', marker='*', s=95, zorder=5, label='实测窗口峰')
        ax.scatter(d_peak+1, series['predicted_q']['D'][d_peak], color=colors['D'], marker='D', s=45, zorder=5, label='D窗口峰')
        ax.axvline(0, linestyle=':', color='#777777')
        ax.set_xticks(range(13))
        ax.set(xlim=(-.3, 12.3), xlabel='相对起报时间（小时；0为起报Q）', ylabel='流量（m³/s）',
               title=f"{case['case']}｜{case['station_uid'].split(':', 1)[-1]} / {case['event_group'].split(':')[-1]}｜三种子峰值APE差 D−C = {case['mean_ape_difference_pp']:+.2f} 百分点")
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .895), ncol=6, fontsize=9)
    fig.suptitle('真实洪峰案例：改善、普通与退步均保留', fontsize=16, y=.975)
    fig.text(.5, .93, '事后峰前6小时诊断；案例按三种子 D−C 峰值APE平均差选取；曲线均为固定 seed2', ha='center', fontsize=10)
    fig.text(.5, .055, '未来1–12小时流量；峰标记限定未来窗口；没有平均三种子曲线，不是集成预测或业务触发。', ha='center', fontsize=10)
    fig.text(.5, .025, f"来源：原run保存验证预测与原事件主峰评分；执行 {EXECUTION_COMMIT[:7]}；APE差单位为百分点", ha='center', fontsize=9)
    figures.append(('page_2_cases', fig))
    with PdfPages(out/'three_branch_formal_report.pdf') as pdf:
        for name, figure in figures:
            figure.savefig(out/(name+'.png'), dpi=160)
            figure.savefig(out/(name+'.pdf'))
            pdf.savefig(figure)
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_dir, out = args.run_dir.resolve(), args.out.resolve()
    if out == run_dir or out.is_relative_to(run_dir):
        raise ValueError('Use a new output directory outside the original run')
    if out.exists():
        raise FileExistsError(out)
    data = collect(run_dir)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'plot_source.json').open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    draw(data, out)


if __name__ == '__main__':
    main()
