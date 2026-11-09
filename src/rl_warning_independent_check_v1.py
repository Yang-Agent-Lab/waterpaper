"""Independent saved-action audit; does not import the policy implementation."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def audit(source, output, destination):
    cpu = time.process_time()
    source, output = Path(source), Path(output)
    rows = json.loads((source / 'select_rows.json').read_text())
    bank = np.load(source / 'select_predictions.npz', allow_pickle=False)
    assert digest(source / 'select_predictions.npz') == '240db13a546b4a8ad1914390cb35c9220f0d04d2d42ae98d08ce2753a8a8bc5d'
    assert digest(source / 'best.pt') == 'bbd6f87e0ad452873f1d4b357a67cf2fbd823bb2dd3bdc36e56ab7b2d06c4a3a'
    ids = {str(r['sample_id']): i for i, r in enumerate(rows)}
    assert len(ids) == len(rows)
    cfg = json.loads((output / 'approved_config.json').read_text()) if (output / 'approved_config.json').exists() else None
    # Explicit independent copy of the predeclared engineering costs.
    hit, miss, cover, fa, switching = [2, 4], [4, 8], [.25, .5], [.05, .1], .02
    reports, event_sets = {}, {}
    names = ['always0', 'always1', 'always2', 'fixed', 'hysteresis', 'cost_sensitive', 'ppo']
    for name in names:
        path = output / (name + '_cal_val.json')
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        totals = dict(raw_reward=0., monitor_hours=0, switches=0,
                      hits=[0, 0], misses=[0, 0], onset_count=[0, 0],
                      false_alarm_hours=[0, 0], uncovered_hours=[0, 0],
                      danger_hours=[0, 0], non_danger_hours=[0, 0], action_hours=[0, 0, 0])
        parts = dict(hit=0., miss=0., false_alarm=0., uncovered=0., switch=0.)
        events, leads, valid_hours = {}, [[], []], [0, 0]
        actual_switches = 0
        for trajectory, ep_saved in zip(saved['trajectories'], saved['episodes'], strict=True):
            a = trajectory['actions']
            indices = [ids[str(s)] for s in trajectory['sample_id']]
            n = len(indices)
            assert len(a) == n and n > 0
            ts = [datetime.fromisoformat(rows[i]['origin_time']) for i in indices]
            assert [t.isoformat() for t in ts] == trajectory['times']
            assert all(v == 1 or v == 2 or v == 0 for v in a)
            assert all(ts[k + 1] - ts[k] == timedelta(hours=1) for k in range(n - 1))
            assert all([rows[i]['canonical_station'], rows[i]['source_event_identity']] == trajectory['key'] for i in indices)
            assert all(rows[i]['canonical_station'] in {'da_tl_cn:Luanchuan', 'da_tl_cn:Qingyangcha', 'da_tl_cn:Zhuxipo'} for i in indices)
            thresholds = bank['thresholds_q95_q99_m3_s'][indices[0]]
            assert np.array_equal(bank['thresholds_q95_q99_m3_s'][indices], np.broadcast_to(thresholds, (n, 2)))
            current = [float(rows[i]['current_state']['q_m3_s']) for i in indices]
            future = bank['truth_q_m3_s'][indices, :6]
            assert np.allclose(np.asarray(current[1:]), future[:-1, 0], rtol=1e-6, atol=1e-5)
            truth = current + [float(future[-1, 0])]
            step = [0.] * n
            local_parts = dict(hit=0., miss=0., false_alarm=0., uncovered=0., switch=0.)
            for j in (0, 1):
                last = None
                for t, q in enumerate(truth):
                    if q <= thresholds[j]:
                        continue
                    new = last is None or t - last > 6
                    last = t
                    if not new or t < 6 or any(x > thresholds[j] for x in truth[t - 6:t]):
                        continue
                    totals['onset_count'][j] += 1
                    alarm = next((k for k in range(t - 6, t) if a[k] >= j + 1), None)
                    identity = tuple(trajectory['key']) + ((ts[0] + timedelta(hours=t)).isoformat(), j)
                    lead = None if alarm is None else t - alarm
                    events[identity] = lead
                    if lead is None:
                        totals['misses'][j] += 1
                        step[t - 1] -= miss[j]
                        local_parts['miss'] -= miss[j]
                    else:
                        totals['hits'][j] += 1
                        leads[j].append(lead)
                        reward = hit[j] * (.25 + .75 * lead / 6)
                        step[t - 1] += reward
                        local_parts['hit'] += reward
            previous = 0
            for t, action in enumerate(a):
                totals['action_hours'][action] += 1
                change = abs(action - previous)
                actual_switches += int(action != previous)
                totals['switches'] += change
                step[t] -= switching * change
                local_parts['switch'] -= switching * change
                previous = action
                for j in (0, 1):
                    danger = future[t, 0] > thresholds[j]
                    totals['danger_hours'][j] += int(danger)
                    totals['non_danger_hours'][j] += int(not danger)
                    uncovered = danger and action < j + 1
                    totals['uncovered_hours'][j] += int(uncovered)
                    step[t] -= cover[j] * uncovered
                    local_parts['uncovered'] -= cover[j] * uncovered
                    valid_hours[j] += int(current[t] <= thresholds[j] and all(q <= thresholds[j] for q in future[t]))
                if action and current[t] <= thresholds[action - 1] and all(q <= thresholds[action - 1] for q in future[t]):
                    totals['false_alarm_hours'][action - 1] += 1
                    step[t] -= fa[action - 1]
                    local_parts['false_alarm'] -= fa[action - 1]
            assert np.allclose(step, ep_saved['step_rewards'], rtol=0, atol=1e-10), (name, trajectory['key'], 'step reward')
            for key, value in local_parts.items():
                assert abs(value - ep_saved['reward_parts'][key]) < 1e-9
                parts[key] += value
            totals['raw_reward'] += sum(step)
            totals['monitor_hours'] += n
        for key, value in totals.items():
            assert np.allclose(value, saved['summary'][key], rtol=0, atol=1e-9), (name, key, value, saved['summary'][key])
        totals.update(reward_parts=parts, lead_hours=leads, false_alarm_eligible_hours=valid_hours,
                      action_change_count=actual_switches,
                      switch_field_meaning='sum absolute level differences, including episode initialization')
        reports[name] = totals
        event_sets[name] = events
    simple = [n for n in reports if n != 'ppo']
    best = max(simple, key=lambda n: reports[n]['raw_reward'])
    common = {}
    if 'ppo' in reports:
        assert set(event_sets['ppo']) == set(event_sets[best])
        for j in (0, 1):
            keys = [k for k in event_sets[best] if k[-1] == j and event_sets[best][k] is not None and event_sets['ppo'][k] is not None]
            common[str(j)] = dict(n=len(keys), reference_leads=[event_sets[best][k] for k in keys], ppo_leads=[event_sets['ppo'][k] for k in keys])
    result = dict(status='PASS', independent_of_policy_scorer=True, source_unchanged=True,
                  formal_eligible=False, strongest_simple_rule=best, policies=reports,
                  common_hit_events=common, cpu_seconds=time.process_time() - cpu)
    assert result['cpu_seconds'] < 60
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return result


def plot_comparison(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in [result['strongest_simple_rule'], 'cost_sensitive', 'ppo'] if n in result['policies']]
    names = list(dict.fromkeys(names))
    labels = {'fixed': 'Threshold 0.10', 'hysteresis': 'Hysteresis',
              'cost_sensitive': 'Rolling cost', 'ppo': 'PPO selected'}
    colors = ['#366c9e', '#8d9da7', '#cc6544']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    x = np.arange(2)
    width = .24
    for k, name in enumerate(names):
        r = result['policies'][name]
        offset = (k - (len(names) - 1) / 2) * width
        means = [float(np.mean(v)) if len(v) else np.nan for v in r['lead_hours']]
        for panel, values in enumerate([means, r['false_alarm_hours'], r['uncovered_hours']]):
            axes[panel].bar(x + offset, values, width=width, color=colors[k], label=labels.get(name, name))
        for j in (0, 1):
            axes[0].text(x[j] + offset, (means[j] if np.isfinite(means[j]) else 0) + .15,
                         f"{r['hits'][j]}/{r['onset_count'][j]}", ha='center', fontsize=8)
    for ax, letter, ylabel in zip(axes, ['a', 'b', 'c'], ['Mean timely lead (h)', 'False warning duration (h)', 'Uncovered hazard duration (h)']):
        ax.set_xticks(x, ['Q95', 'Q99'])
        ax.set_ylabel(ylabel)
        ax.text(-.16, 1.02, letter, transform=ax.transAxes, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(0, 7)
    axes[0].legend(loc='lower left', fontsize=8, frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--figure')
    a = p.parse_args()
    result = audit(a.source, a.output, a.destination)
    if a.figure:
        plot_comparison(result, a.figure)
