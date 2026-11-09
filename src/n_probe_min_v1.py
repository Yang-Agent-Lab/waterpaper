"""Fixed CPU-only China six-hour-rise probes; approved revision 1.

Only predictor histories and future Q labels are retained from window records.
The former encoder's calibration inputs are not independently unseen data.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import gzip
import hashlib
import json
import resource
import signal
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import sklearn
import scipy
import joblib
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from esdf_flood.forecasting import Separation

KEEP = ('window_id', 'station_uid', 'country', 'role', 'source_event_id', 'overlap_group_id',
        'temporal_overlap_component', 'origin_time', 'source_rows', 'source_times',
        'candidate_start', 'origin_source_row', 'x_p', 'x_q', 'y_q')


def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def csvout(path, rows):
    if not rows:
        return
    with Path(path).open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def weights(rows):
    """Each available station, then event, then window gets equal mass."""
    counts = Counter((r['station_uid'], r['overlap_group_id']) for r in rows)
    events = Counter(st for st, _ in counts)
    return np.array([1 / (len(events) * events[r['station_uid']] *
                         counts[r['station_uid'], r['overlap_group_id']]) for r in rows])


def raw_features(rows):
    x = np.array([list(zip(r['x_p'], r['x_q'])) for r in rows], dtype=np.float64)
    assert x.shape == (len(rows), 12, 2) and np.isfinite(x).all()
    segments = []
    for lo, hi in ((9, 12), (6, 9), (0, 6)):
        p, q = x[:, lo:hi, 0], x[:, lo:hi, 1]
        segments.extend((p.sum(1), p.max(1), (p > 0).sum(1), q.mean(1),
                         q[:, -1] - q[:, 0], np.maximum(0, np.diff(q, axis=1).max(1))))
    return np.column_stack((x.reshape(len(x), 24), *segments)), x.astype(np.float32)


def targets(rows):
    y = np.array([r['y_q'] for r in rows], dtype=np.float64)
    assert y.shape == (len(rows), 12) and np.isfinite(y).all()
    q = np.array([r['x_q'][-1] for r in rows])
    return np.maximum(0, y[:, :6].max(1) - q), q, y[:, 5], y[:, :6].max(1)


def split_check(rows, fit, cal):
    for key in ('temporal_overlap_component', 'overlap_group_id'):
        assert not ({rows[i][key] for i in fit} & {rows[i][key] for i in cal}), key
    # Cross-station timestamps are deliberately checked, not merely station IDs.
    sides = []
    for ix in (fit, cal):
        intervals = sorted((np.datetime64(rows[i]['source_times'][0], 's').astype('int64'),
                            np.datetime64(rows[i]['source_times'][-1], 's').astype('int64')) for i in ix)
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        sides.append(merged)
    i = j = 0
    while i < len(sides[0]) and j < len(sides[1]):
        a, b = sides[0][i], sides[1][j]
        assert a[1] < b[0] or b[1] < a[0], '24h source intervals cross fit/calibration'
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    by_station = defaultdict(lambda: [set(), set()])
    for side, ix in enumerate((fit, cal)):
        for k in ix:
            by_station[rows[k]['station_uid']][side].update(rows[k]['source_rows'])
    assert all(not (a & b) for a, b in by_station.values()), 'source rows cross split'


def classify(rows, rise, cuts):
    return np.array([int(v > cuts[r['station_uid']]) if cuts[r['station_uid']] is not None else -1
                     for r, v in zip(rows, rise)])


def threshold(scores, labels, rows, limit):
    negative = np.flatnonzero((labels == 0) & np.isfinite(scores))
    if not len(negative):
        return dict(state='NOT_EVALUABLE', threshold=None, calibration_fpr=None, negative_n=0)
    w = weights([rows[i] for i in negative])
    values = np.unique(scores[negative])
    # At the lowest candidate alarm is score > threshold; ties remain unalarmed.
    mass = np.array([w[scores[negative] == v].sum() for v in values])
    fprs = np.maximum(0, 1 - np.cumsum(mass))
    chosen = np.flatnonzero(fprs <= limit + 1e-12)[0]
    return dict(state='evaluable', threshold=float(values[chosen]),
                calibration_fpr=float(w[scores[negative] > values[chosen]].sum()), negative_n=len(negative),
                negative_events=len({rows[i]['overlap_group_id'] for i in negative}),
                negative_stations=len({rows[i]['station_uid'] for i in negative}))


def classification_metrics(rows, labels, scores, cut):
    ix = np.flatnonzero((labels >= 0) & np.isfinite(scores))
    if not len(ix):
        return dict(state='NOT_EVALUABLE', n=0, ap=None, positive_rate=None, fpr=None, recall=None, precision=None)
    rs = [rows[i] for i in ix]
    y, s, w = labels[ix], scores[ix], weights(rs)
    both = len(np.unique(y)) == 2
    answer = dict(state='evaluable' if both else 'NOT_EVALUABLE', n=len(ix),
                  positive_rate=float(w[y == 1].sum()),
                  ap=float(average_precision_score(y, s, sample_weight=w)) if both else None,
                  fpr=None, recall=None, precision=None)
    if cut['threshold'] is not None:
        alarm = s > cut['threshold']
        # FPR/recall independently reweight their applicable negative/positive strata.
        for label, name in ((0, 'fpr'), (1, 'recall')):
            mask = y == label
            if mask.any():
                answer[name] = float(np.dot(weights([r for r, yes in zip(rs, mask) if yes]), alarm[mask]))
        if alarm.any():
            answer['precision'] = float(w[alarm & (y == 1)].sum() / w[alarm].sum())
    return answer


def regression_metrics(rows, truth, prediction):
    if not rows:
        return dict(state='NOT_EVALUABLE', n=0, mae=None, mse=None, physical_bias_m3_s=None,
                    physical_mae_m3_s=None, physical_mse_m6_s2=None)
    error = prediction - truth
    scale = np.array([r['qscale'] for r in rows])
    w = weights(rows)
    return dict(state='evaluable', n=len(rows), mae=float(w @ abs(error)), mse=float(w @ error ** 2),
                physical_bias_m3_s=float(w @ (error * scale)), physical_mae_m3_s=float(w @ abs(error * scale)),
                physical_mse_m6_s2=float(w @ (error * scale) ** 2))


def n_features(x, model):
    features, pred = [], []
    with torch.no_grad():
        for start in range(0, len(x), 64):
            t = torch.from_numpy(x[start:start + 64])
            shared, specific = model.shared_encoder(t), model.real_encoder(t.reshape(len(t), 24))
            z = torch.cat((shared, specific), dim=1)
            features.append(z.numpy().copy())
            pred.append(model.real_predictor(z).numpy().copy())
    return np.concatenate(features).astype(np.float64), np.concatenate(pred).astype(np.float64)


def self_check():
    row = dict(x_p=list(range(12)), x_q=list(range(12)), y_q=list(range(12, 24)))
    a, x = raw_features([row])
    assert a.shape == (1, 42) and x.shape == (1, 12, 2)
    assert np.array_equal(a[0, 24:30], [30, 11, 3, 10, 2, 1])
    assert targets([row])[0][0] == 6
    rs = [dict(station_uid='a', overlap_group_id='a1')] * 2 + [dict(station_uid='a', overlap_group_id='a2'), dict(station_uid='b', overlap_group_id='b1')]
    assert np.allclose(weights(rs), [.125, .125, .25, .5])
    scores = np.array([.1, .1, .4, .7])
    cut = threshold(scores, np.zeros(4), rs, .05)
    assert cut['threshold'] == .7 and cut['calibration_fpr'] == 0
    assert classify([{'station_uid': 'a'}], np.array([1.]), {'a': 1.})[0] == 0


def figures(out, rows, rises, scores, labels, cuts, eventstats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = out / 'figures'
    figdir.mkdir()
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    events = sorted({r['overlap_group_id'] for r in rows})
    f, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for name in ('A', 'B', 'N'):
        es = {r['event']: r for r in eventstats if r['model'] == name and r['stratum'] == 'all'}
        axes[0].plot([es[e]['ap'] if es[e]['ap'] is not None else np.nan for e in events], range(len(events)), 'o-', label=name)
        axes[1].plot([es[e]['mae'] for e in events], range(len(events)), 'o-', label=name)
    for ax in axes:
        ax.set_yticks(range(len(events)), [e.replace('da_tl_cn:', '').replace('洪水_', 'event ') for e in events])
        ax.legend()
    axes[0].set(xlabel='Weighted average precision (missing: no two classes)', title='Every development event: rise recognition')
    axes[1].set(xlabel='Six-hour rise MAE / original train Q95', title='Every development event: amplitude')
    f.suptitle('China development | fixed seed-2 N | A: history, B: N representation | no test')
    f.savefig(figdir / '01_all_events.png', dpi=150)
    plt.close(f)
    hard = sorted({r['overlap_group_id'] for r in rows if any(s in r['station_uid'] for s in ('Liqingdian', 'Luanchuan'))})
    f, axes = plt.subplots(len(hard), 2, figsize=(14, max(4, 2.8 * len(hard))), squeeze=False, constrained_layout=True)
    source = []
    for axrow, event in zip(axes, hard):
        ix = sorted([i for i, r in enumerate(rows) if r['overlap_group_id'] == event], key=lambda i: rows[i]['origin_time'])
        t = [np.datetime64(rows[i]['origin_time']) for i in ix]
        axrow[0].plot(t, [rows[i]['r6'] for i in ix], 'k-', label='Observed r6')
        for name in ('N', 'A', 'B'):
            axrow[0].plot(t, rises[name][ix], label=name)
            axrow[1].plot(t, scores[name][ix], label=name)
            if cuts[name]['threshold'] is not None:
                axrow[1].axhline(cuts[name]['threshold'], ls='--', lw=.8, label=f'{name} calibration line')
        axrow[0].set(title=event.replace('洪水_', 'event '), ylabel='Rise / train Q95', xlabel='Issue time (source clock unknown)')
        axrow[1].set(title='Risk: classifier A/B probabilities, N rise score', ylabel='Risk score (scales differ)', xlabel='Issue time')
        for ax in axrow:
            ax.legend(fontsize=6, ncol=3)
            ax.tick_params(axis='x', rotation=20)
        for i in ix:
            source.append(dict(window_id=rows[i]['window_id'], event=event, origin_time=rows[i]['origin_time'],
                               truth=rows[i]['r6'], label=int(labels[i]),
                               **{f'rise_{k}': float(v[i]) for k, v in rises.items()},
                               **{f'score_{k}': float(v[i]) for k, v in scores.items()}))
    f.suptitle('All Liqingdian / Luanchuan development events | calibration lines from train only')
    f.savefig(figdir / '02_difficult_events.png', dpi=150)
    plt.close(f)
    csvout(out / 'figure_source_difficult.csv', source)


def run(args, config, out):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_path = Path(config['manifest'])
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_data_commit'] == config['data_commit']
    assert manifest['scope'] == 'projected_train_validation_only'
    sources, train, dev = [], [], []
    allowed_root = Path(manifest['stageB_root']) / 'stations'
    for station in manifest['stations']:
        if station['country'] != 'China':
            continue
        uid = station['station_uid']
        for role, dest in (('train', train), ('validation', dev)):
            path = Path(station['files'][role]['path'])
            assert path.is_relative_to(allowed_root) and path.name == role + '_windows.jsonl.gz'
            assert path.parent.name == uid.replace(':', '__')
            sources.append(dict(station=uid, role=role, path=str(path), sha256=sha(path),
                                qscale=station['scales']['qscale'], pscale=station['scales']['pscale']))
            with gzip.open(path, 'rt') as f:
                for line in f:
                    raw = json.loads(line)
                    row = {k: raw[k] for k in KEEP}
                    assert row['country'] == 'China' and row['role'] == role and row['station_uid'] == uid
                    assert row['temporal_overlap_component'], 'NOT_EVALUABLE: absent overlap component'
                    row['qscale'] = station['scales']['qscale']
                    dest.append(row)
    train.sort(key=lambda r: r['window_id'])
    dev.sort(key=lambda r: r['window_id'])
    assert len(train) == 8557 and len(dev) == 1356
    assert len({r['overlap_group_id'] for r in dev}) == 16 and len({r['station_uid'] for r in dev}) == 9
    assert len({r['window_id'] for r in train + dev}) == 9913
    groups = [r['temporal_overlap_component'] for r in train]
    fit, cal = next(GroupShuffleSplit(n_splits=1, test_size=config['calibration_fraction'],
                                    random_state=config['split_seed']).split(np.zeros(len(train)), groups=groups))
    split_check(train, fit, cal)
    ar, xt = raw_features(train)
    ad, xd = raw_features(dev)
    root = Path(config['n_root'])
    weight = root / 'best.pt'
    before = sha(weight)
    model = Separation(64).cpu()
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bd, nd = n_features(xd, model)
    archived = {}
    with gzip.open(root / 'validation_predictions.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['country'] == 'China' and r['primary_origin']:
                archived[r['window_id']] = r
    assert set(archived) == {r['window_id'] for r in dev}
    for i, row in enumerate(dev):
        old = archived[row['window_id']]
        assert np.array_equal(old['observed_normalized_q'], row['y_q'])
        assert old['qscale'] == row['qscale'] and old['q_now_normalized'] == row['x_q'][-1]
    saved_nd = np.array([archived[r['window_id']]['predicted_normalized_q'] for r in dev])
    difference = nd - saved_nd
    absolute = abs(difference)
    discrepancy = float(absolute.max())
    worst_i, worst_h = np.unravel_index(absolute.argmax(), absolute.shape)
    ydev = np.array([r['y_q'] for r in dev])
    wd = weights(dev)
    replay_peak_mae = float(wd @ abs(nd.max(1) - ydev.max(1)))
    saved_peak_mae = float(wd @ abs(saved_nd.max(1) - ydev.max(1)))
    peak_metric_difference = abs(replay_peak_mae - saved_peak_mae)
    replay_ok = np.allclose(nd, saved_nd, atol=3e-6, rtol=1e-6) and peak_metric_difference <= 3e-6
    csvout(out / 'replay_per_window.csv', [dict(window_id=r['window_id'], station=r['station_uid'],
        max_absolute_difference=float(absolute[i].max()), **{f'signed_difference_h{h+1}':float(difference[i,h]) for h in range(12)}) for i,r in enumerate(dev)])
    dump(out / 'replay.json', dict(n=1356, max_absolute_difference=discrepancy, atol=3e-6, rtol=1e-6,
        absolute_difference_quantiles=dict(zip(['0','50','90','99','100'],np.quantile(absolute,[0,.5,.9,.99,1]).tolist())),
        values_exceeding_original_1e6=int((absolute>1e-6).sum()), worst_window_id=dev[worst_i]['window_id'],
        worst_lead_hour=int(worst_h+1), replay_12h_peak_mae=replay_peak_mae, archived_12h_peak_mae=saved_peak_mae,
        peak_metric_difference=peak_metric_difference, state='evaluable' if replay_ok else 'failed',
        labels_scales_qnow_exact=True, weights_sha256=before))
    assert replay_ok, ('N dev replay mismatch', discrepancy, peak_metric_difference)
    br, nt = n_features(xt, model)
    assert ar.shape[1] == 42 and br.shape[1] == 128
    rt, qt, q6t, peakt = targets(train)
    rd, qd, q6d, peakd = targets(dev)
    cuts = {}
    for st in sorted({r['station_uid'] for r in train}):
        positive = rt[[i for i in fit if train[i]['station_uid'] == st and rt[i] > 0]]
        cuts[st] = float(np.quantile(positive, config['positive_rise_quantile'])) if len(positive) else None
    lt, ld = classify(train, rt, cuts), classify(dev, rd, cuts)
    fit_set, cal_set = set(fit), set(cal)
    inventory, splitrows = [], []
    for name, rows, labels, ix in (('fit', train, lt, fit), ('calibration', train, lt, cal), ('development', dev, ld, range(len(dev)))):
        for st in sorted(cuts):
            ids = [i for i in ix if rows[i]['station_uid'] == st]
            inv = dict(split=name, station=st, cutoff=cuts[st], n=len(ids), state='evaluable' if cuts[st] is not None else 'NOT_EVALUABLE')
            for label, key in ((0, 'negative'), (1, 'positive')):
                z = [i for i in ids if labels[i] == label]
                inv[key + '_windows'] = len(z)
                inv[key + '_events'] = len({rows[i]['overlap_group_id'] for i in z})
            inventory.append(inv)
    for i, r in enumerate(train):
        splitrows.append(dict(window_id=r['window_id'], station=r['station_uid'], event=r['overlap_group_id'],
                              component=r['temporal_overlap_component'], split='fit' if i in fit_set else 'calibration',
                              r6=float(rt[i]), label=int(lt[i])))
    used = np.array(fit)
    if args.smoke:
        selected = []
        for component in sorted(set(groups)):
            available = [i for i in fit if groups[i] == component]
            selected.extend(available[:config['smoke_fit_per_component']])
        used = np.array(sorted(selected))
    assert set(used) <= fit_set
    used_ids = {train[i]['window_id'] for i in used}
    for r in splitrows:
        r['used_in_probe_fit'] = r['window_id'] in used_ids
    csvout(out / 'splits.csv', splitrows)
    csvout(out / 'label_inventory.csv', inventory)
    dump(out / 'station_cutoffs.json', cuts)
    preds_t, preds_d, scores_t, scores_d, standard = {}, {}, {}, {}, {}
    fit_rows = [train[i] for i in used]
    wf = weights(fit_rows) * len(used)
    for name, tr, dv in (('A', ar, ad), ('B', br, bd)):
        mu = np.average(tr[used], axis=0, weights=wf)
        scale = np.sqrt(np.average((tr[used] - mu) ** 2, axis=0, weights=wf))
        scale[scale == 0] = 1
        zt, zd = (tr - mu) / scale, (dv - mu) / scale
        standard[name] = dict(mean=mu.tolist(), scale=scale.tolist(), fit_weight_mean=float(wf.mean()), fit_n=len(used))
        ridge = Ridge(**config['ridge']).fit(zt[used], rt[used], sample_weight=wf)
        rawt, rawd = ridge.predict(zt), ridge.predict(zd)
        preds_t[name], preds_d[name] = np.maximum(0, rawt), np.maximum(0, rawd)
        joblib.dump(ridge, out / (name + '_ridge.joblib'))
        np.savez(out / (name + '_ridge_raw.npz'), train=rawt, development=rawd)
        good = used[lt[used] >= 0]
        if len(np.unique(lt[good])) != 2:
            scores_t[name], scores_d[name] = np.full(len(train), np.nan), np.full(len(dev), np.nan)
            dump(out / (name + '_logistic_status.json'), dict(state='NOT_EVALUABLE', reason='fit lacks either class'))
        else:
            wc = weights([train[i] for i in good]) * len(good)
            clf = LogisticRegression(**config['logistic'])
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter('always')
                clf.fit(zt[good], lt[good], sample_weight=wc)
            scores_t[name], scores_d[name] = clf.predict_proba(zt)[:, 1], clf.predict_proba(zd)[:, 1]
            joblib.dump(clf, out / (name + '_logistic.joblib'))
            dump(out / (name + '_logistic_status.json'), dict(state='evaluable', fit_n=len(good), iterations=clf.n_iter_.tolist(),
                                                           warnings=[str(r.message) for r in records]))
    preds_t.update(N=np.maximum(0, nt[:, :6].max(1) - qt), persistence=np.zeros(len(train)))
    preds_d.update(N=np.maximum(0, saved_nd[:, :6].max(1) - qd), persistence=np.zeros(len(dev)))
    known = used[lt[used] >= 0]
    prevalence = float(weights([train[i] for i in known]) @ lt[known]) if len(known) else None
    for ss, qq, xx, pp in ((scores_t, qt, xt, preds_t), (scores_d, qd, xd, preds_d)):
        ss.update(N=pp['N'], fit_prevalence=np.full(len(qq), prevalence if prevalence is not None else np.nan),
                  current_Q=qq, recent_3h_rise=np.maximum(0, xx[:, -1, 1] - xx[:, -4, 1]))
    thresholds = {name: threshold(v[cal], lt[cal], [train[i] for i in cal], config['false_positive_limit'])
                  for name, v in scores_t.items()}
    dump(out / 'calibration_thresholds.json', thresholds)
    dump(out / 'standardization.json', standard)
    dump(out / 'feature_definitions.json', dict(A='12x2 interleaved P/Q history flattened + 3 segments of 6 features',
        segments=[[-2, 0], [-5, -3], [-11, -6]], segment_features=['Psum', 'Pmax', 'Ppositive_count', 'Qmean', 'Qlast_minus_first', 'max_positive_hourly_Qdifference'],
        B='torch.cat(shared_encoder(history), real_encoder(history.flatten)), 128 dimensions', future_P_access=False))
    metrics, eventstats, stationstats, output_rows = [], [], [], []
    for role, rows, rise, label, qq, q6, peak, predictions, scores in (
        ('fit', [train[i] for i in used], rt[used], lt[used], qt[used], q6t[used], peakt[used], {k:v[used] for k,v in preds_t.items()}, {k:v[used] for k,v in scores_t.items()}),
        ('calibration', [train[i] for i in cal], rt[cal], lt[cal], qt[cal], q6t[cal], peakt[cal], {k:v[cal] for k,v in preds_t.items()}, {k:v[cal] for k,v in scores_t.items()}),
        ('development', dev, rd, ld, qd, q6d, peakd, preds_d, scores_d)):
        for i, r in enumerate(rows):
            output_rows.append(dict(split=role, window_id=r['window_id'], station=r['station_uid'], event=r['overlap_group_id'],
                origin_time=r['origin_time'], qscale=r['qscale'], qnow=float(qq[i]), actual_Q6=float(q6[i]), actual_peak6=float(peak[i]),
                r6=float(rise[i]), label=int(label[i]), future_high95=bool(peak[i] > 1),
                **{f'rhat_{k}':float(v[i]) for k,v in predictions.items()},
                **{f'score_{k}':float(v[i]) if np.isfinite(v[i]) else None for k,v in scores.items()}))
        strata = dict(all=np.ones(len(rows), dtype=bool), positive_rise=rise > 0, strong_rise=label == 1,
                      qnow_at_most_1=qq <= 1, future_high95=peak > 1)
        for level, identifiers, sink in (('overall', ['all'], metrics), ('event', sorted({r['overlap_group_id'] for r in rows}), eventstats),
                                        ('station', sorted({r['station_uid'] for r in rows}), stationstats)):
            for identifier in identifiers:
                groupmask = np.ones(len(rows), dtype=bool) if level == 'overall' else np.array([r['overlap_group_id' if level == 'event' else 'station_uid'] == identifier for r in rows])
                for stratum, mask in strata.items():
                    ix = np.flatnonzero(mask & groupmask)
                    sub = [rows[i] for i in ix]
                    for name in dict.fromkeys(list(predictions) + list(scores)):
                        result = dict(split=role, level=level, event=identifier if level=='event' else None,
                                      station=identifier if level=='station' else None, stratum=stratum, model=name)
                        if name in predictions:
                            result.update(regression_metrics(sub, rise[ix], predictions[name][ix]))
                        else:
                            result.update(mae=None, mse=None, physical_bias_m3_s=None, physical_mae_m3_s=None, physical_mse_m6_s2=None)
                        if name in scores:
                            cl = classification_metrics(sub, label[ix], scores[name][ix], thresholds[name])
                            result.update({('classification_state' if k=='state' else 'classification_n' if k=='n' else k):v for k,v in cl.items()})
                        else:
                            result.update(classification_state='NOT_EVALUABLE', classification_n=0, ap=None, positive_rate=None, fpr=None, recall=None, precision=None)
                        sink.append(result)
    csvout(out / 'window_predictions.csv', output_rows)
    csvout(out / 'metrics.csv', metrics)
    csvout(out / 'event_metrics.csv', eventstats)
    csvout(out / 'station_metrics.csv', stationstats)
    for i, r in enumerate(dev):
        r['r6'] = float(rd[i])
    figures(out, dev, preds_d, scores_d, ld, thresholds, [r for r in eventstats if r['split']=='development'])
    assert sha(weight) == before, 'Original N weight changed'
    dump(out / 'input_manifest.json', dict(change_id=config['change_id'], code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_commit=config['data_commit'], manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
        weight_path=str(weight), weight_sha256_before=before, weight_sha256_after=sha(weight),
        archived_predictions_path=str(root/'validation_predictions.jsonl.gz'), archived_predictions_sha256=sha(root/'validation_predictions.jsonl.gz'),
        inputs=sources, train=8557, development=1356, fit_n=len(fit), calibration_n=len(cal), probe_fit_n=len(used),
        smoke=args.smoke, source_case_audit=config['source_case_audit'],
        limitations=['Development diagnosis: encoder saw all original train and dev selected checkpoint',
                     'Single fixed seed; readout performance is not complete-model performance',
                     'No test, future P, optimizer update to N, GPU, search, or additional seed'],
        versions=dict(python=os.sys.version, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--prior-wall-seconds', type=float, default=0)
    args = parser.parse_args()
    start = time.monotonic()
    cpu_start = time.process_time()
    config = json.loads(Path('configs/n_probe_min_v1.json').read_text())
    remaining = config['budget_seconds'] - args.prior_wall_seconds
    assert remaining > 0 and args.prior_wall_seconds >= 0
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'progress.json', dict(state='running', pid=os.getpid(), smoke=args.smoke, prior_wall_seconds=args.prior_wall_seconds))
    def timeout(signum, frame):
        raise TimeoutError('Cumulative 3600s computation budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(remaining)))
    state, error = 'complete', None
    try:
        self_check()
        if not args.self_check:
            run(args, config, out)
    except BaseException as exc:
        state = 'partial' if isinstance(exc, TimeoutError) else 'failed'
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        elapsed = time.monotonic() - start
        dump(out / 'completion.json', dict(state=state, error=error, wall_seconds=elapsed,
            cpu_seconds=time.process_time()-cpu_start, cumulative_wall_seconds=args.prior_wall_seconds+elapsed,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpu=False,
            self_check=args.self_check, smoke=args.smoke, optimizer_steps_N=0, test_access=False))
        signal.alarm(0)


if __name__ == '__main__':
    main()
