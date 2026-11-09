"""Pure scoring reused verbatim from 9c1f05d, originating parent audit239af24.

No model, bank loader, network, sender or execution entrypoint is imported.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np

def hierarchy_weights(rows):
    """Within one country: station, then inherited event/block, then origin equal."""
    from collections import Counter
    count=Counter((r['canonical_station'],r['event']) for r in rows)
    events=Counter(station for station,event in count)
    return np.asarray([1/(len(events)*events[r['canonical_station']]*count[r['canonical_station'],r['event']]) for r in rows])
def weights(keys):
    event_counts = Counter(keys)
    station_counts = Counter(k[0] for k in event_counts)
    return np.array([1.0 / (len(station_counts) * station_counts[k[0]] * event_counts[k])
                     for k in keys], dtype=np.float64)


def phases(rows, truth, observed):
    keys = [(r['canonical_station'], r['source_event_identity']) for r in rows]
    times = [[datetime.fromisoformat(r['origin_time']) + timedelta(hours=h)
              for h in range(1, 25)] for r in rows]
    masks = {'all': np.ones((len(rows), 24), dtype=bool)}
    support = {}
    for qi, qname in enumerate(('q95', 'q99')):
        timeline = defaultdict(dict)
        for i, key in enumerate(keys):
            for h, time in enumerate(times[i]):
                value = (float(truth[i, h]), bool(observed[i, h, qi]))
                assert time not in timeline[key] or timeline[key][time] == value
                timeline[key][time] = value
        selected = {name: defaultdict(set) for name in ('entry_first3', 'middle')}
        for key, values in timeline.items():
            blocks, block = [], []
            for time in sorted(values):
                if block and time - block[-1] != timedelta(hours=1):
                    blocks.append(block)
                    block = []
                block.append(time)
            if block:
                blocks.append(block)
            for block in blocks:
                i = 0
                while i < len(block):
                    if not values[block[i]][1]:
                        i += 1
                        continue
                    start = i
                    while i + 1 < len(block) and values[block[i + 1]][1]:
                        i += 1
                    end = i
                    if start > 0:
                        selected['entry_first3'][key].update(block[start:min(start + 3, end + 1)])
                    if start > 0 and end < len(block) - 1 and end - start + 1 > 6:
                        selected['middle'][key].update(block[start + 3:end - 2])
                    i += 1
        for name, points in selected.items():
            label = qname + '_' + name
            masks[label] = np.array([[time in points[keys[i]] for time in times[i]]
                                     for i in range(len(rows))], dtype=bool)
            support[label] = dict(stations=len({k[0] for k, v in points.items() if v}),
                                  source_events=sum(bool(v) for v in points.values()),
                                  unique_valid_hours=sum(len(v) for v in points.values()))
        masks[qname + '_positive'] = observed[:, :, qi]
        masks[qname + '_negative'] = ~observed[:, :, qi]
    return keys, masks, support


def summarize(probability, observed, keys, masks):
    assert probability.shape == observed.shape == (1706, 24, 2)
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    result, long_rows = {}, []
    for phase, mask in masks.items():
        per_lead = []
        for h in range(24):
            ix = np.flatnonzero(mask[:, h])
            if not len(ix):
                per_lead.append(None)
                continue
            w = weights([keys[i] for i in ix])
            value = (w[:, None] * (probability[ix, h] - observed[ix, h]) ** 2).sum(axis=0)
            per_lead.append(value)
            long_rows.append(dict(phase=phase, lead_hours=h + 1, origins=len(ix),
                                  q95_brier=float(value[0]), q99_brier=float(value[1])))
        bands = {}
        for lo, hi in ((1, 24), (1, 3), (4, 6), (7, 12), (13, 24)):
            use = [v for v in per_lead[lo - 1:hi] if v is not None]
            if not use:
                bands[f'{lo}-{hi}h'] = dict(status='NOT_EVALUABLE')
                continue
            mean = np.mean(use, axis=0)
            bands[f'{lo}-{hi}h'] = dict(q95_brier=float(mean[0]), q99_brier=float(mean[1]),
                                        mean_brier=float(mean.mean()), evaluated_leads=len(use))
            if phase.startswith(('q95_', 'q99_')):
                qi = 0 if phase.startswith('q95_') else 1
                bands[f'{lo}-{hi}h']['phase_threshold_brier'] = float(mean[qi])
        result[phase] = bands
    return result, long_rows
def marginal_scores(draws,y,thresholds,rows,observed_override=None):
    draws=np.asarray(draws,dtype=float);y=np.asarray(y,dtype=float)
    thresholds=np.asarray(thresholds,dtype=float)
    if draws.ndim!=3 or y.shape!=(draws.shape[0],draws.shape[2]) or thresholds.shape!=(len(y),2):
        raise ValueError('expected [N,S,F], [N,F], [N,2]')
    if not np.isfinite(draws).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite forecast or observed target')
    probability=(draws[:,:,:,None]>thresholds[:,None,None,:]).mean(axis=1)
    observed=(y[:,:,None]>thresholds[:,None,:]) if observed_override is None else np.asarray(observed_override,dtype=bool)
    if observed.shape!=(len(y),y.shape[1],2):
        raise ValueError('canonical observed labels must have shape [N,F,2]')
    brier=(probability-observed)**2
    sorted_draws=np.sort(draws,axis=1);s=draws.shape[1]
    coefficient=(2*np.arange(1,s+1)-s-1)[None,:,None]
    crps=np.abs(draws-y[:,None,:]).mean(axis=1)-(coefficient*sorted_draws).sum(axis=1)/(s*s)
    summary=[]
    for country in sorted({r['country'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['country']==country]
        weights=hierarchy_weights([rows[i] for i in ids])
        for lead in range(y.shape[1]):
            for j,name in enumerate(('Q95','Q99')):
                bins=[];p=probability[ids,lead,j];truth=observed[ids,lead,j]
                for k in range(10):
                    mask=(p>=k/10)&((p<(k+1)/10) if k<9 else (p<=1))
                    weight=weights[mask].sum()
                    bins.append(dict(lower=k/10,upper=(k+1)/10,n=int(mask.sum()),
                        mean_probability=float(np.average(p[mask],weights=weights[mask])) if weight else None,
                        observed_fraction=float(np.average(truth[mask],weights=weights[mask])) if weight else None,
                        state='observed' if weight else 'empty',reason=None if weight else 'no_sample_in_bin'))
                summary.append(dict(country=country,role=rows[ids[0]]['role'],horizon=lead+1,
                    target='pointwise_exceedance_'+name,brier=float(weights@brier[ids,lead,j]),
                    crps=float(weights@crps[ids,lead]),crps_unit='station_fit_Q95_normalized',
                    reliability_bins=bins,n_origins=len(ids),independent_events=False))
    china=[r['brier'] for r in summary if r['country']=='CN']
    return dict(probability=probability,observed=observed,crps=crps,summary=summary,
        selection_score=float(np.mean(china)) if china else None,
        selection_rule='China station then inherited-event equal, mean hourly Q95/Q99 Brier')
