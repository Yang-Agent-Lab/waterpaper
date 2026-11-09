"""Fixed China development selection and prediction; no independent-test claim."""
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime, timedelta
from src.pointwise_benchmark_v1.metrics import marginal_scores
from src.d500_process_adapter_v1.sampling import NoisePlan
from src.pointwise_benchmark_v1.dataset import hierarchy_weights
from src.d500_process_adapter_v1.reference_scoring import phases, weights as phase_weights


@torch.no_grad()
def predict(model, bank, draws, device, is_nb, seed=900002, chunk=8, plan_receipts=None):
    model.eval()
    output = []
    for start in range(0, len(bank.select_rows), chunk):
        ids = np.arange(start, min(start+chunk, len(bank.select_rows)))
        x, _ = bank.batch('select', ids, device)
        if is_nb:
            samples = model.predict_samples(x, draws, seed=seed+start)
            if plan_receipts is not None:
                plan_receipts.append(dict(kind='deterministic_quantile_midpoint_grid', sample_ids=[bank.select_rows[i]['sample_id'] for i in ids], k=draws))
        else:
            mu, _ = model.process.conditions(x)
            plan = NoisePlan.create(mu, [bank.select_rows[i]['sample_id'] for i in ids], draws, seed+start)
            if plan_receipts is not None:
                plan_receipts.append(plan.manifest())
            pieces = []
            for first in range(0, draws, 16):
                value, _ = model.process.sample(x, plan, draws=range(first, min(draws, first+16)),
                                                enabled=getattr(model,'adapter_enabled',True))
                pieces.append(value)
            samples = torch.cat(pieces, dim=1)
        if not torch.isfinite(samples).all():
            raise FloatingPointError('nonfinite prediction before physical projection')
        output.append(samples.cpu().numpy())
    return np.maximum(np.concatenate(output), 0.)


def selection(draws, bank):
    a = bank.arrays['select']; scale = a['flow_scale']
    y = a['future_q']/scale[:, None]
    thresholds = np.stack((a['q95_fit'], a['q99_fit']), axis=-1)/scale[:, None]
    observed = a['future_q'][:, :, None] > np.stack((a['q95_fit'], a['q99_fit']), axis=-1)[:, None, :]
    scores = marginal_scores(draws, y, thresholds, bank.select_rows, observed)
    return scores['selection_score']


def evaluation_top_masks(rows, truth):
    """Station-specific quantiles of unique target hours, strict exceedance."""
    by_station = defaultdict(dict)
    time_grid = []
    for i, row in enumerate(rows):
        station = row['canonical_station']
        origin = datetime.fromisoformat(row['origin_time'])
        times = [origin + timedelta(hours=h) for h in range(1, 25)]
        time_grid.append(times)
        for h, target_time in enumerate(times):
            value = float(truth[i,h])
            prior = by_station[station].get(target_time)
            if prior is not None and not np.isclose(value, prior, rtol=2e-6, atol=2e-6):
                raise ValueError('inconsistent station/target-hour truth')
            by_station[station][target_time] = value
    thresholds = {station:np.quantile(list(values.values()), [.90,.95,.99]).tolist()
                  for station,values in by_station.items()}
    masks = {}
    for j, label in enumerate(('eval_top10','eval_top5','eval_top1')):
        masks[label] = np.array([[truth[i,h] > thresholds[row['canonical_station']][j]
                                 for h in range(24)] for i,row in enumerate(rows)], dtype=bool)
    return masks, thresholds, {s:len(v) for s,v in by_station.items()}


def supplemental_masks(bank):
    rows = bank.select_rows; a = bank.arrays['select']; truth = a['future_q']
    observed = truth[:,:,None] > np.stack((a['q95_fit'],a['q99_fit']),axis=-1)[:,None,:]
    _, phase_masks, support = phases(rows, truth, observed)
    masks = {name:mask for name,mask in phase_masks.items() if name != 'all'}
    stations = np.asarray([r['canonical_station'] for r in rows])
    for station in sorted(set(stations)):
        masks['station:'+station] = np.broadcast_to((stations==station)[:,None], truth.shape)
    pairs = [(r['canonical_station'], r['source_event_identity']) for r in rows]
    for station, event in sorted(set(pairs)):
        member = np.asarray([value==(station,event) for value in pairs])
        masks['source:'+station+':'+event] = np.broadcast_to(member[:,None],truth.shape)
    return masks, support


def phase_brier(draws, bank):
    """Inherited source/station equal phase scoring without the old 1706-row assertion."""
    a=bank.arrays['select'];truth=a['future_q'];scale=a['flow_scale']
    thresholds=np.stack((a['q95_fit'],a['q99_fit']),axis=-1)
    observed=truth[:,:,None]>thresholds[:,None,:]
    probability=((draws*scale[:,None,None])[:,:,:,None]>thresholds[:,None,None,:]).mean(1)
    keys,masks,support=phases(bank.select_rows,truth,observed)
    result={}
    for phase,mask in masks.items():
        if phase=='all':continue
        leads=[]
        for h in range(24):
            ids=np.flatnonzero(mask[:,h])
            if not len(ids):
                leads.append(None);continue
            weights=phase_weights([keys[i] for i in ids])
            score=(weights[:,None]*(probability[ids,h]-observed[ids,h])**2).sum(0)
            leads.append(dict(horizon=h+1,origins=len(ids),q95=float(score[0]),q99=float(score[1])))
        values=[x for x in leads if x is not None]
        result[phase]=dict(state='EVALUABLE' if values else 'NOT_EVALUABLE',per_h=leads,
                           q95_mean=float(np.mean([x['q95'] for x in values])) if values else None,
                           q99_mean=float(np.mean([x['q99'] for x in values])) if values else None,
                           weighting='station then source event then origin, recomputed per lead')
    return result,support


def score_cells(draws, bank, masks=None):
    """Pointwise weighted summaries; explicit unsupported masks stay absent."""
    a = bank.arrays['select']; scale = a['flow_scale']
    x = np.asarray(draws, dtype=np.float64)*scale[:, None, None]
    y = a['future_q']; thresholds = np.stack((a['q95_fit'], a['q99_fit']), axis=-1)
    if x.shape[0] != len(y) or x.shape[2] != 24 or not np.isfinite(x).all():
        raise ValueError('invalid scored predictions')
    median = np.quantile(x, .5, axis=1)
    lo, hi = np.quantile(x, [.05, .95], axis=1)
    error = median-y
    prob = (x[:,:,:,None] > thresholds[:,None,None,:]).mean(1)
    truth = y[:,:,None] > thresholds[:,None,:]
    k = x.shape[1]; coefficient = (2*np.arange(1,k+1)-k-1)[None,:,None]
    def crps(z, target):
        return np.abs(z-target[:,None,:]).mean(1) - (np.sort(z,axis=1)*coefficient).sum(1)/(k*k)
    physical_crps = crps(x,y)
    tw = [crps(np.maximum(x, thresholds[:,None,None,i]),
               np.maximum(y, thresholds[:,None,i])) / scale[:,None] for i in range(2)]
    weights = hierarchy_weights(bank.select_rows)[:,None]*np.ones_like(y)/24
    values = dict(mae=np.abs(error), mse=error**2, crps=physical_crps,
                  crps_q95=physical_crps/scale[:,None], twcrps95_q95=tw[0], twcrps99_q95=tw[1],
                  brier95=(prob[:,:,0]-truth[:,:,0])**2,
                  brier99=(prob[:,:,1]-truth[:,:,1])**2,
                  coverage90=((y>=lo)&(y<=hi)).astype(float), width90=hi-lo)
    groups = {'all':np.ones_like(y,dtype=bool), 'ordinary_cell':~truth[:,:,0],
              'ordinary_origin':np.broadcast_to((y.max(1)<=thresholds[:,0])[:,None],y.shape),
              'above_fit_q95':truth[:,:,0], 'above_fit_q99':truth[:,:,1]}
    groups.update(masks or {})
    result = {}
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError('bad mask '+name)
        w = weights*mask; denominator = w.sum()
        if not denominator:
            result[name] = dict(state='NOT_EVALUABLE', cells=0)
            continue
        row = {key:float((value*w).sum()/denominator) for key,value in values.items()}
        row.update(state='EVALUABLE', cells=int(mask.sum()), origins=int(mask.any(1).sum()),
                   rmse=float(np.sqrt(row['mse'])), coverage90_deviation=float(row['coverage90']-.9))
        if np.all(mask == mask[:,:1]):
            origin_mask = mask[:,0]; origin_weights = hierarchy_weights(bank.select_rows)*origin_mask
            origin_weights /= origin_weights.sum()
            for horizon in (6,24):
                predicted = median[:,:horizon]; actual = y[:,:horizon]
                delta = predicted.max(1)-actual.max(1)
                time_delta = predicted.argmax(1)-actual.argmax(1)
                row['peak_h%d_mae'%horizon] = float(origin_weights@np.abs(delta))
                row['peak_h%d_bias'%horizon] = float(origin_weights@delta)
                row['peak_h%d_time_mae'%horizon] = float(origin_weights@np.abs(time_delta))
        result[name] = row
    for h in range(24):
        m = np.zeros_like(y,dtype=bool); m[:,h] = True
        w = weights*m; denominator=w.sum()
        result['h%d'%(h+1)] = {key:float((value*w).sum()/denominator) for key,value in values.items()}
        result['h%d'%(h+1)]['rmse'] = float(np.sqrt(result['h%d'%(h+1)]['mse']))
    w = hierarchy_weights(bank.select_rows)
    for h in (6,24):
        predicted = median[:,:h]; actual = y[:,:h]
        pmax = predicted.max(1); ymax = actual.max(1); delta=pmax-ymax
        pt = predicted.argmax(1)+1; yt=actual.argmax(1)+1
        result['peak_h%d'%h] = dict(mae=float(w@np.abs(delta)), bias=float(w@delta),
             relative_absolute_error=float(w@(np.abs(delta)/np.maximum(ymax,1e-8))),
             peak_time_mae=float(w@np.abs(pt-yt)), peak_time_bias=float(w@(pt-yt)),
             definition='peak of pointwise median curve', unit='m3/s, hours')
    result['nse_by_station'] = {}
    stations = np.asarray([r['canonical_station'] for r in bank.select_rows])
    for station in sorted(set(stations)):
        ids = np.flatnonzero(stations == station)
        ys = y[ids]; es = error[ids]
        station_weights = hierarchy_weights([bank.select_rows[i] for i in ids])[:,None]/24
        if np.all(ys == ys.flat[0]):
            result['nse_by_station'][station] = dict(state='NOT_EVALUABLE', reason='exact_constant_truth')
        else:
            mean = float((station_weights*ys).sum())
            sst = float((station_weights*(ys-mean)**2).sum())
            if sst <= 0 or not np.isfinite(sst):
                raise FloatingPointError('invalid nonconstant station NSE denominator')
            result['nse_by_station'][station] = dict(state='EVALUABLE',
                nse=float(1-(station_weights*es**2).sum()/sst))
    return result
