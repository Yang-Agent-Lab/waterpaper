"""Hourly marginal scores; no time cummax and no assumed joint flood paths."""
import numpy as np
from .dataset import hierarchy_weights


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
