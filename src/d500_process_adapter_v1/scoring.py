"""Reuse accepted FP64 marginal/joint/phase definitions, separate U training scores."""
import numpy as np
from .joint_scores import row_metrics,aggregate_metrics,csvsave
from .reference_scoring import phases,summarize,weights

def energy_score(draws,truth):
    n,k,h=draws.shape;first=np.linalg.norm(draws-truth[:,None,:],axis=-1).mean(1)/np.sqrt(h)
    pair=np.zeros(n)
    for i in range(k-1):pair+=np.linalg.norm(draws[:,i:i+1,:]-draws[:,i+1:,:],axis=-1).sum(1)/np.sqrt(h)
    return first-pair/(k*k)

def score(draws_q95,truth_q95,thresholds_q95,rows,truth_physical=None,q95_physical=None):
    if q95_physical is None:raise ValueError("explicit physical Q95 required for m3/s MAE")
    n,k,h=draws_q95.shape;w=np.full((n,k),1/k,np.float64)
    values,prob=row_metrics(draws_q95,w,truth_q95,thresholds_q95)
    values["mae_m3_s"]*=np.asarray(q95_physical,np.float64)
    values["energy_score_q95"]=energy_score(draws_q95,truth_q95)
    summary=aggregate_metrics(values,rows)
    observed=truth_q95[:,:,None]>thresholds_q95[:,None,:]
    phase=None;long=[];support={}
    if n==1706:
        keys,masks,support=phases(rows,truth_q95 if truth_physical is None else truth_physical,observed)
        phase,long=summarize(prob,observed,keys,masks)
    constraints=dict(brier=summary["brier"],crps_q95=summary["crps_q95"],q99_brier=summary["q99_brier"])
    for q in (95,99):
        constraints[f"q{q}_entry_first3_brier"]=(phase.get(f"q{q}_entry_first3",{}).get("1-24h",{}).get("phase_threshold_brier") if phase else None)
    return dict(summary=summary,constraints=constraints,phase=phase,phase_leads=long,phase_support=support,
                probability=prob,observed=observed,origin_values=values,
                recession_phase="NOT_EVALUABLE_no_separate_exit_phase_in_original_accepted_definition")

def mc_tolerance(draws,truth,thresholds,rows,truth_physical):
    # Fixed16 without-replacement subsets, prior to any candidate updates/results.
    assert draws.shape==(1706,256,24)
    original=constraints_only(draws,truth,thresholds,rows,truth_physical)
    if any(v is None for v in original.values()):raise ValueError("required original phase constraint not evaluable")
    rng=np.random.default_rng(20261022);subsets=[];deltas={k:0. for k in original}
    # MC needs only constrained marginals/CRPS and phase, not expensive auxiliary ES/variograms.
    keys,masks,_=phases(rows,truth_physical,truth[:,:,None]>thresholds[:,None,:])
    w=weights(keys)
    for j in range(16):
        ix=rng.choice(256,128,replace=False);x=draws[:,ix]
        prob=(x[:,:,:,None]>thresholds[:,None,None,:]).mean(1)
        ph,_=summarize(prob,truth[:,:,None]>thresholds[:,None,:],keys,masks)
        sx=np.sort(x,axis=1);coeff=(2*np.arange(1,129)-129)[None,:,None]
        crps=np.abs(x-truth[:,None]).mean(1)-(sx*coeff).sum(1)/(128*128)
        v=dict(brier=ph["all"]["1-24h"]["mean_brier"],q99_brier=ph["all"]["1-24h"]["q99_brier"],crps_q95=float(w@crps.mean(1)),
               q95_entry_first3_brier=ph["q95_entry_first3"]["1-24h"]["phase_threshold_brier"],q99_entry_first3_brier=ph["q99_entry_first3"]["1-24h"]["phase_threshold_brier"])
        for metric,value in v.items():deltas[metric]=max(deltas[metric],abs(value-original[metric]))
        subsets.append(dict(index=j,member_indices=ix.tolist(),constraints=v))
    return dict(seed=20261022,full256=original,delta=deltas,subsets=subsets,meaning="finite-member sensitivity, not confidence/significance/practical noninferiority margin")

def constraints_only(draws,truth,thresholds,rows,truth_physical):
    n,k,h=draws.shape
    probability=(draws[:,:,:,None]>thresholds[:,None,None,:]).mean(1)
    observed=truth[:,:,None]>thresholds[:,None,:]
    keys,masks,_=phases(rows,truth_physical,observed)
    summary,_=summarize(probability,observed,keys,masks)
    sorted_draws=np.sort(draws,axis=1);coefficient=(2*np.arange(1,k+1)-k-1)[None,:,None]
    crps=np.abs(draws-truth[:,None]).mean(1)-(sorted_draws*coefficient).sum(1)/(k*k)
    return dict(brier=summary["all"]["1-24h"]["mean_brier"],q99_brier=summary["all"]["1-24h"]["q99_brier"],
                crps_q95=float(weights(keys)@crps.mean(1)),q95_entry_first3_brier=summary["q95_entry_first3"]["1-24h"]["phase_threshold_brier"],
                q99_entry_first3_brier=summary["q99_entry_first3"]["1-24h"]["phase_threshold_brier"])
