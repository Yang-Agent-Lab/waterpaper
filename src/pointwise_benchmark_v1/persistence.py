"""Persistence plus fit-exposure empirical marginal errors, without optimization."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .coverage import save
from .dataset import sha256
from .metrics import marginal_scores


def empirical_residual_quantiles(bank,exposure,draws):
    arrays=bank.arrays['fit'];residual=arrays['y']-arrays['x'][:,-1,1,None]
    weight=np.bincount(exposure.reshape(-1),minlength=len(residual)).astype(float)
    if weight.sum()<=0:raise ValueError('nonempty fit exposure is required for empirical errors')
    levels=(np.arange(draws)+.5)/draws;result=[]
    for lead in range(residual.shape[1]):
        order=np.argsort(residual[:,lead],kind='stable');cdf=np.cumsum(weight[order])/weight.sum()
        indices=np.minimum(np.searchsorted(cdf,levels,side='left'),len(order)-1)
        result.append(residual[order[indices],lead])
    return np.stack(result,axis=1),weight


def run_persistence(bank,exposure,condition,folder,render):
    folder=Path(folder);condition_sha=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
    if (folder/'completion.json').exists():
        old=json.loads((folder/'completion.json').read_text())
        if old['condition_sha256']!=condition_sha or old['bank_identity']!=bank.identity or old['prediction_sha256']!=sha256(folder/'select_predictions.npz'):
            raise ValueError('existing persistence result identity differs')
        return dict(state='completed_configuration_skipped',model='persistence',route=bank.route)
    if folder.exists():raise FileExistsError('partial empirical baseline is preserved; use a new reviewed run directory')
    folder.mkdir(parents=True)
    residuals,weight=empirical_residual_quantiles(bank,exposure,condition['final_draws'])
    current=bank.arrays['select']['x'][:,-1,1]
    raw=current[:,None,None]+residuals[None,:,:];projected=np.maximum(raw,0)
    scored=marginal_scores(projected,bank.arrays['select']['y'],bank.arrays['select']['thresholds_normalized'],bank.rows['select'])
    np.savez_compressed(folder/'fit_empirical_residuals.npz',marginal_residual_quantiles=residuals,fit_exposure_counts=weight)
    np.savez_compressed(folder/'select_predictions.npz',raw_draws_normalized=raw,draws_normalized=projected,
        truth_normalized=bank.arrays['select']['y'],probability=scored['probability'],observed=scored['observed'],crps=scored['crps'])
    save(folder/'scores.json',scored['summary']);save(folder/'select_rows.json',bank.rows['select']);save(folder/'epochs.json',[])
    config=dict(name='persistence_fit_empirical_error',route=bank.route,parameter_count=0,
        loss_definition='no neural training; route-shared fit-exposure weighted empirical residual marginals',
        residual_definition='future normalized Q at each lead minus current normalized Q in fit samples only',
        pool='global within information route; weighted by the same persisted fit exposure counts',
        station_conditioning='current Q and inherited station Q95 normalization; no station-specific residual fit',
        sample_method='weighted empirical inverse CDF on deterministic midpoint probability grid',
        temporal_joint_distribution='not established; marginal grid does not certify joint flood trajectories',
        physical_projection='raw values saved; uniform zero clipping for scoring only',fit_scales_recomputed=False)
    save(folder/'model_config.json',config);save(folder/'projection.json',dict(raw_negative_fraction=float((raw<0).mean()),
        raw_negative_fraction_by_lead=(raw<0).mean(axis=(0,1)).tolist(),rule='uniform zero clipping for scoring only'))
    render(folder,'persistence_'+bank.route,SimpleNamespace(config=config),bank,[],[],projected,scored,condition,dict(epoch=0))
    result=dict(state='complete',model='persistence',route=bank.route,training_state='not_required_no_optimization',
        selection_state='not_applicable_no_tuned_checkpoint',training_ended=True,selection_ended=True,final_scoring_ended=True,
        updates=0,bank_identity=bank.identity,condition_sha256=condition_sha,
        prediction_sha256=sha256(folder/'select_predictions.npz'),fit_residual_exposures=int(weight.sum()))
    save(folder/'completion.json',result);return result
