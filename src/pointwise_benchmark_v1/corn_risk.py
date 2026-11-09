"""Deterministic ordered-risk heads on a frozen state_variance_only model."""
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import hierarchy_weights
from .models import random_state


HEADS=('corn_risk','binary_risk')
NUM_CLASSES=3
FEATURE_DIM=35
HIDDEN_DIM=32
RISK_HEAD_PARAMETERS=(FEATURE_DIM+1)*HIDDEN_DIM+(HIDDEN_DIM+1)*2
VARIANCE_FLOOR=1e-8
PRIOR_SMOOTHING=1e-6


class DirectRiskHead(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind not in HEADS:raise ValueError('risk head must be corn_risk or binary_risk')
        self.kind=kind
        self.lin1=nn.Linear(FEATURE_DIM,HIDDEN_DIM)
        self.activation=nn.SiLU()
        self.lin2=nn.Linear(HIDDEN_DIM,2)

    def forward(self,features):
        if features.ndim!=3 or features.shape[-1]!=FEATURE_DIM:
            raise ValueError('risk head input must be [batch,lead,35]')
        return self.lin2(self.activation(self.lin1(features)))

    def probabilities(self,features):
        logits=self(features)
        if self.kind=='binary_risk':return torch.sigmoid(logits)
        q1=torch.sigmoid(logits[...,0])
        q2=torch.sigmoid(logits[...,1])
        return torch.stack((q1,q1*q2),dim=-1)


def risk_labels(y,thresholds):
    """Return ordered classes and [Q95,Q99] exceedance labels."""
    if y.ndim!=2 or thresholds.shape!=(len(y),2):raise ValueError('risk labels require y[B,F] and thresholds[B,2]')
    if not torch.isfinite(y).all() or not torch.isfinite(thresholds).all():raise ValueError('risk labels contain nonfinite values')
    if (thresholds[:,1]<thresholds[:,0]).any():raise ValueError('fit Q99 threshold must be >= fit Q95 for every origin')
    q95=y>thresholds[:,0,None]
    q99=y>thresholds[:,1,None]
    if (q99&~q95).any():raise AssertionError('Q99 exceedance must imply Q95 exceedance')
    labels=q95.to(torch.long)+q99.to(torch.long)
    targets=torch.stack((q95,q99),dim=-1).to(y.dtype)
    return labels,targets


def corn_loss_parts(logits,labels):
    """Paper CORN objective: conditional BCE normalized by the total risk-set rows.

    Method definition: https://arxiv.org/html/2111.08851v3
    Author reference implementation: https://github.com/Raschka-research-group/coral-pytorch
    """
    if logits.ndim!=2 or logits.shape[1]!=NUM_CLASSES-1 or labels.shape!=(len(logits),):
        raise ValueError('CORN logits must be [N,2] with one ordinal label per row')
    if not len(labels):raise ValueError('CORN first risk set cannot be empty')
    if (labels<0).any() or (labels>=NUM_CLASSES).any():raise ValueError('ordinal labels must be 0, 1, or 2')
    first_target=(labels>0).to(logits.dtype)
    numerator=F.binary_cross_entropy_with_logits(logits[:,0],first_target,reduction='sum')
    denominator=len(labels)
    second_mask=labels>0
    second_count=int(second_mask.sum().item())
    if second_count:
        second_target=(labels[second_mask]>1).to(logits.dtype)
        numerator=numerator+F.binary_cross_entropy_with_logits(logits[second_mask,1],second_target,reduction='sum')
        denominator+=second_count
    return numerator/denominator,numerator,denominator


def independent_binary_loss_parts(logits,targets):
    if logits.shape!=targets.shape or logits.ndim!=2 or logits.shape[1]!=2:
        raise ValueError('independent binary loss requires logits/targets [N,2]')
    numerator=F.binary_cross_entropy_with_logits(logits,targets,reduction='sum')
    return numerator/logits.numel(),numerator,logits.numel()


def batch_risk_loss(head,features,labels,targets):
    logits=head(features).reshape(-1,2)
    if head.kind=='corn_risk':return corn_loss_parts(logits,labels.reshape(-1))
    return independent_binary_loss_parts(logits,targets.reshape(-1,2).to(logits.dtype))


def fit_prior(y,thresholds,rows):
    """Equal-country, within-country station/event weighted fit-only class prior."""
    y=np.asarray(y,dtype=np.float64);thresholds=np.asarray(thresholds,dtype=np.float64)
    if y.ndim!=2 or thresholds.shape!=(len(rows),2) or len(rows)!=len(y):raise ValueError('fit prior identity shape mismatch')
    if not np.isfinite(y).all() or not np.isfinite(thresholds).all() or (thresholds[:,1]<thresholds[:,0]).any():
        raise ValueError('fit prior requires finite ordered fit thresholds and targets')
    labels=(y>thresholds[:,0,None]).astype(np.int8)+(y>thresholds[:,1,None]).astype(np.int8)
    country_priors=[]
    countries=sorted({row['country'] for row in rows})
    for country in countries:
        ids=np.asarray([i for i,row in enumerate(rows) if row['country']==country],dtype=np.int64)
        weights=hierarchy_weights([rows[i] for i in ids])/y.shape[1]
        weighted=np.bincount(labels[ids].reshape(-1),weights=np.repeat(weights,y.shape[1]),minlength=NUM_CLASSES)
        country_priors.append(weighted/weighted.sum())
    prior=np.mean(country_priors,axis=0)
    prior=(prior+PRIOR_SMOOTHING)/(1.+NUM_CLASSES*PRIOR_SMOOTHING)
    p95=float(prior[1]+prior[2]);p99=float(prior[2])
    if not 0.<p99<p95<1.:raise ValueError('smoothed fit-only priors must be strictly ordered and finite')
    return dict(class_probability=prior.tolist(),p95=p95,p99=p99,
        corn_conditional_q99_given_q95=p99/p95,smoothing=PRIOR_SMOOTHING,
        weighting='equal country; within country station, inherited event, origin; average across leads')


def build_risk_head(kind,seed,prior,device):
    if kind not in HEADS:raise ValueError('unknown direct risk head')
    with random_state(torch.device(device),seed=seed):
        head=DirectRiskHead(kind).to(device)
    q1=prior['p95']
    q2=prior['p99']/prior['p95'] if kind=='corn_risk' else prior['p99']
    bias=torch.tensor([math.log(q1/(1.-q1)),math.log(q2/(1.-q2))],dtype=head.lin2.bias.dtype,device=device)
    with torch.no_grad():
        head.lin2.weight.zero_()
        head.lin2.bias.copy_(bias)
    if sum(p.numel() for p in head.parameters())!=RISK_HEAD_PARAMETERS:
        raise AssertionError('risk readout parameter count changed')
    return head


def frozen_risk_features(base,x):
    """Read [state, scaled mean, log floored variance, normalized lead] with no gradients."""
    if base.training or any(module.training for module in base.modules()):
        raise RuntimeError('frozen forecasting model and every submodule must remain in eval mode')
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError('frozen forecasting parameters must have requires_grad=False')
    with torch.no_grad():
        mu,g=base.conditions(x)
        state=base._current_state
        if mu.shape!=(len(x),base.forecast_hours,1) or g.shape!=mu.shape or state.shape!=(len(x),32):
            raise ValueError('frozen model returned unexpected risk representation shapes')
        mu_scaled=mu.sign()*torch.log1p(mu.abs())
        log_variance=torch.log(g.clamp_min(VARIANCE_FLOOR))
        state_by_lead=state[:,None,:].expand(-1,base.forecast_hours,-1)
        lead=torch.arange(1,base.forecast_hours+1,device=x.device,dtype=x.dtype).view(1,-1,1)/base.forecast_hours
        lead=lead.expand(len(x),-1,-1)
        features=torch.cat((state_by_lead,mu_scaled,log_variance,lead),dim=-1)
        if features.shape!=(len(x),base.forecast_hours,FEATURE_DIM) or not torch.isfinite(features).all():
            raise FloatingPointError('invalid frozen risk features')
        return features


def module_state_sha256(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape,dtype=np.int64).tobytes());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def direct_probability_scores(probability,y,thresholds,rows):
    """Brier/reliability scores from direct probabilities; no draws or CRPS are fabricated."""
    probability=np.asarray(probability,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    thresholds=np.asarray(thresholds,dtype=np.float64)
    if y.ndim!=2 or probability.ndim!=3 or thresholds.shape!=(len(rows),2) or probability.shape!=(len(rows),y.shape[1],2) or y.shape[0]!=len(rows):
        raise ValueError('direct risk scoring shapes/identities mismatch')
    if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any() or not np.isfinite(y).all() or not np.isfinite(thresholds).all():
        raise ValueError('direct risk predictions/truth must be finite probabilities')
    if (thresholds[:,1]<thresholds[:,0]).any():raise ValueError('score thresholds must be ordered')
    observed=np.stack((y>thresholds[:,0,None],y>thresholds[:,1,None]),axis=-1).astype(np.float64)
    if (observed[:,:,1]>observed[:,:,0]).any():raise AssertionError('observed Q99 events must be nested within Q95')
    summary=[]
    for country in sorted({row['country'] for row in rows}):
        ids=np.asarray([i for i,row in enumerate(rows) if row['country']==country],dtype=np.int64)
        country_rows=[rows[i] for i in ids];weights=hierarchy_weights(country_rows)
        for lead in range(y.shape[1]):
            for threshold,label in enumerate(('Q95','Q99')):
                p=probability[ids,lead,threshold];truth=observed[ids,lead,threshold].astype(bool)
                error=(p-truth.astype(np.float64))**2
                bins=[]
                for bin_index in range(10):
                    lower=bin_index/10;upper=(bin_index+1)/10
                    mask=(p>=lower)&((p<upper) if bin_index<9 else (p<=upper))
                    total_weight=float(weights[mask].sum())
                    bins.append(dict(lower=lower,upper=upper,n=int(mask.sum()),
                        mean_probability=float(np.average(p[mask],weights=weights[mask])) if total_weight else None,
                        observed_fraction=float(np.average(truth[mask],weights=weights[mask])) if total_weight else None,
                        state='observed' if total_weight else 'empty',reason=None if total_weight else 'no_sample_in_bin'))
                positive_weight=float(weights[truth].sum());negative_weight=float(weights[~truth].sum())
                row=dict(country=country,role=country_rows[0]['role'],horizon=lead+1,
                    target='pointwise_exceedance_'+label,brier=float(weights@error),
                    crps=None,crps_unit=None,n_origins=len(ids),independent_events=False,
                    positive_brier=float(np.average(error[truth],weights=weights[truth])) if positive_weight else None,
                    negative_brier=float(np.average(error[~truth],weights=weights[~truth])) if negative_weight else None,
                    prevalence=float(np.average(truth,weights=weights)),
                    mean_probability=float(np.average(p,weights=weights)),reliability_bins=bins,
                    state='observed',reason='CRPS_NOT_APPLICABLE: direct binary risk head has no continuous flow distribution')
                summary.append(row)
    return dict(summary=summary,observed=observed,
        china_primary_brier=float(np.mean([row['brier'] for row in summary if row['country']=='CN'])),
        crps_state='NOT_APPLICABLE',ordering_violation_rate=float((probability[:,:,1]>probability[:,:,0]).mean()),
        ordering_violation_rate_by_lead=(probability[:,:,1]>probability[:,:,0]).mean(axis=0).astype(float).tolist(),
        probability_semantics='marginal pointwise exceedance probability; time direction unrestricted')


def save_direct_risk_predictions(path,probability,observed,y,thresholds,sample_ids):
    """Persist direct risks while preserving canonical target/threshold precision."""
    np.savez_compressed(path,probability=np.asarray(probability,dtype=np.float32),
        observed=np.asarray(observed,dtype=np.uint8),truth_normalized=np.asarray(y),
        thresholds_normalized=np.asarray(thresholds),sample_ids=np.asarray(sample_ids,dtype='U64'))
