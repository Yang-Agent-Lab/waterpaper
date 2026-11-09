"""Non-diagonal U estimates in physical Q/Q95 units, with finite norm subgradient at0."""
import math
import torch

def distribution_scores(samples,target,q_scale,q95):
    assert samples.ndim==3 and samples.shape[1]>=2
    k,h=samples.shape[1:]
    # Bank target/sampler output in inherited q_scale units; recover physical Q then Q95.
    # Existing final evaluation projects physical negative samples to0, retained here.
    z=(samples*q_scale[:,None,None]).clamp_min(0)/q95[:,None,None]
    truth=target*q_scale[:,None]/q95[:,None]
    a,b=torch.triu_indices(k,k,offset=1,device=samples.device)
    diff=z[:,a]-z[:,b]
    crps=(z-truth[:,None,:]).abs().mean((1,2))-diff.abs().mean((1,2))*.5
    # Mean over unordered pairs equals the ordered off-diagonal expectation.
    es=torch.linalg.vector_norm(z-truth[:,None,:],dim=-1).mean(1)/math.sqrt(h)-.5*torch.linalg.vector_norm(diff,dim=-1).mean(1)/math.sqrt(h)
    return crps.mean(),es.mean()
