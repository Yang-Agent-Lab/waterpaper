"""One common actual loss/update and original-protocol evaluator for T/M/A."""
import numpy as np
import torch
from src.pointwise_benchmark_v1.metrics import marginal_scores
from .sampling import NoisePlan
from .losses import distribution_scores

def objective(model,batch,seed,lambda_c,lambda_e,distribution,checkpoint_grad=False):
    mse,nll,balance=model.native_loss(batch["x"],batch["y"],seed)
    crps=mse.new_zeros(());es=crps
    if distribution:
        # Prespecified first8 origins for the full20-step K8 auxiliary path.
        # Native epsilon MSE above continues to use the complete batch32.
        aux={key:(value[:8] if isinstance(value,(torch.Tensor,list)) else value) for key,value in batch.items()}
        mu,_=model.conditions(aux["x"])
        plan=NoisePlan.create(mu,aux["sample_ids"],8,seed+10000000)
        samples,_=model.sample(aux["x"],plan,checkpoint_grad=checkpoint_grad)
        crps,es=distribution_scores(samples,aux["y"],aux["q_scale"],aux["q95"])
    balance_weight=.01 if lambda_c>0 or lambda_e>0 else 0.
    total=mse+lambda_c*crps+lambda_e*es+balance_weight*balance
    return total,dict(noise_mse=mse,nll_frozen_constant=nll,crps_u=crps,es_u=es,balance=balance,
                      balance_weight=mse.new_tensor(balance_weight),native_origins=mse.new_tensor(len(batch["x"])),
                      auxiliary_origins=mse.new_tensor(min(8,len(batch["x"])) if distribution else 0))

def step(model,batch,optimizer,seed,lambda_c,lambda_e,distribution,checkpoint_grad=False,clip=1.):
    model.train();optimizer.zero_grad(set_to_none=True)
    total,parts=objective(model,batch,seed,lambda_c,lambda_e,distribution,checkpoint_grad)
    if not torch.isfinite(total):raise FloatingPointError("nonfinite full objective")
    total.backward();health={}
    for name,parameter in model.adapter.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():raise FloatingPointError("nonfinite gradient:"+name)
            health[name]=float(parameter.grad.float().norm())
    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),clip,error_if_nonfinite=True)
    optimizer.step()
    for name,p in model.adapter.state_dict().items():
        if not torch.isfinite(p).all():raise FloatingPointError("nonfinite updated state:"+name)
    return dict(total=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}),health

@torch.no_grad()
def evaluate(model,data,indices,k,seed,device,chunk=32,enabled=True):
    model.eval();outputs=[]
    for start in range(0,len(indices),32):
        ix=np.asarray(indices[start:start+32]);b=data.batch(ix,device);mu,_=model.conditions(b["x"])
        plan=NoisePlan.create(mu,b["sample_ids"],k,seed+start)
        pieces=[]
        for first in range(0,k,chunk):
            samples,_=model.sample(b["x"],plan,draws=range(first,min(k,first+chunk)),enabled=enabled)
            pieces.append(samples.cpu().numpy())
        outputs.append(np.concatenate(pieces,1))
    native=np.concatenate(outputs);scale=data.scale[indices];q95=data.q95[indices]
    physical=np.maximum(native*scale[:,None,None],0.)
    normalized=physical/q95[:,None,None];truth=data.y_physical[indices]/q95[:,None]
    thresholds=np.stack([data.q95[indices],data.q99[indices]],-1)/q95[:,None]
    rows=[data.rows[int(i)] for i in indices]
    metrics=marginal_scores(normalized,truth,thresholds,rows)
    return dict(metrics=metrics,draws_normalized_q95=normalized,truth_normalized_q95=truth,
                probability=metrics["probability"],projection_frequency=float((native<0).mean()))
