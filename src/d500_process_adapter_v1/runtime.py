import copy
import hashlib
import random
import numpy as np
import torch

def fingerprint(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
def restore_rng(value):
    random.setstate(value["python"]);np.random.set_state(value["numpy"]);torch.set_rng_state(value["torch"])
    if value["cuda"]:torch.cuda.set_rng_state_all(value["cuda"])
def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)
def snapshot(model,optimizer,scheduler,completed,cursor,config):
    if not hasattr(model,"_verified_base_fingerprint"):
        model._verified_base_fingerprint=fingerprint(model.base)
    return cpu_copy(dict(adapter=model.adapter.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        completed=completed,cursor=cursor,rng=rng(),config=config,base_fingerprint=model._verified_base_fingerprint))
def restore(state,model,optimizer,scheduler):
    assert state["base_fingerprint"]==fingerprint(model.base)
    model.adapter.load_state_dict(state["adapter"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"]);return state["completed"],state["cursor"]
