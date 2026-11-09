"""Separate exact serialized-state restoration from CUDA kernel update reproducibility."""
import warnings
import numpy as np
import torch
from torch.nn import functional as F

ATOL=1e-7
RTOL=1e-5

def reflection_backward_evidence(device):
    previous=torch.are_deterministic_algorithms_enabled()
    previous_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    caught=[];error=None
    report=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                operation="F.pad(input[2,3,24],(4,4),mode='reflect').square().sum().backward()",
                original_deterministic=previous,original_warn_only=previous_warn)
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                x=torch.linspace(-1.,1.,144,device=device,dtype=torch.float32).reshape(2,3,24).requires_grad_()
                F.pad(x,(4,4),mode='reflect').square().sum().backward()
                if device.type=='cuda':torch.cuda.synchronize(device)
            except RuntimeError as exc:error=dict(type=type(exc).__name__,message=str(exc))
            caught=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]
    finally:
        torch.use_deterministic_algorithms(previous,warn_only=previous_warn)
    messages=([error['message']] if error else [])+[w['message'] for w in caught]
    explicit=any('determin' in message.lower() and ('reflect' in message.lower() or 'pad' in message.lower()) for message in messages)
    report.update(error=error,warnings=caught,explicit_cuda_reflection_nondeterminism=device.type=='cuda' and explicit,
                  restored_deterministic=torch.are_deterministic_algorithms_enabled(),
                  restored_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                  interpretation='specific tested kernel reports a deterministic-algorithm limitation; does not establish that it uniquely caused model replay differences')
    assert report['restored_deterministic']==previous and report['restored_warn_only']==previous_warn
    return report

def tensor_bits_equal(a,b):
    if a.dtype!=b.dtype or a.shape!=b.shape:return False
    a=a.detach().cpu().contiguous().reshape(-1);b=b.detach().cpu().contiguous().reshape(-1)
    return torch.equal(a.view(torch.uint8),b.view(torch.uint8))

def exact_state_report(actual,saved):
    mismatches=[];tensors=[]
    def walk(a,b,path):
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            equal=tensor_bits_equal(a,b)
            tensors.append(dict(path=path,shape=list(a.shape),dtype=str(a.dtype),bitwise_equal=equal))
            if not equal:mismatches.append(path)
        elif isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
            if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():mismatches.append(path)
        elif isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):mismatches.append(path+'.keys')
            for key in a.keys()&b.keys():walk(a[key],b[key],path+'.'+str(key))
        elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):mismatches.append(path+'.sequence')
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'.'+str(i))
        elif type(a)!=type(b) or a!=b:mismatches.append(path)
    walk(actual,saved,'state')
    return dict(restore_state_exact=not mismatches,mismatches=mismatches,tensor_checks=tensors,
                checked='adapter; optimizer moments/step/param groups; scheduler; Python/numpy/torch/CUDA RNG; snapshot cursor/config/base identity',comparison='bit patterns and dtype/shape for tensors and numpy arrays; exact nested scalar/container values')

def next_update_report(actual,expected,nondeterminism):
    parameters=[];all_bits=True;all_close=True
    assert set(actual)==set(expected)
    for name,value in actual.items():
        a=value.detach().cpu();b=expected[name].detach().cpu()
        assert a.shape==b.shape and a.dtype==b.dtype
        bits=tensor_bits_equal(a,b);all_bits=all_bits and bits
        if a.is_floating_point():
            aa=a.double();bb=b.double();difference=(aa-bb).abs()
            rel=difference/bb.abs().clamp_min(torch.finfo(b.dtype).tiny)
            close=bool(torch.isclose(a,b,atol=ATOL,rtol=RTOL).all())
            maxabs=float(difference.max()) if a.numel() else 0.;maxrel=float(rel.max()) if a.numel() else 0.
        else:close=bits;maxabs=float((a.double()-b.double()).abs().max()) if a.numel() else 0.;maxrel=None
        abytes=a.contiguous().reshape(-1).view(torch.uint8).reshape(-1,a.element_size())
        bbytes=b.contiguous().reshape(-1).view(torch.uint8).reshape(-1,b.element_size())
        parameters.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),maxabs=maxabs,maxrel_reference_clamped_to_dtype_tiny=maxrel,different_value_elements=int((a!=b).sum()),different_bit_pattern_elements=int((abytes!=bbytes).any(1).sum()),elements=a.numel(),bitwise_equal=bits,close=close))
        all_close=all_close and close
    evidence=nondeterminism['explicit_cuda_reflection_nondeterminism']
    accepted=all_bits or (evidence and all_close)
    return dict(next_update_bitwise_equal=all_bits,next_update_close=all_close,accepted=accepted,
                tolerance_allowed_by_explicit_reflection_evidence=evidence,atol=ATOL,rtol=RTOL,
                tolerance_scope='FP32 next CUDA update only; serialized restoration and writer copy remain bitwise exact',
                parameters=parameters)
