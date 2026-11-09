"""Focused risk-readout algebra and optional author-reference checks."""
import importlib.metadata
import inspect
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .corn_risk import (HEADS, build_risk_head, corn_loss_parts,
    direct_probability_scores, independent_binary_loss_parts,
    save_direct_risk_predictions)


def check_risk_readouts(device='cpu'):
    device=torch.device(device)
    logits=torch.tensor([[-1.2,.8],[.3,-.7],[1.1,.2],[-.1,2.]],device=device,requires_grad=True)
    labels=torch.tensor([0,1,2,0],device=device)
    loss,numerator,denominator=corn_loss_parts(logits,labels)
    first=F.binary_cross_entropy_with_logits(logits[:,0],(labels>0).to(logits.dtype),reduction='sum')
    mask=labels>0
    second=F.binary_cross_entropy_with_logits(logits[mask,1],(labels[mask]>1).to(logits.dtype),reduction='sum')
    reference=(first+second)/(len(labels)+int(mask.sum()))
    if denominator!=6 or not torch.allclose(loss,reference,rtol=0,atol=1e-7) or not torch.allclose(numerator,first+second,rtol=0,atol=1e-7):
        raise AssertionError('CORN loss differs from hand-computed conditional-riskset objective')
    loss.backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():raise AssertionError('CORN hand-case gradient is missing/nonfinite')

    empty_logits=torch.tensor([[.2,-2.],[-.4,1.]],device=device,requires_grad=True)
    empty_labels=torch.zeros(2,dtype=torch.long,device=device)
    empty_loss,_,empty_denominator=corn_loss_parts(empty_logits,empty_labels)
    expected_empty=F.binary_cross_entropy_with_logits(empty_logits[:,0],torch.zeros(2,device=device),reduction='mean')
    if empty_denominator!=2 or not torch.allclose(empty_loss,expected_empty,rtol=0,atol=1e-7) or not torch.isfinite(empty_loss):
        raise AssertionError('empty second CORN risk set must safely retain only the first task')
    empty_loss.backward()
    if empty_logits.grad is None or not torch.isfinite(empty_logits.grad).all() or torch.count_nonzero(empty_logits.grad[:,1]).item():
        raise AssertionError('empty CORN second risk set produced invalid secondary-task gradient')

    targets=torch.tensor([[0.,0.],[1.,0.],[1.,1.]],device=device)
    binary_logits=torch.zeros((3,2),device=device,requires_grad=True)
    binary,binary_num,binary_den=independent_binary_loss_parts(binary_logits,targets)
    expected_binary=F.binary_cross_entropy_with_logits(binary_logits,targets,reduction='mean')
    if binary_den!=6 or not torch.allclose(binary,expected_binary,rtol=0,atol=1e-7) or not torch.isfinite(binary_num):
        raise AssertionError('independent binary BCE differs from direct two-marginal objective')
    binary.backward()

    prior=dict(p95=.16,p99=.04)
    corn=build_risk_head('corn_risk',2,prior,device)
    independent=build_risk_head('binary_risk',2,prior,device)
    if not torch.equal(corn.lin1.weight,independent.lin1.weight) or not torch.equal(corn.lin1.bias,independent.lin1.bias):
        raise AssertionError('risk heads do not share the same seed2 hidden-layer initialization')
    features=torch.randn(3,24,35,device=device,generator=torch.Generator(device=device).manual_seed(41))
    corn_p=corn.probabilities(features);binary_p=independent.probabilities(features)
    if not torch.allclose(corn_p,binary_p,rtol=0,atol=1e-7):raise AssertionError('fit-prior initial marginal probabilities are not identical')
    if (corn_p[:,:,1]>corn_p[:,:,0]).any():raise AssertionError('CORN unconditional thresholds must be ordered')
    if sum(p.numel() for p in corn.parameters())!=sum(p.numel() for p in independent.parameters()) or sum(p.numel() for p in corn.parameters())!=1218:
        raise AssertionError('risk heads must have the same approved parameter count (1218)')
    if any(torch.count_nonzero(p).item() for p in corn.lin2.weight) or any(torch.count_nonzero(p).item() for p in independent.lin2.weight):
        raise AssertionError('readout output weights must be zero for matched prior initialization')

    # Verify that the independent head's raw outputs are never silently sorted.
    with torch.no_grad():
        independent.lin2.weight.zero_();independent.lin2.bias.copy_(torch.tensor([-2.,2.],device=device))
    raw=independent.probabilities(features)
    y=torch.zeros(3,24,device=device);thresholds=torch.tensor([[1.,2.]]*3,device=device)
    rows=[dict(country='CN',canonical_station='s',event='e',role='select') for _ in range(3)]
    scores=direct_probability_scores(raw.detach().cpu().numpy(),y.cpu().numpy(),thresholds.cpu().numpy(),rows)
    if scores['ordering_violation_rate']!=1.:raise AssertionError('independent head probability violation was hidden or projected')

    # Preserve exact canonical targets around a threshold; float32 copies can
    # collapse a strict exceedance to equality and change its label.
    y64=np.asarray([[np.nextafter(1.0,2.0),np.nextafter(2.0,3.0)]],dtype=np.float64)
    thresholds64=np.asarray([[1.0,2.0]],dtype=np.float64)
    observed64=np.stack((y64>thresholds64[:,0,None],y64>thresholds64[:,1,None]),axis=-1)
    with tempfile.TemporaryDirectory() as temp_dir:
        path=f'{temp_dir}/risk_predictions.npz'
        save_direct_risk_predictions(path,np.full((1,2,2),.1),observed64,y64,thresholds64,['precision-case'])
        with np.load(path,allow_pickle=False) as saved:
            if saved['truth_normalized'].dtype!=np.dtype('float64') or saved['thresholds_normalized'].dtype!=np.dtype('float64'):
                raise AssertionError('canonical risk labels/thresholds must retain float64 precision')
            if not np.array_equal(saved['truth_normalized'],y64) or not np.array_equal(saved['thresholds_normalized'],thresholds64):
                raise AssertionError('risk prediction serialization changed canonical target values')
            if not np.array_equal(saved['observed'].astype(bool),observed64):
                raise AssertionError('risk prediction serialization changed canonical labels')
            rounded=np.stack((saved['truth_normalized'].astype(np.float32)>saved['thresholds_normalized'].astype(np.float32)[:,0,None],
                saved['truth_normalized'].astype(np.float32)>saved['thresholds_normalized'].astype(np.float32)[:,1,None]),axis=-1)
            if np.array_equal(rounded,observed64):
                raise AssertionError('precision boundary fixture did not expose float32 label collapse')

    author_reference=dict(state='not_installed')
    try:
        from coral_pytorch.losses import corn_loss as reference_corn_loss
    except Exception as exc:
        try:version=importlib.metadata.version('coral-pytorch')
        except importlib.metadata.PackageNotFoundError:version='not_installed'
        author_reference=dict(state='import_unavailable',version=version,error=type(exc).__name__+': '+str(exc))
    else:
        source_path=inspect.getsourcefile(reference_corn_loss)
        try:source_line=inspect.getsourcelines(reference_corn_loss)[1]
        except (OSError,TypeError):source_line=None
        try:
            reference_value=reference_corn_loss(logits.detach(),labels,num_classes=3)
        except Exception as exc:
            try:version=importlib.metadata.version('coral-pytorch')
            except importlib.metadata.PackageNotFoundError:version='unknown'
            author_reference=dict(state='reference_call_failed',version=version,
                source_path=source_path,source_line=source_line,
                input_shape=list(logits.shape),label_shape=list(labels.shape),error=type(exc).__name__+': '+str(exc))
        else:
            try:version=importlib.metadata.version('coral-pytorch')
            except importlib.metadata.PackageNotFoundError:version='unknown'
            author_reference=dict(state='verified' if torch.allclose(reference_value,reference.detach(),rtol=1e-6,atol=1e-7) else 'value_differs_from_paper_formula',
                version=version,source_path=source_path,source_line=source_line,
                input_shape=list(logits.shape),label_shape=list(labels.shape),
                absolute_difference=float((reference_value-reference.detach()).abs()))
    return dict(state='passed',device=str(device),parameter_count=1218,
        checks=['hand-calculated paper CORN risk-set BCE and denominator',
            'empty secondary risk set skips safely with finite gradients',
            'independent two-marginal BCE',
            'same fit-prior initialized probabilities and equal parameter count',
            'CORN threshold nesting and raw independent-head violation preserved',
            'canonical target/threshold precision preserved in prediction serialization'],
        author_reference=author_reference)
