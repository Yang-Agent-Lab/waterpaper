"""Synthetic-only state boundary, equal-capacity, zero-origin and fit sampling checks."""
import io
import numpy as np
import pytest
import torch
from pointwise_benchmark_v1.models import build_model
from pointwise_benchmark_v1.state_models import build_state_model,history_state,state_variance
from pointwise_benchmark_v1.state_sampling import extreme_exposure


def test_history_only_and_rain_slots():
    x=torch.arange(2*24*2,dtype=torch.float32).reshape(2,24,2)/20
    q=history_state(x,False);pq=history_state(x,True)
    assert q.shape==(2,19) and torch.equal(q[:,:13],pq[:,:13]) and not q[:,13:].any()
    expected=torch.log1p(x[:,-3:,0].sum(1))
    assert torch.allclose(pq[:,14],expected)
    changed=x.clone();changed[:,:,0]+=100
    assert torch.equal(q,history_state(changed,False))
    assert not torch.equal(pq,history_state(changed,True))
    with pytest.raises(ValueError):history_state(torch.cat([x,x[:,:1]],dim=1),True)


def test_equal_parameters_zero_start_and_real_two_steps():
    torch.set_num_threads(1)
    baseline=build_model('diffpts','pq',24,24,seed=2,variance_floor=1e-8)
    state_q=build_state_model('state_q','pq',24,24)
    state_pq=build_state_model('state_pq','pq',24,24)
    assert state_q.config['parameter_count']==state_pq.config['parameter_count']
    assert state_q.config['parameter_count']-baseline.config['parameter_count']==7504
    for key,value in baseline.state_dict().items():
        assert torch.equal(value,state_q.state_dict()[key]) and torch.equal(value,state_pq.state_dict()[key])
    x=torch.linspace(.1,2,2*24*2).reshape(2,24,2);y=x[:,:,-1].flip(1)
    baseline.eval();expected=baseline.predict_samples(x,5,seed=13)
    for model in (state_q,state_pq):
        model.eval()
        # inverse-softplus/softplus round trip has ordinary floating-point error.
        assert torch.allclose(expected,model.predict_samples(x,5,seed=13),rtol=2e-5,atol=2e-6)
        baseline_loss,_=baseline.loss(x,y,generator=torch.Generator().manual_seed(19))
        initial_loss,_=model.loss(x,y,generator=torch.Generator().manual_seed(19))
        assert torch.allclose(baseline_loss,initial_loss,rtol=2e-6,atol=2e-6)
        initial=model.state_mean.weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for step in range(2):
            model.train();optimizer.zero_grad()
            value,details=model.loss(x,y,generator=torch.Generator().manual_seed(10+step))
            assert torch.isfinite(value);value.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
        assert not torch.equal(initial,model.state_mean.weight)
        model.eval();pred=model.predict_samples(x,5,seed=13)
        saved=io.BytesIO();torch.save(model.state_dict(),saved);saved.seek(0)
        restored=build_state_model(model.name,'pq',24,24)
        restored.load_state_dict(torch.load(saved,weights_only=True))
        assert torch.equal(pred,restored.predict_samples(x,5,seed=13))
        changed=x.clone();changed[:,:,0]+=1
        model.conditions(x);s=model._current_state.clone()
        model.conditions(changed)
        assert torch.equal(s,model._current_state) if model.name=='state_q' else not torch.equal(s,model._current_state)
        assert details['loss_definition']==baseline.config['loss_definition']


def test_extreme_resampling_keeps_station_positions_and_balances():
    rows=[dict(country=country,canonical_station=station,event=event,role='fit')
          for country,station,events in [('CN','a',['e1','e1','e2','e2']),('DE','b',['e3','e3'])]
          for event in events]
    y=np.asarray([[0]*24,[2]*24,[0]*24,[2]*24,[0]*24,[0]*24],dtype=float)
    thresholds=np.ones((6,2));original=np.asarray([[[0,0,0,2,2,4,4,4]]])
    sampled,receipt=extreme_exposure(rows,y,thresholds,original)
    repeated,_=extreme_exposure(rows,y,thresholds,original)
    assert np.array_equal(sampled,repeated)
    assert receipt['station_positions_identical'] and sampled.shape==original.shape
    assert all(rows[int(i)]['canonical_station']==rows[int(j)]['canonical_station'] for i,j in zip(original.flat,sampled.flat))
    for count in receipt['positive_counts_by_station']:
        if count['canonical_station']=='a':assert abs(count['positive']-count['negative'])<=1
        else:assert count['positive']==0 and count['negative']==3
    bad=[dict(r) for r in rows];bad[0]['role']='select'
    with pytest.raises(PermissionError):extreme_exposure(bad,y,thresholds,original)


def test_stable_variance_zero_residual_and_head_gradient():
    g=torch.tensor([1e-8,1e-4,.1,1.,100.,1e20])
    delta=torch.zeros_like(g,requires_grad=True)
    value=state_variance(g,delta)
    assert torch.allclose(value,g,rtol=2e-6,atol=1e-12)
    value.sum().backward()
    assert torch.isfinite(delta.grad).all() and (delta.grad[1:]>0).all()
    assert torch.isfinite(state_variance(g,torch.full_like(g,1000))).all()
