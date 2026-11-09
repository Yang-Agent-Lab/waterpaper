"""Focused numeric-floor checks; server execution only."""
import torch
import pytest
from src.pointwise_benchmark_v1.models import build_model


@pytest.mark.parametrize('route',['pq','q_only'])
def test_paper_floor_is_shared_and_zero_variance_loss_gradient_finite(route):
    torch.set_num_threads(1)
    model=build_model('diffpts',route,24,24,variance_floor=1e-8)
    channels=2 if route=='pq' else 1
    x=torch.linspace(.1,1.,2*24*channels).reshape(2,24,channels);y=torch.ones(2,24)
    with torch.no_grad():
        for p in model.variance.parameters():p.zero_()
        list(model.variance.modules())[-1].bias.fill_(-1000.)
    captured={}
    loss,_=model.loss(x,y,generator=torch.Generator().manual_seed(2),observer=captured.update)
    assert torch.isfinite(loss) and torch.all(captured['gx']==1e-8)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    original=model.author_functions['p_sample_loop'];seen={}
    def sampler(*args):
        seen['gx']=args[4];return [torch.zeros_like(args[3])]
    model.author_functions['p_sample_loop']=sampler
    try:assert model.predict_samples(x,3).shape==(2,3,24)
    finally:model.author_functions['p_sample_loop']=original
    assert torch.all(seen['gx']==1e-8)


def test_positive_variance_original_loss_unchanged_and_P_path_preserved():
    raw=build_model('diffpts','pq',24,24,seed=2)
    fixed=build_model('diffpts','pq',24,24,seed=2,variance_floor=1e-8)
    fixed.load_state_dict(raw.state_dict());raw.eval();fixed.eval()
    x=torch.linspace(.1,1.,96).reshape(2,24,2);y=torch.ones(2,24)
    mu,gx=fixed.conditions(x);old_mu,old_gx=raw.conditions(x)
    assert torch.equal(mu,old_mu) and torch.equal(gx,old_gx)
    a,_=raw.loss(x,y,generator=torch.Generator().manual_seed(9))
    b,_=fixed.loss(x,y,generator=torch.Generator().manual_seed(9))
    assert torch.equal(a,b)
    changed=x.clone();changed[:,:,0]+=1
    next_mu,next_gx=fixed.conditions(changed)
    assert not torch.allclose(mu,next_mu) and torch.equal(gx,next_gx)


def test_old_failed_run_never_reported_as_running(tmp_path):
    import json
    from src.pointwise_benchmark_v1.status import training_status
    root=tmp_path/'artifacts';folder=root/'diffpts_q_only';folder.mkdir(parents=True)
    (tmp_path/'run_record.json').write_text(json.dumps(dict(state='failed')))
    (root/'failure.json').write_text(json.dumps(dict(state='failed')))
    (folder/'progress.json').write_text(json.dumps(dict(state='training',epoch=5,updates=871)))
    cfg=dict(models={'pq':[],'q_only':['diffpts']},epochs=50,updates_per_epoch=200,forecast_hours=24,select_draws=128,final_draws=256)
    rows=training_status(root,cfg)
    assert next(r for r in rows if r['configuration_id']=='diffpts_q_only')['state']=='failed_stopped'


def test_single_failure_continues_other_independent_configuration(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from src.pointwise_benchmark_v1 import runner
    calls=[]
    def fake_train(name,*args):
        calls.append(name)
        if name=='bad':raise runner.NumericalFailure('fixture',1,1,'loss')
        return dict(state='complete')
    monkeypatch.setattr(runner,'train_one',fake_train)
    monkeypatch.setattr(runner,'SampleBank',lambda *args:SimpleNamespace(identity='fixture'))
    monkeypatch.setattr(runner,'load_exposure',lambda *args:np.zeros((1,2,4),dtype=int))
    cfg=dict(condition_id='synthetic_fixture',expected_bank_sha256={'q_only':'fixture'},banks={'q_only':'unused'},
        execution_plan=[dict(id=name,model=name,route='q_only',action='train') for name in ('bad','good')])
    assert runner.execute_plan(cfg,tmp_path,'cpu')==1
    assert calls==['bad','good']
