"""Small synthetic checks; --torch checks state isolation without real data."""
import argparse
import json
import tempfile
from pathlib import Path
from .summary import summarize,write_summary


def check_summary():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);training=root/'source';training.mkdir();arm=training/'Risk-C';arm.mkdir()
        (arm/'best.json').write_text(json.dumps(dict(epoch=14,step=2996)))
        records=[]
        for step in (50,100,150):
            records.append(dict(arm='Risk-C',kind='health',step=step,metrics=dict(saturation_fraction=.99,module_health=dict(zero=dict(state='sampled',gradient_norm=0),missing=dict(state='sampled',gradient_norm=None),small=dict(state='sampled',gradient_norm=1e-12),protected=dict(state='NOT_APPLICABLE')))))
        for epoch in range(4):
            records.append(dict(arm='Risk-C',kind='epoch',epoch=epoch,metrics=dict(fit_brier=[0,0,4-epoch],select_brier=[0,0,epoch])))
        records.append(dict(arm='Risk-C',kind='shadow_only',epoch=10,diagnostic=dict(result=dict(roles={}))))
        path=training/'metrics.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records));before=path.read_bytes()
        report=summarize(training);codes={r['code'] for r in report['observations']}
        assert {'zero_gradient','missing_gradient','saturation','fit_select_divergence','best_diagnostic_missing'} <= codes
        assert 'configured_small_gradient' not in codes
        configured=summarize(training,dict(near_zero_gradient_by_module=dict(small=1e-10)))
        assert any(r['code']=='configured_small_gradient' for r in configured['observations'])
        write_summary(training,root/'output');write_summary(training,root/'output')
        assert path.read_bytes()==before
        assert not any(r.get('module')=='protected' for r in report['observations'])
        records.append(dict(arm='Risk-C',kind='selected_checkpoint',epoch=14,identity=dict(checkpoint_sha256='abc'),diagnostic=dict(result=dict(roles={}))))
        (arm/'best.json').write_text(json.dumps(dict(epoch=14,step=2996,checkpoint_sha256='abc')))
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        report=summarize(training)
        assert any(r['code']=='best_diagnostic_coverage' and r['identity_state']=='content_hash_verified' for r in report['observations'])
        (arm/'best.json').write_text(json.dumps(dict(epoch=14,step=2996,checkpoint_sha256='different')))
        report=summarize(training)
        assert not any(r['code']=='best_diagnostic_coverage' for r in report['observations'])
        assert any(r['code']=='best_diagnostic_missing' for r in report['observations'])


def check_torch():
    import random
    import numpy as np
    import torch
    from protected_flood_risk_v1.diagnostics import isolated_diagnostic,cached_diagnostic
    model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    weights={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    gradients={k:v.grad.clone() for k,v in model.named_parameters()}
    import copy
    state=copy.deepcopy(optimizer.state_dict());np_rng=np.random.get_state();py_rng=random.getstate()
    try:
        with isolated_diagnostic(model,optimizer):
            model.weight.fill_(42);torch.rand(3);np.random.random();random.random()
            raise RuntimeError('synthetic failure')
    except RuntimeError:pass
    assert model.training and torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in weights.items())
    assert all(torch.equal(dict(model.named_parameters())[k].grad,v) for k,v in gradients.items())
    assert random.getstate()==py_rng and np.array_equal(np.random.get_state()[1],np_rng[1])
    assert all(torch.equal(optimizer.state_dict()['state'][k]['exp_avg'],v['exp_avg']) for k,v in state['state'].items())
    identity=dict(code_version='fixture',data_role_id='fit/select',protocol='fixture',checkpoint_sha256='a',algorithm_version='fixture',sample_ids=['x'],schema_version=1,content_digest='fixture')
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);calls=[]
        first=cached_diagnostic(root/'old',identity,lambda:calls.append(1) or dict(state='fixture'))
        second=cached_diagnostic(root/'new',identity,lambda:1/0,[root/'old'])
        assert not first['cache_hit'] and second['cache_hit'] and calls==[1]
        third=cached_diagnostic(root/'new',dict(identity,checkpoint_sha256='b'),lambda:calls.append(2) or {})
        assert not third['cache_hit'] and calls==[1,2]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--torch',action='store_true');args=parser.parse_args()
    check_summary()
    if args.torch:check_torch()
    print('PASS: synthetic diagnostic checks; no real samples or model replay')
