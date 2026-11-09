"""Focused novel-property checks, executed only on teacherbox after commit."""
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from protected_flood_risk_v1.diagnostics import isolated_diagnostic
from protected_flood_risk_v1.training import atomic_json,state_hash
from .model import blocks,maxima,nll,crps,build
from .data import slots,load,eligibility,original_q
from .runner import SOURCE


class Protected(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(32,64)
    def encode(self,x,f):return self.linear(torch.cat((x.flatten(1),f),1))


def run(output):
    torch.set_num_threads(1);torch.manual_seed(2)
    y=torch.tensor([[5.,4,3,2,1,0,9,8,7,6,5,4],[0.]*12])
    assert blocks(y)[0].tolist()==[5,4,2,9]
    assert maxima(y,True)[0].tolist()==[5,5,5,9]
    assert float(nll(torch.tensor(0.),torch.tensor(0.),torch.tensor(1.)))==math.log(2)
    q=torch.tensor(3.,dtype=torch.float64);z=torch.log1p(q)
    assert torch.allclose(nll(q,torch.tensor(0.),torch.tensor(1.)),.5*z*z+.5*math.log(2*math.pi)+z)
    draws=torch.tensor([[[1.,2.],[3.,6.],[5.,4.]]]);truth=torch.tensor([[2.,3.]])
    brute=(draws-truth[:,None]).abs().mean(1)-.5*(draws[:,:,None]-draws[:,None,:]).abs().mean((1,2))
    assert torch.allclose(crps(draws,truth),brute)
    models=build(Protected());assert len({state_hash(m.adapter) for m in models.values()})==1
    x=torch.rand(2,12,2);f=torch.rand(2,8)
    for arm,m in models.items():
        h=m.encode(x,f);g=torch.Generator().manual_seed(19)
        before=torch.get_rng_state().clone();a=m.sample(h,16,g)
        assert torch.equal(before,torch.get_rng_state()) and torch.isfinite(a).all() and (a.diff(dim=-1)>=0).all()
        # Fixed forecast RNG never consults labels, including teacher-likelihood calls.
        if arm!='S0':
            target=torch.ones(2,12 if arm=='S1' else 4)
            mu1,sg1=m.distribution(h,target);mu2,sg2=m.distribution(h,target*9)
            if arm=='S2':assert torch.equal(mu1,mu2) and torch.equal(sg1,sg2)
            else:assert not torch.equal(mu1[:,1:],mu2[:,1:])
            b=m.sample(h,16,torch.Generator().manual_seed(19));assert torch.equal(a,b)
        opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad])
        identity=state_hash(m);before=torch.get_rng_state().clone()
        with isolated_diagnostic(m,opt):torch.randn(20)
        assert identity==state_hash(m) and torch.equal(before,torch.get_rng_state())
    fit=dict(rows=[dict(country='China',station_uid='A'),dict(country='China',station_uid='B')],x=np.array([[[0,3]],[[0,0]]]))
    a,ha=slots(fit,2,128,2);b,hb=slots(fit,2,128,2)
    assert ha==hb and np.array_equal(a,b) and set(a.ravel())=={0,1}
    cfg=json.loads(Path(SOURCE).read_text())['config'];pools,_=load(cfg,['fit','select'])
    masks={};records={}
    for role,pool in pools.items():
        original,provenance=original_q(pool)
        masks[role],records[role]=eligibility(pool,'._protected_risk_318b1fa/data/catalog/china_event_catalog_v1.csv',original)
    result=dict(state='PASS' if masks['select'].any() else 'NOT_EVALUABLE',math='blocks, atom, Jacobian, sorted CRPS',protocol='paired initialization/slots, includes current-high, S2 no stochastic feedback, free forecast label-invariance, RNG/optimizer isolation',real_select_eligible=int(masks['select'].sum()),real_fit_eligible=int(masks['fit'].sum()),eligibility=records,development_read=False)
    atomic_json(Path(output),result)
    assert masks['select'].any(),'No legal select mainpeak-minus6 origins; stop before training'
    print(json.dumps({k:v for k,v in result.items() if k!='eligibility'}))


def replay_smoke(source,output):
    import subprocess
    from .runner import diagnostic
    from .data import fixed
    from .report import plot
    from prob_flood_v1.model import Forecaster
    from protected_flood_risk_v1.runner import subset
    from protected_flood_risk_v1.training import content_digest
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'effective_config.json').read_text())
    assert config['smoke'] is True
    raw_cfg=json.loads(Path(SOURCE).read_text())['config'];pools,_=load(raw_cfg,['fit','select'])
    chosen=json.loads((source/'eligibility.json').read_text());fixed_ids={}
    for role,pool in pools.items():
        ids={r['window_id'] for r in json.loads((source/(role+'_rows.json')).read_text())}
        pools[role]=subset(pool,[i for i,r in enumerate(pool['rows']) if r['window_id'] in ids])
        legal_ids={r['window_id'] for r in chosen['records'][role] if r['state']=='evaluable'}
        mask=np.array([r['window_id'] in legal_ids for r in pools[role]['rows']]);fixed_ids[role],_=fixed(pools[role],mask)
    folder=source/'seed_2'/'S3';cfg=json.loads((folder/'config.json').read_text());cfg['code_version']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original=Forecaster('P0',json.loads((Path(raw_cfg['source'])/'configs/prob_flood_v1.json').read_text()))
    original.load_state_dict(torch.load(raw_cfg['protected_checkpoint'],map_location='cpu',weights_only=False))
    model=build(original)['S3'].cuda();selected=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected['model'])
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad]);opt.load_state_dict(selected['optimizer'])
    d=diagnostic(model,opt,pools,fixed_ids,out,cfg,{r:content_digest(p) for r,p in pools.items()})
    for role,result in d['result'].items():
        assert abs(sum(result['observed_highest_peak_segment_probabilities'])-1)<1e-10
        assert abs(sum(result['highest_peak_segment_probabilities'])-1)<1e-10
        assert -3<=result['expected_highest_segment_shift']<=3
    atomic_json(out/'selected_diagnostic.json',d)
    result=json.loads((source/'completion.json').read_text())
    pools['development']=pools['select'];plot(out,result['summaries'],pools,source=source)
    ordinary=np.flatnonzero(pools['select']['y'][:,:6].max(1)<=1)
    assert len(ordinary)>0
    atomic_json(out/'replay_receipt.json',dict(state='PASS',source=str(source),code_version=cfg['code_version'],scope='selected S3 diagnostic + all three figures; no retraining',data_roles=['fit','select'],development_read=False,ordinary_support=len(ordinary)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke-source');args=p.parse_args()
    if args.smoke_source:replay_smoke(args.smoke_source,args.output)
    else:run(args.output)
