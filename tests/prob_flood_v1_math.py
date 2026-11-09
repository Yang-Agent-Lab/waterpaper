"""Small exact fixtures, no real inputs or fitting."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import json
from unittest.mock import patch
from prob_flood_v1.model import censored_nll,crps,Forecaster
from prob_flood_v1.data import features,components,assign_roles,validate_roles
from prob_flood_v1.score import calibrated,probability_metrics,event_discovery
from prob_flood_v1.run import Budget,BudgetLimit


def run():
    z=torch.tensor([0.,1.]);mu=torch.zeros(2);sigma=torch.ones(2)
    got=censored_nll(mu,sigma,z)
    assert torch.allclose(got,torch.tensor([np.log(2),.5+.5*np.log(2*np.pi)],dtype=torch.float32),atol=1e-6)
    samples=torch.tensor([[0.,2.]])
    assert torch.allclose(crps(samples,torch.tensor([1.])),torch.tensor([0.]))
    assert torch.allclose(crps(samples[:,:1],torch.tensor([1.])),torch.tensor([1.]))
    x=np.arange(24).reshape(1,12,2).astype(float)
    f=features(x);assert f.shape==(1,8) and f[0,0]==22 and f[0,1]==60 and f[0,6]==2 and f[0,7]==6
    rs=[]
    for j in range(10):
        rs.append(dict(station_uid='a',country='Germany',event=str(j),source_rows=list(range(j*25,j*25+24)),
                       source_times=[f'{j:04d}',f'{j:04d}z'],temporal_component=None))
    gs=components(rs);roles,_=assign_roles(rs,gs,20261009);validate_roles(rs,gs,roles)
    altered=roles.copy();altered[1]='calibration' if roles[0]!='calibration' else 'fit'
    try:validate_roles([rs[0],rs[0]],np.array([0,0]),np.array(['fit','select']))
    except AssertionError:pass
    else:raise AssertionError('leakage check did not reject shared source')
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/prob_flood_v1.json').read_text())
    torch.set_num_threads(1)
    x=torch.zeros(2,12,2);f=torch.ones(2,8)
    for arm in ['P1','P5']:
        model=Forecaster(arm,config);model.eval();counter=[]
        handle=model.denoiser.register_forward_hook(lambda *args:counter.append(1))
        first=model.sample(x,f,4,torch.Generator().manual_seed(7))
        second=model.sample(x,f,4,torch.Generator().manual_seed(7))
        assert first.shape==(2,4,12) and torch.equal(first,second) and (first>=0).all()
        if arm=='P1':assert torch.equal(first,model.sample(x,f*100,4,torch.Generator().manual_seed(7)))
        else:assert len(counter)==2*config['diffusion_sampling_steps'],'Diffusion free generation must run all 20 steps'
        handle.remove()
    raw=dict(risk=np.array([[0,.1,.4,.8],[.2,.3,.5,1.]]))
    p=calibrated(raw,dict(state='evaluable',slope=2.,intercept=-1.),'P1',config['risk_logit_clip'])
    assert (np.diff(p,axis=1)>=0).all()
    score_rows=[dict(country='China',station_uid='s',event='e')]*2
    baseline=probability_metrics(np.array([False,True]),np.array([True,False]),score_rows)
    assert baseline['brier']==1.,'Boolean persistence scores must enter numeric Brier arithmetic'
    times=[str(np.datetime64('2000-01-01T00:00:00')+np.timedelta64(i,'h')) for i in range(24)]
    eventrow=dict(country='China',station_uid='s',event='e',origin_time=times[11],source_times=times)
    history=np.zeros((1,12));history[0,4]=2.;future=np.zeros((1,12));future[0,2]=2.
    discovery=event_discovery(np.array([.9]),np.array([True]),np.array([0.]),[eventrow],history,future,{'threshold':.5})
    assert discovery[0]['state']=='NOT_EVALUABLE' and discovery[0]['first_visible_exceedance']==times[4]
    history[0,0]=2.
    discovery=event_discovery(np.array([.9]),np.array([True]),np.array([0.]),[eventrow],history,future,{'threshold':.5})
    assert discovery[0]['state']=='NOT_EVALUABLE','An already-high truncated fragment cannot claim a later re-crossing as its first warning'
    budget=Budget.__new__(Budget);budget.config=config;budget.since=None;budget.model='model';budget.kind='training'
    budget.state=dict(total_gpu_seconds=4500.,evaluation_gpu_seconds=0.,models={'model':4500.},segments=[])
    try:budget.check()
    except BudgetLimit:pass
    else:raise AssertionError('Model budget did not stop at 75 minutes')
    budget.model=None;budget.kind=None;budget.state['evaluation_gpu_seconds']=config['evaluation_gpu_seconds']
    try:budget.check()
    except BudgetLimit:pass
    else:raise AssertionError('Exhausted final-scoring allowance must also stop the next training model')
    state=dict(total_gpu_seconds=40.,evaluation_gpu_seconds=20.,models={'old':5.},segments=[],
               active=dict(model='old',kind='final',gpu=True,started_wall_timestamp=100.,last_wall_timestamp=100.))
    with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value=json.dumps(state)),\
         patch('prob_flood_v1.run.update'),patch('prob_flood_v1.run.time.time',return_value=107.):
        recovered=Budget(Path('/fixture-ledger-not-written'),config,False)
    assert recovered.state['total_gpu_seconds']==47 and recovered.state['evaluation_gpu_seconds']==27
    assert recovered.state['models']['old']==12 and recovered.state['active'] is None
    diffusion=Forecaster('P5',config);a=diffusion.alpha_bar[-1]
    assert a<1e-6,'Terminal noise must be close to pure standard Gaussian'
    noise=torch.randn(50000,generator=torch.Generator().manual_seed(51))
    terminal=a.sqrt()*3+(1-a).sqrt()*noise
    assert abs(float(terminal.mean()))<.02 and abs(float(terminal.std())-1)<.02
    clean=torch.tensor([0.,3.]);eps=torch.tensor([.5,-1.]);z=a.sqrt()*clean+(1-a).sqrt()*eps
    v=a.sqrt()*eps-(1-a).sqrt()*clean
    restored=a.sqrt()*z-(1-a).sqrt()*v
    assert torch.allclose(restored,clean,atol=1e-6),'Stable v reconstruction must not divide by tiny terminal alpha'
    print('Exact NLL/CRPS/history/group/free-generation/monotone-risk/budget fixtures passed')

if __name__=='__main__':run()
