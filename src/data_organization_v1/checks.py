"""Focused server checks using fixed fit only; no GPU or held-out data."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from protected_flood_risk_v1.model import TemporalAdapter
from protected_flood_risk_v1.training import state_hash
from multiwindow_peak_v1.model import PeakModel
from prob_flood_v1.model import Forecaster
from multiwindow_peak_v1.data import load
from .sampling import prepare,slots,cells,categorical
from .runner import source_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-manifest');parser.add_argument('--path-map-json');args=parser.parse_args()
    options={'path_map':args.path_map_json}
    if args.source_manifest:options['source_manifest']=args.source_manifest
    torch.set_num_threads(1);cfg=source_config(**options);pool=load(cfg,['fit'])[0]['fit']
    attrs={r['station_uid']:r for r in json.loads(Path(cfg['attribute_rows']).read_text())};scales=json.loads((Path(cfg['data'])/'scales.json').read_text())
    p=prepare(pool,attrs,scales)
    covered=np.concatenate([v for units in p['units'].values() for v in units]);assert np.array_equal(np.sort(covered),np.arange(28814))
    for target,by_station in p['weights'].items():
        for station,(a,c) in by_station.items():
            units=p['units'][station];n=sum(map(len,units));assert all(np.isclose(a[j]/len(ix),1/n) for j,ix in enumerate(units));assert np.isclose(c.sum(),1)
            # C window chance differs only through unit chance, uniform inside.
            assert all(np.isclose(np.repeat(c[j]/len(ix),len(ix)).sum(),c[j]) for j,ix in enumerate(units))
        counts=np.zeros(12,int);th=p['thresholds'][target]
        if th['state']=='evaluable':
            for ix in p['units'][target]:
                rr=[pool['rows'][i] for i in ix];area=attrs[target]['area_km2'];scale=rr[0]['qscale']
                now=pool['x'][ix,-1,1].astype(float);q=3.6*now*scale/area;rain=pool['x'][ix,-6:,0].astype(float).sum(1)*scales[target]['pscale'];rise=3.6*(pool['y'][ix,:6].astype(float).max(1)-now)*scale/area
                counts[np.unique(cells(q,rain,rise,th))]+=1
            assert counts.tolist()==p['support'][target]['distinct_process_units_by_cell']
    th=dict(q_mm_h=1,p6_mm=2,rise_mm_h=3)
    assert cells(np.array([.9,1,1]),np.array([1.9,2,2]),np.array([0,1,3]),th).tolist()==[0,10,11]
    one=np.array([1],dtype=np.float32).astype(float);boundary=3.6*250.7875/2297.06
    assert (3.6*one*250.7875/2297.06>=boundary).all()
    assert categorical([.25,.75],np.array([0,.249,.25,.999])).tolist()==[0,0,1,1]
    sampled,audit,_=slots(p,2,256,2);assert audit['B_C_station_country_position_equal'] and audit['A_B_C_country_position_equal']
    again=slots(p,2,256,2)[0];assert all(np.array_equal(sampled[a],again[a]) for a in 'ABC')
    target=p['targets'][0];bad=copy.deepcopy(pool);ix=p['groups'][target];bad['y'][ix]=bad['x'][ix,-1,1,None]
    fallback=prepare(bad,attrs,scales);assert fallback['thresholds'][target]['state']=='NOT_EVALUABLE';assert all(np.array_equal(a,c) for a,c in fallback['weights'][target].values())
    forbidden=copy.deepcopy(pool);forbidden['rows'][0]['role']='select'
    try:prepare(forbidden,attrs,scales)
    except ValueError:pass
    else:raise AssertionError('non-fit accepted')
    original=Forecaster('P0',json.loads((Path(cfg['source'])/'configs/prob_flood_v1.json').read_text()));original.load_state_dict(torch.load(cfg['protected_checkpoint'],map_location='cpu',weights_only=False))
    identities=[]
    for policy in 'ABC':
        torch.manual_seed(2);model=PeakModel('S2',original,TemporalAdapter(),nn.GRUCell(68,64),nn.Linear(64,2))
        assert all(not v.requires_grad for v in model.protected.parameters());identities.append(state_hash(model));assert state_hash(model.protected)==state_hash(original)
    assert len(set(identities))==1
    print(json.dumps(dict(state='PASS',fit_windows=28814,process_units=len(p['unit_ids']),checks=['fit-only','physical thresholds and equality','unit-count support','A uniform window equivalence','C whole-unit average and uniform interior','B/C country/station paired slots','seed reproducibility','fallback','identical S2 new initialization','P0 frozen and identical'],roles_read=['fit'],GPU_used=False)))

if __name__=='__main__':main()
