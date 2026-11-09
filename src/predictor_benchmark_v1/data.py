"""Prepared arrays only, explicitly retaining foreign select and forbidding early final roles."""
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from protected_flood_risk_v1.data import digest,validate_metadata,join_static
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.data import slots
from data_organization_v1.runner import source_config
from .statistics import group_key

COUNTS={'fit':{'China':4554,'Germany':12199,'Iceland':12061},'select':{'China':2150,'Germany':4528,'Iceland':3896},'calibration':{'China':1853,'Germany':3273,'Iceland':4043},'development':{'China':1356}}

def authorize_role(role,final_authorized):
    if role not in COUNTS or (role not in ('fit','select') and not final_authorized):raise PermissionError('role not authorized: '+role)

def load(cfg,roles,countries=None,*,final_authorized=False):
    folder=Path(cfg['data']);manifest=json.loads((folder/'manifest.json').read_text())
    if digest(folder/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('manifest mismatch')
    attrs={r['station_uid']:r for r in json.loads(Path(cfg['attribute_rows']).read_text())};norm=json.loads(Path(cfg['normalization']).read_text())
    fit_rows=json.loads((folder/'fit_rows.json').read_text());fit_stations={r['station_uid'] for r in fit_rows};pools={}
    inputs={str(p):digest(p) for p in (folder/'manifest.json',Path(cfg['attribute_rows']),Path(cfg['normalization']),folder/'fit_rows.json',folder/'scales.json',folder/'rise_cutoffs.json')}
    export_path=folder.parent/'manifest.json'
    if digest(export_path)!='e8355e8240975589b0da5081dc471f40f387da2251ad0a05ee6e242e0f42ca35':raise ValueError('fixed export manifest changed')
    export=json.loads(export_path.read_text())
    expected={r['relative']:r['sha256'] for r in export['files']}
    for name in ('scales.json','rise_cutoffs.json','fit_rows.json'):
        if digest(folder/name)!=expected['prepared/'+name]:raise ValueError('prepared fixed metadata mismatch '+name)
    inputs[str(export_path)]=digest(export_path)
    for role in roles:
        authorize_role(role,final_authorized);path=folder/(role+'.npz');rp=folder/(role+'_rows.json')
        if digest(path)!=manifest['derived_files'][path.name]:raise ValueError('NPZ mismatch')
        if digest(rp)!=expected['prepared/'+rp.name]:raise ValueError('metadata mismatch')
        rows=json.loads(rp.read_text());validate_metadata(rows,role)
        if dict(Counter(r['country'] for r in rows))!=COUNTS[role]:raise ValueError('role/country counts changed '+role)
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'x','y','features'}:raise ValueError('unexpected input arrays')
            arrays={k:z[k].copy() for k in z.files}
        n=len(rows)
        if arrays['x'].shape!=(n,12,2) or arrays['y'].shape!=(n,12) or arrays['features'].shape!=(n,8):raise ValueError('shape mismatch')
        if not all(np.isfinite(a).all() for a in arrays.values()):raise FloatingPointError('nonfinite input')
        inputs[str(path)]=digest(path);inputs[str(rp)]=digest(rp)
        pool=dict(rows=rows,input_manifest=dict(inputs),static=np.asarray(join_static(rows,attrs,norm['fit_station_uids'],fit_stations),dtype=np.float32),**arrays)
        if countries is not None:pool=subset(pool,[i for i,r in enumerate(rows) if r['country'] in countries])
        pools[role]=pool
    return pools,dict(inputs=inputs,roles=list(roles),countries=countries,future_P_read=False,test_read=False)

def fixed_probe(pool,per_group=8,max_groups_per_station=2,per_country=256):
    groups={}
    for i,r in enumerate(pool['rows']):groups.setdefault((r['country'],*group_key(r)),[]).append(i)
    chosen=[];station_groups=Counter();eligible={}
    for key,ix in sorted(groups.items(),key=lambda kv:hashlib.sha256(repr(kv[0]).encode()).hexdigest()):
        station=key[:2]
        if station_groups[station]>=max_groups_per_station:continue
        station_groups[station]+=1;eligible[key]=ix[:per_group]
    counts=Counter()
    # Round robin across fixed hashed station/block groups, before prediction.
    for offset in range(per_group):
        for key,ix in eligible.items():
            if offset<len(ix) and counts[key[0]]<per_country:
                chosen.append(ix[offset]);counts[key[0]]+=1
    return sorted(chosen)

def smoke_subset(pool):
    # Fixed before seeing predictions: 16 windows per station/event group.
    return subset(pool,fixed_probe(pool,16))
