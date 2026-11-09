"""Fixed display cases selected from observed targets only, shared by all models."""
import numpy as np
PREFERRED=[dict(station='Liqingdian',role='development',case_kind='difficult'),dict(station='Luanchuan',role='development',case_kind='difficult'),dict(station='Luxi',role='development',case_kind='ordinary'),dict(station='Zhuxipo',role='development',case_kind='difficult')]
RULE='development only; difficult=max observed H6 window peak, earliest origin tie; Luxi ordinary=earliest origin with observed H6<=fit Q95; same window IDs for all models'

def choose_cases(rows,truth):
    chosen=[]
    for spec in PREFERRED:
        ix=[i for i,r in enumerate(rows) if r['station_uid'].split(':')[-1]==spec['station']]
        if spec['case_kind']=='ordinary':ix=[i for i in ix if truth[i,2]<=1]
        if not ix:continue
        key=(lambda i:(-float(truth[i,2]),rows[i]['origin_time'],rows[i]['window_id'])) if spec['case_kind']=='difficult' else (lambda i:(rows[i]['origin_time'],rows[i]['window_id']))
        chosen.append((spec['case_kind'],min(ix,key=key)))
    return chosen

def make_cases(rows,truth,draw,model,role):
    if role!='development':return []
    median=np.median(draw,axis=1);lo,hi=np.quantile(draw,[.05,.95],axis=1);result=[]
    for kind,i in choose_cases(rows,truth):
        r=rows[i];scale=r['qscale'];result.append(dict(case_id=role+':'+kind+':'+r['window_id'],window_id=r['window_id'],origin_time=r['origin_time'],station=r['station_uid'],event_id=r['source_event_id'],country='China',role=role,model=model,x_kind='horizon_window_peak',x=[1,3,6,12],truth=(truth[i]*scale).tolist(),median=(median[i]*scale).tolist(),lower=(lo[i]*scale).tolist(),upper=(hi[i]*scale).tolist(),unit='m3/s',case_kind=kind))
    return result
