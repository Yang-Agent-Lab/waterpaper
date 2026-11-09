"""Small adapter to the existing loopback MLflow API; logs are authoritative."""
import json
import time
from pathlib import Path
from experiment_diagnostics.mlflow_bridge import TrackingAPI, digest, load, save
from multiwindow_peak_v1.tracking import numeric


def sync(folder, cfg):
    folder=Path(folder); api=TrackingAPI(cfg['mlflow_uri'])
    receipt=load(folder/'mlflow.json', {'sent':{}})
    name='waterpaper-local-rain-2x2-v1'
    found=api.request('mlflow/experiments/search',{'filter':f"name = '{name}'",'max_results':10}).get('experiments',[])
    eid=found[0]['experiment_id'] if found else api.request('mlflow/experiments/create',{'name':name})['experiment_id']
    key=digest(cfg)
    matches=api.request('mlflow/runs/search',{'experiment_ids':[eid],'filter':f"tags.bridge_key = '{key}'",'max_results':2}).get('runs',[])
    if len(matches)>1:raise ValueError('duplicate MLflow run identity')
    tags={k:str(cfg[k]) for k in ('change_id','code_version','run_id','seed','phase','condition','station','smoke')}
    tags.update(bridge_key=key,project_slug='waterpaper',information_status='oracle_rain_diagnostic',
                **{'mlflow.runName':cfg['run_id']+' '+folder.name})
    run=matches[0] if matches else api.request('mlflow/runs/create',{'experiment_id':eid,'start_time':int(time.time()*1000),'tags':[{'key':k,'value':v} for k,v in tags.items()]})['run']
    rid=run['info']['run_id']; receipt.update(run_id=rid,experiment_id=eid)
    for line in (folder/'metrics.jsonl').read_text().splitlines() if (folder/'metrics.jsonl').exists() else []:
        row=json.loads(line); rec=str(row['step']); fingerprint=digest(row)
        if row['step']%50 and row['step']!=cfg['updates']:continue
        if rec in receipt['sent']:
            if receipt['sent'][rec]!=fingerprint:raise ValueError('changed synchronized metric')
            continue
        points=[]
        for k,v in numeric(row):
            if k=='step':continue
            hist=api.request('mlflow/metrics/get-history?run_id='+rid+'&metric_key='+k).get('metrics',[])
            if not any(x['step']==row['step'] and x['value']==v for x in hist):
                points.append(dict(key=k,value=v,step=row['step'],timestamp=int(time.time()*1000)))
        api.request('mlflow/runs/log-batch',dict(run_id=rid,metrics=points));receipt['sent'][rec]=fingerprint
        save(folder/'mlflow.json',receipt)
    api.request('mlflow/runs/update',dict(run_id=rid,status='FINISHED' if (folder/'complete.json').exists() else 'RUNNING'))
    receipt['state']='synced';save(folder/'mlflow.json',receipt)
    return receipt


def safe_sync(folder,cfg):
    try:return sync(folder,cfg)
    except Exception as exc:
        save(Path(folder)/'tracking_pending.json',dict(state='partial',error=str(exc),resume='sync only; never retrain'))
        return dict(state='partial')
