"""Reuse loopback MLflow API with this experiment's metric/identity adapter."""
import json
import time
from pathlib import Path
from experiment_diagnostics.mlflow_bridge import TrackingAPI,digest,load,save


def numeric(value,prefix=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from numeric(v,prefix+'/'+str(k) if prefix else str(k))
    elif isinstance(value,list):
        for k,v in enumerate(value):yield from numeric(v,prefix+'/'+str(k))
    elif isinstance(value,(float,int)) and not isinstance(value,bool):yield prefix,value


def sync(folder,config):
    folder=Path(folder);api=TrackingAPI('http://127.0.0.1:5510')
    receipt=load(folder/'mlflow.json',{'sent':{}})
    key=digest(dict(run=config['run_id'],arm=config['arm'],seed=config['seed'],code=config['code_version']))
    name='waterpaper-multiwindow-peak-v1'
    found=api.request('mlflow/experiments/search',{'filter':f"name = '{name}'",'max_results':10}).get('experiments',[])
    eid=found[0]['experiment_id'] if found else api.request('mlflow/experiments/create',{'name':name})['experiment_id']
    matches=api.request('mlflow/runs/search',{'experiment_ids':[eid],'filter':f"tags.bridge_key = '{key}'",'max_results':2}).get('runs',[])
    if len(matches)>1:raise ValueError('duplicate MLflow identity')
    tags=dict(bridge_key=key,project_slug='waterpaper',change_id=config['change_id'],arm=config['arm'],seed=config['seed'],source_run_id=config['run_id'],source_training_commit=config['code_version'],smoke=config['smoke'])
    tags['mlflow.runName']=f"{config['run_id']} {config['arm']} seed{config['seed']}"
    tags['mlflow.note.content']='主峰前6小时起报，四窗口峰幅及概率；select主指标是M6中位数绝对误差/本站Q95，越低越好。S0仅点预测。开发集在全部候选选择结束后评价。'
    run=matches[0] if matches else api.request('mlflow/runs/create',{'experiment_id':eid,'start_time':int(time.time()*1000),'tags':[{'key':k,'value':str(v)} for k,v in tags.items()]})['run']
    rid=run['info']['run_id'];receipt.update(run_id=rid,experiment_id=eid,ui_path=f'/#/experiments/{eid}/runs/{rid}');save(folder/'mlflow.json',receipt)
    records=[]
    for filename,kind in [('epochs.jsonl','epoch'),('health.jsonl','health')]:
        if (folder/filename).exists():
            for line in (folder/filename).read_text().splitlines():
                row=json.loads(line);records.append((kind,str(row['epoch'] if kind=='epoch' else row['step']),row))
    selected=load(folder/'selected_diagnostic.json')
    if selected:records.append(('selected','0',selected['result']))
    final=load(folder/'final_summary.json')
    if final:records.append(('final','0',{r['role']+'/'+str(r['horizon'])+'h':r for r in final}))
    for kind,identifier,row in records:
        rec_id=kind+':'+identifier;fingerprint=digest(row)
        if rec_id in receipt['sent']:
            if receipt['sent'][rec_id]!=fingerprint:raise ValueError('changed synchronized epoch')
            continue
        points=[]
        for k,v in numeric(row):
            metric=kind+'/'+k;step=int(identifier)
            hist=api.request('mlflow/metrics/get-history?run_id='+rid+'&metric_key='+metric).get('metrics',[])
            if not any(x['step']==step and x['value']==v for x in hist):points.append(dict(key=metric,value=v,step=step,timestamp=int(time.time()*1000)))
        api.request('mlflow/runs/log-batch',dict(run_id=rid,metrics=points))
        receipt['sent'][rec_id]=fingerprint;save(folder/'mlflow.json',receipt)
    api.request('mlflow/runs/update',dict(run_id=rid,status='FINISHED' if (folder/'selection_complete.json').exists() else 'RUNNING'))
    artifacts=receipt.setdefault('artifacts',{})
    paths=[folder/'config.json',folder/'selected_diagnostic.json',folder/'selection_complete.json',folder/'final_summary.json',folder/'strata_metrics.csv']+list((folder.parent.parent/'figures').glob('*.png'))
    for path in paths:
        if not path.exists() or path.stat().st_size>2_000_000:continue
        content=path.read_bytes();sha=__import__('hashlib').sha256(content).hexdigest();name=path.parent.name+'/'+path.name
        if artifacts.get(name)!=sha:api.artifact(run,name,content);artifacts[name]=sha
    receipt['state']='synced';save(folder/'mlflow.json',receipt);return receipt
