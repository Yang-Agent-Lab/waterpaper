"""Add display-only selection fields from saved epoch logs, without inference."""
import argparse,json,csv
import numpy as np
from pathlib import Path
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.training import atomic_json
from .runner import read_jsonl
from .cases import make_cases,PREFERRED,RULE

def repair(artifacts,output):
    out=Path(artifacts);target=Path(output)
    if target.exists():raise FileExistsError('new report input copy required')
    source=out/'report_bundle.json';bundle=json.loads(source.read_text());lookup={}
    canonical_truth=None
    for model in bundle['models']:
        for row in read_jsonl(out/'seed_2'/model['id']/'epochs.jsonl'):lookup[model['id'],row['epoch']]=row['selection_score']
    for row in bundle['epochs']:
        if row['country']=='China' and row['role']=='select':row['selection_metric']=lookup[row['model'],row['epoch']]
    bundle['metadata']['selection_label']='中国6h CRPS/Q95'
    rows=json.loads((out/'development_rows.json').read_text());ids=[r['window_id'] for r in rows]
    saved=list(csv.DictReader((out/'point_windows.csv').open()));bundle['cases']=[]
    evidence={str(out/'development_rows.json'):digest(out/'development_rows.json'),str(out/'point_windows.csv'):digest(out/'point_windows.csv')}
    for model in bundle['models']:
        lookup={(r['window_id'],int(r['horizon'])):float(r['truth_m3_s']) for r in saved if r['data_policy']==model['id']}
        truth=np.array([[lookup[r['window_id'],h]/r['qscale'] for h in (1,3,6,12)] for r in rows])
        if canonical_truth is None:canonical_truth=truth
        elif not np.array_equal(truth,canonical_truth):raise ValueError('saved observed targets differ across model reports')
        path=out/'seed_2'/model['id']/'development_final.npz'
        with np.load(path,allow_pickle=False) as z:
            if list(z['window_ids'].astype(str))!=ids:raise ValueError('saved development ID mismatch')
            bundle['cases'].extend(make_cases(rows,truth,z['samples'],model['id'],'development'))
        evidence[str(path)]=digest(path)
    bundle['metadata']['preferred_cases']=PREFERRED;bundle['metadata']['case_identity']=RULE
    bundle['metadata']['display_only_repair']=dict(original_bundle=str(source),original_sha256=digest(source),evidence=evidence,source='existing epochs selection_score and saved development targets/scenarios only; no inference or training')
    atomic_json(target,bundle);return bundle
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);p.add_argument('--out',required=True);a=p.parse_args();repair(a.artifacts,a.out)
