"""Approved A/B/C execution. Prepare is CPU-only; formal launch is explicit."""
import argparse,copy,fcntl,hashlib,json,os,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from prob_flood_v1.model import Forecaster
from protected_flood_risk_v1.model import TemporalAdapter
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.training import atomic_json,state_hash,content_digest
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.model import PeakModel
from multiwindow_peak_v1.data import load,eligibility,fixed
from multiwindow_peak_v1.runner import SOURCE,predict
from .sampling import prepare,slots,save
from .training import train_one
from .runtime_io import original_q,remap,map_config

CHANGE='WP-20261013-DATA-ORGANIZATION-V1'
MANIFEST='e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6'
P0='0908370cc4f1ed487495e3f49fa3bd881e5a1cf3b9156c45eff141a6b6896d58'
CATALOG=Path('._protected_risk_318b1fa/data/catalog/china_event_catalog_v1.csv')


def source_config(source_manifest=SOURCE,path_map=None,q_only_dir=None):
    mapping=json.loads(Path(path_map).read_text()) if path_map else {}
    cfg=map_config(json.loads(Path(source_manifest).read_text())['config'],mapping)
    cfg['_path_map']=mapping;cfg['_source_manifest']=str(source_manifest);cfg['_catalog']=remap(str(CATALOG),mapping)
    cfg['_q_only_dir']=q_only_dir
    if cfg['data_manifest_sha256']!=MANIFEST or cfg['protected_checkpoint_sha256']!=P0:raise ValueError('wrong fixed input identity')
    for key in ('protected_checkpoint','attribute_rows','normalization'):
        if digest(cfg[key])!=cfg[key+'_sha256']:raise ValueError('input changed '+key)
    if digest(Path(cfg['data'])/'manifest.json')!=MANIFEST:raise ValueError('prepared manifest changed')
    if json.loads((Path(cfg['data'])/'manifest.json').read_text())['producer_code_sha']!='ab720ba0fe75bdd5b694dcd4c43302d9d29e8ec2':raise ValueError('wrong producer')
    return cfg


def execute(output,mode,stop_after=None,cpu_smoke=False,source_manifest=SOURCE,path_map=None,q_only_dir=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=source_config(source_manifest,path_map,q_only_dir);catalog=Path(cfg['_catalog']);smoke=mode=='smoke';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    torch.set_num_threads(1)
    roles=['fit'] if mode=='prepare' else ['fit','select'];pools,source=load(cfg,roles)
    if len(pools['fit']['rows'])!=28814:raise ValueError('fit membership changed')
    attrs={r['station_uid']:r for r in json.loads(Path(cfg['attribute_rows']).read_text())};scales=json.loads((Path(cfg['data'])/'scales.json').read_text())
    p=prepare(pools['fit'],attrs,scales);sampled,audit,latent=slots(p,2 if smoke else 100,256 if smoke else 28928,2)
    save(out/'sampling',p,sampled,audit,latent)
    runtime=dict(change_id=CHANGE,code_version=head,data_manifest_sha256=MANIFEST,run_id=out.parent.name,smoke=smoke,epochs=2 if smoke else 100,batch_size=128,lr=.001,clip=10,patience=None,seeds=[2],arms=['S2'],data_policies=list('ABC'),epoch_size=256 if smoke else 28928,train_scenarios=16,select_scenarios=128,final_scenarios=512)
    if (out/'effective_config.json').exists() and json.loads((out/'effective_config.json').read_text())!=runtime:raise ValueError('resume identity changed')
    atomic_json(out/'effective_config.json',runtime)
    atomic_json(out/'source_manifest.json',dict(source_manifest=source_manifest,source_manifest_sha256=digest(source_manifest),path_map=cfg['_path_map'],source=source,protected_checkpoint_sha256=P0,runtime=runtime,future_P_read=False,test_read=False,threshold_roles=['fit'],static_context='retrospective area/annual rain; China climate labels not used'))
    if mode=='prepare':
        result=dict(state='complete',phase='fit-only preparation',fit_windows=28814,fit_stations=len(p['groups']),process_units=len(p['unit_ids']),thresholds=p['thresholds'],slots_shape=audit['shape'],actual_window_difference_count=audit['actual_window_difference_count'],B_C_station_country_position_equal=True,A_B_C_country_position_equal=True,roles_read=['fit'])
        atomic_json(out/'completion.json',result);print(json.dumps(result),flush=True);return result
    if cpu_smoke and not smoke:raise ValueError('CPU restricted to engineering smoke')
    if not cpu_smoke:
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not selected or ',' in selected:raise RuntimeError('one explicit selected CUDA_VISIBLE_DEVICES required')
        uuid=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        gpu_lock=Path(cfg['gpu_lock']).with_name('data_organization_'+uuid+'.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','-i',selected,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU occupied before launch; preserve prepare and retry later')
        if torch.cuda.device_count()!=1:raise RuntimeError('exactly one GPU required')
    device=torch.device('cpu' if cpu_smoke else 'cuda');torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    atomic_json(out/'device.json',dict(device=str(device),engineering_smoke=smoke,cpu_smoke=cpu_smoke))
    original=Forecaster('P0',json.loads((Path(cfg['source'])/'configs/prob_flood_v1.json').read_text()));original.load_state_dict(torch.load(cfg['protected_checkpoint'],map_location='cpu',weights_only=False));original.eval()
    if digest(Path(cfg['source'])/'src/prob_flood_v1/model.py')!=digest(__import__('prob_flood_v1.model',fromlist=['']).__file__):raise ValueError('P0 math changed')
    legal={};elig={};q_provenance={};fixed_ids={};rules={}
    for role,pool in pools.items():
        q,q_provenance[role]=original_q(pool,config=cfg);legal[role],elig[role]=eligibility(pool,catalog,q);fixed_ids[role],rules[role]=fixed(pool,legal[role])
    if smoke:
        ix=sorted(set(fixed_ids['select']+list(range(64))));legal['select']=legal['select'][ix];pools['select']=subset(pools['select'],ix);fixed_ids['select'],rules['select']=fixed(pools['select'],legal['select'])
    for role,pool in pools.items():atomic_json(out/(role+'_rows.json'),pool['rows'])
    atomic_json(out/'Q_only_evaluation_provenance.json',q_provenance);atomic_json(out/'eligibility.json',elig)
    role_content={r:content_digest(v) for r,v in pools.items()};selections=[];skipped=[];new=0
    def new_model():
        random.seed(2);np.random.seed(2);torch.manual_seed(2);torch.cuda.manual_seed_all(2)
        return PeakModel('S2',original,TemporalAdapter(),nn.GRUCell(68,64),nn.Linear(64,2))
    initial={}
    for policy in 'ABC':
        model=new_model();initial[policy]=dict(adapter=state_hash(model.adapter),decoder=state_hash(model.decoder),head=state_hash(model.head),protected=state_hash(model.protected));atomic_json(out/'initialization.json',initial)
        if len({json.dumps(v,sort_keys=True) for v in initial.values()})!=1:raise AssertionError('unpaired initialization')
        current={**runtime,'seed':2,'arm':'S2','data_policy':policy,'slots_sha256':audit['exposure'][policy]['sha256'],'role_content_identity':role_content,'Q_evaluation_provenance':q_provenance,'catalog_sha256':digest(catalog),'initial_trainable_hash':initial[policy]}
        model=model.to(device);result,reused=train_one(model,pools,legal,fixed_ids,sampled[policy],out,current,role_content);selections.append(result)
        if reused:skipped.append(policy)
        else:new+=1
        aggregate(out);model.cpu();torch.cuda.empty_cache()
        if stop_after and new>=stop_after:
            result=dict(state='partial',reason='intentional smoke interruption',completed=len(selections),skipped=skipped);atomic_json(out/'progress.json',result);return result
    atomic_json(out/'all_selections_stopped.json',dict(state='complete',n=3,time=time.time()))
    from .score import final_scores
    final_scores(out,pools,legal,elig,cfg,scales,new_model,device,smoke)
    from .plot import plot
    plot(out)
    result=dict(state='complete',smoke=smoke,runs=3,skipped=skipped,scientific_status='smoke_not_scientific' if smoke else 'single_seed_exploratory')
    atomic_json(out/'completion.json',result);atomic_json(out/'progress.json',result);return result


def aggregate(out):
    rows=[json.loads(p.read_text()) for p in sorted(out.glob('seed_2/*/selection_complete.json'))]
    atomic_json(out/'selections.json',dict(state='complete' if len(rows)==3 else 'partial',expected=3,completed=len(rows),runs=rows));return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--mode',choices=['prepare','smoke','formal'],required=True);parser.add_argument('--stop-after',type=int);parser.add_argument('--cpu-smoke',action='store_true');parser.add_argument('--source-manifest',default=SOURCE);parser.add_argument('--path-map-json');parser.add_argument('--q-only-dir');a=parser.parse_args()
    if a.stop_after and a.mode!='smoke':raise ValueError('only smoke may intentionally stop')
    try:execute(a.output,a.mode,a.stop_after,a.cpu_smoke,a.source_manifest,a.path_map_json,a.q_only_dir)
    except Exception as e:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'failure.json',dict(state='failed',error_type=type(e).__name__,error=str(e),time=time.time()));raise

if __name__=='__main__':main()
