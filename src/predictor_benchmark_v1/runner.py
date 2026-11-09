"""Approved three-model benchmark; formal launch remains an explicit parent decision."""
import argparse,csv,fcntl,json,os,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from protected_flood_risk_v1.training import atomic_json,state_hash,content_digest
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.runner import predict
from multiwindow_peak_v1.model import crps
from multiwindow_peak_v1.data import eligibility,fixed
from data_organization_v1.runtime_io import original_q
from .statistics import mean,reliability
from .data import source_config,load,slots,fixed_probe,smoke_subset
from .model import build_model,MODELS,HEAD_SEED
from .training import train_one
from .score import final_scores
from .cases import make_cases,PREFERRED,RULE

ROOT='._inputs/data_org_v1'
CHANGE='WP-20261014-PREDICTOR-BENCHMARK-V1'
H=(1,3,6,12)

def read_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def read_csv(path):return list(csv.DictReader(path.open())) if path.exists() else []
def numeric(value):return float(value) if value not in ('','None','null',None) else None

def completed_result(out,source_manifest,code):
    path=out/'completion.json'
    if not path.exists():return None
    record=json.loads(path.read_text())
    if record.get('state')!='complete' or record.get('code_version')!=code or record.get('models')!=list(MODELS):raise ValueError('invalid completion identity')
    hashes=record.get('artifact_hashes',{})
    required={'source_manifest.json','report_bundle.json','report_manifest.json'}|{f'seed_2/{name}/selection_complete.json' for name in MODELS}|{f'seed_2/{name}/best.pt' for name in MODELS}
    required|={f'figures/{name}' for name in ('01_learning.png','02_predictions.png','03_risk.png','04_modules.png','training_report.pdf','report_manifest.json')}
    if not required.issubset(hashes):raise ValueError('completion missing required hashes')
    for name,expected in hashes.items():
        target=out/name
        if not target.resolve().is_relative_to(out.resolve()) or not target.is_file() or digest(target)!=expected:raise ValueError('completed artifact changed '+name)
    source=json.loads((out/'source_manifest.json').read_text())
    report=json.loads((out/'report_manifest.json').read_text())
    if [r['file'] for r in report.get('pages',[])]!=['01_learning.png','02_predictions.png','03_risk.png','04_modules.png']:raise ValueError('completed report pages changed')
    if digest(source_manifest)!=source['source_manifest_sha256']:raise ValueError('completed source manifest changed')
    for path,expected in source['source']['inputs'].items():
        if digest(path)!=expected:raise ValueError('completed input changed '+path)
    for name in MODELS:
        folder=out/'seed_2'/name;selected=json.loads((folder/'selection_complete.json').read_text())
        if selected['state']!='complete' or digest(folder/'best.pt')!=selected['best_file_sha256']:raise ValueError('completed selection changed')
    return {**record,'reused':True,'new_training_updates':0,'new_prediction_calls':0}

def report_bundle(out,pools,runtime,models,scales):
    epochs=[];health=[];interventions=[];cases=[];risk=[];sources={}
    for name in MODELS:
        folder=out/'seed_2'/name;selection=json.loads((folder/'selection_complete.json').read_text())
        for row in read_jsonl(folder/'epochs.jsonl'):
            for country in row['select']:
                item=dict(model=name,role='select',country=country,epoch=row['epoch'],step=row['step'],train_loss=row['train_loss'],train_nll=row['train_nll'],train_crps=row['train_crps'],fit_loss=row['fit'][country]['loss'],fit_nll=row['fit'][country]['nll'],fit_crps=row['fit'][country]['crps'],select_loss=row['select'][country]['loss'],select_nll=row['select'][country]['nll'],select_crps=row['select'][country]['crps'],selected=row['epoch']==selection['best_epoch'])
                if country=='China':item['selection_metric']=row['selection_score']
                epochs.append(item)
        health.extend(read_jsonl(folder/'health.jsonl'));interventions.extend(json.loads((folder/'selected_diagnostic.json').read_text())['result'])
        # Fixed cases selected by observed input/target before comparing models.
        for role in ('select','development'):
            rp=out/(role+'_rows.json')
            if not rp.exists():continue
            rows=json.loads(rp.read_text());npz=folder/('best_select.npz' if role=='select' else 'development_final.npz')
            with np.load(npz,allow_pickle=False) as z:draw=z['samples'];ids=z['window_ids'].astype(str)
            if list(ids)!=[r['window_id'] for r in rows]:raise ValueError('prediction/row identity mismatch')
            pool=pools[role];truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);median=np.median(draw,axis=1);lo,hi=np.quantile(draw,[.05,.95],axis=1)
            cases.extend(make_cases(rows,truth,draw,name,role))
            if role=='select':
                cc=crps(torch.from_numpy(draw),torch.from_numpy(truth)).numpy()
                for country in sorted({r['country'] for r in rows}):
                    ix=np.array([i for i,r in enumerate(rows) if r['country']==country]);rr=[rows[i] for i in ix]
                    for j,h in enumerate(H):
                        ratio=np.array([scales[r['station_uid']]['q99']/r['qscale'] for r in rr])
                        prob=(draw[ix,:,j]>ratio[:,None]).mean(1);target=truth[ix,j]>ratio
                        bins=reliability(prob,target,rr)
                        risk.append(dict(model=name,country=country,role='select',horizon=h,target='window_exceedance_Q99',brier=mean((prob-target)**2,rr),crps=mean(cc[ix,j],rr),detected=None,events=None,false_alarm_segments=None,lead_hours=None,reliability_bins=[dict(mean_probability=b['predicted'],observed_fraction=b['observed'],n=b['windows']) for b in bins],state='partial',reason='uncalibrated development holdout; event discovery NOT_EVALUABLE'))
        sources[name]=dict(model_id=name,weight_hash=selection['weight_hash'],checkpoint_sha256=selection['best_file_sha256'],encoder_class=type(models[name].encoder).__name__ if models[name].encoder is not None else 'P0+TemporalAdapter')
    for row in read_csv(out/'summary.csv'):
        if row['scope']!='overall' or float(row['fpr_target'])!=.05:continue
        bins=[r for r in read_csv(out/'reliability.csv') if r['data_policy']==row['data_policy'] and r['target']==row['target'] and r['horizon']==row['horizon']]
        risk.append(dict(model=row['data_policy'],country='China',role='development',horizon=int(row['horizon']),target=row['target'],brier=numeric(row['brier']),crps=None,detected=numeric(row['discovered_events']),events=numeric(row['evaluable_events']),false_alarm_segments=numeric(row['false_alarm_segments']),lead_hours=numeric(row['detected_mean_lead_hours']),reliability_bins=[dict(mean_probability=numeric(b['predicted']),observed_fraction=numeric(b['observed']),n=int(b['windows'])) for b in bins],state=row['state'],reason='smoke fit/select standins; CRPS in point_metrics.csv' if runtime['smoke'] else 'CRPS in point_metrics.csv; event quantities conditional on observation eligibility'))
    bundle=dict(schema_version=1,run_id=runtime['run_id'],models=[dict(id=name,label=name) for name in MODELS],epochs=epochs,cases=cases,risk=risk,health=health,interventions=interventions,metadata=dict(selection_label='中国6h CRPS/Q95',preferred_cases=[dict(station='Liqingdian',role='development',case_kind='difficult'),dict(station='Luanchuan',role='development',case_kind='difficult'),dict(station='Luxi',role='development',case_kind='ordinary'),dict(station='Zhuxipo',role='development',case_kind='difficult')],statistical_unit='China source event then station equal; foreign temporal block then station equal, not independent floods',runtime=runtime,model_sources=sources,units='case flow m3/s; CRPS and MAE divided by station fit Q95',case_identity=RULE,selection='China select H6 CRPS; event then station equal',curves='train minibatch average; fit_probe/select event-station equal, 128 scenarios',intervention='temporary no-gradient changes, delta=disabled-full; Brier target window Q95, not event/combined Q99',input_roles={r:content_digest(p) for r,p in pools.items()},final_role_sources='fit/select standins' if runtime['smoke'] else 'calibration then development',scientific_state='engineering_smoke_not_scientific' if runtime['smoke'] else 'single_seed_exploratory',limitations=['P0 pretraining cost excluded from new update count','foreign select previously used in broader project; same-station development holdout','prefix source unknown','temporary intervention is not removal-and-retraining contribution']))
    atomic_json(out/'report_bundle.json',bundle);return bundle

def execute(output,smoke,stop_after,cpu,source_manifest,path_map,q_only):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if cpu and not smoke:raise ValueError('CPU only for engineering smoke')
    cfg=source_config(source_manifest,path_map,q_only);head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();torch.set_num_threads(1)
    completed=completed_result(out,source_manifest,head)
    if completed is not None:return completed
    if (out/'scoring_manifest.json').exists():raise ValueError('scoring already saved; use saved report_bundle with training_report_v1 CLI in a new directory, no rescoring')
    if not cpu:
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not selected or ',' in selected:raise RuntimeError('one explicit GPU required')
        uuid=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        gpu_lock=Path('./external-home')/('predictor_benchmark_'+uuid+'.lock');gpu_fp=gpu_lock.open('a');fcntl.flock(gpu_fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','-i',selected,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        if occupied or torch.cuda.device_count()!=1:raise RuntimeError('GPU occupied or visibility invalid')
    device=torch.device('cpu' if cpu else 'cuda');torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    pools,source=load(cfg,['fit','select']);full_counts={r:len(p['rows']) for r,p in pools.items()}
    if smoke:pools['select']=smoke_subset(pools['select'])
    probes={r:fixed_probe(p) for r,p in pools.items()};shared,fingerprint=slots(pools['fit'],2 if smoke else 100,256 if smoke else 28928,2)
    runtime=dict(change_id=CHANGE,code_version=head,run_id=out.parent.name,smoke=smoke,seed=2,epochs=2 if smoke else 100,epoch_size=256 if smoke else 28928,batch_size=128,learning_rate=.001,clip=10,train_scenarios=16,select_scenarios=128,final_scenarios=512,head_seed=HEAD_SEED,slots_sha256=fingerprint,full_role_counts=full_counts,data_manifest_sha256=cfg['data_manifest_sha256'],role_hashes={r:content_digest(p) for r,p in pools.items()},probe_ids={r:[pools[r]['rows'][i]['window_id'] for i in ix] for r,ix in probes.items()},device=str(device))
    if (out/'effective_config.json').exists() and json.loads((out/'effective_config.json').read_text())!=runtime:raise ValueError('resume config changed')
    atomic_json(out/'effective_config.json',runtime);atomic_json(out/'source_manifest.json',dict(source=source,source_manifest=source_manifest,source_manifest_sha256=digest(source_manifest),path_map=cfg['_path_map'],torch=torch.__version__,numpy=np.__version__,test_read=False,future_P_read=False));np.save(out/'slots.npy',shared)
    for role,pool in pools.items():atomic_json(out/(role+'_rows.json'),pool['rows'])
    original=Forecaster('P0',json.loads((Path(cfg['source'])/'configs/prob_flood_v1.json').read_text()));original.load_state_dict(torch.load(cfg['protected_checkpoint'],map_location='cpu',weights_only=False));original.eval()
    if digest(Path(cfg['source'])/'src/prob_flood_v1/model.py')!=digest(__import__('prob_flood_v1.model',fromlist=['']).__file__):raise ValueError('P0 source math changed')
    def new_model(name):return build_model(name,original,2)
    models={name:new_model(name) for name in MODELS};initial={name:dict(decoder=state_hash(m.decoder),head=state_hash(m.head),protected=state_hash(m.protected) if m.protected is not None else None,model=state_hash(m)) for name,m in models.items()}
    if len({v['decoder'] for v in initial.values()})!=1 or len({v['head'] for v in initial.values()})!=1:raise AssertionError('head initialization differs')
    atomic_json(out/'initialization.json',initial);new=0;skipped=[];selections=[]
    for name,model in models.items():
        random.seed(2);np.random.seed(2);torch.manual_seed(2)
        result,reused=train_one(model.to(device),pools,probes,shared,out,runtime);selections.append(result)
        if reused:skipped.append(name)
        else:new+=1
        model.cpu();torch.cuda.empty_cache();atomic_json(out/'selections.json',dict(completed=len(selections),expected=3,rows=selections))
        if stop_after and new>=stop_after:atomic_json(out/'smoke_interruption.json',dict(state='partial',skipped=skipped,new_completed=new));return
    atomic_json(out/'all_selections_stopped.json',dict(state='complete',models=list(MODELS),time=time.time()))
    # Old scoring functions, thresholds, eligibility and strict alarm rules, with
    # independent model-specific constructors/checkpoints. No historical source edits.
    scales=json.loads((Path(cfg['data'])/'scales.json').read_text())
    final_scores(out,pools,{}, {},cfg,scales,new_model,device,smoke)
    if smoke:dev=subset(pools['select'],[i for i,r in enumerate(pools['select']['rows']) if r['country']=='China'])
    else:dev=load(cfg,['development'],countries=['China'],final_authorized=True)[0]['development']
    pools['development']=dev
    bundle=report_bundle(out,pools,runtime,models,scales)
    from training_report_v1.plot import render_report
    manifest=render_report(bundle,out/'figures');atomic_json(out/'report_manifest.json',manifest)
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in ('execution.lock','progress.json','failure.json','completion.json') and 'diagnostic_cache' not in p.parts]
    complete=dict(state='complete',code_version=head,smoke=smoke,models=list(MODELS),skipped=skipped,updates_per_model=4 if smoke else 22600,report_bundle_sha256=digest(out/'report_bundle.json'),artifact_hashes={str(p.relative_to(out)):digest(p) for p in files},scientific_state='not_assessable_smoke' if smoke else 'exploratory_single_seed')
    atomic_json(out/'completion.json',complete);atomic_json(out/'progress.json',complete);return complete

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['smoke','formal'],required=True);p.add_argument('--cpu-smoke',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--source-manifest',default=ROOT+'/source_manifest.json');p.add_argument('--path-map-json',default=ROOT+'/target_path_map_072b561.json');p.add_argument('--q-only-dir',default=ROOT+'/Q_only');a=p.parse_args()
    if a.stop_after and a.mode!='smoke':raise ValueError('stop-after only allowed for smoke')
    try:execute(a.output,a.mode=='smoke',a.stop_after,a.cpu_smoke,a.source_manifest,a.path_map_json,a.q_only_dir)
    except Exception as e:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'failure.json',dict(state='failed',type=type(e).__name__,error=str(e),time=time.time()));raise

if __name__=='__main__':main()
