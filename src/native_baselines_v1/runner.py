"""Approved native baseline workflow; formal launch requires a parent acceptance file."""
import argparse,csv,fcntl,json,os,subprocess,time,importlib.metadata,hashlib
from pathlib import Path
import numpy as np
import torch
from protected_flood_risk_v1.training import atomic_json,state_hash,content_digest
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.runner import write_torch
from multiwindow_peak_v1.model import crps
from predictor_benchmark_v1.data import source_config,load,slots,fixed_probe,smoke_subset
from predictor_benchmark_v1.cases import make_cases,RULE
from predictor_benchmark_v1.statistics import mean,reliability
from predictor_benchmark_v1.runner import completed_result as old_completed_result,numeric
from .model import MODELS,ALL_MODELS,build_model,predict,raw_predictions,quantile_diagnostics,targets
from .training import train_one,evaluate,interventions
from .score import final_scores

ROOT='._inputs/data_org_v1'
CHANGE='WP-20261014-NATIVE-BASELINES-V1'
LEGACY=Path('._predictor_benchmark_v1/results/experiments/predictor_benchmark_formal_fbffc65_20261014/artifacts')

def exposure_slots(pool,smoke):
    if smoke:return slots(pool,2,256,2)
    previous=json.loads((LEGACY/'effective_config.json').read_text())
    if json.loads((LEGACY/'fit_rows.json').read_text())!=pool['rows']:raise ValueError('legacy slot row identities differ')
    shared=np.load(LEGACY/'slots.npy',allow_pickle=False);fingerprint=hashlib.sha256(shared.tobytes()).hexdigest()
    if shared.shape!=(100,28928) or shared.dtype!=np.int64 or fingerprint!=previous['slots_sha256']:raise ValueError('legacy exposure slots changed')
    if shared.min()<0 or shared.max()>=len(pool['rows']):raise ValueError('invalid legacy slot index')
    return shared,fingerprint

def reuse_legacy(out,source_manifest,smoke):
    record=dict(state='not_read_smoke' if smoke else 'running',models=['current_s2','direct_lstm','direct_tcn'],note='old models share S2 probability head; comparison is complete procedures, not isolated architecture')
    native=list(csv.DictReader((out/'summary.csv').open()));combined=native
    if not smoke and (LEGACY/'completion.json').exists():
        old_completed_result(LEGACY,source_manifest,'fbffc659143e4663788bda2650a750c1d9feb425')
        record.update(state='complete',artifact_path=str(LEGACY),source_summary_sha256=digest(LEGACY/'summary.csv'))
        combined=native+list(csv.DictReader((LEGACY/'summary.csv').open()))
    with (out/'comparison_summary.csv').open('x') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(dict.fromkeys(k for r in combined for k in r)));writer.writeheader();writer.writerows(combined)
    atomic_json(out/'legacy_result_reuse.json',record)

def saved_point_crps(rows):
    result={}
    for row in rows:
        if row.get('scope') not in ('',None,'overall') or row.get('station') or row.get('stratum'):continue
        value=numeric(row.get('crps'))
        if value is None:continue
        key=(row['data_policy'],int(row['horizon']))
        if key in result and result[key]!=value:raise ValueError('conflicting saved overall CRPS')
        result[key]=value
    return result

def report_bundle(out,pools,runtime,models,scales,bundle_path=None):
    epochs=[];health=[];diagnostics=[];cases=[];risk=[];sources={}
    for name in ALL_MODELS:
        folder=out/'seed_2'/name;selection=json.loads((folder/'selection_complete.json').read_text())
        log=folder/'epochs.jsonl'
        for row in [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []:
            for country in row['select']:
                item=dict(model=name,role='select',country=country,epoch=row['epoch'],step=row['step'],train_loss=row['train_loss'],fit_loss=row['fit'][country]['loss'],select_loss=row['select'][country]['loss'],selected=row['epoch']==selection['best_epoch'])
                if country=='China':item['selection_metric']=row['selection_score']
                epochs.append(item)
        if (folder/'health.jsonl').exists():health.extend(json.loads(s) for s in (folder/'health.jsonl').read_text().splitlines())
        diagnostics.extend(json.loads((folder/'selected_diagnostic.json').read_text())['result'])
        for role in ('select','development'):
            rows=pools[role]['rows'];path=folder/('best_select.npz' if role=='select' else 'development_final.npz')
            with np.load(path,allow_pickle=False) as z:
                draw=z['samples'];ids=z['window_ids'].astype(str)
            if list(ids)!=[r['window_id'] for r in rows]:raise ValueError('report row identity')
            truth=targets(torch.from_numpy(pools[role]['y'])).numpy();cases.extend(make_cases(rows,truth,draw,name,role))
            if role=='select':
                cc=crps(torch.from_numpy(draw),torch.from_numpy(truth)).numpy()
                for country in sorted({r['country'] for r in rows}):
                    ix=np.array([i for i,r in enumerate(rows) if r['country']==country]);rr=[rows[i] for i in ix]
                    for j,h in enumerate((1,3,6,12)):
                        ratio=np.array([scales[r['station_uid']]['q99']/r['qscale'] for r in rr]);prob=(draw[ix,:,j]>ratio[:,None]).mean(1);truth01=truth[ix,j]>ratio;bins=reliability(prob,truth01,rr)
                        risk.append(dict(model=name,country=country,role='select',horizon=h,target='window_exceedance_Q99',brier=mean((prob-truth01)**2,rr),crps=mean(cc[ix,j],rr),detected=None,events=None,false_alarm_segments=None,lead_hours=None,reliability_bins=[dict(mean_probability=b['predicted'],observed_fraction=b['observed'],n=b['windows']) for b in bins],state='partial',reason='same-station select; foreign alarm eligibility NOT_EVALUABLE'))
        sources[name]=dict(model_id=name,weight_hash=selection['weight_hash'],checkpoint_sha256=selection['best_file_sha256'],native_config=models[name].config)
    rel=list(csv.DictReader((out/'reliability.csv').open()))
    point_crps=saved_point_crps(csv.DictReader((out/'point_metrics.csv').open()))
    for row in csv.DictReader((out/'summary.csv').open()):
        if row['scope']!='overall' or float(row['fpr_target'])!=.05:continue
        num=lambda k:float(row[k]) if row[k] not in ('','None') else None
        bins=[b for b in rel if all(b[k]==row[k] for k in ('data_policy','target','horizon'))]
        risk.append(dict(model=row['data_policy'],country='China',role='development',horizon=int(row['horizon']),target=row['target'],brier=num('brier'),crps=point_crps.get((row['data_policy'],int(row['horizon']))),detected=num('discovered_events'),events=num('evaluable_events'),false_alarm_segments=num('false_alarm_segments'),lead_hours=num('detected_mean_lead_hours'),reliability_bins=[dict(mean_probability=numeric(b['predicted']),observed_fraction=numeric(b['observed']),n=int(b['windows'])) for b in bins],state=row['state'],reason='fit/select standins' if runtime['smoke'] else 'saved overall window-peak CRPS; observation eligibility limits event counts'))
    bundle=dict(schema_version=1,run_id=runtime['run_id'],models=[dict(id=n,label=n) for n in ALL_MODELS],epochs=epochs,cases=cases,risk=risk,health=health,interventions=diagnostics,metadata=dict(selection_label='中国6h CRPS/Q95',loss_label='原生MQLoss',loss_components=[],runtime=runtime,model_sources=sources,case_identity=RULE,statistical_unit='source event then station equal; foreign blocks then station equal',units='case flow m3/s; losses/CRPS normalized by station fit Q95',input_roles={r:content_digest(p) for r,p in pools.items()},scientific_state='engineering_smoke_not_scientific' if runtime['smoke'] else 'single_seed_exploratory',limitations=['four outputs are nested peaks, not hourly flow','quantile tails fixed at 0.005/0.995; no learned joint scenarios','sort quantiles then cumulative max across windows; raw negatives retained','temporary input intervention is not retrained module contribution','all new models use native MQLoss; old S2 loss not comparable','Iceland P reanalysis; prefix source unknown']))
    bundle['metadata']['loss_components_note']='单一损失，无独立分项'
    bundle['metadata']['crps_population']='all_valid_origins'
    bundle['metadata']['crps_label']='CRPS(全窗)'
    bundle['metadata']['risk_population_note']='CRPS covers all valid origins at the given horizon; Q99 Brier and discovery use their eligible alert subset, not the same denominator'
    atomic_json(bundle_path or out/'report_bundle.json',bundle);return bundle

def execute(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();smoke=args.mode=='smoke';torch.set_num_threads(1)
    if not smoke:
        if not args.parent_acceptance:raise PermissionError('formal requires parent acceptance')
        decision=json.loads(Path(args.parent_acceptance).read_text())
        if decision.get('accepted_commit')!=head or decision.get('change_id')!=CHANGE or decision.get('formal_authorized') is not True:raise PermissionError('parent acceptance identity')
    if (out/'completion.json').exists():
        complete=json.loads((out/'completion.json').read_text())
        if complete['code_version']!=head:raise ValueError('completed code changed')
        for path,sha in complete['artifact_hashes'].items():
            if digest(out/path)!=sha:raise ValueError('completed artifact changed '+path)
        source=json.loads((out/'source_manifest.json').read_text())
        if digest(args.source_manifest)!=source['source_manifest_sha256']:raise ValueError('completed source manifest changed')
        for path,sha in source['source']['inputs'].items():
            if digest(path)!=sha:raise ValueError('completed input changed '+path)
        return dict(complete,reused=True,new_training_updates=0)
    if (out/'scoring_manifest.json').exists():raise ValueError('scoring complete; reuse saved report in new report directory')
    device=torch.device('cpu' if args.cpu_smoke else 'cuda')
    if args.cpu_smoke and not smoke:raise PermissionError('CPU only smoke')
    if device.type=='cuda':
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not selected or ',' in selected:raise RuntimeError('one explicit GPU required')
        uuid=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();gpu_lock=open('./external-home/predictor_benchmark_'+uuid+'.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','-i',selected,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg=source_config(args.source_manifest,args.path_map_json,args.q_only_dir);pools,source=load(cfg,['fit','select']);full_counts={r:len(p['rows']) for r,p in pools.items()}
    if smoke:pools['select']=smoke_subset(pools['select'])
    probes={r:fixed_probe(p) for r,p in pools.items()};shared,fingerprint=exposure_slots(pools['fit'],smoke)
    package_config=json.loads(Path('configs/native_baselines_v1.json').read_text())
    runtime=dict(change_id=CHANGE,code_version=head,run_id=out.parent.name,smoke=smoke,seed=2,epochs=2 if smoke else 100,epoch_size=256 if smoke else 28928,batch_size=128,learning_rate=.001,clip=10,slots_sha256=fingerprint,full_role_counts=full_counts,role_hashes={r:content_digest(p) for r,p in pools.items()},probe_ids={r:[pools[r]['rows'][i]['window_id'] for i in ix] for r,ix in probes.items()},data_manifest_sha256=cfg['data_manifest_sha256'],device=str(device),neuralforecast_tag=package_config['neuralforecast_tag'],neuralforecast_commit=package_config['neuralforecast_commit'],torch=torch.__version__)
    if (out/'effective_config.json').exists() and json.loads((out/'effective_config.json').read_text())!=runtime:raise ValueError('resume runtime changed')
    atomic_json(out/'effective_config.json',runtime);atomic_json(out/'source_manifest.json',dict(source=source,source_manifest_sha256=digest(args.source_manifest),test_read=False,future_P_read=False));np.save(out/'slots.npy',shared)
    for role,pool in pools.items():atomic_json(out/(role+'_rows.json'),pool['rows'])
    configs={n:build_model(n).config for n in ALL_MODELS};atomic_json(out/'native_model_configs.json',configs)
    import neuralforecast
    atomic_json(out/'environment.json',dict(python=os.sys.version,executable=os.sys.executable,versions={name:importlib.metadata.version(name) for name in ('neuralforecast','torch','numpy','pandas','scipy','matplotlib','pytorch-lightning','ray','coreforecast','utilsforecast')},neuralforecast_file=neuralforecast.__file__,source_tag=package_config['neuralforecast_tag'],source_commit=package_config['neuralforecast_commit'],dependency_source=package_config['author_dependency_source'],installation_index='https://pypi.tuna.tsinghua.edu.cn/simple',cuda_runtime=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(0) if device.type=='cuda' else None,driver=subprocess.check_output(['nvidia-smi','--query-gpu=index,driver_version','--format=csv,noheader'],text=True).strip(),old_environment_modified=False))
    selections=[];skipped=[];new=0
    for name in ALL_MODELS:
        model=build_model(name).to(device);folder=out/'seed_2'/name
        if name=='persistence':
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'selection_complete.json').exists():
                score,draw,raw=evaluate(model,pools['select']);write_torch(folder/'best.pt',dict(model=model.state_dict(),model_id=name));np.savez_compressed(folder/'best_select.npz',samples=draw,raw_quantiles=raw,window_ids=np.array([r['window_id'] for r in pools['select']['rows']]))
                result=dict(state='complete',model=name,updates=0,best_epoch=None,best_score=score['China']['crps6'],best_file_sha256=digest(folder/'best.pt'),weight_hash=state_hash(model));atomic_json(folder/'selection_complete.json',result);atomic_json(folder/'selected_diagnostic.json',interventions(model,pools,probes,'selected'))
            else:
                result=json.loads((folder/'selection_complete.json').read_text())
                if digest(folder/'best.pt')!=result['best_file_sha256']:raise ValueError('persistence checkpoint changed')
        else:
            result,reused=train_one(model,pools,probes,shared,out,runtime)
            if reused:skipped.append(name)
            else:new+=1
        selections.append(result);model.cpu();torch.cuda.empty_cache();atomic_json(out/'selections.json',dict(completed=len(selections),expected=6,rows=selections))
        if args.stop_after and new>=args.stop_after:return dict(state='partial',skipped=skipped,new_completed=new)
    atomic_json(out/'all_selections_stopped.json',dict(state='complete',models=list(ALL_MODELS),time=time.time()))
    scales=json.loads((Path(cfg['data'])/'scales.json').read_text());final_scores(out,pools,{}, {},cfg,scales,build_model,device,smoke);reuse_legacy(out,args.source_manifest,smoke)
    pools['development']=subset(pools['select'],[i for i,r in enumerate(pools['select']['rows']) if r['country']=='China']) if smoke else load(cfg,['development'],countries=['China'],final_authorized=True)[0]['development']
    models={n:build_model(n) for n in ALL_MODELS};bundle=report_bundle(out,pools,runtime,models,scales)
    from training_report_v1.plot import render_report
    atomic_json(out/'report_manifest.json',render_report(bundle,out/'figures'))
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in ('execution.lock','completion.json','progress.json','failure.json')]
    complete=dict(state='complete',code_version=head,models=list(ALL_MODELS),skipped=skipped,updates_per_trained_model=4 if smoke else 22600,artifact_hashes={str(p.relative_to(out)):digest(p) for p in files},scientific_state='not_assessable_smoke' if smoke else 'exploratory_single_seed')
    atomic_json(out/'completion.json',complete);atomic_json(out/'progress.json',complete);return complete

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['smoke','formal'],required=True);p.add_argument('--cpu-smoke',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--parent-acceptance');p.add_argument('--source-manifest',default=ROOT+'/source_manifest.json');p.add_argument('--path-map-json',default=ROOT+'/target_path_map_072b561.json');p.add_argument('--q-only-dir',default=ROOT+'/Q_only');args=p.parse_args()
    if args.stop_after and args.mode!='smoke':raise ValueError('stop-after only smoke')
    try:
        result=execute(args);print(json.dumps(result),flush=True)
    except Exception as e:
        Path(args.output).mkdir(parents=True,exist_ok=True);atomic_json(Path(args.output)/'failure.json',dict(state='failed',error=str(e),type=type(e).__name__,time=time.time()));raise

if __name__=='__main__':main()
