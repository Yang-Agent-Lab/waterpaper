"""Complete a failed display from retained predictions; no fit, inference, or rescoring."""
import argparse,json,subprocess
from pathlib import Path
from protected_flood_risk_v1.training import atomic_json
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.runner import subset
from predictor_benchmark_v1.data import source_config,load,smoke_subset
from training_report_v1.plot import render_report
from .runner import ROOT,report_bundle
from .model import ALL_MODELS,build_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);p.add_argument('--report-output');a=p.parse_args();out=Path(a.artifacts);target=Path(a.report_output) if a.report_output else out
    runtime=json.loads((out/'effective_config.json').read_text());scoring=json.loads((out/'scoring_manifest.json').read_text())
    if scoring['smoke']!=runtime['smoke']:raise ValueError('role identity mismatch')
    if (target/'report_bundle.json').exists() or (target/'figures').exists():raise ValueError('display already exists; preserve it')
    target.mkdir(parents=True,exist_ok=True)
    cfg=source_config(ROOT+'/source_manifest.json',ROOT+'/target_path_map_072b561.json',ROOT+'/Q_only')
    pools,_=load(cfg,['fit','select'])
    if runtime['smoke']:
        pools['select']=smoke_subset(pools['select']);pools['development']=subset(pools['select'],[i for i,r in enumerate(pools['select']['rows']) if r['country']=='China'])
    else:pools['development']=load(cfg,['development'],countries=['China'],final_authorized=True)[0]['development']
    selections=[]
    for name in ALL_MODELS:
        folder=out/'seed_2'/name;record=json.loads((folder/'selection_complete.json').read_text())
        if digest(folder/'best.pt')!=record['best_file_sha256']:raise ValueError('selected checkpoint changed')
        selections.append(record)
    scales=json.loads((Path(cfg['data'])/'scales.json').read_text());models={n:build_model(n) for n in ALL_MODELS}
    bundle=report_bundle(out,pools,runtime,models,scales,target/'report_bundle.json');manifest=render_report(bundle,target/'figures');atomic_json(target/'report_manifest.json',manifest)
    files=[p for p in target.rglob('*') if p.is_file() and p.name not in ('execution.lock','completion.json','progress.json')]
    complete=dict(state='complete',code_version=runtime['code_version'],report_code_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),models=list(ALL_MODELS),updates_per_trained_model=4 if runtime['smoke'] else 22600,source_artifacts=str(out),artifact_hashes={str(p.relative_to(target)):digest(p) for p in files},scientific_state='not_assessable_smoke' if runtime['smoke'] else 'exploratory_single_seed',repair='saved output display only; original failure and wrapper retained; no inference/training/scoring')
    atomic_json(target/'completion.json',complete);atomic_json(target/'progress.json',complete);print(json.dumps(dict(state='complete',report_pages=[p['file'] for p in manifest['pages']],inference_calls=0,new_training_updates=0)))

if __name__=='__main__':main()
