"""Merge only small scored reports from hosts after exact cohort/unit identity checks."""
import argparse,csv,json,subprocess
from pathlib import Path
from .run import assemble,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--input',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False);reports={};leads={};sources=[]
    for source in map(Path,a.input):
        parent_manifest=source/'manifest.json';parent=json.loads(parent_manifest.read_text())
        run_record=source.parent/'run_record.json';record=json.loads(run_record.read_text())
        assert record['state']=='complete' and record['exit_code']==0 and record['head']==parent['commit']
        for item in cfg['models']:
            name=item['id'];folder=source/name
            if not (folder/'metrics.json').exists():continue
            assert name not in reports
            report=json.loads((folder/'metrics.json').read_text());identity=report['identity']
            assert identity['N']==cfg['expected_origins'] and identity['K']==cfg['expected_scenarios'] and identity['H']==cfg['expected_horizon']
            for key,value in cfg['canonical_expected'].items():assert identity[key]==value
            reports[name]=report;curve=list(csv.DictReader((folder/'lead_metrics.csv').open()))
            leads[name]=[dict(r,lead_hour=int(r['lead_hour']),crps_q95=float(r['crps_q95']),q95_twcrps_q95=float(r['q95_twcrps_q95'])) for r in curve]
            sources.append(dict(model=name,directory=str(folder),metrics_sha256=sha(folder/'metrics.json'),identity_sha256=sha(folder/'identity.json'),lead_metrics_sha256=sha(folder/'lead_metrics.csv'),actual_scoring_commit=parent['commit'],parent_manifest=str(parent_manifest),parent_manifest_sha256=sha(parent_manifest),run_record=str(run_record),run_record_sha256=sha(run_record),original_run_record=record))
    assemble(cfg,out,reports,leads)
    from .plot import plot
    plot(out,cfg)
    save(out/'manifest.json',dict(experiment_id=cfg['experiment_id'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_reports=sources,canonical_expected=cfg['canonical_expected'],model_forward=0,training_updates=0,random_draws=0,cloud_sync=False))
    save(out/'completion.json',dict(state='MERGED_METRICS_COMPLETE_PAUSED',models=list(reports),next_job='NONE',independent_acceptance='pending'))
if __name__=='__main__':main()
