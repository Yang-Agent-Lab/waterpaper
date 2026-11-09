"""Copy fixed prepared bytes and fit/select Q-only metadata, never raw mixed-role gz."""
import argparse,json,shutil,time
from pathlib import Path
from protected_flood_risk_v1.training import atomic_json
from protected_flood_risk_v1.data import digest
from multiwindow_peak_v1.data import load
from .runner import source_config,SOURCE,CATALOG
from .runtime_io import original_q

EVENT=Path('._event_risk_25efa99/results/experiments/event_risk_formal_25efa99_20261013/artifacts')
PEAK=Path('._multiwindow_97c3930/results/experiments/multiwindow_formal_97c3930_20261011/artifacts')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);cfg=source_config();manifest=[];mapping={}
    def cp(source,relative):
        source=Path(source);destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        before=digest(source);after=digest(destination)
        if before!=after:raise ValueError('copy identity mismatch')
        manifest.append(dict(source=str(source),relative=str(relative),sha256=after,bytes=destination.stat().st_size));atomic_json(out/'progress.json',dict(state='running',copied=len(manifest),last_file=str(relative)))
        return destination
    cp(SOURCE,'source_manifest.json')
    prepared=Path(cfg['data']);mapping[str(prepared)]=str(out/'prepared')
    for filename in ['manifest.json','scales.json','rise_cutoffs.json']+[r+suffix for r in ('fit','select','calibration','development') for suffix in ('.npz','_rows.json')]:cp(prepared/filename,Path('prepared')/filename)
    for key in ('attribute_rows','normalization','protected_checkpoint'):
        destination=cp(cfg[key],Path('fixed')/Path(cfg[key]).name);mapping[cfg[key]]=str(destination)
    mapping[cfg['source']]=str(out/'source')
    for filename in ('configs/prob_flood_v1.json','src/prob_flood_v1/model.py'):cp(Path(cfg['source'])/filename,Path('source')/filename)
    mapping[str(CATALOG)]=str(cp(CATALOG,'fixed/china_event_catalog_v1.csv'))
    mapping[cfg['gpu_lock']]=str(out/'fixed/gpu0.lock')
    mapping[str(EVENT)]=str(out/'accepted_event')
    for filename in ('event_risk.csv','thresholds.json','manifest.json'):cp(EVENT/filename,Path('accepted_event')/filename)
    cp(EVENT.parent/'run_record.json','accepted_event/run_record.json')
    mapping[str(PEAK/'final_eligibility.json')]=str(cp(PEAK/'final_eligibility.json','accepted_event/development_eligibility.json'))
    # Q-only author metadata is selected by role/event before parsing Q/time.
    pools,_=load(cfg,['fit','select']);qout=out/'Q_only';qout.mkdir()
    for role,pool in pools.items():
        records,provenance=original_q(pool,config=cfg)
        value=dict(role=role,records=[dict(station=s,source_event_id=e,issues=sorted(v['issues']),points=[[t.isoformat(),source,q] for t,(source,q) in sorted(v['points'].items())]) for (s,e),v in sorted(records.items())],source_provenance=provenance)
        atomic_json(qout/(role+'.json'),value);manifest.append(dict(relative='Q_only/'+role+'.json',sha256=digest(qout/(role+'.json')),bytes=(qout/(role+'.json')).stat().st_size,source='event-filtered original Q-only metadata'))
    atomic_json(out/'path_map.json',mapping)
    atomic_json(out/'manifest.json',dict(state='complete',files=manifest,total_bytes=sum(r['bytes'] for r in manifest),test_copied=False,raw_gz_copied=False,future_P_read=False,Q_roles_parsed=['fit','select'],final_role_arrays='copied bytes only; values not parsed',complete_time=time.time()))
    atomic_json(out/'progress.json',dict(state='complete',files=len(manifest),bytes=sum(r['bytes'] for r in manifest)));print(json.dumps(dict(state='complete',files=len(manifest),bytes=sum(r['bytes'] for r in manifest))),flush=True)

if __name__=='__main__':main()
