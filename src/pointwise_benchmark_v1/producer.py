"""Materialize only a fixed shared sample plan, not every overlapping source window."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time
import multiprocessing
import numpy as np
from src.five_country_model_data_v1.prepare import hour,iso
from src.pointwise_flood_risk_v1.adapter import adapt,context
from src.workspace_consolidation_v1.relocation import RelocatedReader
from .coverage import save
from .dataset import sha256,SampleBank
from .legacy_blocks import read_blocks,FOREIGN
from .planning import units,make_plan
from .source_slices import batch_records


def samples_for_station(reader,source,station,plans,permission,permission_sha):
    ctx=context(reader,source,station,plans[0]['route'],'q95')
    if source in FOREIGN:
        events={p['event'] for p in plans}
        blocks=[b for b in permission['blocks'] if b['source']==source and b['station']==station and b['event'] in events]
        stats={};raw=read_blocks(blocks,stats);records={}
        for b in blocks:
            values=[]
            for r in raw[b['event']]:
                values.append(reader.record(source,station,iso(r['native_hour'],'+00:00'),r['p'],r['q'],
                    role=b['role'],event=b['event'],label='start' if source=='lamah_ice' else 'unknown',
                    source_row=r['source_row'],p_ok=r['p_mask'],q_ok=r['q_mask'],support='catchment_mean',
                    quality={'state':'original_allowed_block_numeric_and_QC','qc_flags':r['quality'],
                             'derived_from_prepared':False,'original_allocated_block':b['event']}))
            records[(b['role'],b['event'])]=values
        # Raw observations newly recovered beyond the old quota export constitute
        # a new data version even though roles and fitted scales are unchanged.
        ctx['data_version']=hashlib.sha256(('restored_original_allocated_blocks_v1:'+permission_sha).encode()).hexdigest()
    elif source=='da_tl_cn':
        records={};stats={}
        for role in {p['role'] for p in plans}:
            groups=defaultdict(list)
            for r in reader._legacy(source,station,role,None):groups[r['event']].append(r)
            for event,rows in groups.items():records[(role,event)]=rows
    else:
        native=batch_records(reader,source,station,plans);records={(role,'continuous'):rows for role,rows in native.items()};stats={}
    indexed={key:{hour(r['interval_end']):r for r in rows} for key,rows in records.items()}
    for plan in plans:
        key=(plan['role'],plan['event']);origin=plan['origin_hour']
        lookup=indexed[key]
        try:points=[lookup[t] for t in range(origin-plan['history_hours']+1,origin+plan['forecast_hours']+1)]
        except KeyError as error:raise ValueError('planned window missing an actual source hour: '+str(plan)) from error
        sample=adapt(points,lookup[origin]['interval_end'],ctx,route=plan['route'],history_hours=plan['history_hours'],forecast_hours=plan['forecast_hours'])
        sample['threshold_source']['evidence']=str(reader.path_map.logical(sample['threshold_source']['evidence']))
        sample['sample_id']=hashlib.sha256(json.dumps([sample['identity'],sample['origin_time'],sample['history_hours'],
            sample['forecast_hours'],sample['route'],sample['data_version'],sample['threshold_source']],sort_keys=True).encode()).hexdigest()
        sc=ctx['table']['scale'] if ctx['table'] else reader._load_legacy('fit')[1][source+':'+station]
        q99=sc['q99'];qscale=ctx['qscale']
        row=dict(sample_id=sample['sample_id'],plan_id=plan['plan_id'],source=source,station=station,
            canonical_station=plan['canonical_station'],country=plan['country'],role=plan['role'],route=plan['route'],
            event=plan['source_group_id'],source_event_identity=sample['identity'][2],grouping_unit=plan['grouping_unit'],
            origin_time=sample['origin_time'],input_start=sample['input_times'][0],target_end=sample['target_times'][-1],
            current_information_included=sample['current_information_included'],target_leads=sample['lead_hours'],
            current_state=sample['current_state'],
            q_scale=qscale,q95=ctx['threshold_value'],q99=q99,threshold_source=sample['threshold_source'],
            threshold_sources={'Q95':sample['threshold_source'],
                'Q99':dict(sample['threshold_source'],quantile='q99',value=q99,field='q99')},
            data_version=sample['data_version'],timezone=sample['timezone'],information_mode='hindcast',
            availability='actual_release_unknown',offline_metadata=sample['offline_metadata'],
            source_rows=[r['source_row'] for r in points],predictor_fields='historical P/Q values only; identity/event/offline completeness excluded')
        yield plan,row,np.asarray(sample['x_normalized'],dtype=np.float32),np.asarray(sample['mask'],dtype=bool),np.asarray(sample['y_q_normalized'],dtype=np.float64),np.asarray([ctx['threshold_value']/qscale,q99/qscale],dtype=np.float64),np.asarray(sample['y_q'],dtype=np.float64),stats
    reader._prepared.clear()


_WORKER_READER=None


def worker_initialize(reader_kwargs):
    global _WORKER_READER
    _WORKER_READER=RelocatedReader(**reader_kwargs)


def station_worker(payload):
    source,station,plans,blocks,permission_sha=payload
    return list(samples_for_station(_WORKER_READER,source,station,plans,dict(blocks=blocks),permission_sha))


def station_results(reader,groups,permission,permission_sha,workers,reader_kwargs):
    by_station=defaultdict(list)
    for block in permission['blocks']:by_station[(block['source'],block['station'])].append(block)
    if workers==1:
        for (source,station),plans in sorted(groups.items()):
            yield source,station,list(samples_for_station(reader,source,station,plans,dict(blocks=by_station[(source,station)]),permission_sha))
        return
    # Only at most `workers` source jobs are in flight. Results are written by
    # immutable plan_id, so completion order cannot change exposure or identities.
    jobs=iter(sorted(groups.items()))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_initialize,initargs=(reader_kwargs,)) as executor:
        pending={}
        def submit():
            try:(source,station),plans=next(jobs)
            except StopIteration:return False
            future=executor.submit(station_worker,(source,station,plans,by_station[(source,station)],permission_sha))
            pending[future]=(source,station);return True
        for _ in range(workers):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                source,station=pending.pop(future)
                yield source,station,future.result()
                submit()


def bank(reader,plan,permission,permission_sha,out,config,workers,reader_kwargs):
    out.mkdir(parents=True,exist_ok=False);rows=plan['unique_samples'];n=len(rows);h=config['history_hours'];f=config['forecast_hours']
    x=np.empty((n,h,2),np.float32);mask=np.empty((n,h,2),bool);y=np.empty((n,f),np.float64);thresholds=np.empty((n,2),np.float64);y_raw=np.empty((n,f),np.float64)
    actual=[None]*n;by=defaultdict(list);master={r['plan_id']:i for i,r in enumerate(rows)};scans=[];started=time.monotonic()
    for r in rows:by[(r['source'],r['station'])].append(r)
    for count,(source,station,station_samples) in enumerate(station_results(reader,by,permission,permission_sha,workers,reader_kwargs),1):
        station_stats=None
        for requested,row,xx,mm,yy,tt,raw_q,stats in station_samples:
            index=master[requested['plan_id']];actual[index]=row;x[index]=xx;mask[index]=mm;y[index]=yy;thresholds[index]=tt;y_raw[index]=raw_q
            station_stats=stats
        if station_stats:scans.append(dict(source=source,station=station,**station_stats))
        save(out/'progress.json',dict(state='materializing_shared_sample_bank',stations_complete=count,stations_total=len(by),
            unique_samples=n,route=plan['route'],cpu_workers=workers,elapsed_seconds=time.monotonic()-started))
    if any(r is None for r in actual):raise ValueError('planned sample not materialized')
    manifest=dict(state='accepted',acceptance_scope='producer source/role/mask/time/hash checks; not scientific model acceptance',
        target='pointwise_Q',route=plan['route'],history_hours=h,forecast_hours=f,training_countries=config['training_countries'],
        roles={},source_qualification_receipt=dict(permission_sha256=permission_sha,permission_source='original allocated blocks',
            coverage_source='existing five-country fit/select coverage',all_actual_samples_revalidated=True,
            input_manifest_path=str((out.parent/'input_manifest.json').resolve()),input_manifest_sha256=sha256(out.parent/'input_manifest.json')),
        fit_scale_identity=dict(legacy_sha256=sha256(reader.legacy_dir/'scales.json'),prepared_manifest_sha256=sha256(reader.prepared_dir/'manifest.json')),
        raw_Q_restoration_is_new_data_version=True,roles_reallocated=False,fit_scales_recomputed=False,
        raw_supervision_preserved=True,raw_supervision_unit='m3/s',
        calibration_test_values_read=False,foreign_select_rule=plan['foreign_select_rule'],china_select_rule=plan['china_select_rule'],
        actual_fit_exposures=plan['actual_fit_exposures'],report_case_ids=[],report_case_kinds={},
        scientific_scope='DE/IS old quota selected blocks; continuous permitted US/JP; no annual flood-prevalence claim')
    fit_ids=[i for i,r in enumerate(actual) if r['role']=='fit'];select_ids=plan['select_indices']
    if any(actual[i]['role']!='select' for i in select_ids):raise ValueError('selection plan contains non-select sample')
    fit_lookup={old:new for new,old in enumerate(fit_ids)}
    for role,indices in [('fit',fit_ids),('select',select_ids)]:
        np.savez_compressed(out/(role+'.npz'),x=x[indices],mask=mask[indices],y=y[indices],y_q_m3_s=y_raw[indices],thresholds_normalized=thresholds[indices])
        save(out/(role+'_rows.json'),[actual[i] for i in indices])
        manifest['roles'][role]={role+'.npz':sha256(out/(role+'.npz')),role+'_rows.json':sha256(out/(role+'_rows.json'))}
    for station in ('Liqingdian','Luanchuan','Luxi','Zhuxipo'):
        candidates=[actual[i] for i in select_ids if actual[i]['country']=='CN' and actual[i]['station']==station]
        row=min(candidates,key=lambda r:r['origin_time']) if candidates else None
        if row is not None:
            manifest['report_case_ids'].append(row['sample_id']);manifest['report_case_kinds'][row['sample_id']]='fixed_source_station_earliest_qualified_origin'
    if not manifest['report_case_ids']:raise ValueError('no predeclared common China report cases')
    save(out/'manifest.json',manifest)
    smoke=plan['mode']=='engineering_smoke';shape=(1,2,4) if smoke else (config['epochs'],config['updates_per_epoch'],config['batch_size'])
    exposure=np.asarray([fit_lookup[i] for i in plan['fit_schedule']],dtype=np.int64).reshape(shape)
    np.savez_compressed(out/'exposure.npz',indices=exposure,bank_identity=np.asarray(sha256(out/'manifest.json')))
    verified=SampleBank(out,plan['route'],h,f)
    verified.exposure(out/'exposure.npz',*shape)
    save(out/'source_scan_stats.json',scans);save(out/'exposure_manifest.json',dict(shape=list(shape),
        exposure_sha256=sha256(out/'exposure.npz'),bank_sha256=sha256(out/'manifest.json'),
        country_counts=plan['actual_fit_exposures'],shared_by_every_model_in_route=True,
        fit_unique_samples=len(fit_ids),select_actual_origins=len(select_ids),
        select_country_counts=dict(Counter(actual[i]['country'] for i in select_ids)),
        select_country_stations={c:len({actual[i]['canonical_station'] for i in select_ids if actual[i]['country']==c}) for c in config['training_countries']}))
    save(out/'progress.json',dict(state='complete',elapsed_seconds=time.monotonic()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('--current-entry',required=True);p.add_argument('--coverage',required=True)
    p.add_argument('--permission',required=True);p.add_argument('--config',default='configs/pointwise_benchmark_producer_v1.json')
    p.add_argument('--metadata-access-receipt',required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['plan','smoke','full'],default='plan')
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if not 1<=a.workers<=4 or (a.mode=='smoke' and a.workers!=1):raise ValueError('1..4 CPU workers; engineering smoke must use one')
    config=json.loads(Path(a.config).read_text());permission_sha=sha256(a.permission)
    if (config['history_hours'],config['forecast_hours'])!=(24,24):raise ValueError('current approved data protocol is H24/F24')
    if config['training_countries']!=['CN','DE','IS','US','JP'] or not np.isclose(sum(config['country_weights'].values()),1):
        raise ValueError('five-country fixed exposure weights required')
    if permission_sha!=config['permission_sha256']:raise PermissionError('permission list identity differs from committed condition')
    access=json.loads(Path(a.metadata_access_receipt).read_text())
    if access['state']!='complete' or access['permission_sha256']!=permission_sha or access['access_method']!='lexical_top_level_projection_without_decoding_skipped_values' or access['unrequested_peak_values_decoded'] or access['raw_hourly_values_read']:
        raise PermissionError('new restricted metadata access receipt bound to this permission list is required')
    permission=json.loads(Path(a.permission).read_text());coverage=json.loads(Path(a.coverage).read_text())
    if permission['state']!='metadata_checked' or coverage['state']!='complete':raise ValueError('completed input qualifications required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);plans={}
    for route in ('pq','q_only'):
        plans[route]=make_plan(units(coverage,permission,route),route,config,smoke=a.mode=='smoke')
        save(out/(route+'_plan.json'),plans[route])
    save(out/'input_manifest.json',dict(config=config,config_sha256=sha256(a.config),permission_sha256=permission_sha,
        coverage_path=str(Path(a.coverage).resolve()),coverage_sha256=sha256(a.coverage),
        current_entry_path=str(Path(a.current_entry).resolve()),current_entry_sha256=sha256(a.current_entry),
        metadata_access_receipt_path=str(Path(a.metadata_access_receipt).resolve()),metadata_access_receipt_sha256=sha256(a.metadata_access_receipt),
        role_values_allowed=['fit','select'],mode=a.mode,cpu_workers=a.workers,old_files_modified=False,model_training=False))
    if a.mode=='plan':return
    reader_kwargs=json.loads(Path(a.current_entry).read_text())['reader_kwargs'];reader=RelocatedReader(**reader_kwargs)
    if sha256(reader.prepared_dir/'manifest.json')!='adb945fa9e7e054ab25ad74ed85c4c6eb3ede21098bfc9ed82a19986a3fbf5c7':
        raise ValueError('prepared full-source data manifest differs from the accepted version')
    for route,plan in plans.items():bank(reader,plan,permission,permission_sha,out/route,config,a.workers,reader_kwargs)
    if a.mode=='smoke':
        training=dict(state='APPROVED_SMOKE',history_hours=24,forecast_hours=24,seed=2,
            training_countries=config['training_countries'],models=config['models'],epochs=1,updates_per_epoch=2,batch_size=4,
            learning_rate=.001,gradient_clip_norm=10.,evaluation_batch_size=4,select_draws=8,final_draws=8,
            fit_probe_samples=2,cpu_threads=2,device='cuda:0',
            banks={route:str((out/route).resolve()) for route in plans},
            exposures={route:str((out/route/'exposure.npz').resolve()) for route in plans},
            scientific_state='engineering_smoke_only_no_benchmark_claim',smoke_weights_not_reused_for_formal=True)
        save(out/'smoke_training_config.json',training)
    save(out/'completion.json',dict(state='complete',mode=a.mode,routes=['pq','q_only'],old_files_modified=False,model_training=False))
    print(json.dumps(dict(state='complete',mode=a.mode)),flush=True)


if __name__=='__main__':main()
