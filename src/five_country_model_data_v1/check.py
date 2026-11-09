"""Small real-source integration check; no fit, threshold, training or test access."""
import argparse
import copy
import json
import math
import socket
import subprocess
import sys
from importlib.metadata import version
from collections import Counter
from datetime import timedelta
from pathlib import Path
from .reader import Reader, build_sample, stamp, ROUTE_PRIORITY
from .catalog import station_index


def reject(fn):
    try:
        fn()
    except (ValueError,PermissionError):
        return
    raise AssertionError('invalid input accepted')


def self_test(reader):
    rows=[reader.record('camelsh_us_v6','01011000',f'2020-01-01T{i:02d}:00:00',None,10+i,
        role='fit',label='end',quality={'test':'synthetic'},available_at=f'2020-01-01T{i:02d}:00:00') for i in range(20)]
    for h in (3,8):
        sample=build_sample(rows,'2020-01-01T10:00:00',history_hours=h,forecast_hours=3)
        assert len(sample['x'])==h and all(math.isnan(x[0]) for x in sample['x'])
        assert not any(x[0] for x in sample['mask']) and 'future_P' not in sample
        assert sample['input_times'][-1]==sample['origin']
        assert all(stamp(t)>stamp(sample['origin']) for t in sample['target_times'])
    reject(lambda:build_sample(rows[:8],rows[6]['time'],history_hours=3,forecast_hours=3))
    gap=rows[:6]+rows[7:]
    reject(lambda:build_sample(gap,rows[7]['time'],history_hours=3,forecast_hours=3))
    duplicate=rows+[rows[3]]
    reject(lambda:build_sample(duplicate,rows[10]['time'],history_hours=3,forecast_hours=3))
    for field,value in (('role','select'),('event','other'),('q_mask',False),('interval_minutes',1440)):
        changed=copy.deepcopy(rows);changed[9][field]=value
        reject(lambda:build_sample(changed,rows[10]['time'],history_hours=3,forecast_hours=3))
    reject(lambda:list(reader.read('da_tl_cn','Jinping',role='test')))
    reject(lambda:list(reader.read('camelsh_us_v6','01011000',role='evaluation_reserved')))
    reject(lambda:list(reader.read('camelsh_us_v6','01011000',role='unassigned')))
    forbidden=copy.deepcopy(rows)
    for r in forbidden:r['role']='test'
    reject(lambda:build_sample(forbidden,rows[10]['time'],history_hours=3,forecast_hours=3))
    boundary=copy.deepcopy(rows)
    for r in boundary:r['role_bounds']=[None,'2020-01-01T12:00:00']
    reject(lambda:build_sample(boundary,rows[10]['time'],history_hours=3,forecast_hours=3))
    unknown=copy.deepcopy(rows);unknown[9]['available_at']=None
    reject(lambda:build_sample(unknown,rows[10]['time'],history_hours=3,forecast_hours=3,mode='operational'))
    changed=copy.deepcopy(rows)
    for r in changed:r['timezone']='UTC'
    build_sample(changed,rows[10]['time'],history_hours=3,forecast_hours=3,mode='operational')
    changed[9]['available_at']='2020-01-02T00:00:00'
    reject(lambda:build_sample(changed,rows[10]['time'],history_hours=3,forecast_hours=3,mode='operational'))
    class Fixture:
        def read(self,*args,**kwargs):
            return iter(copy.deepcopy(rows))
    fixture=Fixture()
    q_samples=list(Reader.iter_samples(fixture,'camelsh_us_v6','01011000',role='fit',
        route='q_only',history_hours=4,max_samples=2,technical_only=True))
    assert len(q_samples)==2 and all(not m[0] for s in q_samples for m in s['mask'])
    assert set(q_samples[0]['window_max_targets'])=={'1','3','6','12'}
    assert not list(Reader.iter_samples(fixture,'camelsh_us_v6','01011000',role='fit',
        route='pq',history_hours=4,max_samples=2,technical_only=True))
    # Crosswalk identity is checked against the accepted mapping, not a guessed USGS ID.
    assert reader.canonical('waterbench_iowa_v2_7087806','1609').startswith('USGS:')
    assert reader.canonical('japan_mlit_tone_existing','303031283301030:0202')==reader.canonical('jp_jma_maebashi_mlit_kurihashi_2015_2024','303031283301030')
    return dict(state='passed',checks=['variable_history','missing_P_NaN_mask','current_included',
        'future_Q_supervision_only','gap_boundary','duplicate_identity','role_event_boundary',
        'Q_quality','auxiliary_interval','test_forbidden','reserved_roles_forbidden','availability',
        'USGS_and_Kurihashi_canonical_identity'])


def find_sample(records):
    # First qualifying origin in source order, not a score-selected favourable event.
    for r in records:
        try:
            a=build_sample(records,r['interval_end'],history_hours=4,forecast_hours=12)
            b=build_sample(records,r['interval_end'],history_hours=2,forecast_hours=12)
            assert set(a['window_max_targets'])=={'1','3','6','12'}
            return a,b
        except ValueError:
            continue
    return None


def run(manifest, output):
    reader=Reader(manifest)
    checks=self_test(reader)
    index=station_index(reader)
    output.mkdir(parents=True,exist_ok=False)
    (output/'station_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
    results=[]
    legacy={key:next(r for r in index if r['source']==key and 'fit' in r['roles']) for key in ('da_tl_cn','camels_de_1h','lamah_ice')}
    choices={key:dict(station=r['station'],role='fit',event=r['roles']['fit']['events'][0],limit=256) for key,r in legacy.items()}
    choices.update(camelsh_us_v6=dict(station='05464500',start='2012-01-01T00:00:00',limit=256),
        waterbench_iowa_v2_7087806=dict(station='1609',limit=256),
        japan_mlit_tone_existing=dict(station='303031283301030:0202',limit=256),
        jp_jma_maebashi_mlit_kurihashi_2015_2024=dict(station='303031283301030',limit=256),
        hei_luoma_daily=dict(station='*',limit=24), geodata_poyang_samples=dict(station='*',limit=24),
        gxsw_recent_capture=dict(station='*',limit=24))
    for key in choices:
        if reader.sources[key]['country_code'] in ('US','JP'):
            choices[key]['role']='fit'
    for key,args in choices.items():
        records=list(reader.read(key,**args))
        sample=None if key in ('hei_luoma_daily','geodata_poyang_samples','gxsw_recent_capture') else find_sample(records)
        if not records:
            raise ValueError('no real rows: '+key)
        if key not in ('hei_luoma_daily','geodata_poyang_samples','gxsw_recent_capture') and not sample:
            # Event boundary and missing prefix may make first event unsuitable; no test fallback.
            raise ValueError('no qualifying non-test sample: '+key)
        if sample:
            reject(lambda:build_sample(records,sample[0]['origin'],history_hours=4,forecast_hours=12,mode='operational'))
            route='q_only' if key in ('camelsh_us_v6','japan_mlit_tone_existing') else 'pq'
            actual_iterator=list(reader.iter_samples(key,args['station'],role='fit',route=route,
                history_hours=4,start=args.get('start'),event=args.get('event'),max_samples=1,record_limit=256,technical_only=True))
            if not actual_iterator:
                raise ValueError('actual model iterator produced no sample: '+key)
        first=records[0]
        results.append(dict(source=key,rows_read=len(records),args=args,
            canonical_station=first['canonical_station'],role=first['role'],
            q_qualified=sum(r['q_mask'] for r in records),p_qualified=sum(r['p_mask'] for r in records),
            model_sample='technical_hindcast_only' if sample else 'auxiliary_no_hourly_model_sample',
            samples=sample,first_record=first,
            reason='actual available_at unknown; no operational claim' if sample else reader.sources[key]['recommended_route']))
        if sample:
            results[-1]['model_iterator_example']=actual_iterator[0]
        if key not in ('hei_luoma_daily','geodata_poyang_samples','gxsw_recent_capture'):
            selected=list(reader.read(key,args['station'],role='select',limit=256))
            # Legacy source-event selectors differ by role, so choose first preserved event.
            event=selected[0]['event'] if selected else None
            selected=[r for r in selected if r['event']==event]
            sel_sample=find_sample(selected)
            if not sel_sample:
                raise ValueError('no qualifying select sample: '+key)
            results[-1]['select_example']={'rows_read':len(selected),'samples':sel_sample}
    clean=lambda x: {k:clean(v) for k,v in x.items()} if isinstance(x,dict) else [clean(v) for v in x] if isinstance(x,(list,tuple)) else None if isinstance(x,float) and not math.isfinite(x) else x
    receipt=dict(state='complete',host=socket.gethostname(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=sys.executable,dependency_versions={k:version(k) for k in ('numpy','openpyxl','h5py')},checks=checks,
        sources=results,index_entries=len(index),source_entries=dict(Counter(r['source'] for r in index)),
        role_locator_station_counts={role:dict(Counter(r['source'] for r in index if role in r['roles'])) for role in ('fit','select','calibration','development')},
        route_priority=ROUTE_PRIORITY,read_paths=sorted(set(reader.access)),test_read=False,
        training_started=False,thresholds_fitted=False,new_roles_assigned=True,
        limits=['No raw legacy quality recovery; saved float32 reconstruction',
                'US/JP user-approved source-clock roles; fit/select access only, no evaluation read',
                'NaN serialized as null in JSON; memory samples use NaN plus mask',
                'Locator index is metadata coverage, not all-station observation validation'])
    (output/'receipt.json').write_text(json.dumps(clean(receipt),ensure_ascii=False,indent=2,allow_nan=False))
    assert len(results)==10
    assert {reader.sources[r['source']]['country_code'] for r in results if r['samples']}=={'CN','DE','IS','US','JP'}
    print(json.dumps({'state':'complete','sources':len(results),'index_entries':len(index),'samples':sum(bool(r['samples']) for r in results)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/existing_data_integration_v2.json');p.add_argument('--output',required=True);a=p.parse_args()
    run(a.manifest,Path(a.output))
