"""Fit-only scales and censored Q95 episodes; vectorized, one source station at a time."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import socket
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from .reader import Reader, csv_rows, number, stamp, truth, OLD


def hour(value):
    return int(np.datetime64(stamp(value).replace(tzinfo=None),'h').astype('int64'))


def iso(value, zone=''):
    return str(np.datetime64(int(value),'h')).replace('T','T')+':00:00'+zone


def clean(values, missing=()):
    values=np.asarray(values,dtype=float).copy()
    for sentinel in missing:
        if np.isfinite(sentinel):
            values[values==sentinel]=np.nan
    values[~np.isfinite(values)|(values<0)]=np.nan
    return values


def h5_missing(dataset):
    missing=[]
    for key in ('_FillValue','missing_value'):
        missing.extend(np.asarray(dataset.attrs.get(key,[])).reshape(-1).tolist())
    # HDF5's explicit creation fill is present even when no netCDF attribute exists.
    if dataset.id.get_create_plist().fill_value_defined()==2:
        missing.append(float(dataset.fillvalue))
    return missing


def scales(q,p=None):
    q=clean(q);valid=q[np.isfinite(q)]
    if not len(valid):
        return dict(state='NOT_EVALUABLE',reason='no_qualified_fit_Q',q_count=0,q95=None,q99=None,q_scale=None)
    q95,q99=np.quantile(valid,[.95,.99],method='linear')
    result=dict(state='qualified' if q95>0 else 'degenerate_scale',q_count=len(valid),
        q95=float(q95),q99=float(q99),q_scale=float(q95) if q95>0 else 1.,
        q_unit='m3/s',extreme_eligible=bool(q95>0),method='fit_only_numpy_linear')
    if p is not None:
        p=clean(p);positive=p[np.isfinite(p)&(p>0)]
        result.update(p_count=int(np.isfinite(p).sum()),p_positive_count=len(positive),
            p_scale=float(np.quantile(positive,.95,method='linear')) if len(positive) else 1.,
            p_scale_state='fit_positive_P_Q95' if len(positive) else 'no_positive_fit_P_uses_1',p_unit='mm/hour')
    return result


def episodes(t,q,q95,identity):
    """A gap interrupts episodes. 24 valid subthreshold hours establish each edge."""
    t=np.asarray(t,dtype=np.int64);q=clean(q)
    if len(t)!=len(q) or (len(t)>1 and np.any(np.diff(t)<=0)):
        raise ValueError('time duplicate/out_of_order or array length mismatch')
    valid=np.isfinite(q)
    prev=np.r_[False,valid[:-1]&(np.diff(t)==1)] if len(t) else np.array([],bool)
    after=np.r_[valid[1:]&(np.diff(t)==1),False] if len(t) else np.array([],bool)
    starts=np.flatnonzero(valid&~prev);ends=np.flatnonzero(valid&~after)
    segments=[];process=[]
    for a,b in zip(starts,ends):
        a,b=int(a),int(b)
        segments.append(dict(start_hour=int(t[a]),end_hour=int(t[b]),hours=b-a+1,
            left_boundary='role_or_source_edge' if a==0 else 'missing_Q_or_clock_gap',
            right_boundary='role_or_source_edge' if b==len(q)-1 else 'missing_Q_or_clock_gap'))
        if q95 is None or q95<=0:
            continue
        exceed=np.flatnonzero(q[a:b+1]>q95)+a
        if not len(exceed):
            continue
        groups=np.split(exceed,np.flatnonzero(np.diff(exceed)>=25)+1)
        for indices in groups:
            first,last=int(indices[0]),int(indices[-1]);peak=first+int(np.argmax(q[first:last+1]))
            left=first-a>=24;right=b-last>=24
            gap=(not left and a>0) or (not right and b<len(q)-1)
            pid=hashlib.sha256(json.dumps([identity,int(t[first]),int(t[last])]).encode()).hexdigest()
            process.append(dict(process_id=pid,first_exceed_hour=int(t[first]),last_exceed_hour=int(t[last]),
                peak_hour=int(t[peak]),peak_q=float(q[peak]),above_hours=len(indices),
                left_censored=not left,right_censored=not right,gap_censored=bool(gap),
                complete=bool(left and right),formal_process_eligible=bool(left and right),
                q95=float(q95),identity=identity,interpretation='Q95_exceedance_process_not_independent_storm'))
    invalid=~valid
    badstarts=np.flatnonzero(invalid&~np.r_[False,invalid[:-1]]) if len(q) else []
    badends=np.flatnonzero(invalid&~np.r_[invalid[1:],False]) if len(q) else []
    return dict(valid_segments=segments,processes=process,valid_Q_count=int(valid.sum()),
        invalid_Q_count=int(invalid.sum()),invalid_Q_runs=[dict(start_hour=int(t[a]),end_hour=int(t[b]),rows=int(b-a+1)) for a,b in zip(badstarts,badends)],
        clock_gaps=[dict(before_hour=int(t[i]),after_hour=int(t[i+1])) for i in np.flatnonzero(np.diff(t)!=1)],
        source_time_start_hour=int(t[0]) if len(t) else None,source_time_end_hour=int(t[-1]) if len(t) else None,
        qualified_Q_start_hour=int(t[np.flatnonzero(valid)[0]]) if np.any(valid) else None,
        qualified_Q_end_hour=int(t[np.flatnonzero(valid)[-1]]) if np.any(valid) else None)


def input_identity(reader,row):
    key,station=row['source'],row['station'];p=reader.entry(key)
    if key=='waterbench_iowa_v2_7087806':p=p/(station+'.csv.gz')
    stat=p.stat()
    result=dict(path=str(p),size=stat.st_size,mtime_ns=stat.st_mtime_ns,source_version=reader.sources[key]['source_version'])
    if key=='camelsh_us_v6':
        with zipfile.ZipFile(p) as z:
            info=z.getinfo(row['locator']['member'])
            result.update(member=info.filename,member_crc=info.CRC,member_bytes=info.file_size)
    return result


def attach_nesting(reader,rows):
    path=reader.entry('waterbench_iowa_v2_7087806').parent/'nesting.json'
    raw=path.read_bytes();relations=json.loads(raw);parent={str(k):str(k) for k in relations['ancestors']}
    def root(node):
        while parent[node]!=node:
            parent[node]=parent[parent[node]];node=parent[node]
        return node
    for a,b in relations['direct_edges']:
        parent[root(str(a))]=root(str(b))
    groups={}
    for node in parent:groups.setdefault(root(node),[]).append(node)
    canonical_groups={}
    for nodes in groups.values():
        canonical=sorted(reader.canonical('waterbench_iowa_v2_7087806',n) for n in nodes)
        group='WB-nesting:'+hashlib.sha256(json.dumps(canonical).encode()).hexdigest()[:16]
        for station in canonical:canonical_groups[station]=group
    reference={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
        'definition':'connected_components_of_existing_98_edges_not_independent_storms','components':len(groups)}
    for row in rows:
        row['nesting_group']=canonical_groups.get(row['canonical_station'],'unknown')
        row['nesting_reference']=reference if row['canonical_station'] in canonical_groups else None
    return reference


def apply_route_policy(reader,rows):
    # Latest accepted Kurihashi pair includes Q repairs absent in historical MLIT cache.
    for row in rows:
        if row['canonical_station']=='MLIT:303031283301030':
            primary=reader.policy['identity_policy']['MLIT:303031283301030']['primary_Q_source_both_routes']
            row['q_only_pool_eligible']=row['source']==primary
    return rows


def read_arrays(reader,row):
    """No dictionary-per-hour CAMELSH expansion. Only fit/select values are selected."""
    key,station=row['source'],row['station'];p=reader.entry(key)
    periods=reader.role_bounds[key]
    end=hour(periods['select'][1]);begin=hour(periods['select'][0]);zone='+09:00' if reader.sources[key]['country_code']=='JP' else ''
    fill=[]
    if key=='camelsh_us_v6':
        import h5py
        with zipfile.ZipFile(p) as z,h5py.File(io.BytesIO(z.read(row['locator']['member']))) as f:
            units=f['time'].attrs['units'].decode()
            if not units.startswith('hours since ') or f['streamflow'].attrs['units'].decode()!='m3 s-1':
                raise ValueError('unexpected netCDF time/Q units')
            base=hour(units.removeprefix('hours since '))
            # Time-coordinate metadata is not an observation; values are sliced before reserved years.
            native_time=np.asarray(f['time'][:],dtype=float)
            if np.any(~np.isfinite(native_time)) or np.any(native_time!=np.floor(native_time)):
                raise ValueError('invalid time coordinates')
            t=base+native_time.astype(np.int64)
            stop=int(np.searchsorted(t,end))
            if np.any(np.diff(t)<=0):raise ValueError('duplicate/out-of-order time')
            fill=h5_missing(f['streamflow']);q=clean(f['streamflow'][:stop],fill);t=t[:stop];rain=None
    else:
        ts=[];qs=[];ps=[]
        if key=='japan_mlit_tone_existing':
            with sqlite3.connect(f'file:{p}?mode=ro',uri=True) as c:
                rows=c.execute('SELECT t,q,conflict FROM obs WHERE id=? AND t<? ORDER BY t',(station,iso(end,zone)))
                for t,q,conflict in rows:
                    ts.append(hour(t));qs.append(number(q) if not conflict else math.nan)
            rain=None
        else:
            if key=='waterbench_iowa_v2_7087806':p=p/(station+'.csv.gz')
            for r in csv_rows(p):
                t=hour(r['datetime_native'] if key=='waterbench_iowa_v2_7087806' else r['time_jst'])
                if t>=end:raise ValueError('reserved timestamp reached before permitted endpoint; Q/P not parsed')
                if key=='waterbench_iowa_v2_7087806':
                    q=number(r['discharge_m3_per_s']) if truth(r['q_observed_available']) else math.nan
                    pv=number(r['precipitation_mm_per_hour']) if truth(r['p_observed_available']) else math.nan
                else:
                    q=number(r['Q_m3_s']) if not truth(r['Q_conflict']) else math.nan
                    pv=number(r['P_mm']) if r['P_quality']=='8' and truth(r['P_usable']) else math.nan
                ts.append(t);qs.append(q);ps.append(pv)
                # Uniform prepared CSV keeps missing observations as rows; stop before reserved row.
                if t==end-1:break
            rain=clean(ps)
        t=np.asarray(ts,dtype=np.int64);q=clean(qs)
    if len(t)>1 and np.any(np.diff(t)<=0):raise ValueError('duplicate/out-of-order time')
    return {'fit':(t[t<begin],q[t<begin],rain[t<begin] if rain is not None else None),
            'select':(t[(t>=begin)&(t<end)],q[(t>=begin)&(t<end)],rain[(t>=begin)&(t<end)] if rain is not None else None)},fill,zone


def station_prepare(reader,row,identity,code,policy_sha):
    started=time.monotonic();arrays,fill,zone=read_arrays(reader,row)
    fit_t,fit_q,fit_p=arrays['fit'];scale=scales(fit_q,fit_p)
    roles={}
    for role,(t,q,p) in arrays.items():
        role_identity=dict(source=row['source'],canonical_station=row['canonical_station'],station=row['station'],role=role)
        roles[role]=episodes(t,q,scale['q95'],role_identity)
        roles[role]['P_valid_count']=int(np.isfinite(p).sum()) if p is not None else None
    return dict(schema='five_country_prepared_v1',state='complete' if scale['state']!='NOT_EVALUABLE' else 'NOT_EVALUABLE',
        source=row['source'],station=row['station'],canonical_station=row['canonical_station'],input_identity=identity,
        code=code,role_policy_sha256=policy_sha,scale=scale,roles=roles,timezone=reader.sources[row['source']]['time_zone'],
        source_clock_suffix=zone,declared_missing_values=[float(v) for v in fill if np.isfinite(v)],
        time_coordinate_units='integer source-native-clock hours since 1970-01-01 00:00:00; append source_clock_suffix for JST',
        quality='finite_nonnegative_source_qualification_not_gauge_certification',
        p_spatial_support='catchment_mean' if row['source']=='waterbench_iowa_v2_7087806' else 'point:Maebashi:s47624' if fit_p is not None else 'not_available',
        nesting_group=row.get('nesting_group','unknown'),nesting_reference=row.get('nesting_reference'),
        q_only_pool_eligible=row['q_only_pool_eligible'],pq_pool_eligible=row['pq_pool_eligible'],
        elapsed_seconds=time.monotonic()-started,fit_select_rows=len(fit_t)+len(arrays['select'][0]),
        interpretation='threshold_processes_not_independent_storms; not_operational_availability_validation')


def station_name(source,station):
    return hashlib.sha256((source+':'+station).encode()).hexdigest()+'.json.gz'


def qualify_sample(sample,table,route):
    role=sample['identity'][3];info=table['roles'][role];scale=table['scale']
    first=hour(sample['input_times'][0]);last=hour(sample['target_times'][-1])
    if not table['pq_pool_eligible' if route=='pq' else 'q_only_pool_eligible']:
        return 'duplicate_or_unsupported_source_for_route'
    if not any(s['start_hour']<=first and last<=s['end_hour'] for s in info['valid_segments']):
        return 'outside_prepared_valid_segment'
    touched=[p for p in info['processes'] if p['first_exceed_hour']<=last and p['last_exceed_hour']>=first]
    if any(not p['complete'] for p in touched):
        return 'window_intersects_censored_Q95_process'
    if route=='pq' and 'p_scale' not in scale:
        return 'prepared_route_has_no_P_scale'
    sample['x_normalized']=[[x[0]/scale['p_scale'] if route=='pq' else math.nan,x[1]/scale['q_scale']] for x in sample['x']]
    sample['window_max_targets_normalized']={h:v/scale['q_scale'] for h,v in sample['window_max_targets'].items()}
    sample['scale_source']='prepared_fit_only_numpy_linear'
    sample['boundary_event_integrity']='Q95_exceedance_process_edges_checked_not_independent_storm_validation'
    # Future-dependent episode identities are offline metadata and never enter x.
    sample['offline_process_metadata']={'process_ids':[p['process_id'] for p in touched],
        'extreme_eligible':scale['extreme_eligible'],'ordinary_window':not bool(touched),
        'nesting_group':table.get('nesting_group','unknown')}
    return None


def self_test():
    t=np.arange(120);q=np.zeros(120);q[25]=2;q[49]=3
    a=episodes(t,q,1,{'test':1});assert len(a['processes'])==1 and a['processes'][0]['complete']
    q[49]=0;q[50]=3;a=episodes(t,q,1,{'test':1});assert len(a['processes'])==2
    q=np.zeros(120);q[1]=2;q[118]=3;a=episodes(t,q,1,{'test':1})
    assert a['processes'][0]['left_censored'] and a['processes'][-1]['right_censored']
    q=np.zeros(120);q[25]=2;q[30]=np.nan;q[35]=3;a=episodes(t,q,1,{'test':1})
    assert len(a['processes'])==2 and all(p['gap_censored'] for p in a['processes'])
    fit=np.arange(100,dtype=float);before=scales(fit);select=np.array([1e9]);assert scales(fit)==before and select.max()>before['q99']
    assert scales([0,0])['q_scale']==1 and not scales([0,0])['extreme_eligible']
    assert scales([np.nan])['state']=='NOT_EVALUABLE' and 'p_scale' not in scales([1,2])
    assert scales([1,2],[0,np.nan])['p_scale']==1
    assert np.isnan(clean([9.96921e36],[9.96921e36])[0])
    try:episodes([1,1],[1,2],1,{})
    except ValueError:pass
    else:raise AssertionError('duplicate time accepted')
    from types import SimpleNamespace
    from unittest.mock import patch
    key='camelsh_us_v6';row={'source':key,'station':'00000001','canonical_station':'USGS:00000001',
        'q_only_pool_eligible':True,'pq_pool_eligible':False}
    fake=SimpleNamespace(sources={key:{'time_zone':'unknown'}})
    base=hour('2014-01-01T00:00:00');fit_t=base+np.arange(100)
    def calculate(select_q):
        arrays={'fit':(fit_t,np.arange(100,dtype=float),None),
            'select':(hour('2015-10-01T00:00:00')+np.arange(len(select_q)),np.array(select_q),None)}
        with patch.object(sys.modules[__name__],'read_arrays',return_value=(arrays,[],'')):
            return station_prepare(fake,row,{},'fixture','fixture')
    first=calculate([1,2]);second=calculate([1e9,1e10])
    assert first['scale']==second['scale'] and first['scale']['q95']==94.05
    assert second['roles']['select']['processes'][0]['left_censored']
    sample={'identity':[key,'USGS:00000001','continuous','select'],'input_times':['2015-10-01T00:00:00'],
        'target_times':['2015-10-01T01:00:00'],'x':[[math.nan,1]],'window_max_targets':{'1':2}}
    assert qualify_sample(sample,second,'q_only')=='window_intersects_censored_Q95_process'
    assert 'x_normalized' not in sample
    return dict(state='passed',checks=['23_vs_24_subthreshold_hours','left_right_edges','gap_censoring','fit_only_scales',
        'select_extreme_does_not_change_fit_scale','year_role_edge_censoring','censored_window_excluded',
        'no_P','zero_Q','FillValue','duplicate_clock'])


def run(args):
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    reader=Reader(args.manifest);policy=Path(args.manifest).with_name('five_country_model_data_roles_v1.json')
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    rows=json.loads(Path(args.index).read_text());rows=apply_route_policy(reader,[r for r in rows if r['source'] in reader.role_bounds])
    nesting_reference=attach_nesting(reader,rows)
    if args.smoke:
        path=reader.entry('japan_mlit_tone_existing')
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as c:
            other=c.execute("SELECT id FROM obs WHERE t<'2022-01-01T00:00:00+09:00' AND q>0 AND conflict=0 AND id!='303031283301030:0202' GROUP BY id HAVING min(t)<'2021-01-01T00:00:00+09:00' AND max(t)>='2021-01-01T00:00:00+09:00' LIMIT 1").fetchone()
        if other is None:raise ValueError('no non-Kurihashi fit/select Q station for smoke')
        rows=[r for r in rows if (r['source'],r['station']) in {('camelsh_us_v6','05464500'),('waterbench_iowa_v2_7087806','1609'),
            ('japan_mlit_tone_existing','303031283301030:0202'),('japan_mlit_tone_existing',other[0]),
            ('jp_jma_maebashi_mlit_kurihashi_2015_2024','303031283301030')}]
        assert len(rows)==5
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);(output/'stations').mkdir(exist_ok=True)
    started=time.monotonic();checks=self_test();summary=[];counts={};total_rows=0
    for i,row in enumerate(rows):
        key=row['source'];identity=input_identity(reader,row);file=output/'stations'/station_name(key,row['station'])
        if file.exists():
            with gzip.open(file,'rt') as f:result=json.load(f)
            if result.get('input_identity')!=identity or result.get('code')!=code or result.get('role_policy_sha256')!=policy_sha:
                raise ValueError('resume identity differs: '+str(file))
        else:
            try:result=station_prepare(reader,row,identity,code,policy_sha)
            except Exception as e:
                result=dict(state='failed',source=key,station=row['station'],canonical_station=row['canonical_station'],
                    input_identity=identity,code=code,role_policy_sha256=policy_sha,error_type=type(e).__name__,error=str(e))
            with gzip.open(file,'xt',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
        counts[result['state']]=counts.get(result['state'],0)+1
        total_rows+=result.get('fit_select_rows',0)
        summary.append(dict(source=key,station=row['station'],canonical_station=row['canonical_station'],state=result['state'],
            prepared_file=str(file),prepared_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            q_only_pool_eligible=row['q_only_pool_eligible'],pq_pool_eligible=row['pq_pool_eligible']))
        progress=dict(state='running',stations_done=i+1,stations_total=len(rows),counts=counts,source=key,last_station=row['station'],
            elapsed_seconds=time.monotonic()-started,fit_select_rows=total_rows,code=code,units='stations and source hourly rows; larger counts indicate preparation progress, not model quality')
        (output/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2))
        if (i+1)%100==0 or args.smoke:print(json.dumps(progress),flush=True)
    manifest=dict(schema='five_country_prepared_v1',state='partial' if counts.get('failed') else 'complete',smoke=args.smoke,
        code=code,host=socket.gethostname(),python=sys.executable,role_policy_sha256=policy_sha,source_index=args.index,
        stations=summary,counts=counts,fit_select_rows=total_rows,elapsed_seconds=time.monotonic()-started,self_checks=checks,
        legacy=dict(directory=str(reader.legacy_dir),policy='reuse_existing_fit_select_arrays_scales_source_events_no_new_fit'),
        reserved_values_read=False,training_started=False,threshold_processes_are_independent_storms=False,
        operational_available_at='unknown',pool_composition='mutually_exclusive_PQ_or_Q_only_no_direct_concatenation')
    manifest['nesting_reference']=nesting_reference
    with (output/'manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    if args.smoke and not counts.get('failed'):
        prepared_reader=Reader(args.manifest,prepared_dir=output)
        model_checks=[]
        for row in rows:
            if not row['pq_pool_eligible'] and not row['q_only_pool_eligible']:
                continue
            key=row['source'];route='pq' if row['pq_pool_eligible'] else 'q_only'
            for role in ('fit','select'):
                start='2012-01-01T00:00:00' if role=='fit' and key=='camelsh_us_v6' else None
                if key=='japan_mlit_tone_existing':
                    table=prepared_reader.prepared_station(key,row['station'])
                    segment=next(s for s in table['roles'][role]['valid_segments'] if s['hours']>=32)
                    start=iso(segment['start_hour'],table['source_clock_suffix'])
                samples=list(prepared_reader.iter_samples(key,row['station'],role=role,route=route,history_hours=4,
                    start=start,max_samples=2,record_limit=4096))
                if not samples:raise ValueError('prepared iterator has no qualified real sample: '+key+':'+role)
                assert all(set(s['window_max_targets'])=={'1','3','6','12'} for s in samples)
                assert all('offline_process_metadata' in s and len(s['x_normalized'])==4 for s in samples)
                model_checks.append(dict(source=key,station=row['station'],role=role,route=route,
                    samples=len(samples),sample_ids=[s['sample_id'] for s in samples],
                    raw_q=samples[0]['x'][-1][1],normalized_q=samples[0]['x_normalized'][-1][1],
                    offline_metadata=samples[0]['offline_process_metadata']))
                if key=='jp_jma_maebashi_mlit_kurihashi_2015_2024':
                    q_only=list(prepared_reader.iter_samples(key,row['station'],role=role,route='q_only',history_hours=4,
                        start=samples[0]['input_times'][0],max_samples=2,record_limit=512))
                    assert q_only and q_only[0]['y_q']==samples[0]['y_q']
                    assert q_only[0]['window_max_targets_normalized']==samples[0]['window_max_targets_normalized']
                    assert all(math.isnan(x[0]) for x in q_only[0]['x_normalized']) and all(not m[0] for m in q_only[0]['mask'])
                    model_checks.append(dict(source=key,station=row['station'],role=role,route='q_only',samples=len(q_only),
                        raw_q=q_only[0]['x'][-1][1],normalized_q=q_only[0]['x_normalized'][-1][1],
                        pair_route_same_Q_and_scale=True,P_hidden=True))
        with (output/'model_iterator_checks.json').open('x') as f:json.dump(model_checks,f,ensure_ascii=False,indent=2)
    progress.update(state=manifest['state']);(output/'progress.json').write_text(json.dumps(progress,indent=2))
    if counts.get('failed'):raise RuntimeError('station failures preserved; preparation partial')
    print(json.dumps({k:manifest[k] for k in ('state','counts','fit_select_rows','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/existing_data_integration_v2.json')
    p.add_argument('--index',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
