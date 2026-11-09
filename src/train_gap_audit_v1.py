"""Fixed-fit rain/flow coverage and descriptive donor feasibility; no model or role expansion."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda:fp.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix('.new');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)


def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in row.items()})


def numeric(v):
    try:x=float(v)
    except (TypeError,ValueError):return np.nan
    return x if np.isfinite(x) and x>=0 else np.nan


def longest_missing(values):
    best=run=0
    for v in values:
        run=run+1 if not np.isfinite(v) else 0;best=max(best,run)
    return best


def load_fit(cfg,out):
    root=Path(cfg['prepared']);manifest=json.loads((root/'manifest.json').read_text())
    if digest(root/'manifest.json')!=cfg['prepared_manifest_sha256']:raise ValueError('prepared manifest differs')
    if digest(root/'fit.npz')!=manifest['derived_files']['fit.npz']:raise ValueError('fit array hash differs')
    rows=json.loads((root/'fit_rows.json').read_text());scales=json.loads((root/'scales.json').read_text())
    if len(rows)!=cfg['expected_fit_windows'] or any(r['role']!='fit' for r in rows):raise ValueError('fit role/membership differs')
    if len({r['window_id'] for r in rows})!=len(rows):raise ValueError('duplicate fit identity')
    target=[r for r in rows if r['station_uid']==cfg['target']]
    if len(target)!=cfg['expected_target_windows'] or sorted({r['source_event_id'] for r in target})!=cfg['expected_target_source_events']:raise ValueError('target fit scope differs')
    with np.load(root/'fit.npz',allow_pickle=False) as z:x=z['x'].copy();y=z['y'].copy()
    if x.shape!=(len(rows),12,2) or y.shape!=(len(rows),12) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('fit arrays invalid')
    attrs={r['station_uid']:{k:r.get(k) for k in ('area_km2','mean_annual_precip_mm_equiv','original_precip_value','original_precip_unit','precipitation_reference_period','source')} for r in json.loads(Path(cfg['attribute_rows']).read_text())}
    inputs=[dict(path=str(root/name),sha256=digest(root/name),role='fit metadata/arrays only') for name in ('manifest.json','fit.npz','fit_rows.json','scales.json')]
    inputs.append(dict(path=cfg['attribute_rows'],sha256=digest(cfg['attribute_rows']),role='retrospective static attributes; validation counts/values not used'))
    return rows,x,y,scales,attrs,manifest,inputs


def extract_fit_future(rows,x,y,scales,manifest,out):
    by_station=defaultdict(list)
    for i,r in enumerate(rows):by_station[r['station_uid']].append(i)
    inputs={r['station']:r for r in manifest['inputs'] if r['old_role']=='train'}
    future=np.full((len(rows),12),np.nan);tags={};access=[];failures=[]
    pattern=re.compile(r'"window_id"\s*:\s*"([^\"]+)"')
    for j,(station,ix) in enumerate(sorted(by_station.items())):
        item=inputs.get(station)
        if not item:
            failures.append(dict(station=station,state='partial',reason='no upstream train source metadata'));continue
        path=Path(item['path']);sha=digest(path)
        if sha!=item['sha256']:
            failures.append(dict(station=station,state='partial',reason='upstream hash changed; related extraction stopped'));continue
        allowed={rows[i]['window_id']:i for i in ix};seen=set();temporary={};local_tags={};failure=None
        with gzip.open(path,'rt') as fp:
            for line in fp:
                identity=pattern.search(line)
                # Reject mixed new select/cal identities before parsing any numerical field.
                if not identity or identity.group(1) not in allowed:continue
                record=json.loads(line);i=allowed[identity.group(1)];r=rows[i]
                if record['window_id'] in seen:failure='duplicate upstream fit window';break
                seen.add(record['window_id'])
                if record['station_uid']!=station or record['source_rows']!=r['source_rows'] or record['source_times']!=r['source_times'] or record['role']!='train':failure='source/role identity mismatch';break
                a=np.array(record['x_p'])*item['original_pscale'];q=np.array(record['x_q'])*item['original_qscale'];fq=np.array(record['y_q'])*item['original_qscale']
                if not (np.allclose(a,x[i,:,0]*scales[station]['pscale'],rtol=3e-6,atol=1e-5) and np.allclose(q,x[i,:,1]*r['qscale'],rtol=3e-6,atol=1e-4) and np.allclose(fq,y[i]*r['qscale'],rtol=3e-6,atol=1e-4)):failure='physical historical P/Q or label Q reconstruction mismatch';break
                if 'offline_generation_future_p' in record:
                    p=np.asarray(record['offline_generation_future_p'],float)*item['original_pscale']
                    if p.shape!=(12,) or not np.isfinite(p).all() or (p<0).any():failure='invalid offline fit future P';break
                    temporary[i]=p
                local_tags[i]=record.get('selection_tags',[])
        if failure:
            failures.append(dict(station=station,state='partial',reason=failure+'; station future P extraction stopped'))
        else:
            if len(seen)!=len(allowed):failures.append(dict(station=station,state='partial',reason='missing upstream allowed fit IDs',missing_fit_ids=len(allowed)-len(seen)))
            for i,p in temporary.items():future[i]=p
            tags.update(local_tags)
        access.append(dict(path=str(path),sha256=sha,role='original train filtered by current fit window_id BEFORE json.loads',fit_ids_requested=len(allowed),fit_ids_found=len(seen),future_P_rows_used=0 if failure else len(temporary),source_values_of_other_roles_parsed=False,source_rows_sha256=hashlib.sha256(json.dumps({rows[i]['window_id']:rows[i]['source_rows'] for i in ix},sort_keys=True).encode()).hexdigest(),state='partial' if failure or len(seen)!=len(allowed) else 'complete'))
        if j%50==0:write_json(out/'progress.json',dict(state='running',phase='fit allowlist future-P extraction',stations_completed=j+1,stations_total=len(by_station)))
    return future,tags,access,failures


def fragments(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['station_uid']].append(i)
    mapping={};records=[]
    for station,ix in sorted(groups.items()):
        ordered=sorted(ix,key=lambda i:rows[i]['source_times'][0]);parts=[];part=[];end=None
        for i in ordered:
            start=datetime.fromisoformat(rows[i]['source_times'][0]);finish=datetime.fromisoformat(rows[i]['source_times'][-1])
            if end is not None and start>end+timedelta(hours=1):parts.append(part);part=[]
            part.append(i);end=finish if not part[:-1] else max(end,finish)
        if part:parts.append(part)
        for j,ids in enumerate(parts):
            key=station+':fit_process_fragment:'+str(j)
            for i in ids:mapping[i]=key
            records.append(dict(fragment_id=key,station=station,country=rows[ids[0]]['country'],n_fit_windows=len(ids),source_start=min(rows[i]['source_times'][0] for i in ids),source_end=max(rows[i]['source_times'][-1] for i in ids),unique_fit_source_rows=len({s for i in ids for s in rows[i]['source_rows']}),source_event_labels='|'.join(sorted({rows[i]['source_event_id'] or rows[i]['event'] for i in ids})),complete_physical_event='unknown',independent_storm='unknown',unit='same-station adjacent/overlapping24h fit windows; may be sparse sampled fragments'))
    return mapping,records


def features(rows,x,y,scales,attrs,future,tags,cfg,fragment):
    target_attr=attrs[cfg['target']];result=[]
    for i,r in enumerate(rows):
        station=r['station_uid'];a=attrs.get(station,{})
        area=numeric(a.get('area_km2'));annual=numeric(a.get('mean_annual_precip_mm_equiv'))
        area_ratio=area/target_attr['area_km2'];rain_ratio=annual/target_attr['mean_annual_precip_mm_equiv']
        near=cfg['near_background'];wide=cfg['wide_background_sensitivity']
        inside=lambda rule:rule['area_ratio'][0]<=area_ratio<=rule['area_ratio'][1] and rule['annual_rain_ratio'][0]<=rain_ratio<=rule['annual_rain_ratio'][1]
        layer='unknown' if not np.isfinite(area_ratio+rain_ratio) else 'near' if inside(near) else 'mid_wide_sensitivity_only' if inside(wide) else 'far'
        hp=x[i,:,0].astype(float)*scales[station]['pscale'];hq=x[i,:,1].astype(float)*r['qscale'];fq=y[i].astype(float)*r['qscale']
        now=hq[-1];rise=fq[:6].max()-now;peak_hour=int(np.argmax(fq[:6]))+1;past_slope=(hq[-1]-hq[-4])/3
        spec=lambda q:3.6*q/area if np.isfinite(area) and area>0 else np.nan
        fp=future[i,:6];has_future=bool(np.isfinite(fp).all());total=float(fp.sum()) if has_future else np.nan
        row=dict(window_id=r['window_id'],role='fit',station=station,country=r['country'],event=r['event'],source_event_id=r['source_event_id'],origin_time=r['origin_time'],fragment_id=fragment[i],temporal_overlap_component=r['temporal_overlap_component'],selection_tags='|'.join(map(str,tags.get(i,[]))),area_km2=area,annual_P_mm_equiv=annual,annual_P_original_unit=a.get('original_precip_unit'),annual_P_reference=a.get('precipitation_reference_period'),static_attribute_source=a.get('source'),area_ratio_to_target=area_ratio,annual_P_ratio_to_target=rain_ratio,background_layer=layer,background_wide_sensitive=inside(wide),P_provenance=cfg['P_provenance'][r['country']],P3_mm=float(hp[-3:].sum()),P6_mm=float(hp[-6:].sum()),P12_mm=float(hp.sum()),qscale_m3_s=r['qscale'],Qnow_m3_s=float(now),Qnow_over_fitQ95=float(x[i,-1,1]),past3h_Q_slope_m3_s_per_h=float(past_slope),past3h_Q_slope_normalized_per_h=float((x[i,-1,1]-x[i,-4,1])/3),specific_Qnow_mm_h=float(spec(now)),specific_past3h_slope_mm_h_per_h=float(spec(past_slope)),future6_peak_m3_s=float(fq[:6].max()),future6_rise_m3_s=float(rise),future6_rise_normalized=float(rise/r['qscale']),specific_future6_rise_mm_h=float(spec(rise)),future6_peak_hour=peak_hour,future12_highest_peak_block=int(np.argmax([fq[:1].max(),fq[1:3].max(),fq[3:6].max(),fq[6:12].max()]))+1,futureP6_state='available_offline_fit_only' if has_future else 'NOT_EVALUABLE',futureP6_mm=total,futureP6_max_mm_h=float(fp.max()) if has_future else np.nan,futureP6_max_hour_fraction=float(fp.max()/total) if has_future and total>0 else np.nan,rain_peak_to_Q_peak_hours=float(peak_hour-(int(fp.argmax())+1)) if has_future and total>0 else np.nan,raw_missing_type='not_evaluated_all_raw; prepared fit finite does not prove source complete')
        result.append(row)
    li=[r for r in result if r['station']==cfg['target']]
    wet=[r['P6_mm'] for r in li if r['P6_mm']>0];rises=[r['future6_rise_normalized'] for r in li if r['future6_rise_normalized']>0]
    rain_threshold=float(np.quantile(wet,.75)) if wet else None;rise_threshold=float(np.quantile(rises,.75)) if rises else None
    physical_positive=[r['specific_future6_rise_mm_h'] for r in li if r['specific_future6_rise_mm_h']>0]
    physical_threshold=float(np.quantile(physical_positive,.75)) if physical_positive else None
    thresholds=dict(rain_P6_mm=rain_threshold,rain_population='positive P6 of162 target fit origins; overlapping windows not independent',rain_positive_origins=len(wet),rise_normalized=rise_threshold,rise_specific_mm_h=physical_threshold,rise_population='positive future6 maxQ-Qnow in target fit, normalized or3.6deltaQ/area',rise_positive_origins=len(rises),quantile=.75,origin_Q_boundary='Qnow / station fit Q95 <=1 vs >1; China event-conditioned and foreign long-period Q95 are not same physical rarity',no_development_values_used=True)
    for row in result:
        row['Q_state']='current_low' if row['Qnow_over_fitQ95']<=1 else 'current_high'
        row['rain_state']='strong_history_rain' if rain_threshold is not None and row['P6_mm']>=rain_threshold else 'weaker_history_rain' if rain_threshold is not None else 'NOT_EVALUABLE'
        row['response_state']='nonrise' if row['future6_rise_normalized']<=0 else 'strong_positive_rise' if rise_threshold is not None and row['future6_rise_normalized']>=rise_threshold else 'positive_rise' if rise_threshold is not None else 'NOT_EVALUABLE'
        v=row['specific_future6_rise_mm_h']
        row['response_state_physical']='NOT_EVALUABLE' if not np.isfinite(v) or physical_threshold is None else 'nonrise' if v<=0 else 'strong_positive_rise' if v>=physical_threshold else 'positive_rise'
        row['history_cell']=row['Q_state']+'/'+row['rain_state'];row['joint_cell']=row['history_cell']+'/'+row['response_state']
        row['joint_cell_physical']=row['history_cell']+'/'+row['response_state_physical']
    return result,thresholds


def target_quality(cfg,rows,x,y,scales,future,out):
    entries=[json.loads(s) for s in Path(cfg['source_index']).read_text().splitlines()]
    entry=next(r for r in entries if r['station_uid']==cfg['target']);path=Path(entry['output'])
    target=[r for r in rows if r['station_uid']==cfg['target']];allowed={r['source_event_id'] for r in target};events=defaultdict(list)
    with gzip.open(path,'rt',newline='') as fp:
        reader=csv.reader(fp);header=next(reader);event_col=header.index('source_event_id')
        columns={k:header.index(k) for k in ('source_row','time_axis','q_m3_s','p_mm_h','pq_numeric_valid','qc_flags')}
        for values in reader:
            event=values[event_col]
            if event not in allowed:continue
            # Whole author records are authorized only for these same3 fit events' quality.
            r={k:values[j] for k,j in columns.items()};r['P']=numeric(r['p_mm_h']);r['Q']=numeric(r['q_m3_s']);events[event].append(r)
    lookup={(event,r['source_row']):r for event,items in events.items() for r in items}
    for i,meta in enumerate(rows):
        if meta['station_uid']!=cfg['target']:continue
        p=np.r_[x[i,:,0]*scales[cfg['target']]['pscale'],future[i]];q=np.r_[x[i,:,1]*meta['qscale'],y[i]*meta['qscale']]
        for source,stamp,pv,qv in zip(meta['source_rows'],meta['source_times'],p,q):
            raw=lookup.get((meta['source_event_id'],str(source)))
            if raw is None or raw['time_axis']!=stamp or not np.isclose(raw['Q'],qv,rtol=4e-6,atol=1e-4) or (np.isfinite(pv) and not np.isclose(raw['P'],pv,rtol=4e-6,atol=1e-5)):raise ValueError('target original/fit source row,time,P/Q mismatch; quality extraction stopped')
    result=[]
    for event,records in sorted(events.items()):
        records.sort(key=lambda r:r['time_axis']);times=[datetime.fromisoformat(r['time_axis']) for r in records]
        used={str(s) for r in target if r['source_event_id']==event for s in r['source_rows']}
        missing_hours=sum(max((b-a).total_seconds()/3600-1,0) for a,b in zip(times,times[1:]));maxgap=max([max((b-a).total_seconds()/3600-1,0) for a,b in zip(times,times[1:])]+[0])
        result.append(dict(station=cfg['target'],source_event_id=event,role='same fit source event author-record quality only',author_rows=len(records),fit_windows=sum(r['source_event_id']==event for r in target),fit_unique24h_source_rows=len(used),author_rows_not_in_fit_windows=sum(r['source_row'] not in used for r in records),prefix_source_status='unknown',excluded_initial_author_rows=12,source_P_missing_or_invalid=sum(not np.isfinite(r['P']) for r in records),source_Q_missing_or_invalid=sum(not np.isfinite(r['Q']) for r in records),source_PQ_flag_invalid=sum(r['pq_numeric_valid'].lower() not in ('true','1') for r in records),longest_P_missing_run_rows=longest_missing([r['P'] for r in records]),longest_Q_missing_run_rows=longest_missing([r['Q'] for r in records]),missing_grid_hours=missing_hours,longest_time_gap_missing_hours=maxgap,source_start=records[0]['time_axis'],source_end=records[-1]['time_axis'],qc_flags='|'.join(sorted({r['qc_flags'] for r in records})),physical_flood_completeness='unknown; no numeric missing does not establish complete physical flood',China_P_time_accumulation_semantics='unknown'))
    table(out/'target_source_event_quality.csv',result)
    return result,dict(path=str(path),sha256=digest(path),allowed_source_event_ids=sorted(allowed),fields_accessed=list(columns),other_role_values_parsed=False,use='quality evidence only; no prefix restored into fit inputs')


def donor_candidates(data,cfg):
    li=[r for r in data if r['station']==cfg['target']];references=[]
    physical=('P3_mm','P6_mm','P12_mm','specific_Qnow_mm_h','specific_past3h_slope_mm_h_per_h')
    normalized=('P3_mm','P6_mm','P12_mm','Qnow_over_fitQ95','past3h_Q_slope_normalized_per_h')
    scale={}
    for keys in (physical,normalized):
        values=np.array([[r[k] for k in keys] for r in li]);spread=np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0);spread[spread==0]=1;scale[keys]=spread
    for cell in sorted({r['history_cell'] for r in li}):
        group=[r for r in li if r['history_cell']==cell];center=np.median([[r[k] for k in physical] for r in group],axis=0)
        chosen=min(group,key=lambda r:(float(np.mean(abs((np.array([r[k] for k in physical])-center)/scale[physical]))),r['window_id']))
        references.append(dict(reference_rule='nearest physical historical-feature center of actual target-fit history_cell; no future selection',**chosen))
    low=[r for r in li if r['Q_state']=='current_low']
    if low:
        critical=max(low,key=lambda r:(r['future6_rise_m3_s'],r['window_id']))
        references.append(dict(reference_rule='maximum target-fit6h rise among current-low origins; descriptive train-label case, foreign ranking still history-only',**critical))
    candidates=[];sensitivity=[]
    for ref in references:
        for country in ('Germany','Iceland'):
            for layer in ('near','mid_wide_sensitivity_only','far'):
                pool=[r for r in data if r['country']==country and r['background_layer']==layer and r['history_cell']==ref['history_cell'] and all(np.isfinite(r[k]) for k in physical)]
                scored=[]
                for r in pool:
                    p=float(np.mean(abs(np.array([r[k]-ref[k] for k in physical])/scale[physical])))
                    n=float(np.mean(abs(np.array([r[k]-ref[k] for k in normalized])/scale[normalized])))
                    scored.append((p,n,r))
                by_fragment={}
                for p,n,r in sorted(scored,key=lambda v:(v[0],v[2]['window_id'])):by_fragment.setdefault(r['fragment_id'],(p,n,r))
                physical_order=sorted(by_fragment.values(),key=lambda v:(v[0],v[2]['window_id']));norm_order=sorted(by_fragment.values(),key=lambda v:(v[1],v[2]['window_id']));nranks={r['window_id']:j+1 for j,(p,n,r) in enumerate(norm_order)}
                k=cfg['donors_per_reference_country_layer'];pids={r['window_id'] for p,n,r in physical_order[:k]};nids={r['window_id'] for p,n,r in norm_order[:k]}
                sensitivity.append(dict(reference_window_id=ref['window_id'],reference_rule=ref['reference_rule'],reference_source_event=ref['source_event_id'],country=country,background_layer=layer,history_cell=ref['history_cell'],fit_windows=len(pool),deduplicated_process_fragments=len(physical_order),matched_history_physical_strong_rise_windows=sum(r['response_state_physical']=='strong_positive_rise' for r in pool),matched_history_normalized_strong_rise_windows=sum(r['response_state']=='strong_positive_rise' for r in pool),top_k=k,physical_vs_normalized_top_k_overlap=len(pids&nids),state='evaluable' if pool else 'NOT_FOUND_IN_FIXED_FIT_POOL',selection='history only, target-fit IQR scaling; one representative per same-station fragment; no future response used'))
                for rank,(p,n,r) in enumerate(physical_order[:k],1):
                    row=dict(reference_window_id=ref['window_id'],reference_rule=ref['reference_rule'],reference_source_event=ref['source_event_id'],reference_P6_mm=ref['P6_mm'],reference_Qnow_over_fitQ95=ref['Qnow_over_fitQ95'],reference_specific_Qnow_mm_h=ref['specific_Qnow_mm_h'],reference_specific_future6_rise_mm_h=ref['specific_future6_rise_mm_h'],rank_physical=rank,rank_normalized_in_same_fragment_pool=nranks[r['window_id']],history_distance_physical=p,history_distance_normalized=n,selection_uses_future_response=False,**r)
                    row['response_specific_ratio_to_reference']=r['specific_future6_rise_mm_h']/ref['specific_future6_rise_mm_h'] if ref['specific_future6_rise_mm_h']>0 else np.nan
                    candidates.append(row)
    return references,candidates,sensitivity


def coverage(data,cfg):
    cells=[q+'/'+p+'/'+r for q in ('current_low','current_high') for p in ('weaker_history_rain','strong_history_rain') for r in ('nonrise','positive_rise','strong_positive_rise')]
    groups=dict(target=[r for r in data if r['station']==cfg['target']],other_China=[r for r in data if r['country']=='China' and r['station']!=cfg['target']])
    for layer in ('near','mid_wide_sensitivity_only','far','unknown'):groups['foreign_'+layer]=[r for r in data if r['country']!='China' and r['background_layer']==layer]
    result=[]
    for basis,key in [('physical_specific_rise','joint_cell_physical'),('normalized_Q95_sensitivity','joint_cell')]:
        for name,rows in groups.items():
            for cell in cells:
                selected=[r for r in rows if r[key]==cell];values=[r['specific_future6_rise_mm_h'] for r in selected if np.isfinite(r['specific_future6_rise_mm_h'])]
                result.append(dict(response_basis=basis,group=name,joint_cell=cell,n_fit_windows=len(selected),n_stations=len({r['station'] for r in selected}),n_same_station_fragments=len({r['fragment_id'] for r in selected}),n_foreign_source_blocks=len({(r['station'],r['event']) for r in selected}) if name.startswith('foreign') else None,specific_future6_rise_median_mm_h=float(np.median(values)) if values else None,specific_future6_rise_q25_mm_h=float(np.quantile(values,.25)) if values else None,specific_future6_rise_q75_mm_h=float(np.quantile(values,.75)) if values else None,state='seen_in_fixed_fit_pool' if selected else 'not_seen_in_fixed_fit_pool',interpretation='no claim an empty target cell ought to exist; fragments can occupy multiple cells and are not independent floods'))
    return result


def plot(out,data,coverage_rows,refs,candidates,x,y,future,rows,scales,cfg,thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir()
    groups=list(dict.fromkeys(r['group'] for r in coverage_rows));cells=list(dict.fromkeys(r['joint_cell'] for r in coverage_rows))
    fig,axes=plt.subplots(1,2,figsize=(18,7))
    maximum=max(r['n_fit_windows'] for r in coverage_rows)
    for ax,basis,title in zip(axes,('physical_specific_rise','normalized_Q95_sensitivity'),('MAIN: common physical runoff-depth rise threshold','SENSITIVITY: each-station Q95 normalized rise')):
        matrix=np.array([[next(r['n_fit_windows'] for r in coverage_rows if r['response_basis']==basis and r['group']==g and r['joint_cell']==c) for c in cells] for g in groups])
        image=ax.imshow(np.log1p(matrix),aspect='auto',cmap='Blues',vmin=0,vmax=np.log1p(maximum))
        for i in range(len(groups)):
            for j in range(len(cells)):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if np.log1p(matrix[i,j])>np.log1p(matrix.max())*.7 else 'black')
        short=[c.replace('current_','Q_').replace('weaker_history_rain','P6 weak').replace('strong_history_rain','P6 strong').replace('strong_positive_rise','rise strong').replace('positive_rise','rise positive').replace('nonrise','nonrise') for c in cells]
        ax.set(xticks=range(len(cells)),xticklabels=short,yticks=range(len(groups)),yticklabels=groups,title=title+'; fit-window counts')
        ax.tick_params(axis='x',rotation=75);fig.colorbar(image,ax=ax,label='log(1+count), zero is observed no support in this fixed pool')
    fig.suptitle(f"TRAIN ONLY: Li fit strong-rise cutoffs: physical {thresholds['rise_specific_mm_h']:.3g} mm/h vs normalized {thresholds['rise_normalized']:.3g}\nHistory Q/Q95 has different populations across countries; empty cells are not presumed gaps. Foreign blocks/fragments are not independent complete storms.",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(figures/'01_fit_condition_coverage.png',dpi=160);plt.close(fig)
    # Foreign examples are chosen only by static layer and physical historical distance.
    examples=[];seen=set()
    ordered=sorted((r for r in candidates if r['background_layer'] in ('near','mid_wide_sensitivity_only')),key=lambda r:(0 if r['background_layer']=='near' else 1,r['history_distance_physical'],r['window_id']))
    for country in ('Germany','Iceland'):
        item=next((r for r in ordered if r['country']==country),None)
        if item:examples.append(item);seen.add(item['station'])
    for r in ordered:
        if len(examples)>=cfg['foreign_examples_max']:break
        if r['station'] not in seen:examples.append(r);seen.add(r['station'])
    li=[r for r in data if r['station']==cfg['target']]
    event_cases=[max((r for r in li if r['source_event_id']==event),key=lambda r:(r['future6_rise_m3_s'],r['window_id'])) for event in sorted({r['source_event_id'] for r in li})]
    cases=event_cases+examples;lookup={r['window_id']:i for i,r in enumerate(rows)}
    fig,axes=plt.subplots(math.ceil(len(cases)/2),2,figsize=(15,3.8*math.ceil(len(cases)/2)),squeeze=False)
    source_rows=[]
    for ax,case in zip(axes.flat,cases):
        i=lookup[case['window_id']];meta=rows[i];q=np.r_[x[i,:,1]*meta['qscale'],y[i]*meta['qscale']];p=np.r_[x[i,:,0]*scales[meta['station_uid']]['pscale'],future[i]];hours=np.arange(-11,13)
        pax=ax.twinx();pax.bar(hours,p,color='tab:blue',alpha=.2,width=.8,label='P (source-specific)');pax.set_ylabel('P (mm/h; interval semantics unknown for China)')
        ax.plot(hours,q,color='black',label='Q');ax.axvline(.5,color='red',ls='--');ax.axvspan(.5,6.5,color='orange',alpha=.07)
        ax.set(xlabel='Hours relative to origin; future P is offline fit description only',ylabel='Q (m3/s)',title=case['country']+' | '+meta['station_uid'].split(':')[-1]+' | '+meta['origin_time'])
        ax.text(.02,.96,'Li event largest6h rise (train-label descriptive case)' if case['country']=='China' else case['background_layer']+'; history-only matched',transform=ax.transAxes,va='top',fontsize=8)
        for h,pv,qv in zip(hours,p,q):source_rows.append(dict(window_id=case['window_id'],station=case['station'],country=case['country'],role='fit',hour_relative_origin=int(h),P_mm_h=float(pv) if np.isfinite(pv) else None,Q_m3_s=float(qv),P_provenance=cfg['P_provenance'][case['country']],future_P_forecasting_input=False,source_time=meta['source_times'][int(h+11)],source_row=meta['source_rows'][int(h+11)]))
    for ax in list(axes.flat)[len(cases):]:ax.axis('off')
    fig.suptitle('Fixed fit24h examples: all3 Liqingdian fit events + up to4 distinct foreign stations\nChina supplied P / Germany author gap-filled P / Iceland ERA5-Land reanalysis P; future P only offline, Q future are train labels. No complete-storm or causal claim.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(figures/'02_representative_fit_PQ.png',dpi=160);plt.close(fig)
    table(out/'figure_PQ_source_rows.csv',source_rows)
    table(out/'representative_fit_windows.csv',[dict(case='target_event_reference' if r['country']=='China' else 'foreign_history_match',**r) for r in cases])
    return [r['window_id'] for r in cases],len(examples)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if cfg['role']!='fit' or cfg['training'] is not False or cfg['new_data_download'] is not False:raise ValueError('fit-only no-training scope required')
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    write_json(out/'config.json',cfg);write_json(out/'progress.json',dict(state='running',phase='load fixed fit only'))
    try:
        assert longest_missing([np.nan,1,np.nan,np.nan])==2
        rows,x,y,scales,attrs,manifest,inputs=load_fit(cfg,out)
        future,tags,access,failures=extract_fit_future(rows,x,y,scales,manifest,out)
        mapping,processes=fragments(rows)
        data,thresholds=features(rows,x,y,scales,attrs,future,tags,cfg,mapping)
        write_json(out/'progress.json',dict(state='running',phase='target quality and descriptive historical matching'))
        quality_state='complete'
        try:quality,quality_access=target_quality(cfg,rows,x,y,scales,future,out)
        except ValueError as exc:
            quality_state='partial';quality=[dict(state='NOT_EVALUABLE',reason=str(exc),station=cfg['target'])];quality_access=dict(path=cfg['source_index'],state='partial',reason=str(exc));table(out/'target_source_event_quality.csv',quality)
        refs,donors,sensitivity=donor_candidates(data,cfg);cells=coverage(data,cfg)
        table(out/'fit_window_features.csv',data);table(out/'fit_process_fragments.csv',processes);table(out/'condition_coverage.csv',cells)
        table(out/'history_references.csv',refs);table(out/'candidate_donors.csv',donors);table(out/'matching_sensitivity.csv',sensitivity)
        table(out/'station_background.csv',[{k:v for k,v in r.items() if k in ('station','country','area_km2','annual_P_mm_equiv','annual_P_original_unit','annual_P_reference','static_attribute_source','area_ratio_to_target','annual_P_ratio_to_target','background_layer')} for r in {r['station']:r for r in data}.values()])
        units=[]
        for country in ('Germany','Iceland'):
            for layer in ('near','mid_wide_sensitivity_only','far','unknown'):
                selected=[r for r in data if r['country']==country and r['background_layer']==layer]
                misleading=[r for r in selected if r['response_state']=='strong_positive_rise' and r['response_state_physical'] not in ('strong_positive_rise','NOT_EVALUABLE')]
                units.append(dict(country=country,background_layer=layer,n_fit_windows=len(selected),normalized_strong_windows=sum(r['response_state']=='strong_positive_rise' for r in selected),common_physical_strong_windows=sum(r['response_state_physical']=='strong_positive_rise' for r in selected),normalized_strong_but_physical_not_strong=len(misleading),normalized_strong_but_physical_not_strong_fragments=len({r['fragment_id'] for r in misleading}),physical_NOT_EVALUABLE=sum(r['response_state_physical']=='NOT_EVALUABLE' for r in selected)))
        table(out/'response_scale_sensitivity.csv',units)
        write_json(out/'thresholds.json',thresholds)
        write_json(out/'access_manifest.json',dict(code_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=inputs,upstream_fit_allowlist_sources=access,target_quality=quality_access,source_index_sha256=digest(cfg['source_index']),cleaning_rules_sha256=digest(cfg['cleaning_rules']),roles_read=['fit'],other_role_values_parsed=False,future_P_use='authorized offline fit description only',raw_foreign_values_read=False,static_attributes='retrospective; Germany2001-2024/Iceland1989-2009/China unknown, not historical-as-of',scope='fixed selected fit pool, no new inference/training/generation/forecast download'))
        write_json(out/'progress.json',dict(state='running',phase='two server figures'))
        figure_ids,foreign_examples=plot(out,data,cells,refs,donors,x,y,future,rows,scales,cfg,thresholds)
        li=[r for r in data if r['station']==cfg['target']];event_cases=[max((r for r in li if r['source_event_id']==e),key=lambda r:r['future6_rise_m3_s']) for e in cfg['expected_target_source_events']]
        critical=[r for r in refs if r['reference_rule'].startswith('maximum target-fit')]
        gaps=[r['joint_cell'] for r in cells if r['response_basis']=='physical_specific_rise' and r['group']=='target' and r['n_fit_windows']==0]
        tag_counts=Counter(tag for selected in tags.values() for tag in selected)
        result=dict(state='partial' if failures or quality_state=='partial' else 'complete',change_id=cfg['change_id'],fit_windows=len(rows),country_windows=dict(Counter(r['country'] for r in rows)),country_stations={c:len({r['station_uid'] for r in rows if r['country']==c}) for c in sorted({r['country'] for r in rows})},target_fit_windows=len(li),target_fit_source_events=cfg['expected_target_source_events'],target_quality_state=quality_state,target_event_max_rise_cases=[{k:r[k] for k in ('window_id','source_event_id','origin_time','Qnow_m3_s','future6_peak_m3_s','future6_rise_m3_s','P12_mm','futureP6_mm','specific_future6_rise_mm_h')} for r in event_cases],critical_current_low_case=critical,critical_history_support=[r for r in sensitivity if r['reference_rule'].startswith('maximum target-fit')],thresholds=thresholds,candidate_target_empty_cells=gaps,gap_interpretation='not seen in only3 target fit events; no claim these cells ought to exist or are missing from global archives',offline_future_P_available_windows=int(np.isfinite(future).all(1).sum()),future_P_failures=failures,actual_fit_upstream_selection_tags=dict(tag_counts),foreign_fragments=sum(r['country']!='China' for r in processes),foreign_blocks_are_not_independent_floods=True,foreign_examples=foreign_examples,figure_window_ids=figure_ids,response_scale_sensitivity=units,conclusion_status='descriptive feasibility; parent scientific interpretation pending',actual_elapsed_seconds=time.monotonic()-started)
        write_json(out/'summary.json',result)
        (out/'READING_GUIDE.md').write_text('''# 训练资料覆盖与国外匹配说明

这里只分析固定fit池。李青店三个来源事件不是独立性已经证实的三场暴雨；国外block及同站相邻/重叠24小时片段也不是完整独立洪水。selection_tags保留实际旧train选样规则，不将稀疏fit窗口误称完整原始档案。

图1左侧按同一个李青店训练径流深度上涨阈值分格，右侧按各站Q95归一化作敏感性。数字为fit窗口数；去重片段/站点数在condition_coverage.csv。空格只表示这批资料没有观察到，不能证明该条件理应存在或全球资料缺失。阈值均来自李青店fit：正历史P6的75%分位数、正未来6h涨幅的75%分位数。起报Q/Q95≤1与>1是研究粗层，不是业务灾害等级或同等稀有度。

径流深度3.6Q/A单位mm/h，面积A单位km²、Q单位m³/s；涨幅同样用3.6ΔQ/A。若国外只在ΔQ/Q95上较大、物理上涨仍小，response_scale_sensitivity.csv会直接重新计数而不据归一化格认领强donor。

匹配先按回顾性面积与年雨近/中/远分层，再在同历史粗格按P3/P6/P12、起报径流深度与过去3h趋势的物理距离排序；缩放仅来自李青店fit四分位距。各实际历史格用历史特征中心最近窗，另保留低起报最大涨幅train描述案例。外国排序不使用未来峰、涨幅或误差。宽背景和归一化距离仅作描述性敏感性，不调优筛选规则。

图2三个李青店案例按每事件最大6h涨幅选择，这是明确使用train标签的描述案例；国外最多四站仍按历史匹配选择，保留不同响应。红线右侧Q是训练标签，P是此次获准的离线未来雨量，绝不进入预测。中国提供P、德国作者填补P、冰岛ERA5-Land再分析P不可统称实测；中国累计区间和时区语义仍unknown。雨峰—流量峰时差只描述24h窗口，不是因果关系或流域真实传播时间。

target_source_event_quality.csv查的是同三个作者源事件，不用prepared零缺失证明源记录完整；前缀unknown不是missing，未恢复模型输入。外国raw质量未展开，按来源/既有处理与窗口校验标记，完整物理洪水和跨站同场暴雨仍unknown。没有训练、生成样本、外部数据下载或其他角色值读取。
''')
        completion=dict(state=result['state'],elapsed_seconds=result['actual_elapsed_seconds'],resume='finished; parent reads summary/access manifest and two figures; no automatic training or source expansion')
        write_json(out/'completion.json',completion);write_json(out/'progress.json',completion)
        print(json.dumps({k:result[k] for k in ('state','fit_windows','target_fit_windows','offline_future_P_available_windows','foreign_examples','actual_elapsed_seconds')}),flush=True)
    except Exception as exc:
        failure=dict(state='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started,resume='preserve failed run; fix only implementation/identity issue, do not widen data roles')
        write_json(out/'completion.json',failure);write_json(out/'progress.json',failure);raise


if __name__=='__main__':main()
