import csv
import hashlib
import gzip
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
from evaluate_event_peaks_v1 import record_info,select_origin
from protected_flood_risk_v1.data import real_data_adapter
from protected_flood_risk_v1.data import digest


def load(config,roles):
    return real_data_adapter(config['data'],roles,config['attribute_rows'],config['normalization'],config['data_manifest_sha256'],execution_authorized=True)


def original_q(pool,final_authorized=False):
    """Evaluation metadata only; event filter precedes any Q/time field access."""
    index=Path('./data/catalog/cleaning_v2_stations.jsonl')
    sources={r['station_uid']:r for r in map(json.loads,index.read_text().splitlines()) if r['country']=='China'}
    needed=defaultdict(set)
    for r in pool['rows']:
        if r['role'] not in ('fit','select') and not final_authorized:raise PermissionError('original final-role Q requires completed selections')
        if r['country']=='China':needed[r['station_uid']].add(r['source_event_id'])
    records={};provenance=[]
    for station,events in sorted(needed.items()):
        path=Path(sources[station]['output'])
        extracted=[]
        with gzip.open(path,'rt',newline='') as fp:
            reader=csv.reader(fp);header=next(reader)
            event_col=header.index('source_event_id')
            cols={k:header.index(k) for k in ('source_row','time_axis','q_m3_s')}
            for values in reader:
                event=values[event_col]
                if event not in events:continue
                # No P field is accessed; other-role Q is never extracted.
                row={k:values[j] for k,j in cols.items()}
                extracted.append(dict(source_event_id=event,**row))
                record=records.setdefault((station,event),dict(points={},issues=set()))
                try:
                    point=(row['source_row'],float(row['q_m3_s']))
                    stamp=datetime.fromisoformat(row['time_axis'])
                    if not np.isfinite(point[1]) or point[1]<0:raise ValueError('invalid Q')
                except ValueError:
                    record['issues'].add('invalid_original_author_Q_or_time');continue
                existing=record['points']
                if stamp in existing and existing[stamp]!=point:record['issues'].add('original_duplicate_time_source_Q_conflict')
                existing.setdefault(stamp,point)
        provenance.append(dict(station=station,path=str(path),file_sha256=digest(path),source_member=sources[station]['source_member'],event_ids=sorted(events),Q_only_selected_records_sha256=hashlib.sha256(json.dumps(extracted,sort_keys=True).encode()).hexdigest(),selected_Q_rows=len(extracted),prefix_source_status='unknown',use='author-record mainpeak metadata only; excluded12h never enters model or windows'))
    return records,dict(index=str(index),index_sha256=digest(index),sources=provenance,P_accessed=False,other_role_Q_extracted=False)


def eligibility(pool,catalog_path,author_records=None):
    catalog={r['event_uid']:r for r in csv.DictReader(open(catalog_path))}
    grouped=defaultdict(list)
    for i,r in enumerate(pool['rows']):
        if r['country']=='China':grouped[(r['station_uid'],r['source_event_id'])].append(i)
    mask=np.zeros(len(pool['rows']),bool);records=[]
    for (station,event),ix in sorted(grouped.items()):
        key=station+':'+event;cat=catalog.get(key)
        if cat is None:raise ValueError('missing original catalog identity '+key)
        points={};windows={};issues=set()
        for i in ix:
            r=pool['rows'][i];origin=datetime.fromisoformat(r['origin_time']);windows[origin]=i
            for t,source,q in zip(r['source_times'],r['source_rows'],np.r_[pool['x'][i,:,1],pool['y'][i]]):
                t=datetime.fromisoformat(t);q=float(q)*r['qscale']
                if t in points and (str(points[t][0])!=str(source) or not np.isclose(points[t][1],q,rtol=2e-6,atol=1e-5)):issues.add('duplicate_q_or_source_conflict')
                points.setdefault(t,(str(source),q))
        retained_count=len(points)
        if author_records is not None:
            original=author_records[(station,event)]['points']
            issues.update(author_records[(station,event)]['issues'])
            for t,(source,q) in points.items():
                if t not in original or str(original[t][0])!=str(source) or not np.isclose(original[t][1],q,rtol=2e-6,atol=1e-5):
                    raise ValueError('original/prepared retained Q or source identity mismatch')
            points=original
        info=record_info(points,cat,sorted(issues))
        origin,reasons=select_origin('retrospective_peak_anchored',6,windows,info,cat)
        reasons=sorted(set(info['reasons']+reasons))
        legal=not reasons
        chosen=windows[origin] if legal else None
        if chosen is not None:mask[chosen]=True
        info.pop('reasons',None)
        records.append(dict(station=station,source_event_id=event,event=pool['rows'][ix[0]]['event'],state='evaluable' if legal else 'NOT_EVALUABLE',reasons=reasons,window_id=pool['rows'][chosen]['window_id'] if legal else None,high95=bool(info['observed_peak_m3_s']>pool['rows'][ix[0]]['qscale']),prepared_union_rows=retained_count,prefix_source_status='unknown',peak_scope='author record; physical flood completeness unknown',**info))
    return mask,records


def slots(pool,epochs,size,seed):
    groups=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(pool['rows']):groups[r['country']][r['station_uid']].append(i)
    countries=sorted(groups);stations={c:sorted(groups[c]) for c in countries};rng=np.random.default_rng(seed)
    result=np.empty((epochs,size),np.int64)
    for e in range(epochs):
        for j in range(size):
            c=countries[rng.integers(len(countries))];s=stations[c][rng.integers(len(stations[c]))]
            ix=groups[c][s];result[e,j]=ix[rng.integers(len(ix))]
    return result,hashlib.sha256(result.tobytes()).hexdigest()


def fixed(pool,legal):
    chosen=set(np.flatnonzero(legal));rules=[]
    now=pool['x'][:,-1,1];peak=pool['y'][:,:6].max(1)
    for s in sorted({r['station_uid'] for r in pool['rows'] if r['country']=='China'}):
        ix=[i for i,r in enumerate(pool['rows']) if r['station_uid']==s]
        for name,eligible in [('ordinary',[i for i in ix if peak[i]<=1]),('strong_rise',ix),('current_high_upgrade',[i for i in ix if now[i]>1 and peak[i]>now[i]]),('recession',[i for i in ix if peak[i]<now[i]])]:
            i=max(eligible,key=lambda i:peak[i]-now[i]) if eligible and name=='strong_rise' else eligible[0] if eligible else None
            if i is not None:chosen.add(i)
            rules.append(dict(station=s,rule=name,window_id=pool['rows'][i]['window_id'] if i is not None else None,state='evaluable' if i is not None else 'NOT_EVALUABLE'))
    return sorted(chosen),rules
