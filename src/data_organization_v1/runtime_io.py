"""Explicit storage relocation; only original selected-role Q fields are read."""
import csv,gzip,hashlib,json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
from protected_flood_risk_v1.data import digest


def remap(value,mapping):
    if not isinstance(value,str):return value
    for old,new in sorted(mapping.items(),key=lambda v:-len(v[0])):
        if value==old or value.startswith(old+'/'):return new+value[len(old):]
    return value


def map_config(value,mapping):
    if isinstance(value,dict):return {k:map_config(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [map_config(v,mapping) for v in value]
    return remap(value,mapping)


def original_q(pool,final_authorized=False,config=None):
    """Evaluation metadata only; event filter precedes any Q/time field access."""
    config=config or {}
    if config.get('_q_only_dir'):
        roles={r['role'] for r in pool['rows']}
        if len(roles)!=1 or not roles.issubset({'fit','select'}):raise ValueError('exported author Q is fit/select only')
        path=Path(config['_q_only_dir'])/(next(iter(roles))+'.json');raw=json.loads(path.read_text());records={}
        wanted={(r['station_uid'],r['source_event_id']) for r in pool['rows'] if r['country']=='China'}
        for r in raw['records']:
            key=(r['station'],r['source_event_id'])
            if key not in wanted:continue
            records[key]=dict(points={datetime.fromisoformat(t):(str(s),float(q)) for t,s,q in r['points']},issues=set(r['issues']))
        if set(records)!=wanted:raise ValueError('Q-only exported membership changed')
        return records,dict(export_path=str(path),sha256=digest(path),source_provenance=raw['source_provenance'],P_accessed=False,other_role_Q_extracted=False)
    index=Path(remap('./data/catalog/cleaning_v2_stations.jsonl',config.get('_path_map',{})))
    sources={r['station_uid']:r for r in map(json.loads,index.read_text().splitlines()) if r['country']=='China'}
    needed=defaultdict(set)
    for r in pool['rows']:
        if r['role'] not in ('fit','select') and not final_authorized:raise PermissionError('original final-role Q requires completed selections')
        if r['country']=='China':needed[r['station_uid']].add(r['source_event_id'])
    records={};provenance=[]
    for station,events in sorted(needed.items()):
        path=Path(remap(sources[station]['output'],(config or {}).get('_path_map',{})))
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
