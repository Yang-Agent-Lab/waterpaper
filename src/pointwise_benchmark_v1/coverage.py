"""Read-only fit/select coverage from accepted identities, without new preparation.

Ranges are inclusive source-native hourly interval ends. No values from other
roles are opened. Original source events/time blocks are never joined.
"""
import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.five_country_model_data_v1.prepare import hour
from src.five_country_model_data_v1.reader import csv_rows, number, stamp, truth
from src.workspace_consolidation_v1.relocation import RelocatedReader


def merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if a > b:
            continue
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([int(a), int(b)])
    return out


def intersect(left, right):
    out = []; i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]; c, d = right[j]
        if max(a, c) <= min(b, d):
            out.append([max(a, c), min(b, d)])
        if b < d:
            i += 1
        else:
            j += 1
    return out


def runs(times):
    times = sorted(set(times))
    return merge([[t, t] for t in times])


def count(ranges):
    return sum(b-a+1 for a, b in ranges)


def hits(times, ranges):
    times = np.asarray(sorted(set(times)), dtype=np.int64)
    return int(sum(np.searchsorted(times, b, side='right') - np.searchsorted(times, a) for a, b in ranges))


def eligible(qruns, pruns, history, future, route):
    origin = merge([[a+history-1, b-future] for a, b in qruns])
    if route == 'pq':
        origin = intersect(origin, merge([[a+history-1, b] for a, b in pruns]))
    return origin


def coverage_row(source, station, canonical, country, role, event, qruns, pruns, histories, future,
                 *, routes=('pq', 'q_only'), high_times=None, threshold_eligible=True):
    result = []
    for route in routes:
        by_h = {h: eligible(qruns, pruns, h, future, route) for h in histories}
        common = by_h[histories[0]]
        for h in histories[1:]:
            common = intersect(common, by_h[h])
        for h, origins in by_h.items():
            target = merge([[a+1, b+future] for a, b in origins])
            row = dict(source=source, station=station, canonical_station=canonical, country=country,
                role=role, event=event, route=route, history_hours=h, forecast_hours=future,
                origin_ranges=origins, origin_count=count(origins), common_all_history_origin_ranges=common,
                common_all_history_origin_count=count(common), unique_supervision_hour_count=count(target),
                supervision_occurrences=count(origins)*future, q_valid_hours=count(qruns),
                p_valid_hours=count(pruns), longest_continuous_Q_hours=max((b-a+1 for a,b in qruns),default=0),
                longest_continuous_P_hours=max((b-a+1 for a,b in pruns),default=0),
                q_valid_ranges=qruns,p_valid_ranges=pruns,threshold_eligible=threshold_eligible,
                event_independence='inherited_source_event_or_block' if source in ('da_tl_cn','camels_de_1h','lamah_ice') else 'continuous_segment_not_independent_storm',
                reason=None if origins else 'insufficient_same_event_continuous_Q_or_historical_P')
            if high_times is not None and threshold_eligible:
                row.update(current_high_Q_origins=hits(high_times, origins),
                    unique_high_Q_supervision_hours=hits(high_times, target),
                    unique_ordinary_Q_supervision_hours=count(target)-hits(high_times, target),
                    high_Q_supervision_occurrences=sum(hits(high_times, [[a+lead,b+lead]]) for a,b in origins for lead in range(1,future+1)))
            else:
                row.update(current_high_Q_origins=None, unique_high_Q_supervision_hours=None,
                    unique_ordinary_Q_supervision_hours=None, high_Q_supervision_occurrences=None)
            result.append(row)
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['country'],r['source'],r['role'],r['route'],r['history_hours'])].append(r)
    return [dict(country=k[0],source=k[1],role=k[2],route=k[3],history_hours=k[4],
        source_station_entries=len({r['station'] for r in rs}),
        eligible_canonical_stations=len({r['canonical_station'] for r in rs if r['origin_count']}),
        source_events_or_segments=len(rs), eligible_events_or_segments=sum(bool(r['origin_count']) for r in rs),
        origins=sum(r['origin_count'] for r in rs),common_all_history_origins=sum(r['common_all_history_origin_count'] for r in rs),
        unique_supervision_hours_within_event=sum(r['unique_supervision_hour_count'] for r in rs),
        high_Q_supervision_occurrences=sum(r['high_Q_supervision_occurrences'] for r in rs) if all(r['high_Q_supervision_occurrences'] is not None for r in rs) else None)
        for k,rs in sorted(groups.items())]


def legacy_coverage(reader, histories, future, roles, only_cn):
    out=[]
    for role in roles:
        metadata,scales,xx,yy=reader._load_legacy(role)
        if only_cn:
            stations=sorted({r['station_uid'].split(':',1)[1] for r in metadata if r['station_uid'].startswith('da_tl_cn:')})
            for station in stations:
                events=defaultdict(list)
                for r in reader._legacy('da_tl_cn',station,role,None):
                    events[r['event']].append(r)
                sc=scales['da_tl_cn:'+station]; threshold=sc['qscale']
                for event,records in sorted(events.items()):
                    qr=runs(hour(r['interval_end']) for r in records if r['q_mask'])
                    pr=runs(hour(r['interval_end']) for r in records if r['p_mask'])
                    high=[hour(r['interval_end']) for r in records if r['q_mask'] and r['q']>threshold]
                    out.extend(coverage_row('da_tl_cn',station,reader.canonical('da_tl_cn',station),'CN',role,event,
                        qr,pr,histories,future,high_times=high,threshold_eligible=threshold>0))
        else:
            groups={}
            for i,r in enumerate(metadata):
                source=r['station_uid'].split(':',1)[0]
                if source not in ('camels_de_1h','lamah_ice'):
                    continue
                station=r['station_uid'].split(':',1)[1]
                group=groups.setdefault((source,station,r['event']),[set(),set()])
                shift=1 if source=='lamah_ice' else 0
                times=[hour(t)+shift for t in r['source_times']]
                # Accepted numeric arrays, not original mixed-role observations.
                q=np.concatenate([xx[i,:,1],yy[i]])
                group[0].update(t for t,v in zip(times,q) if np.isfinite(v) and v>=0)
                group[1].update(t for t,v in zip(times,xx[i,:,0]) if np.isfinite(v) and v>=0)
            for (source,station,event),(qt,pt) in sorted(groups.items()):
                out.extend(coverage_row(source,station,reader.canonical(source,station),
                    'DE' if source=='camels_de_1h' else 'IS',role,event,runs(qt),runs(pt),histories,future))
    return out


def p_runs(reader, source, station, roles):
    """Parse time first; stop before any calibration/reserved P value is decoded."""
    paired=source=='jp_jma_maebashi_mlit_kurihashi_2015_2024'
    path=reader.entry(source) if paired else reader.entry(source)/(station+'.csv.gz')
    time_field='time_jst' if paired else 'datetime_native'
    boundaries={role: [hour(v) if v else None for v in reader.role_bounds[source][role]] for role in roles}
    last=max(b for a,b in boundaries.values())
    values={role:[] for role in roles}
    previous=None
    for raw in csv_rows(path):
        t=hour(raw[time_field])
        if previous is not None and t<=previous:
            raise ValueError('non-increasing P source clock')
        previous=t
        if t>=last:
            break
        role=next((k for k,(a,b) in boundaries.items() if (a is None or t>=a) and t<b),None)
        if role is None:
            continue
        ok=truth(raw['P_usable']) and raw['P_quality']=='8' if paired else truth(raw['p_observed_available'])
        p=number(raw['P_mm'] if paired else raw['precipitation_mm_per_hour'])
        if ok and math.isfinite(p):
            values[role].append(t)
        # At the final allowed timestamp, do not even fetch the next CSV row.
        if t==last-1:
            break
    return {role:runs(t) for role,t in values.items()}


def new_coverage(reader,histories,future,roles,output):
    manifest=json.loads((reader.prepared_dir/'manifest.json').read_text())
    rows=[]; excluded=[]
    for index,item in enumerate(manifest['stations']):
        source,station=item['source'],item['station']
        try:
            table=reader.prepared_station(source,station)
        except ValueError as error:
            # NOT_EVALUABLE is an accepted preparation outcome, not invented coverage.
            if item.get('state')=='NOT_EVALUABLE':
                excluded.append(dict(source=source,station=station,state='NOT_EVALUABLE',reason=str(error)))
                continue
            raise
        routes=[route for route in ('pq','q_only') if table[route+'_pool_eligible']]
        if not routes:
            excluded.append(dict(source=source,station=station,state='superseded_in_both_routes'))
            continue
        pr=p_runs(reader,source,station,roles) if 'pq' in routes else {r:[] for r in roles}
        for role in roles:
            qr=merge([[s['start_hour'],s['end_hour']] for s in table['roles'][role]['valid_segments']])
            rows.extend(coverage_row(source,station,table['canonical_station'],reader.sources[source]['country_code'],
                role,'continuous',qr,pr[role],histories,future,routes=routes,
                threshold_eligible=bool(table['scale'].get('extreme_eligible'))))
        if index%100==0:
            save(output/'progress.json',dict(state='new_source_metadata',processed=index+1,total=len(manifest['stations']),rows=len(rows)))
        # Keep the scan bounded by one station, not 6,005 cached artifacts.
        reader._prepared.clear()
    return rows,excluded


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--current-entry',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--history-hours',nargs='+',type=int,default=[24,48,72])
    p.add_argument('--forecast-hours',type=int,default=24)
    p.add_argument('--roles',nargs='+',default=['fit','select'])
    p.add_argument('--cn-only',action='store_true')
    args=p.parse_args()
    if set(args.roles)-{'fit','select'} or min(args.history_hours+[args.forecast_hours])<1:
        raise PermissionError('positive windows and only fit/select roles permitted')
    assert eligible([[0,47]],[[0,23]],24,24,'pq')==[[23,23]]
    assert eligible([[0,46]],[[0,23]],24,24,'q_only')==[]
    assert eligible([[0,47]],[[0,22]],24,24,'pq')==[]
    assert intersect([[0,4],[8,10]],[[3,9]])==[[3,4],[8,9]]
    entry=json.loads(Path(args.current_entry).read_text());reader=RelocatedReader(**entry['reader_kwargs'])
    output=Path(args.output)
    cn=legacy_coverage(reader,args.history_hours,args.forecast_hours,args.roles,True)
    save(output/'cn_coverage.json',dict(state='complete',rows=cn,summary=summarize(cn)))
    print(json.dumps(dict(stage='CN_complete',summary=summarize(cn))),flush=True)
    rows=list(cn);excluded=[]
    if not args.cn_only:
        rows.extend(legacy_coverage(reader,args.history_hours,args.forecast_hours,args.roles,False))
        new,excluded=new_coverage(reader,args.history_hours,args.forecast_hours,args.roles,output);rows.extend(new)
    result=dict(state='complete',scope='CN_only' if args.cn_only else 'five_country_existing_fit_select',
        histories=args.history_hours,forecast_hours=args.forecast_hours,rows=rows,summary=summarize(rows),excluded=excluded,
        high_Q_definition='existing_fit_Q95; exact high-hour coverage calculated for CN only',
        quality='legacy numeric reconstructed; raw flags not retained; clocks are source-native hindcast',
        censored_process_policy='retained_for_valid_continuous_pointwise_samples',
        counts_are_independent_events=False,old_files_modified=False,access=reader.access)
    save(output/'coverage.json',result);save(output/'progress.json',dict(state='complete',rows=len(rows)))
    print(json.dumps(dict(stage='complete',summary=result['summary'],excluded=len(excluded))),flush=True)


if __name__=='__main__':
    main()
