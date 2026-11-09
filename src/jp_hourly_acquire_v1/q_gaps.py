"""Fixed one-station missing-clock manifest; never overwrite retained Q values."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import time

from .__main__ import Collector,decode,monthly_rows,numeric


def manifest(source):
    rows=list(csv.DictReader(source.open()))
    gaps=[row['time_jst'] for row in rows if row['raw_Q']=='not_in_existing_snapshot']
    intervals=[]
    for clock in gaps:
        if not intervals or (datetime.fromisoformat(clock)-datetime.fromisoformat(intervals[-1][-1])).total_seconds()!=3600:
            intervals.append([])
        intervals[-1].append(clock)
    months=Counter((datetime.fromisoformat(t)-timedelta(hours=1)).strftime('%Y-%m') for t in gaps)
    return {'Q_id':'303031283301030','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'n_unretrieved_hours':len(gaps),'missing_clocks':gaps,
            'intervals':[{'start':s[0],'end':s[-1],'n_hours':len(s)} for s in intervals],
            'months':[{'month':month,'n_unretrieved_hours':count} for month,count in sorted(months.items(),key=lambda item:(-item[1],item[0]))]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--source')
    ap.add_argument('--manifest')
    ap.add_argument('--selfcheck',action='store_true')
    ap.add_argument('--raw-cache')
    ap.add_argument('--additional-raw-cache')
    ap.add_argument('--cache-only',action='store_true')
    args=ap.parse_args()
    assert numeric('-') is None and numeric('12.3')==12.3
    assert len(monthly_rows('<tr><td>1</td>'+''.join('<td>2.0</td>' for _ in range(24))+'</tr>',2018,1))==24
    assert len(monthly_rows('<tr><td>2018/01/01</td>'+''.join('<td>2.0</td>' for _ in range(24))+'</tr>',2018,1))==24
    if args.selfcheck:
        print('monthly source/missing distinction selfcheck passed');return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    if args.source:
        plan=manifest(Path(args.source))
        (out/'manifest.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
        print(json.dumps({'n_unretrieved_hours':plan['n_unretrieved_hours'],'months':plan['months']},ensure_ascii=False));return
    plan=json.loads(Path(args.manifest).read_text());source=Path(plan['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_sha256']
    old_rows=list(csv.DictReader(source.open()));old={row['time_jst']:row for row in old_rows}
    wanted=set(plan['missing_clocks']);new={};conflicts=[];month_results=[]
    raw_cache=Path(args.raw_cache) if args.raw_cache else None
    extra_cache=Path(args.additional_raw_cache) if args.additional_raw_cache else None
    previous=json.loads((raw_cache/'receipt.json').read_text()) if raw_cache else None
    if extra_cache:
        previous=json.loads((extra_cache/'receipt.json').read_text())
    c=Collector(out);c.used=previous['accounted_bytes'] if previous else 9652146
    result={'host':socket.gethostname(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'partial','manifest':args.manifest,'Q_id':plan['Q_id'],'prior_accounted_bytes':c.used,'reused_raw_cache':args.raw_cache,
            'additional_raw_cache':args.additional_raw_cache,'cache_only':args.cache_only}
    try:
        root='https://www1.river.go.jp'
        if not raw_cache and not args.cache_only:
            c.get(root+'/cgi-bin/SrchWaterData.exe?ID=303031283301030&KIND=6&PAGE=0','official_flow_month_form',True)
        for item in plan['months']:
            month=item['month'];year,mon=map(int,month.split('-'))
            begin=datetime(year,mon,1)
            next_month=datetime(year+1,1,1) if mon==12 else datetime(year,mon+1,1)
            last=next_month-timedelta(days=1)
            suffix=f'/cgi-bin/DspWaterData.exe?KIND=6&ID=303031283301030&BGNDATE={begin:%Y%m%d}&ENDDATE={last:%Y%m%d}&KAWABOU=NO'
            text=None
            cached_file=None
            for folder in (raw_cache,extra_cache):
                if folder:
                    for candidate in folder.glob(f'Q_month_{month}_route*.raw'):
                        if candidate.stat().st_size:
                            text=decode(candidate.read_bytes())[0];cached_file=str(candidate);break
                if text: break
            if not text and not args.cache_only:
                for number,scheme in enumerate(('https','http'),1):
                    text=c.get(scheme+'://www1.river.go.jp'+suffix,f'Q_month_{month}_route{number}',True)
                    if text: break
            rows=monthly_rows(text,year,mon) if text else []
            # Exact monthly grid prevents accepting a shifted/non-hourly layout.
            expected=(next_month-begin).days*24
            plain=re.sub(r'<[^>]+>','',text or '')
            unit_confirmed=bool(re.search(r'(?:m\s*(?:3|\^3)|㎥)\s*/\s*(?:s|sec)',plain,re.I))
            credible=bool(text and plan['Q_id'] in text and '流量' in text and unit_confirmed and rows and len({r['time_jst'] for r in rows})==len(rows))
            valid=credible and len(rows)==expected
            month_results.append({**item,'n_rows':len(rows),'unit_m3_s_confirmed':unit_confirmed,'qualified_monthly_table':valid,'credible_dated_rows':credible,'qualification_reason':'verified station/unit/date and 24 cells per retained day; incomplete month remains partial' if credible else 'no verified dated rows','cached_HTML':cached_file})
            if credible:
                (out/f'parsed_Q_{month}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
                for row in rows:
                    clock=row['time_jst'];value=row['value']
                    if clock in wanted:
                        new[clock]={**row,'new_source':str(out/f'parsed_Q_{month}.json')}
                    elif clock in old:
                        existing=numeric(old[clock]['raw_Q'])
                        if existing is not None and value is not None and abs(existing-value)>1e-6:
                            conflicts.append({'time_jst':clock,'retained_Q':existing,'new_Q':value,'new_raw_Q':row['raw_value'],'month':month})
            if c.deadline<=time.monotonic():
                result['stop_reason']='600s collector limit; remaining manifest months not requested';break
        conflict_times={row['time_jst'] for row in conflicts}
        output=[]
        for row in old_rows:
            clock=row['time_jst'];updated=dict(row)
            updated['Q_acquisition_state']='retained_snapshot'
            if clock in new:
                data=new[clock]
                updated.update(Q_m3_s=data['value'],raw_Q=data['raw_value'],Q_source=data['new_source'],Q_conflict=False,
                               Q_acquisition_state='newly_retrieved_numeric' if data['value'] is not None else 'newly_retrieved_source_missing_marker')
            elif clock in wanted:
                updated['Q_acquisition_state']='still_unretrieved'
            if clock in conflict_times:
                updated['Q_conflict']=True
            output.append(updated)
        columns=list(old_rows[0])+['Q_acquisition_state']
        with (out/'point_P_Q_after_Q_gap_acquisition.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(output)
        (out/'duplicate_value_conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2))
        yearly={};segments=[]
        for row in output:
            label=(datetime.fromisoformat(row['time_jst'])-timedelta(hours=1)).year
            count=yearly.setdefault(str(label),Counter())
            missing=row['Q_acquisition_state']=='still_unretrieved';qval=numeric(str(row['Q_m3_s']))
            count['P_hours']+=1;count['Q_retrieved_hours']+=int(not missing);count['Q_still_unretrieved_hours']+=int(missing)
            count['Q_source_missing_or_flagged_hours']+=int(not missing and qval is None)
            count['Q_conflict_hours']+=int(row['time_jst'] in conflict_times)
            good=str(row['P_usable'])=='True' and qval is not None and qval>=0 and row['time_jst'] not in conflict_times
            count['usable_point_P_Q_hours']+=int(good)
            if good:
                if not segments or (datetime.fromisoformat(row['time_jst'])-datetime.fromisoformat(segments[-1][-1])).total_seconds()!=3600:
                    segments.append([])
                segments[-1].append(row['time_jst'])
        (out/'continuous_usable_segments.json').write_text(json.dumps([{'start':s[0],'end':s[-1],'hours':len(s)} for s in segments],indent=2))
        result.update(status='bounded_Q_gap_routes_attempted',months=month_results,n_newly_retrieved_clocks=len(new),
                      n_new_numeric_Q=sum(r['value'] is not None for r in new.values()),n_still_unretrieved=len(wanted-new.keys()),
                      n_duplicate_value_conflicts=len(conflicts),yearly_coverage=yearly,
                      original_missing_Q_unchanged=True,n_contiguous_segments=len(segments),longest_contiguous_hours=max(map(len,segments),default=0))
    except Exception as error:
        result['error']=repr(error)
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),requests=c.requests,accounted_bytes=c.used)
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='requests'},ensure_ascii=False))


if __name__=='__main__':
    main()
