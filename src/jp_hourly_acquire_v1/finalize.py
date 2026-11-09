"""Uniform official CSV P quality, retaining all prior Q values and conflicts."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import socket

from .__main__ import Collector,numeric
from .annual_csv import csv_rows,params


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--q-output',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--october-csv-cache')
    ap.add_argument('--accounted-bytes',type=int)
    ap.add_argument('--selfcheck',action='store_true')
    args=ap.parse_args()
    assert len(csv_rows('2019/10/31 24:00:00,0,1,8,1'))==1
    if args.selfcheck:
        print('final CSV quality contract selfcheck passed');return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    q_output=Path(args.q_output)
    prior=json.loads((q_output/'receipt.json').read_text())
    c=Collector(out);c.used=args.accounted_bytes or prior['accounted_bytes']
    result={'host':socket.gethostname(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'partial',
            'Q_gap_source':str(q_output),'upstream_boundary':'unknown','training':False}
    try:
        if args.october_csv_cache:
            from .__main__ import decode
            cached=Path(args.october_csv_cache).resolve(strict=True)
            text=decode(cached.read_bytes())[0]
            result['reused_official_October_CSV']=str(cached)
            october_source=cached
        else:
            text=c.get('https://www.data.jma.go.jp/risk/obsdl/show/table','official_2019_october_uniform_CSV',True,
                       data=params('2019-10-01','2019-10-31'))
            october_source=(out/'official_2019_october_uniform_CSV.raw').resolve()
        month=csv_rows(text) if text else []
        if len(month)!=744 or not text or '前橋' not in text or '降水量' not in text:
            raise ValueError('October official CSV did not validate; HTML quality remains partial')
        lookup={row['time_jst']:row for row in month}
        assert len(lookup)==744
        old=list(csv.DictReader((q_output/'point_P_Q_after_Q_gap_acquisition.csv').open()))
        diffs=[];rows=[]
        for row in old:
            record=dict(row)
            if row['time_jst'] in lookup:
                fields=lookup[row['time_jst']]['raw_fields'];assert len(fields)==5
                value=numeric(fields[1]);before=numeric(row['P_mm'])
                if value!=before:
                    diffs.append({'time_jst':row['time_jst'],'cached_HTML_P':before,'new_CSV_P':value})
                record.update(P_mm=value,raw_P=fields[1],P_no_phenomenon=fields[2],P_quality=fields[3],P_homogeneity=fields[4],
                              P_usable=value is not None and value>=0 and fields[3]=='8',
                              P_source=str(october_source))
            rows.append(record)
        (out/'HTML_CSV_numeric_differences.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2))
        dataset=out/'point_P_Q_2015_2024_uniform_CSV_quality.csv'
        with dataset.open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(old[0]));writer.writeheader();writer.writerows(rows)
        years={};segments=[];missing=[];conflicts=[]
        for row in rows:
            year=str((datetime.fromisoformat(row['time_jst'])-timedelta(hours=1)).year)
            count=years.setdefault(year,Counter())
            q=numeric(str(row['Q_m3_s']));quality=row['P_quality'];Pgood=quality=='8' and numeric(str(row['P_mm'])) is not None
            absent=row['Q_acquisition_state']=='still_unretrieved';conflict=str(row['Q_conflict']).lower()=='true'
            count['P_hours']+=1;count['P_quality8_hours']+=int(Pgood)
            count['Q_retrieved_hours']+=int(not absent);count['Q_numeric_hours']+=int(q is not None)
            count['Q_unretrieved_hours']+=int(absent);count['Q_source_missing_or_flagged_hours']+=int(not absent and q is None)
            count['Q_conflict_hours']+=int(conflict)
            good=Pgood and q is not None and q>=0 and not conflict
            count['usable_point_P_Q_hours']+=int(good)
            if absent: missing.append(row['time_jst'])
            if conflict: conflicts.append(row['time_jst'])
            if good:
                if not segments or (datetime.fromisoformat(row['time_jst'])-datetime.fromisoformat(segments[-1][-1])).total_seconds()!=3600:
                    segments.append([])
                segments[-1].append(row['time_jst'])
        summary=[{'start':s[0],'end':s[-1],'n_hours':len(s),'H12_K6_count':max(0,len(s)-17)} for s in segments]
        (out/'continuous_usable_segments.json').write_text(json.dumps(summary,indent=2))
        with (out/'candidate_windows_H12_K6.csv').open('w') as file:
            writer=csv.writer(file);writer.writerow(['history_start_jst','origin_jst','future_Q_start_jst','future_Q_end_jst','qualification'])
            for s in segments:
                for i in range(len(s)-17):
                    writer.writerow([s[i],s[i+11],s[i+12],s[i+17],'point_P_spatial_support_partial'])
        artifact_hashes={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in (dataset,out/'continuous_usable_segments.json',out/'candidate_windows_H12_K6.csv',out/'HTML_CSV_numeric_differences.json')}
        result.update(status='multiyear_point_P_Q_uniform_CSV_quality_saved_spatial_partial',n_rows=len(rows),
            n_HTML_CSV_numeric_differences=len(diffs),october_CSV_quality_counts=dict(Counter(row['raw_fields'][3] for row in month)),
            P_quality_counts=dict(Counter(row['P_quality'] for row in rows)),yearly_coverage=years,
            n_usable_point_pairs=sum(y['usable_point_P_Q_hours'] for y in years.values()),
            n_Q_unretrieved=len(missing),n_Q_conflict=len(conflicts),remaining_Q_unretrieved_times=missing,
            n_contiguous_segments=len(segments),longest_contiguous_hours=max(map(len,segments),default=0),
            n_candidate_H12_K6_windows=sum(s['H12_K6_count'] for s in summary),artifact_sha256=artifact_hashes,
            qualifications='Only CSV quality8 P + finite nonnegative Q, excluding conflicts; candidate identities only',
            limitations=['Point rain gauge, not catchment-area precipitation','Missing Q not interpolated','Original 41 missing Q rows retained',
                         'JMA homogeneity numbers restart at 1 per CSV selection; retain P_source and raw number, not a global station-homogeneity label','No training/test scoring'])
    except Exception as error:
        result['error']=repr(error)
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),accounted_bytes=c.used,requests=c.requests)
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('requests','remaining_Q_unretrieved_times')},ensure_ascii=False))


if __name__=='__main__':
    main()
