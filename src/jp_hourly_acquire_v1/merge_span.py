"""Join one acquired point-rainfall span to one existing discharge station."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import re
import socket

from .__main__ import Collector,JST,numeric


def q_clock(text):
    if 'T24:' in text:
        return (datetime.fromisoformat(text.split('T')[0]).replace(tzinfo=JST)+timedelta(days=1)).isoformat()
    return datetime.fromisoformat(text).isoformat()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--annual',required=True)
    ap.add_argument('--october',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--selfcheck',action='store_true')
    args=ap.parse_args()
    assert q_clock('2019-10-31T24:00:00+09:00')=='2019-11-01T00:00:00+09:00'
    assert numeric('-') is None
    if args.selfcheck:
        print('single-station clock and missing Q selfcheck passed')
        return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    annual=Path(args.annual);october=Path(args.october)
    prior=json.loads((annual/'receipt.json').read_text())
    c=Collector(out);c.used=prior['accounted_bytes']
    result={'host':socket.gethostname(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'partial','P_station':'JMA 前橋 s47624','Q_station':'MLIT 栗橋 303031283301030',
            'upstream_boundary':'unknown','P_role':'point_observation_not_basin_average','training':False}
    try:
        help_text=c.get('https://www.data.jma.go.jp/risk/obsdl/top/help3.html','official_CSV_quality_spec')
        if not help_text or '品質情報' not in help_text or '現象なし' not in help_text:
            raise ValueError('Official CSV quality description not recovered')
        p={}
        for file in sorted(annual.glob('parsed_*.json')):
            for row in json.loads(file.read_text()):
                fields=row['raw_fields']
                assert len(fields)==5
                val=numeric(fields[1]);quality=fields[3]
                p[row['time_jst']]={'time_jst':row['time_jst'],'P_mm':val,'raw_P':fields[1],
                    'P_no_phenomenon':fields[2],'P_quality':quality,'P_homogeneity':fields[4],
                    'P_usable':val is not None and quality=='8','P_source':str(file)}
        for row in json.loads((october/'normalized_point_P.json').read_text()):
            assert row['time_jst'] not in p
            p[row['time_jst']]={'time_jst':row['time_jst'],'P_mm':row['P_mm'],'raw_P':row['raw_P'],
                'P_no_phenomenon':'1' if row['raw_P']=='--' else '0','P_quality':'html_unflagged',
                'P_homogeneity':'unknown_from_HTML','P_usable':row['P_mm'] is not None,'P_source':str(october/'normalized_point_P.json')}
        # Identity and group numbering come from the retained original launch manifest.
        catalog=Path('./data/catalog')
        launch=json.loads((catalog/'japan_tone_bulk_launch_20261013.json').read_text())
        ids=(catalog/'japan_tone_Q_ids_20261013.txt').read_text().splitlines()
        group_size=launch['params']['group'];ngroups=(len(ids)+group_size-1)//group_size
        group=ids.index('303031283301030:0202')//group_size
        chunks=Path('./data/raw/japan_mlit_tone_Q_2015_2024/chunks')
        q={};sources=[];header_only=0;conflicts=0
        for file in sorted(chunks.glob('*.csv')):
            match=re.search(r'_g(\d+)\.csv$',file.name)
            if not match or int(match.group(1))%ngroups!=group:
                continue
            with file.open() as source:
                reader=csv.DictReader(source)
                if not reader.fieldnames or '栗橋' not in reader.fieldnames:
                    header_only+=1
                    continue
                assert reader.fieldnames.count('栗橋')==1
                sources.append(str(file))
                for row in reader:
                    clock=q_clock(row[reader.fieldnames[0]])
                    if not '2015-01-01T00:00:00+09:00'<clock<='2025-01-01T00:00:00+09:00':
                        continue
                    value=row['栗橋'].strip()
                    if clock in q and q[clock]['raw_Q']!=value:
                        conflicts+=1
                        q[clock]['Q_m3_s']=None
                        q[clock]['Q_conflict']=True
                    else:
                        q[clock]={'Q_m3_s':numeric(value),'raw_Q':value,'Q_source':str(file),'Q_conflict':False}
        fields=['time_jst','P_mm','raw_P','P_no_phenomenon','P_quality','P_homogeneity','P_usable','P_source','Q_m3_s','raw_Q','Q_source','Q_conflict']
        joined=[]
        with (out/'point_P_Q_2015_2024.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
            for clock,row in sorted(p.items()):
                record={**row,**q.get(clock,{'Q_m3_s':None,'raw_Q':'not_in_existing_snapshot','Q_source':'unknown','Q_conflict':False})}
                writer.writerow(record);joined.append(record)
        valid=[row for row in joined if row['P_usable'] and row['Q_m3_s'] is not None and row['Q_m3_s']>=0]
        yearly={};monthly={}
        for row in joined:
            # Source days contain hours 1..24; midnight belongs to the preceding source day.
            day=datetime.fromisoformat(row['time_jst'])-timedelta(hours=1)
            for key,table in ((str(day.year),yearly),(day.strftime('%Y-%m'),monthly)):
                counters=table.setdefault(key,Counter())
                counters['P_hours']+=1
                counters['P_usable_hours']+=int(row['P_usable'])
                counters['Q_snapshot_hours']+=int(row['time_jst'] in q)
                counters['Q_numeric_hours']+=int(row['Q_m3_s'] is not None)
                counters['Q_not_retrieved_hours']+=int(row['time_jst'] not in q)
                counters['Q_snapshot_missing_or_flagged_hours']+=int(row['time_jst'] in q and row['Q_m3_s'] is None)
                counters['usable_point_pairs']+=int(row['P_usable'] and row['Q_m3_s'] is not None and row['Q_m3_s']>=0)
        missing_months=[{'Q_id':'303031283301030','month':month,'status':'no_Q_rows_retrieved_in_existing_snapshot',**dict(counts)} for month,counts in monthly.items() if counts['Q_snapshot_hours']==0]
        (out/'Q_missing_month_manifest.json').write_text(json.dumps(missing_months,ensure_ascii=False,indent=2))
        (out/'monthly_coverage.json').write_text(json.dumps(monthly,ensure_ascii=False,indent=2))
        segments=[]
        for row in valid:
            if not segments or (datetime.fromisoformat(row['time_jst'])-datetime.fromisoformat(segments[-1][-1]['time_jst'])).total_seconds()!=3600:
                segments.append([])
            segments[-1].append(row)
        segment_summary=[{'start_jst':s[0]['time_jst'],'end_jst':s[-1]['time_jst'],'n_hours':len(s),'H12_K6_candidate_count':max(0,len(s)-17)} for s in segments]
        (out/'continuous_usable_segments.json').write_text(json.dumps(segment_summary,ensure_ascii=False,indent=2))
        with (out/'candidate_windows_H12_K6.csv').open('w') as file:
            writer=csv.writer(file);writer.writerow(['history_start_jst','origin_jst','future_Q_start_jst','future_Q_end_jst','qualification'])
            for s in segments:
                for i in range(len(s)-17):
                    writer.writerow([s[i]['time_jst'],s[i+11]['time_jst'],s[i+12]['time_jst'],s[i+17]['time_jst'],'point_P_spatial_support_partial'])
        result.update(status='point_hourly_span_saved_spatial_support_partial',n_P_rows=len(p),n_expected_2015_2024_hours=87672,
            n_existing_Q_rows=len(q),n_numeric_Q=sum(r['Q_m3_s'] is not None for r in q.values()),
            n_same_clock_pairs=sum(clock in q for clock in p),n_usable_pairs=len(valid),Q_conflicts=conflicts,
            yearly_coverage=yearly,n_Q_completely_unretrieved_months=len(missing_months),
            Q_missingness_note='Absent Q rows/header-only cache mean acquisition incomplete or source unavailability unknown. Literal missing markers are source-snapshot missing, not proof current publisher has no observation.',
            Q_group_index=group,Q_group_count=ngroups,Q_header_only_chunks=header_only,Q_source_files=sources,
            P_quality_counts=dict(Counter(r['P_quality'] for r in joined)),P_no_phenomenon_counts=dict(Counter(r['P_no_phenomenon'] for r in joined)),
            n_contiguous_segments=len(segments),longest_contiguous_hours=max(map(len,segments),default=0),
            n_H12_K6_candidate_windows=sum(r['H12_K6_candidate_count'] for r in segment_summary),
            first_time=joined[0]['time_jst'],last_time=joined[-1]['time_jst'],
            interval_semantics='P previous-hour total, official hourly=9/P=101 UI label 前1時間; Q as published hourly m3/s',
            qualification_rule='CSV P quality8 only; cached HTML unflagged numeric/no-phenomenon; finite nonnegative Q; consecutive 1h',
            limitations=['One point rain gauge, no basin area weighting','Q snapshot gaps and conflicts retained','No model tensor generation/training/test scoring'])
    except Exception as error:
        result['error']=repr(error)
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),accounted_bytes=c.used,requests=c.requests)
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({key:value for key,value in result.items() if key not in ('requests','Q_source_files')},ensure_ascii=False))


if __name__=='__main__':
    main()
