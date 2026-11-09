"""Independent bounded source-value review of four permitted legacy blocks."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--permission',required=True);p.add_argument('--smoke',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();permission_path=Path(a.permission);raw=permission_path.read_bytes()
    permission=json.loads(raw);smoke=json.loads(Path(a.smoke).read_text())
    digest=hashlib.sha256(raw).hexdigest()
    if smoke['permission_sha256']!=digest or smoke['state']!='complete':raise ValueError('smoke identity/state differs')
    authorized={(b['station_uid'],b['event'],b['role']):b for b in permission['blocks']}
    checks=[];numerical=0
    for entry in smoke['rows']:
        b=entry['permission'];key=(b['station_uid'],b['event'],b['role'])
        if authorized[key]!=b or b['original_role']!='train' or b['role'] not in ('fit','select'):raise ValueError('unauthorized block')
        points=entry['points'];wanted={r['source_row']:r for r in points}
        if sorted(wanted)!=list(range(b['start_source_row'],b['end_source_row']+1)):raise ValueError('missing source rows')
        seen=set()
        with gzip.open(b['physical_observation_file'],'rt',newline='') as f:
            for index,row in enumerate(csv.DictReader(f)):
                number=index+2
                if number in wanted:
                    if int(row['source_row'])!=number:raise ValueError('source row differs')
                    saved=wanted[number]
                    for original,field in [('q_m3_s','q'),('p_mm_h','p')]:
                        actual=float(row[original]);stored=saved[field]
                        if not (actual==stored or math.isnan(actual) and math.isnan(stored)):raise ValueError('raw value differs')
                        numerical+=1
                    if saved['interval_end_hour']!=saved['native_hour']+b['interval_end_shift_hours']:raise ValueError('wrong interval end')
                    seen.add(number)
                if number>=b['end_source_row']:break
        if seen!=set(wanted):raise ValueError('source values missing')
        expected={}
        for route in ('pq','q_only'):
            origins=0
            for i in range(23,len(points)-24):
                history=points[i-23:i+1];future=points[i+1:i+25];window=history+future
                if any(not r['q_mask'] for r in window):continue
                if route=='pq' and any(not r['p_mask'] for r in history):continue
                if any(r['interval_end_hour']!=window[0]['interval_end_hour']+j for j,r in enumerate(window)):continue
                origins+=1
            actual=next(r for r in smoke['coverage'] if (r['source'],r['station'],r['event'],r['role'],r['route'])==(b['source'],b['station'],b['event'],b['role'],route))
            if actual['origin_count']!=origins:raise ValueError('window count differs')
            expected[route]=origins
        checks.append(dict(station=b['station_uid'],role=b['role'],event=b['event'],points=len(points),origins=expected))
    if len(checks)!=4 or {c['role'] for c in checks}!={'fit','select'}:raise ValueError('four-block scope incomplete')
    result=dict(state='accepted_bounded_source_check',permission_sha256=digest,numerical_values_checked=numerical,
        checks=checks,scope='four declared DE/IS blocks only; not whole dataset acceptance',
        old_files_modified=False,model_training=False,forbidden_numeric_values_read=False)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
