"""Build per-target Chinese forecast windows with training-only transforms."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import statistics
import sys


def quantile(values, p):
    a=sorted(values); at=(len(a)-1)*p; i=int(at)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(at-i)


def segments(rows):
    current=[]
    for r in sorted(rows, key=lambda r:(r['time_axis'],int(r['source_row']))):
        valid=r['pq_numeric_valid']=='1'
        contiguous=current and (datetime.fromisoformat(r['time_axis'])-
            datetime.fromisoformat(current[-1]['time_axis'])).total_seconds()==3600
        if not valid or not contiguous:
            if current:
                yield current
            current=[]
        if valid:
            current.append(r)
    if current:
        yield current


def main():
    root=Path('.');cat=root/'data/catalog'
    config=json.loads((root/'configs/china_first_flow_v1.json').read_text())
    out=root/'data/processed/china_first_flow_v1'
    assert not out.exists(), 'Preserve existing dataset; inspect before a new version.'
    assert config['history_hours']==config['forecast_sequence_hours']==12
    manifest=list(csv.DictReader((cat/'china_per_target_split_candidate_v1.csv').open()))
    split_by={r['event_uid']:r['candidate_split'] for r in manifest}
    assert len(split_by)==150
    stations=list(csv.DictReader((cat/'integrated_station_index_v2.csv').open()))
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    results=[]
    for station in stations:
        if station['country']!='China':continue
        uid=station['station_uid'];events=defaultdict(list)
        with gzip.open(station['observation_file'],'rt') as f:
            for r in csv.DictReader(f):
                assert r['station_uid']==uid
                events[uid+':'+r['source_event_id']].append(r)
        overlap_roles=defaultdict(set)
        for eid,rows in events.items():
            for r in rows:overlap_roles[r['overlap_group_id']].add(split_by[eid])
        assert all(len(x)==1 for x in overlap_roles.values())
        fit=[r for eid,rows in events.items() if split_by[eid]=='train'
             for r in rows if r['pq_numeric_valid']=='1']
        scales={}
        for var in ['p_mm_h','q_m3_s']:
            vals=[float(r[var]) for r in fit]
            mean=statistics.mean(vals);std=statistics.pstdev(vals)
            scales[var]=dict(mean=mean,scale=std if std else 1.,constant=std==0)
        q=[float(r['q_m3_s']) for r in fit]
        q95=quantile(q,.95);q99=quantile(q,.99)
        folder=out/station['station_id'];folder.mkdir()
        metadata=dict(station_uid=uid,fit_rows=len(fit),q95_m3_s=q95,q99_m3_s=q99,
                      fit_event_ids=sorted(eid for eid in events if split_by[eid]=='train'),
                      scales=scales,timezone='unknown',task='event_conditioned_high_flow')
        (folder/'training_transforms.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        split_stats={}
        for role in ['train','validation','test']:
            counts=Counter();hours=Counter();event_count=0
            target=folder/(role+'.jsonl.gz')
            with gzip.open(target,'wt') as f:
                for eid,rows in events.items():
                    if split_by[eid]!=role:continue
                    event_count+=1
                    for r in rows:
                        if r['pq_numeric_valid']=='1':
                            hours['valid_pq_hours']+=1
                            hours['q95_hours']+=float(r['q_m3_s'])>q95
                            hours['q99_hours']+=float(r['q_m3_s'])>q99
                    for run in segments(rows):
                        for i in range(len(run)-24+1):
                            block=run[i:i+24]
                            p=[float(r['p_mm_h']) for r in block]
                            q=[float(r['q_m3_s']) for r in block]
                            pn=[(v-scales['p_mm_h']['mean'])/scales['p_mm_h']['scale'] for v in p]
                            qn=[(v-scales['q_m3_s']['mean'])/scales['q_m3_s']['scale'] for v in q]
                            row=dict(event_uid=eid,origin_source_time=block[11]['time_axis'],
                                source_rows=[r['source_row'] for r in block],
                                x_p=pn[:12],x_q=qn[:12],y_q=qn[12:],
                                offline_generation_future_p=pn[12:],
                                high95_hours=[v>q95 for v in q[12:]],
                                high99_hours=[v>q99 for v in q[12:]],
                                generation_high95=any(v>q95 for v in q[12:]))
                            f.write(json.dumps(row,separators=(',',':'))+'\n')
                            counts['windows']+=1
                            counts['high95_windows']+=row['generation_high95']
                            counts['high99_windows']+=any(row['high99_hours'])
            reread=Counter()
            with gzip.open(target,'rt') as f:
                for line in f:
                    r=json.loads(line)
                    assert split_by[r['event_uid']]==role
                    assert all(len(r[k])==12 for k in ['x_p','x_q','y_q','offline_generation_future_p','high95_hours','high99_hours'])
                    assert all(math.isfinite(v) for k in ['x_p','x_q','y_q'] for v in r[k])
                    assert r['generation_high95']==any(r['high95_hours'])
                    reread['windows']+=1
                    reread['high95_windows']+=r['generation_high95']
                    reread['high99_windows']+=any(r['high99_hours'])
            assert counts==reread
            split_stats[role]=dict(events=event_count,**counts,**hours,
                q95_hour_fraction=hours['q95_hours']/hours['valid_pq_hours'],
                q99_hour_fraction=hours['q99_hours']/hours['valid_pq_hours'],
                high95_window_fraction=counts['high95_windows']/counts['windows'],
                output=str(target),output_bytes=target.stat().st_size,readback_verified=True)
        results.append(dict(station_uid=uid,**{k:metadata[k] for k in ['fit_rows','q95_m3_s','q99_m3_s']},splits=split_stats))
    total=sum(s['windows'] for r in results for s in r['splits'].values())
    assert total==15984,total
    report=dict(state='chinese_windows_prepared_models_not_trained',
        checked_at=datetime.now(timezone.utc).isoformat(),config=config,targets=results,
        total_windows=total,source_event_count=150,
        limitations=['Overlapping windows are not independent floods',
                    'Q95/Q99 are fitted within selected training flood episodes, not continuous annual flow',
                    'Chinese timezone and province mappings unresolved; no cross-source timestamp joins',
                    'International training samples and training budget not yet prepared'],
        output_directory=str(out))
    with (cat/'china_windows_preparation_v1.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(state=report['state'],total_windows=total,targets=len(results))))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':
        assert quantile([0,10,20,30,40],.95)==38
        assert quantile([1],.99)==1
        r=[dict(time_axis=f'2000-01-01T{i:02d}:00:00',source_row=str(i),pq_numeric_valid=v)
           for i,v in [(0,'1'),(1,'1'),(2,'0'),(3,'1'),(5,'1')]]
        assert [len(x) for x in segments(r)]==[2,1,1]
        print('passed: linear quantile; invalid rows and gaps break windows')
    else:main()
