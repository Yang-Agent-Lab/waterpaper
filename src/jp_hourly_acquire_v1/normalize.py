"""Normalize only symbols defined by JMA; retain original pages and flags."""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

from .__main__ import Collector, decode, existing_q

LEGEND = "https://www.data.jma.go.jp/stats/data/mdrr/man/remark.html"


def normalize(row):
    out=dict(row)
    if row['raw_P']=='--':
        out.update(P_mm=0.0,quality='no_precipitation_phenomenon_not_imputed')
    elif row['P_mm'] is not None:
        out['quality']='reported_trace_lt_0.5mm' if row['raw_P']=='0.0' else 'reported_numeric'
    else:
        out['quality']='missing_or_flagged_retained_unknown'
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--selfcheck',action='store_true')
    ap.add_argument('--metadata-from')
    args=ap.parse_args()
    assert normalize({'raw_P':'--','P_mm':None})['P_mm']==0.0
    assert normalize({'raw_P':'×','P_mm':None})['P_mm'] is None
    assert normalize({'raw_P':'0.0','P_mm':0.0})['quality']=='reported_trace_lt_0.5mm'
    if args.selfcheck:
        print('JMA no-phenomenon/missing distinction selfcheck passed')
        return
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    c=Collector(out)
    receipt={'host':socket.gethostname(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
             'status':'partial','source':args.source,'upstream_boundary':'unknown','training':False}
    try:
        metadata_root=Path(args.metadata_from) if args.metadata_from else None
        legend=decode((metadata_root/'jma_official_symbol_legend.raw').read_bytes())[0] if metadata_root else c.get(LEGEND,'jma_official_symbol_legend')
        if not legend or '--' not in legend or '該当現象' not in legend or '欠測' not in legend:
            raise ValueError('Official symbol definition not recovered; do not normalize --')
        if metadata_root:
            positions=decode((metadata_root/'jma_official_station_positions.raw').read_bytes())[0]
        else:
            c.get('https://www.data.jma.go.jp/stats/data/mdrr/man/nikkai.html','jma_official_clock')
            positions=c.get('https://www.jma.go.jp/bosai/amedas/const/amedastable.json','jma_official_station_positions')
        metadata=[]
        if positions:
            table=json.loads(positions)
            metadata=[{'amedas_id':key,**value} for key,value in table.items() if value.get('kjName')=='前橋']
        old=json.loads(Path(args.source).read_text())
        p=[normalize(row) for row in old]
        q,sources=existing_q(out)
        qmap={row['time_jst']:row for row in q}
        pairs=[{**row,**qmap[row['time_jst']]} for row in p if row['time_jst'] in qmap]
        valid=[row for row in pairs if row['P_mm'] is not None and row['Q_m3_s'] is not None]
        segments=[]
        for row in valid:
            if not segments or (datetime.fromisoformat(row['time_jst'])-datetime.fromisoformat(segments[-1][-1]['time_jst'])).total_seconds()!=3600:
                segments.append([])
            segments[-1].append(row)
        windows=[]
        for segment in segments:
            for i in range(len(segment)-17):
                windows.append({'history_start_jst':segment[i]['time_jst'],'origin_jst':segment[i+11]['time_jst'],
                                'future_Q_start_jst':segment[i+12]['time_jst'],'future_Q_end_jst':segment[i+17]['time_jst'],
                                'qualification':'point_P_only_spatial_support_partial'})
        with (out/'candidate_windows_H12_K6.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=['history_start_jst','origin_jst','future_Q_start_jst','future_Q_end_jst','qualification'])
            writer.writeheader()
            writer.writerows(windows)
        (out/'normalized_point_P.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
        (out/'point_P_Q_pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2))
        triple=[row for row in valid if row['time_jst'].startswith('2019-10-12T') and row['raw_hour'] in (10,11,12)]
        assert len(triple)==3 and all(datetime.fromisoformat(triple[i+1]['time_jst']).timestamp()-datetime.fromisoformat(triple[i]['time_jst']).timestamp()==3600 for i in (0,1))
        receipt.update(status='hourly_point_P_Q_acquired_spatial_support_partial',P_source='JMA Maebashi block47624 prec42',
                       Q_source='MLIT Kurihashi 303031283301030',P_unit='mm_per_reported_hour',Q_unit='m3/s',time_zone='JST',
                       current_station_metadata=metadata,historical_station_location_verified=False,
                       n_P_rows=len(p),n_Q_rows=len(q),n_numeric_Q=sum(row['Q_m3_s'] is not None for row in q),
                       n_same_clock_pairs=len(pairs),n_numeric_pairs=len(valid),raw_P_symbols=dict(Counter(row['raw_P'] for row in p)),
                       metadata_source=args.metadata_from,
                       usable_contiguous_segments=[{'start_jst':s[0]['time_jst'],'end_jst':s[-1]['time_jst'],'n_hours':len(s)} for s in segments],
                       n_candidate_H12_K6_windows=len(windows),first_pair_time=pairs[0]['time_jst'],last_pair_time=pairs[-1]['time_jst'],
                       quality_counts=dict(Counter(row['quality'] for row in p)),Q_chunk_sources=sources,
                       adjacent_hour_example=[{key:row[key] for key in ('time_jst','raw_P','P_mm','Q_m3_s','raw_Q','quality')} for row in triple],
                       source_legend=LEGEND,interpretation='Point rainfall only; temporal join is not a validated basin-rainfall pairing',
                       missing_not_filled=True,no_phenomenon_converted_by_official_definition=True,
                       P_interval_note='Hourly precipitation as published; raw hour and 24:00 retained. Exact accumulation-end documentation not independently extracted.',
                       limitations=['No basin polygon/area weights','Current position metadata not a 2019 relocation audit','No training or external test evaluation'])
    except Exception as error:
        receipt['error']=repr(error)
    finally:
        receipt.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),accounted_bytes=c.used,requests=c.requests)
        (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        print(json.dumps({key:value for key,value in receipt.items() if key not in ('requests','raw_P_symbols')},ensure_ascii=False))


if __name__=='__main__':
    main()
