"""Official hourly CSV: validate one year, then the same station's approved span."""
import argparse
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import re
import socket
import time

from .__main__ import Collector,JST


def csv_rows(text):
    rows=[]
    for fields in csv.reader(io.StringIO(text)):
        if not fields or not re.fullmatch(r'\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}(?::\d{2})?',fields[0].strip()):
            continue
        date,clock=fields[0].strip().split(' ')
        yy,mm,dd=map(int,date.split('/'))
        hh,minute=map(int,clock.split(':')[:2])
        if minute!=0 or not 0<=hh<=24:
            raise ValueError('Not an hourly timestamp')
        stamp=datetime(yy,mm,dd,tzinfo=JST)+timedelta(hours=hh)
        rows.append({'time_jst':stamp.isoformat(),'raw_fields':fields})
    return rows


def params(start,end):
    y,m,d=start.split('-'); Y,M,D=end.split('-')
    return {'stationNumList':'["s47624"]','aggrgPeriod':'9','elementNumList':'[["101",""]]','interAnnualType':'1',
            'ymdList':json.dumps([y,Y,m,M,d,D]),'option':'{"list":["obs"],"nyear":0}',
            'downloadFlag':'true','rmkFlag':'1','disconnectFlag':'1','youbiFlag':'0','fukenFlag':'0',
            'kijiFlag':'0','csvFlag':'1','jikantaiFlag':'0','jikantaiList':'[]','ymdLiteral':'1'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--selfcheck',action='store_true')
    args=ap.parse_args()
    assert csv_rows('2018/12/31 24:00:00,0,8,1')[0]['time_jst']=='2019-01-01T00:00:00+09:00'
    assert params('2018-01-01','2018-12-31')['aggrgPeriod']=='9'
    if args.selfcheck:
        print('official hourly CSV contract/parser selfcheck passed')
        return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    c=Collector(out)
    c.used=6953033  # Prior Japanese acquisition, metadata, and client-discovery receipts.
    result={'host':socket.gethostname(),'status':'partial','prior_accounted_bytes':c.used,'station':'前橋 s47624',
            'scope':'2015-2024_hourly_point_rainfall','requests_contract':'official top.bundle.js: hourly=9, precipitation=101, POST show/table',
            'periods':[],'existing_2019_october_reused':True,'training':False}
    try:
        root='https://www.data.jma.go.jp/risk/obsdl/'
        helper=c.get(root+'web/dist/whatsnewHistory.js','official_api_helper')
        if not helper or '/risk/obsdl/' not in helper:
            raise ValueError('Official API base could not be confirmed')
        c.get(root+'?vm=r','official_session_bootstrap',True)
        periods=[('2018-01-01','2018-12-31')]
        periods += [(f'{year}-01-01',f'{year}-12-31') for year in (2015,2016,2017)]
        periods += [('2019-01-01','2019-09-30'),('2019-11-01','2019-12-31')]
        periods += [(f'{year}-01-01',f'{year}-12-31') for year in range(2020,2025)]
        for index,(start,end) in enumerate(periods):
            payload=params(start,end)
            text=c.get(root+'show/table','hourly_P_'+start+'_'+end,True,data=payload)
            expected=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).days*24+24
            rows=csv_rows(text) if text else []
            valid=bool(text and '前橋' in text and '降水量' in text and len(rows)==expected and len({r['time_jst'] for r in rows})==expected)
            if valid:
                valid=all((datetime.fromisoformat(b['time_jst'])-datetime.fromisoformat(a['time_jst'])).total_seconds()==3600 for a,b in zip(rows,rows[1:]))
            summary={'start':start,'end':end,'expected_hourly_rows':expected,'n_rows':len(rows),'qualified':valid,
                     'request_fields':payload,'first_three':rows[:3],'last':rows[-1:]}
            result['periods'].append(summary)
            if not valid:
                result['stop_reason']='CSV response failed station/header/hourly/count/continuity check; no more years requested'
                break
            (out/('parsed_'+start+'_'+end+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            if index==0:
                result['first_year_qualified']=True
            time.sleep(2)
        if len(result['periods'])==len(periods) and all(p['qualified'] for p in result['periods']):
            result['status']='hourly_point_P_span_acquired_Q_pairing_and_spatial_support_pending'
    except Exception as error:
        result['error']=repr(error)
    finally:
        result.update(accounted_bytes=c.used,requests=c.requests)
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({key:value for key,value in result.items() if key!='requests'},ensure_ascii=False))


if __name__=='__main__':
    main()
