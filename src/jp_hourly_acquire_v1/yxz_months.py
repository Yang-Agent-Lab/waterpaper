"""Alternate server network, only months without valid cached official tables."""
import argparse
from datetime import datetime,timedelta,timezone
import http.client
import json
import math
from pathlib import Path
import signal
import socket
import time
import urllib.request


def deadline(*_):
    raise TimeoutError('75s wall deadline')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--cache-receipt',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    plan=json.loads(Path(args.manifest).read_text());cached=json.loads(Path(args.cache_receipt).read_text())
    skip={m['month'] for m in cached['months'] if m['qualified_monthly_table']}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    used=cached['accounted_bytes'];end=time.monotonic()+600
    result={'host':socket.gethostname(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'bounded_alternate_network_attempt','prior_accounted_bytes':used,'skipped_cached_months':sorted(skip),
            'requests':[],'Q_id':'303031283301030','GPU_used':False,'dependencies_installed':False}
    try:
        for item in plan['months']:
            month=item['month']
            if month in skip: continue
            year,mon=map(int,month.split('-'));assert 2015<=year<=2024 and 1<=mon<=12
            begin=datetime(year,mon,1);nextmonth=datetime(year+1,1,1) if mon==12 else datetime(year,mon+1,1)
            last=nextmonth-timedelta(days=1)
            for route,scheme in enumerate(('https','http'),1):
                left=int(end-time.monotonic())
                if left<=0 or used+16384>=100*1024*1024: break
                url=f'{scheme}://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=6&ID=303031283301030&BGNDATE={begin:%Y%m%d}&ENDDATE={last:%Y%m%d}&KAWABOU=NO'
                record={'url':url,'month':month,'route':route};raw=bytearray();response=None;complete=False
                result['requests'].append(record);started=time.monotonic();used+=16384
                try:
                    signal.signal(signal.SIGALRM,deadline);signal.alarm(min(75,left))
                    response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=15)
                    record.update(status=response.status,headers={k:v for k,v in response.headers.items() if k.lower()!='set-cookie'})
                    limit=min(2*1024*1024,100*1024*1024-used)
                    if response.headers.get('Content-Length') and int(response.headers['Content-Length'])>limit:
                        raise ValueError('Monthly response exceeds remaining cap')
                    while len(raw)<limit:
                        part=response.read(min(8192,limit-len(raw)))
                        if not part:
                            complete=response.status==200;break
                        raw.extend(part);used+=len(part)
                except http.client.IncompleteRead as error:
                    raw.extend(error.partial or b'');used+=len(error.partial or b'');record['error']=repr(error)
                except Exception as error:
                    record['error']=repr(error)
                finally:
                    signal.alarm(0)
                    if response: response.close()
                    path=out/f'Q_month_{month}_route{route}.{ "raw" if complete else "partial"}'
                    if raw: path.write_bytes(raw)
                    record.update(payload_bytes=len(raw),file=str(path),complete=complete,elapsed_s=round(time.monotonic()-started,3))
                if complete and raw: break
                time.sleep(1)
            if time.monotonic()>=end:
                result['stop_reason']='600s alternate collector limit';break
    finally:
        result.update(accounted_bytes=used,finished_at_utc=datetime.now(timezone.utc).isoformat())
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='requests'},ensure_ascii=False))


if __name__=='__main__':
    main()
