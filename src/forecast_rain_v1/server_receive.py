"""Server-only bounded GRIB transfer; URL is received in memory through SSH stdin."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime,timezone
from pathlib import Path
from .probe import safe_message

JOB='e965fef0-325f-4a82-af2c-e14cb60dea16'
MAX_BYTES=10*1024*1024


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        parsed=urllib.parse.urlsplit(newurl)
        if parsed.scheme!='https' or parsed.username or parsed.password:raise ValueError('unsafe download redirect')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def metadata(path):
    import eccodes
    keys=('edition','centre','subCentre','paramId','shortName','name','units','dataDate','dataTime','validityDate','validityTime','startStep','endStep','stepRange','stepUnits','stepType','typeOfStatisticalProcessing','typeOfEnsembleForecast','number','gridType','Ni','Nj','numberOfDataPoints','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','latitudeOfLastGridPointInDegrees','longitudeOfLastGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees')
    rows=[]
    with path.open('rb') as fp:
        while True:
            handle=eccodes.codes_grib_new_from_file(fp)
            if handle is None:break
            try:
                row={}
                for key in keys:
                    try:row[key]=eccodes.codes_get(handle,key)
                    except eccodes.CodesInternalError:row[key]=None
                rows.append(row)
                if len(rows)>24:raise ValueError('field count exceeds bounded six-day/control/four-step sample')
            finally:eccodes.codes_release(handle)
    if not rows:raise ValueError('no GRIB fields found')
    issues=[]
    for row in rows:
        if row['paramId']!=228228:issues.append('unexpected precipitation parameter')
        if row['dataDate'] is None or not 20100718<=row['dataDate']<=20100723:issues.append('issue date outside bounded request')
        if row['dataTime']!=0 or row['endStep'] not in (6,12,18,24):issues.append('unexpected issue time or forecast step')
        if row['gridType']!='regular_ll' or row['Ni']!=9 or row['Nj']!=7 or row['numberOfDataPoints']!=63:issues.append('grid is not requested regional 0.5-degree sample')
        # Scan direction can reverse either endpoint while retaining the same regional bounds.
        for names,bounds in [(('latitudeOfFirstGridPointInDegrees','latitudeOfLastGridPointInDegrees'),(32.,35.)),(('longitudeOfFirstGridPointInDegrees','longitudeOfLastGridPointInDegrees'),(110.,114.))]:
            values=[row[k] for k in names]
            if any(v is None for v in values) or any(abs(a-b)>1e-6 for a,b in zip(sorted(v for v in values if v is not None),bounds)):issues.append('regional bounds missing or outside request')
        for key in ('iDirectionIncrementInDegrees','jDirectionIncrementInDegrees'):
            if row[key] is None or abs(row[key]-.5)>1e-6:issues.append('grid increment mismatch')
    dates=sorted({r['dataDate'] for r in rows if r['dataDate'] is not None})
    pairs={(r['dataDate'],r['endStep']) for r in rows}
    if len(pairs)!=len(rows) or len(rows)!=len(dates)*4:issues.append('duplicate or incomplete date-step fields')
    date_interpretation='continuous six-day range returned' if dates==list(range(20100718,20100724)) else 'two discrete endpoint dates returned' if dates==[20100718,20100723] else 'other/partial bounded date coverage; inspect metadata'
    return dict(fields=len(rows),rows=rows,actual_issue_dates=dates,actual_end_steps=sorted({r['endStep'] for r in rows if r['endStep'] is not None}),date_interpretation=date_interpretation,validation_issues=sorted(set(issues)),eccodes_python=importlib.metadata.version('eccodes'),scope='regional SAMPLE metadata only; no basin extraction, values analysis or as-issued availability validation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    result=dict(state='failed',job_id=JOB,started_at_utc=datetime.now(timezone.utc).isoformat(),local_meteorological_download=False);created=False
    try:
        payload=json.load(sys.stdin)
        if payload['job_id']!=JOB:raise ValueError('unexpected job identity')
        expected=int(payload['expected_bytes'])
        if not 0<expected<=MAX_BYTES:raise ValueError('size exceeds the bounded regional sample or is unavailable')
        url=payload['download_url'];parsed=urllib.parse.urlsplit(url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:raise ValueError('unsafe download URL')
        args.output.mkdir(parents=True,exist_ok=False)
        created=True
        part=args.output/'regional_sample.grib.part';target=args.output/'regional_sample.grib'
        opener=urllib.request.build_opener(HTTPSRedirect())
        with opener.open(urllib.request.Request(url),timeout=30) as response:
            if response.status!=200:raise ValueError('unexpected download HTTP status')
            length=response.headers.get('Content-Length')
            if length is not None and int(length)!=expected:raise ValueError('download response size differs from API asset metadata')
            count=0;sha=hashlib.sha256()
            with part.open('xb') as fp:
                while True:
                    chunk=response.read(65536)
                    if not chunk:break
                    count+=len(chunk)
                    if count>expected or count>MAX_BYTES:raise ValueError('download exceeds bounded expected size; partial retained')
                    fp.write(chunk);sha.update(chunk)
        if count!=expected:raise ValueError('incomplete download; partial retained')
        with part.open('rb') as fp:
            if fp.read(4)!=b'GRIB':raise ValueError('unexpected non-GRIB result; no archive unpacking attempted')
        part.rename(target)
        details=metadata(target)
        result.update(state='complete' if not details['validation_issues'] else 'metadata_scope_failed',file=str(target),file_bytes=count,sha256=sha.hexdigest(),**details)
    except Exception as exc:result.update(error_type=type(exc).__name__,safe_error=safe_message(exc,''))
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    if created: (args.output/'server_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))
    return 0 if result['state']=='complete' else 2


if __name__=='__main__':sys.exit(main())
