"""One status read for an existing job; successful bounded assets go only to teacher-web."""
import argparse
import contextlib
import importlib.metadata
import io
import json
import logging
import shlex
import subprocess
import sys
import urllib.parse
from datetime import datetime,timezone
from pathlib import Path
from .probe import URL,REQUEST,error_record,safe_message

JOB='e965fef0-325f-4a82-af2c-e14cb60dea16'
MAX_BYTES=10*1024*1024
TOOLS='./external-home/forecast_rain_grib_tools_job_e965fef0'
SERVER_OUTPUT='._forecast_samples/tigge_job_e965fef0_20261011'
SERVER_RUN='results/experiments/forecast_rain_receive_job_e965fef0_20261011'
CORE='./.venv/bin/python'
INSTALLER='./external-home/.local/share/train-engineer/venv/bin/python'


def ssh(command,**kwargs):
    return subprocess.run(['ssh','-T','teacher-web',command],capture_output=True,text=True,timeout=180,**kwargs)


def transfer(download_url,size,checkout):
    # The key is never sent to the server; the signed URL is sent only through stdin.
    install=f'test ! -e {shlex.quote(TOOLS)} && {INSTALLER} -m pip install --disable-pip-version-check --quiet --target {shlex.quote(TOOLS)} eccodes'
    installed=ssh(install)
    if installed.returncode:raise RuntimeError('isolated GRIB dependency setup failed: '+safe_message(installed.stderr,''))
    argv=[CORE,'src/run_versioned.py','--run-dir',SERVER_RUN,'--',CORE,'-m','forecast_rain_v1.server_receive','--output',SERVER_OUTPUT]
    command='cd '+shlex.quote(checkout)+' && PYTHONPATH='+shlex.quote('src:'+TOOLS)+' '+shlex.join(argv)
    received=ssh(command,input=json.dumps(dict(job_id=JOB,expected_bytes=size,download_url=download_url)))
    receipt=ssh('cat '+shlex.quote(SERVER_OUTPUT+'/server_receipt.json'))
    if receipt.returncode:raise RuntimeError('server transfer has no public receipt: '+safe_message(received.stderr,''))
    details=json.loads(receipt.stdout)
    rows=details.get('rows',[])
    grid_keys=('gridType','Ni','Nj','numberOfDataPoints','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','latitudeOfLastGridPointInDegrees','longitudeOfLastGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees')
    grids=list({json.dumps({k:r.get(k) for k in grid_keys},sort_keys=True) for r in rows})
    return dict(receiver_exit_code=received.returncode,server_receipt=SERVER_OUTPUT+'/server_receipt.json',server_versioned_run=checkout+'/'+SERVER_RUN,server_file=details.get('file'),server_state=details['state'],file_bytes=details.get('file_bytes'),sha256=details.get('sha256'),fields=details.get('fields'),actual_issue_dates=details.get('actual_issue_dates'),actual_end_steps=details.get('actual_end_steps'),date_interpretation=details.get('date_interpretation'),units=sorted({str(r.get('units')) for r in rows}),grid_metadata=[json.loads(s) for s in sorted(grids)],validation_issues=details.get('validation_issues'),safe_error=details.get('safe_error'),eccodes_python=details.get('eccodes_python'),credential_sent_to_server=False,signed_URL_persistence=False,local_meteorological_download=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True,type=Path);p.add_argument('--server-checkout',required=True);p.add_argument('--submission-receipt',required=True,type=Path);args=p.parse_args()
    submitted=json.loads(args.submission_receipt.read_text())
    if submitted.get('request_id')!=JOB or submitted.get('state')!='SUBMITTED' or submitted.get('request')!=REQUEST:raise ValueError('existing job or approved bounded request identity mismatch')
    if not args.server_checkout.startswith('._forecast_transfer_'):raise ValueError('dedicated receiver checkout required')
    result=dict(state='preparing',job_id=JOB,code_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_at_utc=datetime.now(timezone.utc).isoformat(),submission_receipt=str(args.submission_receipt),submission_attempts=0,status_read_attempts=0,result_metadata_reads=0,local_meteorological_download=False)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as fp:fp.write(json.dumps(result,indent=2)+'\n')
    logging.disable(logging.CRITICAL);secret=''
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            import cdsapi
            from cdsapi.api import read_config
            from ecmwf.datastores.processing import Results,LinkError
            secret=read_config(str(Path.home()/'.cdsapirc')).get('key')
            if not isinstance(secret,str) or not secret or ':' in secret:raise ValueError('existing local token requires user review')
            client=cdsapi.Client(url=URL,key=secret,verify=True,quiet=True,debug=False,progress=False,wait_until_complete=False,delete=False,retry_max=1,timeout=20,info_callback=lambda *a,**k:None,warning_callback=lambda *a,**k:None,error_callback=lambda *a,**k:None,debug_callback=lambda *a,**k:None)
            result['status_read_attempts']=1
            # Inspected get_job already returns one status response; do not re-query Remote.json.
            job=client.client._retrieve_api.get_job(JOB)
            status=str(job._json_dict.get('status','unknown'))
            result['job_status']=status if status in ('accepted','running','successful','failed','rejected','dismissed','deleted') else 'unknown'
            result['client_versions']={name:importlib.metadata.version(name) for name in ('cdsapi','ecmwf-datastores-client')}
            if status!='successful':
                result['state']='PENDING' if status in ('accepted','running') else 'JOB_FAILED'
                result['safe_job_message']=safe_message({k:job._json_dict[k] for k in ('message','error') if k in job._json_dict},secret)
                result['next_step']='no resubmission; heartbeat may later perform one new status read if pending'
            else:
                remote=job.get_remote()
                try:results_url=job._get_link_href(rel='results')
                except LinkError:results_url=remote.url+'/results'
                results_url=urllib.parse.urljoin(remote.url,results_url)
                parsed=urllib.parse.urlsplit(results_url)
                if parsed.scheme!='https' or parsed.hostname!='ecds.ecmwf.int' or not parsed.path.startswith('/api/'):raise ValueError('results metadata URL is not the authenticated ECDS API')
                result['result_metadata_reads']=1
                asset=Results.from_request('get',results_url,**remote._request_kwargs)
                size=asset.content_length
                result.update(expected_asset_bytes=size,asset_content_type=asset.content_type,download_size_limit_bytes=MAX_BYTES)
                if not 0<size<=MAX_BYTES:
                    result.update(state='BOUNDED_SIZE_STOP',next_step='stop; asset exceeds or lacks the expected small regional size, do not download')
                else:
                    detail=transfer(asset.location,size,args.server_checkout)
                    result['transfer']=detail
                    result['state']='SERVER_SAMPLE_COMPLETE' if detail['server_state']=='complete' and detail['receiver_exit_code']==0 else 'SERVER_TRANSFER_OR_SCOPE_FAILED'
                    result['next_step']='parent verifies server metadata; no local data, analysis or model training performed'
        except Exception as exc:result.update(error_record(exc,secret));result['next_step']='preserve receipt; do not resubmit or retry automatically'
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0 if result['state'] in ('PENDING','SERVER_SAMPLE_COMPLETE') else 2


if __name__=='__main__':sys.exit(main())
