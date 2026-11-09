"""One ECDS request using an existing local token; no download, polling loop or consent."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import re
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

URL='https://ecds.ecmwf.int/api'
DATASET='tigge-forecasts'
REQUEST=dict(class_='ti',date='2010-07-18/2010-07-23',expver='prod',origin='kwbc',type='cf',levtype='sfc',param='228228',time='00:00:00',step='6/12/18/24',grid='0.5/0.5',area='35/110/32/114')
REQUEST['class']=REQUEST.pop('class_')


def safe_message(value,secret):
    text=str(value)
    if secret:text=text.replace(secret,'[credential omitted]')
    text=re.sub(r'https?://[^\s<>"\x27]+','[URL omitted]',text)
    text=re.sub(r'(?i)(bearer|token|key|authorization)\s*[:=]\s*[^\s,;]+',r'\1=[omitted]',text)
    text=re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}','[account omitted]',text)
    return text[:2000]


def error_record(exc,secret):
    response=getattr(exc,'response',None)
    status=response.status_code if response is not None else None
    message=safe_message(exc,secret)
    if response is not None:
        try:message=safe_message(response.json(),secret)
        except (ValueError,TypeError):pass
    lower=message.lower()
    if any(s in lower for s in ('licence','license','terms and conditions','terms of use','accept the terms')):
        state='HUMAN_TERMS_OR_LICENCE_REQUIRED'
        human='User must review/accept the required ECDS terms and TIGGE licence in their own account; this probe stops and does not accept them.'
    elif status==401:
        state='AUTHENTICATION_FAILED';human='User must verify their existing API token/account; no credential change was performed.'
    elif status in (400,422):
        state='SCHEMA_REJECTED';human='Review the bounded request against the returned schema error; do not resubmit automatically.'
    elif status==404:
        state='RESOURCE_NOT_FOUND';human='Dataset/request endpoint was not found; archive availability remains unverified.'
    else:
        state='ACCESS_FAILED';human='Inspect the sanitized error; do not retry or widen scope automatically.'
    return dict(state=state,error_type=type(exc).__name__,http_status=status,safe_error=message,required_human_step=human)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path);args=p.parse_args()
    config=json.loads(args.config.read_text())
    if config['url']!=URL or config['dataset']!=DATASET or config['request']!=REQUEST:raise ValueError('probe scope differs from the approved one bounded request')
    if config['max_submissions']!=1 or config['max_status_reads']!=1 or config['download'] is not False:raise ValueError('request/status/download limits changed')
    # Keep failed receipts immutable; an existing receipt never authorizes another submission.
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as fp:
        result=dict(state='preparing',change_id=config['change_id'],code_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),dataset=DATASET,request=REQUEST,scope=config['scope'],submission_attempts=0,status_read_attempts=0,downloaded=False,credential_source='existing user-local .cdsapirc; key never copied or logged',official_dataset_page=config['official_dataset_page'])
        fp.write(json.dumps(result,indent=2)+'\n');fp.flush()
    secret=''
    logging.disable(logging.CRITICAL)
    # No API library output or traceback may expose a token or signed download URL.
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            import cdsapi
            from cdsapi.api import read_config
            credentials=read_config(str(Path.home()/'.cdsapirc'))
            secret=credentials.get('key')
            if not isinstance(secret,str) or not secret:raise ValueError('existing local token missing')
            if ':' in secret:raise ValueError('legacy token format requires user review; no automatic account or token change')
            check=safe_message('key='+secret+' https://example.invalid/file?signature=hidden',secret)
            assert secret not in check and 'https://' not in check and 'signature=hidden' not in check
            result['client_versions']={name:importlib.metadata.version(name) for name in ('cdsapi','ecmwf-datastores-client')}
            client=cdsapi.Client(url=URL,key=secret,verify=True,quiet=True,debug=False,progress=False,wait_until_complete=False,delete=False,retry_max=1,timeout=config['timeout_seconds'],info_callback=lambda *a,**k:None,warning_callback=lambda *a,**k:None,error_callback=lambda *a,**k:None,debug_callback=lambda *a,**k:None)
            result['submission_attempts']=1
            # No target means no download in the inspected LegacyClient.retrieve implementation.
            job=client.retrieve(DATASET,REQUEST,target=None)
            job_id=job.request_id
            if not re.fullmatch(r'[A-Za-z0-9-]{1,100}',job_id):raise ValueError('unsafe/unexpected request identifier')
            result.update(state='SUBMITTED',request_id=job_id,availability='submission accepted; archived field availability not yet confirmed')
            # One status request, never get_results/get_receipt/download or a polling loop.
            result['status_read_attempts']=1
            try:
                metadata=job.json
                status=str(metadata.get('status','unknown'))
                if not re.fullmatch(r'[A-Za-z_-]{1,40}',status):status='unknown'
                result['job_status']=status
                if status=='successful':result['availability']='server request succeeded; no data downloaded or field/time/unit/basin validation performed'
                elif status in ('failed','rejected','dismissed'):result['availability']='job did not complete successfully; archive field availability unverified'
                for name in ('created','updated','started','finished'):
                    value=metadata.get(name)
                    if isinstance(value,str) and re.fullmatch(r'[0-9TZtz .:+-]+',value):result[name+'_at']=value
            except Exception as exc:result['status_read_error']=error_record(exc,secret)
        except Exception as exc:result.update(error_record(exc,secret))
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    result['next_step']='stop; parent reviews public receipt before any further status check or server-only transfer'
    args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0 if result['state']=='SUBMITTED' else 2


if __name__=='__main__':sys.exit(main())
