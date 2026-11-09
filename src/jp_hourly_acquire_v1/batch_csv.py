"""Inspect official JMA batch-download client before any annual CSV request."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
from urllib.parse import urljoin,urlparse

from .__main__ import Collector


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    c=Collector(out)
    result={'host':socket.gethostname(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'discovery_only_no_year_requested','station':'Maebashi s47624','scope':'hourly_2015_2024_existing_Q_period'}
    try:
        base='https://www.data.jma.go.jp/risk/obsdl/'
        page=c.get(base+'?vm=r','official_obsdl_frontend',True)
        scripts=re.findall(r'<script[^>]+src=["\x27]([^"\x27]+)',page or '',re.I)
        urls=[]
        for source in scripts:
            url=urljoin(base,source)
            if urlparse(url).hostname in ('www.data.jma.go.jp','ds.data.jma.go.jp') and not any(x in url.lower() for x in ('jquery','bootstrap','prototype','calendar')):
                urls.append(url)
        result['script_urls']=urls
        snippets=[]
        for index,url in enumerate(urls[:8]):
            text=c.get(url,f'official_client_{index}')
            if text:
                for pattern in ('aggrgPeriod','stationNumList','elementNumList','ymdList','downloadFlag','download'):
                    matches=list(re.finditer(re.escape(pattern),text))
                    for match in matches[:3]:
                        snippets.append({'source':url,'pattern':pattern,'context':text[max(0,match.start()-150):match.end()+600]})
        result['request_contract_snippets']=snippets
    except Exception as error:
        result['error']=repr(error)
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),requests=c.requests,accounted_bytes=c.used)
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({key:value for key,value in result.items() if key!='requests'},ensure_ascii=False))


if __name__=='__main__':
    main()
