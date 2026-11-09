"""Follow the observed official monthly link and the official JMA alternate host."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import time

from .__main__ import Collector, existing_q, jma_rows, monthly_rows, selfcheck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--selfcheck",action="store_true")
    ap.add_argument("--reuse-jma")
    args = ap.parse_args()
    selfcheck()
    fixture='<table><tr><td>12</td>'+''.join('<td>1.0</td>' for _ in range(24))+'</tr></table>'
    assert len(monthly_rows(fixture,2019,10))==24
    if args.selfcheck:
        print("monthly and JMA parser selfcheck passed")
        return
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    c=Collector(out)
    result={"host":socket.gethostname(),"started_at_utc":datetime.now(timezone.utc).isoformat(),
            "status":"partial","change_reason":"Observed SiteInfo maps monthly rainfall to KIND=2; KIND=1 is arbitrary-period list",
            "upstream_boundary":"unknown","Q_id":"303031283301030","P_id":"103031283301040","P_name":"八斗島"}
    try:
        q,sources=existing_q(out)
        result['existing_Q']={'n_rows':len(q),'n_numeric':sum(r['Q_m3_s'] is not None for r in q),'sources':sources}
        base='https://www1.river.go.jp'
        setup=c.get(base+'/cgi-bin/SrchRainData.exe?ID=103031283301040&KIND=2&PAGE=0','mlit_month_form',True)
        if setup:
            result['form_controls']=re.findall(r'<(?:input|select)[^>]+>',setup,re.I)
        url=base+'/cgi-bin/DspRainData.exe?KIND=2&ID=103031283301040&BGNDATE=20191001&ENDDATE=20191031&KAWABOU=NO'
        text=c.get(url,'mlit_yattajima_month_kind2',True)
        p=monthly_rows(text,2019,10) if text else []
        (out/'mlit_yattajima_rows.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
        result['MLIT']={'n_candidate_hourly_rows':len(p),'unit_mm_header':bool(text and 'mm' in text),
                        'header_review_required':True,'sample':p[:3]}
        all_p=json.loads(Path(args.reuse_jma).read_text()) if args.reuse_jma else []
        days=[] if args.reuse_jma else [12]+[day for day in range(1,32) if day!=12]
        result['reused_JMA_source']=args.reuse_jma
        active_base=None
        for day in days:
            suffix=f'?prec_no=42&block_no=47624&year=2019&month=10&day={day}&view=p1'
            options=[active_base] if active_base else [
                'https://ds.data.jma.go.jp/stats/etrn/view/hourly_s1.php',
                'https://ds.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php']
            rows=[]
            for number,host in enumerate(options):
                text=c.get(host+suffix,f'jma_alternate_201910{day:02d}_{number}')
                rows=jma_rows(text,f'2019-10-{day:02d}') if text else []
                if len(rows)==24:
                    active_base=host
                    break
            if len(rows)!=24:
                result['jma_stop_day']=day
                break
            all_p.extend(rows)
            time.sleep(1)
        all_p.sort(key=lambda row:row['time_jst'])
        (out/'jma_maebashi_P_rows.json').write_text(json.dumps(all_p,ensure_ascii=False,indent=2))
        qmap={r['time_jst']:r for r in q}
        pairs=[{**r,**qmap[r['time_jst']]} for r in all_p if r['time_jst'] in qmap]
        (out/'same_clock_point_P_Q.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2))
        result['JMA']={'n_hourly_rows':len(all_p),'n_numeric_P':sum(r['P_mm'] is not None for r in all_p),
                        'base_url':active_base,'station':'前橋','P_unit':'mm','time_zone':'JST',
                        'n_same_clock_pairs':len(pairs),'n_numeric_pairs':sum(r['P_mm'] is not None and r['Q_m3_s'] is not None for r in pairs),
                        'sample_pairs':[r for r in pairs if r['P_mm'] is not None and r['Q_m3_s'] is not None][:3],
                        'pairing':'point_P_temporal_overlap_only_not_basin_rainfall'}
    except Exception as error:
        result['error']=repr(error)
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),requests=c.requests,accounted_bytes=c.used)
        (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({'status':result['status'],'JMA':result.get('JMA'),'MLIT':result.get('MLIT')},ensure_ascii=False))


if __name__=='__main__':
    main()
