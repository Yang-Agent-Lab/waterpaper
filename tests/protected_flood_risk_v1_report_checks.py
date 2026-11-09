"""Report-only horizon checks on saved real smoke and tiny timestamp boundaries."""
import argparse,json,time
from pathlib import Path
import numpy as np
from protected_flood_risk_v1.report import horizon_event_discovery,table,build_report
from protected_flood_risk_v1.training import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();start=time.monotonic();source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[dict(window_id='origin5h',station_uid='synthetic',event='event',origin_time='2000-01-01T00',source_times=[str(np.datetime64('1999-12-31T13')+np.timedelta64(i,'h')) for i in range(24)])]
    xq=np.zeros((1,12));y=np.zeros((1,12));y[0,4]=2.;threshold=dict(threshold=.5)
    states={}
    for h in (1,3,6):
        truth=np.array([bool(y[0,:h].max()>1)]);record=horizon_event_discovery(np.array([.8]),truth,np.array([0.]),rows,xq,y,threshold,h)[0]
        states[str(h)]=record
    assert states['1']['state']==states['3']['state']=='NOT_EVALUABLE'
    assert states['6']['state']=='evaluable' and states['6']['discovered'] and states['6']['lead_hours']==5
    # Exact boundary is eligible, nonfinite probability are not.
    assert horizon_event_discovery(np.array([.8]),np.array([True]),np.array([0.]),rows,xq,y,threshold,5)[0]['eligible_origins_within_horizon']==1
    assert horizon_event_discovery(np.array([np.nan]),np.array([True]),np.array([0.]),rows,xq,y,threshold,6)[0]['state']=='NOT_EVALUABLE'
    # A high fragment beginning without its predecessor remains NE.
    high_x=xq.copy();high_x[0,0]=2
    assert horizon_event_discovery(np.array([.8]),np.array([True]),np.array([0.]),rows,high_x,y,threshold,6)[0]['state']=='NOT_EVALUABLE'
    training=source/'training';roles=json.loads((training/'roles_manifest.json').read_text());alarm_lines=json.loads((training/'alarm_lines.json').read_text());records=[];pools={}
    for arm in ('Risk-A','Risk-B','Risk-C','Risk-D'):
        best=json.loads((training/arm/'best.json').read_text());folder=training/arm/f"epoch_{best['epoch']}"
        for role,identity in roles.items():
            rows=json.loads((training/identity['rows_file']).read_text())
            inputs=identity['input_hashes'];npz_path=next(Path(path) for path in inputs if Path(path).name==role+'.npz')
            original_rows_path=next(Path(path) for path in inputs if Path(path).name==role+'_rows.json')
            original_rows=json.loads(original_rows_path.read_text());lookup={r['window_id']:i for i,r in enumerate(original_rows)}
            indices=np.array([lookup[r['window_id']] for r in rows])
            with np.load(npz_path) as data:
                arrays={key:data[key][indices] for key in ('x','y','features')}
                xq=arrays['x'][:,:,1];y=arrays['y']
            pools[role]=dict(rows=rows,**arrays)
            with np.load(folder/(role+'_predictions.npz')) as z:saved={k:z[k] for k in z.files}
            for j,h in enumerate((1,3,6)):
                risk=np.where(saved['at_risk'],saved['risk'][:,j],np.nan)
                result=horizon_event_discovery(risk,saved['truth'][:,j],xq[:,-1],rows,xq,y,alarm_lines[arm][str(h)],h)
                for record in result:
                    if record['state']=='evaluable':assert record['eligible_origins_within_horizon']>0
                    if record['lead_hours'] is not None:assert 0<record['lead_hours']<=h
                    records.append(dict(arm=arm,role=role,**record))
    table(out/'event_discovery.csv',records)
    build_report(out/'full_report',pools,training_source=training)
    assert (out/'full_report/report_completion.json').exists()
    assert len(list((out/'full_report/figures').glob('*.png')))==2
    atomic_json(out/'checks.json',dict(state='PASS',report_only=True,full_build_report=True,no_training=True,no_model_inference=True,source=str(source),boundary=states,event_records=len(records),elapsed_seconds=time.monotonic()-start))
    print('PASS: report-only horizon qualification; no training or model inference')

if __name__=='__main__':main()
