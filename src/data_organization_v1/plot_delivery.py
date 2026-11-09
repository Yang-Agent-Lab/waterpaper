"""Read-only display repair for the accepted H6 Q99 saved-table figure."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path

SOURCE_COMMIT='072b561c4657bbe307ee80cd42342ab24eda27de'
COLORS={'A':'#888888','B':'tab:blue','C':'tab:orange'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def table(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(source,out):
    record_path=source.parent/'run_record.json';record=json.loads(record_path.read_text(encoding='utf-8'))
    if record['head']!=SOURCE_COMMIT or record['state']!='complete' or record['exit_code']!=0:raise ValueError('completed scientific source mismatch')
    out.mkdir(parents=True,exist_ok=False)
    summary=[r for r in read(source/'summary.csv') if r['scope']=='overall' and r['target']=='combined_first_q99' and r['horizon']=='6' and r['fpr_target']=='0.05']
    events=[r for r in read(source/'event_risk.csv') if r['target']=='combined_first_q99' and r['horizon']=='6' and r['fpr_target']=='0.05']
    if sorted(r['data_policy'] for r in summary)!=list('ABC'):raise ValueError('expected three saved summary rows')
    eligible={p:{(r['station'],r['source_event_id']) for r in events if r['data_policy']==p and r['state']=='evaluable'} for p in 'ABC'}
    ne={p:{(r['station'],r['source_event_id']) for r in events if r['data_policy']==p and r['state']=='NOT_EVALUABLE'} for p in 'ABC'}
    if not (eligible['A']==eligible['B']==eligible['C']) or len(eligible['A'])!=5:raise ValueError('fixed five evaluable event identity mismatch')
    if not (ne['A']==ne['B']==ne['C']) or len(ne['A'])!=11:raise ValueError('eleven observation NE identities must remain explicit')
    keys=sorted(eligible['A']);shown=[r for r in events if r['data_policy'] in ('B','C') and (r['station'],r['source_event_id']) in keys]
    table(out/'source_summary_H6_Q99.csv',summary);table(out/'source_events_H6_Q99_all.csv',events);table(out/'displayed_BC_events.csv',shown)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(13.5,6.5),gridspec_kw={'width_ratios':[1,1.65]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.23,top=.79,wspace=.33)
    fig.suptitle('China development, seed2 | H6 combined Q99',y=.96,fontsize=15)
    fig.text(.5,.905,'Saved predictions; branch calibration FPR target 5%; source event then station equal',ha='center',fontsize=10)
    values=[]
    for j,p in enumerate('ABC'):
        row=next(r for r in summary if r['data_policy']==p);value=float(row['brier'])
        if not math.isfinite(value):raise ValueError('nonfinite saved Brier')
        values.append(value);left.bar(j,value,color=COLORS[p],width=.65)
        left.text(j,value+.0009,f"{row['discovered_events']}/{row['evaluable_events']} events\nFP runs {row['false_alarm_segments']}",ha='center',va='bottom',fontsize=10)
    left.set(xticks=range(3),xticklabels=['A\nShared','B\nBackground','C\nConditional'],ylabel='Raw Brier (lower is better)',title='Probability and event discovery',ylim=(0,max(values)*1.40))
    left.tick_params(axis='x',pad=8);left.grid(axis='y',alpha=.18);left.set_axisbelow(True)
    for j,key in enumerate(keys):
        for p,offset in [('B',-.12),('C',.12)]:
            row=next(r for r in shown if r['data_policy']==p and (r['station'],r['source_event_id'])==key)
            detected=row['discovered']=='True'
            lead=float(row['lead_hours']) if detected else -1.
            if detected and (not math.isfinite(lead) or lead<=0):raise ValueError('invalid saved detected lead')
            right.scatter(j+offset,lead,color=COLORS[p],marker='o' if detected else 'x',s=65,label=p if j==0 else None,zorder=3)
    right.axhline(-.5,color='#888888',ls=':',lw=.8)
    right.set(xticks=range(len(keys)),xticklabels=[s.split(':')[-1]+'\n'+e.replace('洪水_','Event ') for s,e in keys],yticks=[-1,0,1,2,3,4,5,6],yticklabels=['Miss','0','1','2','3','4','5','6'],ylim=(-1.55,6.65),xlim=(-.55,4.55),ylabel='Actual lead (h)',title='B vs C: same five evaluable events')
    right.tick_params(axis='x',pad=8,labelsize=9);right.grid(axis='y',alpha=.18);right.set_axisbelow(True);right.legend(loc='upper right',frameon=False)
    fig.text(.5,.105,'Five evaluable source events shown; 11 observation-NE events stay in the complete table. Discovery denominator remains 5.',ha='center',fontsize=10)
    fig.text(.5,.055,'Miss is a categorical position, not a negative lead. Source events are not confirmed independent storms.',ha='center',fontsize=9,color='#555555')
    png=out/'02_Q99_policy_comparison.png';fig.savefig(png,dpi=160);plt.close(fig)
    manifest=dict(state='complete',scientific_source_commit=SOURCE_COMMIT,drawing_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs={str(p):sha(p) for p in (record_path,source/'summary.csv',source/'event_risk.csv')},output_png_sha256=sha(png),displayed_evaluable_source_events=5,observation_NE_retained=11,display_policies=['B','C'],full_summary_policies=list('ABC'),palette=COLORS,miss_display='categorical tick named Miss',scores_recomputed=False,new_training=False,new_inference=False,old_outputs_modified=False,matplotlib=matplotlib.__version__)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();run(args.source,args.output)
