"""Two decision-oriented figures from this run's saved small tables."""
import csv,json
from pathlib import Path
import numpy as np


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    runtime=json.loads((out/'effective_config.json').read_text());tag='SMOKE select stand-in' if runtime['smoke'] else 'Exploratory China development seed2'
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for policy in 'ABC':
        rows=[json.loads(v) for v in (out/'seed_2'/policy/'epochs.jsonl').read_text().splitlines()]
        axes[0].plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label=policy)
        axes[1].plot([r['epoch'] for r in rows],[r['select_primary_normalized'] for r in rows],label=policy)
    axes[0].set(xlabel='Epoch',ylabel='NLL + CRPS',title='Identical S2 objective; fit only');axes[1].set(xlabel='Epoch',ylabel='M6 median MAE / fit Q95 (lower better)',title='Select, true mainpeak minus6');axes[1].legend()
    fig.savefig(figdir/'01_training_selection.png',dpi=150);plt.close(fig)
    summary=list(csv.DictReader((out/'summary.csv').open()));event=list(csv.DictReader((out/'event_risk.csv').open()))
    selected=[r for r in summary if r['scope']=='overall' and r['target']=='combined_first_q99' and r['horizon']=='6' and r['fpr_target']=='0.05']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for j,p in enumerate('ABC'):
        r=next(r for r in selected if r['data_policy']==p);b=float(r['brier']) if r['brier'] else np.nan
        axes[0].bar(j,b);axes[0].text(j,b if np.isfinite(b) else 0,f"{r['discovered_events']}/{r['evaluable_events']} events\nFP runs {r['false_alarm_segments']}",ha='center',va='bottom',fontsize=9)
    axes[0].set(xticks=range(3),xticklabels=list('ABC'),ylabel='Raw Brier (lower better)',title=tag+'\nH6 combined Q99, source event then station equal')
    keys=sorted({(r['station'],r['source_event_id']) for r in event if r['target']=='combined_first_q99' and r['horizon']=='6' and r['fpr_target']=='0.05'})
    for j,key in enumerate(keys):
        for policy,offset in [('B',-.12),('C',.12)]:
            r=next(r for r in event if r['data_policy']==policy and r['target']=='combined_first_q99' and r['horizon']=='6' and r['fpr_target']=='0.05' and (r['station'],r['source_event_id'])==key)
            y=float(r['lead_hours']) if r['lead_hours'] else -.6 if r['state']=='evaluable' else -1.2
            axes[1].scatter(j+offset,y,color={'B':'tab:blue','C':'tab:orange'}[policy],label=policy if j==0 else None)
    axes[1].set(xticks=range(len(keys)),xticklabels=[s.split(':')[-1]+'|'+e.replace('洪水_','Event ') for s,e in keys],ylabel='Actual detected lead (h); miss=-0.6, NE=-1.2',title='Paired source events B vs C; same H6 / 5% rule')
    axes[1].tick_params(axis='x',rotation=80);axes[1].legend();fig.savefig(figdir/'02_Q99_policy_comparison.png',dpi=150);plt.close(fig)
    (figdir/'sources.json').write_text(json.dumps(dict(training='seed_2/{A,B,C}/epochs.jsonl',risk='summary.csv',events='event_risk.csv',smoke=runtime['smoke'],statistical_unit='source events then stations; source events not confirmed independent storms',negative_event_markers='miss and NE are categorical display markers, not negative time'),indent=2))
