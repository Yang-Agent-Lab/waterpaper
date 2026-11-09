"""Review-only figure correction from saved small tables; no prediction/scoring."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from .report import digest,dump


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    cases=list(csv.DictReader((source/'case.csv').open()))
    events=list(csv.DictReader((source/'event_risk.csv').open()))
    thresholds=json.loads((source/'thresholds.json').read_text());lines=json.loads((source/'alarm_lines.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col',constrained_layout=True)
    colors={'S1':'tab:orange','S2':'tab:green'}
    for col,(name,event) in enumerate((('Liqingdian','洪水_6'),('Luanchuan','洪水_9'))):
        selected=[r for r in cases if name in r['station'] and r['source_event_id']==event and r['seed']=='2' and r['horizon']=='6']
        station={r['station'] for r in selected}
        if len(station)!=1:raise ValueError('fixed case identity mismatch')
        st=next(iter(station));observed={}
        for r in selected:
            t=datetime.fromisoformat(r['origin_time']);q=float(r['current_Q_m3_s'])
            if t in observed and observed[t]!=q:raise ValueError('saved current-Q conflict')
            observed[t]=q
        times=sorted(observed);axes[0,col].plot(times,[observed[t] for t in times],color='black',label='Observed Q at available issue times')
        for key,color,style in [('q95','tab:blue','--'),('q99','tab:red',':')]:
            axes[0,col].axhline(thresholds[st][key],color=color,ls=style,label='fit '+key)
        axes[0,col].set(title=name+' Event '+event.split('_')[-1],ylabel='Observed Q (m3/s)');axes[0,col].legend(fontsize=8)
        for target,style in [('first_q95','-'),('upgrade_q99','--')]:
            for arm in ('S1','S2'):
                rows=sorted([r for r in selected if r['arm']==arm and r['target']==target],key=lambda r:r['origin_time'])
                axes[1,col].plot([datetime.fromisoformat(r['origin_time']) for r in rows],
                  [float(r['probability']) if r['eligible_current_state']=='True' and r['probability'] else np.nan for r in rows],
                  ls=style,color=colors[arm],label=arm+' '+target)
                line=lines[f'{arm}|2|{target}|6|0.05']['line']
                if line is not None:axes[1,col].axhline(line,ls=style,color=colors[arm],alpha=.4)
            match=[r for r in events if r['station']==st and r['source_event_id']==event and r['arm']=='S2' and r['seed']=='2' and r['target']==target and r['horizon']=='6' and r['fpr_target']=='0.05']
            if len(match)!=1:raise ValueError('saved crossing identity mismatch')
            if match[0]['first_crossing']:
                t=datetime.fromisoformat(match[0]['first_crossing'])
                for ax in axes[:,col]:ax.axvline(t,ls=style,color='purple',alpha=.6)
        axes[1,col].set(ylabel='Raw saved-scenario exceedance probability',xlabel='Issue time / recorded crossing time (timezone unknown)',ylim=(-.03,1.03))
        axes[1,col].legend(fontsize=7)
        # Shared column clock explicitly spans all available observed issue times.
        axes[0,col].set_xlim(min(times),max(times))
        for ax in axes[:,col]:ax.tick_params(axis='x',labelrotation=25)
    fig.suptitle('Fixed development cases, seed2; H6 cumulative peaks; only eligible issue-state probabilities')
    fig.savefig(out/'02_fixed_difficult_cases_review.png',dpi=160)
    with PdfPages(out/'event_risk_figures_review.pdf') as pdf:
        original=plt.imread(source/'figures/01_event_probability_tradeoff.png')
        first=plt.figure(figsize=(15,8));ax=first.add_axes([0,0,1,1]);ax.imshow(original);ax.axis('off');pdf.savefig(first);plt.close(first)
        pdf.savefig(fig)
    plt.close(fig)
    dump(out/'figure_manifest.json',dict(state='complete',purpose='shared-clock and English fixed-event label correction only',
      inputs={str(source/name):digest(source/name) for name in ['case.csv','event_risk.csv','thresholds.json','alarm_lines.json','figures/01_event_probability_tradeoff.png']},
      observed_Q='current_Q_m3_s deduplicated by origin_time from saved case table; observed at available issue times only',
      score_recomputed=False,inference=False,training=False,original_figures_preserved=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();run(args.source,args.out)
if __name__=='__main__':main()
