"""Server-only concise views of retained formal/smoke artifacts."""
import csv
import json
from pathlib import Path
import numpy as np


def overview(out,smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[];curves=[];reliability=[];cases=[]
    for folder in sorted(out.glob('seed_*/P*')):
        arm=folder.name;seed=folder.parent.name
        path=folder/'development'/'summary.csv'
        if path.exists():
            with path.open(encoding='utf-8') as f:
                panels += [dict(arm=arm,seed=seed,**r) for r in csv.DictReader(f)
                           if r['level']=='overall' and r['stratum']=='current_not_high' and r['probability']=='calibrated' and r['horizon']=='6']
        path=folder/'epochs.jsonl'
        if path.exists():curves += [dict(arm=arm,seed=seed,**json.loads(r)) for r in path.read_text(encoding='utf-8').splitlines()]
        path=folder/'development'/'reliability.csv'
        if path.exists():
            with path.open(encoding='utf-8') as f:reliability += [dict(arm=arm,seed=seed,**r) for r in csv.DictReader(f) if r['stratum']=='current_not_high' and r['horizon']=='6']
        path=folder/'development'/'difficult_peak_cases.json'
        if path.exists():cases += [dict(arm=arm,seed=seed,**r) for r in json.loads(path.read_text(encoding='utf-8'))]
    if not panels:return
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    colors=dict(zip(['P0','P1','P2','P3','P4','P5'],['#555555','#2878b5','#42a779','#d78925','#9b57a0','#c75346']))
    for j,arm in enumerate(['P0','P1','P2','P3','P4','P5']):
        rr=[r for r in panels if r['arm']==arm and r['brier']]
        axes[0].scatter([j]*len(rr),[float(r['brier']) for r in rr],label=arm,color=colors[arm])
        for seed in sorted({r['seed'] for r in curves if r['arm']==arm}):
            cc=[r for r in curves if r['arm']==arm and r['seed']==seed]
            axes[1].plot([r['epoch'] for r in cc],[r['selection_score'] for r in cc],alpha=.6,color=colors[arm],label=arm if seed==sorted({r['seed'] for r in curves if r['arm']==arm})[0] else None)
        rr=[r for r in reliability if r['arm']==arm]
        if rr:
            bins=sorted({float(r['left']) for r in rr});points=[]
            for left in bins:
                br=[r for r in rr if float(r['left'])==left]
                points.append((np.mean([float(r['predicted']) for r in br]),np.mean([float(r['observed']) for r in br])))
            axes[2].plot(*np.array(points).T,'o-',label=arm,color=colors[arm])
    axes[0].set(xticks=range(6),xticklabels=['P0','P1','P2','P3','P4','P5'],ylabel='H6 Brier error',title='China development: qnow <= fit Q95')
    axes[1].set(xlabel='Epoch',ylabel='Common select CRPS',title='Every seed retained');axes[1].legend()
    axes[2].plot([0,1],[0,1],'k--',lw=.8);axes[2].set(xlabel='Predicted event probability',ylabel='Observed weighted fraction',title='H6 reliability (seed means in occupied bins)');axes[2].legend()
    fig.suptitle('SMOKE ONLY: no scientific efficacy claim' if smoke else 'Exploratory China development; source events / stations equal')
    fig.savefig(figdir/'01_probability_and_selection.png',dpi=150);plt.close(fig)
    if cases:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
        for ax,station in zip(axes,['Liqingdian','Luanchuan']):
            rr=[r for r in cases if station in r['station']]
            if not rr:ax.set(title=station+' NOT_EVALUABLE');continue
            event=max({r['event'] for r in rr},key=lambda e:max(max(r['true_future_Q']) for r in rr if r['event']==e))
            rr=[r for r in rr if r['event']==event];truth=rr[0]
            ax.plot(range(1,13),truth['true_future_Q'],'k-o',label='Observed')
            for arm in ['P0','P1','P2','P3','P4','P5']:
                aa=[r for r in rr if r['arm']==arm]
                for j,r in enumerate(aa):ax.plot(range(1,13),r['median_future_Q'],alpha=.6,color=colors[arm],label=arm if j==0 else None)
            ax.set(title=station+' | retrospective peak-minus-6h origin',xlabel='Forecast lead (h)',ylabel='Discharge (m3/s)');ax.legend(ncol=3,fontsize=8)
        fig.suptitle('Each line is a retained seed median curve; probability/peak scores use complete trajectories')
        fig.savefig(figdir/'02_difficult_peak_origins.png',dpi=150);plt.close(fig)
