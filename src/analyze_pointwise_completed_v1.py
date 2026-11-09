"""Read saved completed pq forecasts only; no inference, training or reserved data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def weights(rows):
    counts=Counter((r['canonical_station'],r['event']) for r in rows)
    events=Counter(s for s,e in counts)
    return np.array([1/(len(events)*events[r['canonical_station']]*counts[r['canonical_station'],r['event']]) for r in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.run)/'artifacts';out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    names=['diffpts','nbeatsx','lstm','tcn','tft','persistence'];labels=['DiffPTS','N-BEATSx','LSTM','TCN','TFT','Persistence']
    result={};saved={};reference=None;event_rows=[]
    for name in names:
        folder=root/(name+'_pq');completion=json.loads((folder/'completion.json').read_text());assert completion['state']=='complete'
        fullrows=json.loads((folder/'select_rows.json').read_text());ids=[i for i,r in enumerate(fullrows) if r['country']=='CN'];rows=[fullrows[i] for i in ids]
        identity=[r['sample_id'] for r in rows]
        if reference is None:reference=identity
        else:assert identity==reference
        w=weights(rows);scale=np.array([r['q_scale'] for r in rows]);th=np.array([[r['q95']/r['q_scale'],r['q99']/r['q_scale']] for r in rows])
        with np.load(folder/'select_predictions.npz') as z:
            draws=z['draws_normalized'][ids].astype(float);y=z['truth_normalized'][ids];stored_p=z['probability'][ids];stored_crps=z['crps'][ids]
        prob=(draws[:,:,:,None]>th[:,None,None,:]).mean(1);obs=y[:,:,None]>th[:,None,:];assert np.array_equal(prob,stored_p)
        s=draws.shape[1];ordered=np.sort(draws,axis=1);crps=np.abs(draws-y[:,None,:]).mean(1)-((2*np.arange(1,s+1)-s-1)[None,:,None]*ordered).sum(1)/(s*s)
        assert np.allclose(crps,stored_crps,rtol=1e-10,atol=1e-12)
        brier=(prob-obs)**2;hourly=np.einsum('n,nhq->hq',w,brier);hourly_crps=w@crps
        scores=[r for r in json.loads((folder/'scores.json').read_text()) if r['country']=='CN']
        for r in scores:assert abs(r['brier']-hourly[r['horizon']-1,0 if r['target'].endswith('Q95') else 1])<1e-12
        median=np.median(draws,axis=1);lo,hi=np.quantile(draws,[.05,.95],axis=1)
        r=dict(brier=float(hourly.mean()),q95_brier=float(hourly[:,0].mean()),q99_brier=float(hourly[:,1].mean()),
            crps_q95=float(hourly_crps.mean()),best_epoch=completion.get('selected',{}).get('epoch'),
            coverage90=float(w@((y>=lo)&(y<=hi)).mean(1)),interval_width_q95=float(w@(hi-lo).mean(1)),
            median_mae_m3_s=float(w@(np.abs(median-y)*scale[:,None]).mean(1)),lead_groups={},thresholds={})
        for left,right in [(0,6),(6,12),(12,24)]:r['lead_groups'][f'{left+1}-{right}']=float(hourly[left:right].mean())
        wt=np.broadcast_to(w[:,None]/24,y.shape)
        for j,key in enumerate(['Q95','Q99']):
            pos=obs[:,:,j];neg=~pos
            def cond(values,mask):return float((wt*values*mask).sum()/(wt*mask).sum()) if (wt*mask).sum() else None
            r['thresholds'][key]=dict(prevalence=float((wt*pos).sum()),mean_probability=float((wt*prob[:,:,j]).sum()),
                mean_probability_on_positive=cond(prob[:,:,j],pos),mean_probability_on_negative=cond(prob[:,:,j],neg),
                positive_brier=cond(brier[:,:,j],pos),negative_brier=cond(brier[:,:,j],neg),
                fraction_positive_probability_below_10pct=cond(prob[:,:,j]<.1,pos),positive_origin_hour_cells=int(pos.sum()))
        result[name]=r;saved[name]=dict(prob=prob,obs=obs,y=y,median=median,lo=lo,hi=hi,hourly=hourly,crps=hourly_crps,brier=brier)
        for station,event in sorted({(r['canonical_station'],r['event']) for r in rows}):
            ii=[i for i,r in enumerate(rows) if (r['canonical_station'],r['event'])==(station,event)]
            event_rows.append(dict(model=name,station=station,event=event,n_origins=len(ii),brier=float(brier[ii].mean()),
                crps_q95=float(crps[ii].mean()),positive_cells=int(obs[ii].sum())))
        del draws,ordered
    result['always_zero_diagnostic']=dict(brier=float(np.einsum('n,nhq->hq',w,saved['diffpts']['obs']).mean()),
        note='posthoc no-training diagnostic; not an added tuned model')
    diff={(r['station'],r['event']):r for r in event_rows if r['model']=='diffpts'}
    tcn={(r['station'],r['event']):r for r in event_rows if r['model']=='tcn'}
    pairs=[dict(station=k[0],event=k[1],delta_brier=diff[k]['brier']-tcn[k]['brier'],positive_cells=diff[k]['positive_cells']) for k in diff]
    eligible=[r for r in pairs if r['positive_cells']>0]
    chosen=[max(eligible,key=lambda r:r['delta_brier']),min(eligible,key=lambda r:r['delta_brier'])]
    case_info=[]
    colors=['#7544ac','#2474a6','#d49426','#159476','#d65867','#777777']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(out/'comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
        for name,label,color in zip(names,labels,colors):
            for j in range(2):axes[j].plot(range(1,25),saved[name]['hourly'][:,j],label=label,color=color,lw=2 if name=='diffpts' else 1.3)
            axes[2].plot(range(1,25),saved[name]['crps'],label=label,color=color,lw=2 if name=='diffpts' else 1.3)
        for ax,title in zip(axes,['Q95 probability error','Q99 probability error','Distribution error']):
            ax.set(title=title,xlabel='Forecast lead (hours)',ylabel='Brier (lower is better)' if ax!=axes[2] else 'CRPS / station fit Q95 (lower is better)');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('China select | history P+Q | 9 stations / 17 source events / 1706 origins | seed 2')
        fig.savefig(out/'01_comparison.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        for col,event in enumerate(chosen):
            ii=[i for i,r in enumerate(rows) if (r['canonical_station'],r['event'])==(event['station'],event['event'])]
            idx=max(ii,key=lambda i:saved['diffpts']['y'][i].max());row=rows[idx];scale=row['q_scale'];truth=saved['diffpts']['y'][idx]*scale
            axes[0,col].plot(range(1,25),truth,'k-',label='Observed',lw=2)
            for name,color in [('diffpts',colors[0]),('tcn',colors[3])]:
                d=saved[name];axes[0,col].plot(range(1,25),d['median'][idx]*scale,color=color,label=name.upper())
                if name=='diffpts':axes[0,col].fill_between(range(1,25),d['lo'][idx]*scale,d['hi'][idx]*scale,color=color,alpha=.13,label='DiffPTS 90% interval')
                axes[1,col].plot(range(1,25),d['prob'][idx,:,0],color=color,label=name.upper())
            axes[0,col].axhline(row['q95'],ls='--',color='#999999',label='Fit Q95')
            axes[1,col].step(range(1,25),(truth>row['q95']).astype(int),where='mid',color='black',label='Observed Q95 exceedance')
            axes[0,col].set(title=row['station']+' / event '+row['event'].rsplit('_',1)[-1]+'\n'+row['origin_time'],ylabel='Flow (m3/s)')
            axes[1,col].set(xlabel='Forecast lead (hours)',ylabel='Q95 exceedance probability',ylim=(-.03,1.03))
            for ax in axes[:,col]:ax.legend(fontsize=8);ax.grid(alpha=.2)
            case_info.append(dict(**event,sample_id=row['sample_id'],origin=row['origin_time'],
                observed_peak_m3_s=float(truth.max()),q95=row['q95'],
                diffpts_median_at_observed_peak=float(saved['diffpts']['median'][idx,truth.argmax()]*scale),
                tcn_median_at_observed_peak=float(saved['tcn']['median'][idx,truth.argmax()]*scale)))
        fig.suptitle('Posthoc examples: largest disadvantage / largest advantage vs TCN among positive events\nWithin each event: origin with largest observed future peak; descriptive, not independent tests',fontsize=11)
        fig.savefig(out/'02_cases.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        history=json.loads((root/'diffpts_pq/epochs.json').read_text());ep=[r['epoch'] for r in history];best=result['diffpts']['best_epoch']
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for key,label in [('fit_loss','China fixed fit probe'),('select_loss','China select')]:axes[0].plot(ep,[r[key] for r in history],label=label)
        axes[0].set(xlabel='Epoch',ylabel='Same native loss definition',title='DiffPTS learning');axes[0].legend()
        axes[1].plot(ep,[r['selection_metric'] for r in history],color=colors[0]);axes[1].set(xlabel='Epoch',ylabel='China mean Q95/Q99 Brier',title='Checkpoint selection (128 draws)')
        for ax in axes:ax.axvline(best,color='gray',ls='--',label='Selected epoch');ax.grid(alpha=.2)
        fig.savefig(out/'03_learning.png',dpi=160);pdf.savefig(fig);plt.close(fig)
    report=dict(scope='completed pq configs only; q_only DiffPTS failed; no new inference/training',metrics=result,
        events=event_rows,paired_events=pairs,cases=case_info,selected_diffpts_epoch=best,
        diffpts_learning_first_last={key:[history[0][key],history[-1][key]] for key in ('fit_loss','select_loss','selection_metric')},
        saved_prediction_probability_and_crps_independently_recomputed=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','paired_events')},indent=2))


if __name__=='__main__':main()
