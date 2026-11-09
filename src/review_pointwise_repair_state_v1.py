"""Independent review from saved predictions; no model inference or held-out data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def hierarchy(rows):
    cells=Counter((r['canonical_station'],r['event']) for r in rows)
    events=Counter(s for s,e in cells)
    return np.asarray([1/(len(events)*events[r['canonical_station']]*cells[r['canonical_station'],r['event']]) for r in rows])


def main(index_path=None,out_path=None):
    if index_path is None:
        p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);a=p.parse_args()
        index_path=a.index;out_path=a.out
    cfg=read(index_path);out=Path(out_path);out.mkdir(parents=True,exist_ok=False)
    metrics=[];events=[];predictions={};identities={};histories={};metadata={}
    for entry in cfg['entries']:
        folder=Path(entry['path']);complete=read(folder/'completion.json',{});fail=read(folder/'failure.json',{})
        if complete.get('state')!='complete':
            metrics.append(dict(id=entry['id'],route=entry['route'],state='failed' if fail else 'not_complete',details=fail));continue
        fullrows=read(folder/'select_rows.json');ids=[i for i,r in enumerate(fullrows) if r['country']=='CN'];rows=[fullrows[i] for i in ids]
        identity=[r['sample_id'] for r in rows];route=entry['route'];w=hierarchy(rows)
        if route not in identities:identities[route]=identity
        else:assert identities[route]==identity,'not the same within-route China samples'
        with np.load(folder/'select_predictions.npz') as z:
            draws=z['draws_normalized'][ids].astype(float);y=z['truth_normalized'][ids];stored_p=z['probability'][ids];stored_crps=z['crps'][ids]
        threshold=np.asarray([[r['q95']/r['q_scale'],r['q99']/r['q_scale']] for r in rows])
        prob=(draws[:,:,:,None]>threshold[:,None,None,:]).mean(1);obs=y[:,:,None]>threshold[:,None,:]
        assert np.array_equal(prob,stored_p);brier=(prob-obs)**2
        n=draws.shape[1];order=np.sort(draws,axis=1)
        crps=np.abs(draws-y[:,None,:]).mean(1)-((2*np.arange(1,n+1)-n-1)[None,:,None]*order).sum(1)/(n*n)
        assert np.allclose(crps,stored_crps,rtol=1e-10,atol=1e-12)
        hourly=np.einsum('n,nhq->hq',w,brier)
        scores=[r for r in read(folder/'scores.json') if r['country']=='CN']
        for row in scores:assert abs(row['brier']-hourly[row['horizon']-1,0 if row['target'].endswith('Q95') else 1])<1e-12
        bands={}
        for left,right in [(0,3),(3,6),(6,12),(12,24)]:
            band=hourly[left:right]
            bands[f'{left+1}-{right}']=dict(mean_q95_q99_brier=float(band.mean()),
                q95_brier=float(band[:,0].mean()),q99_brier=float(band[:,1].mean()))
        record=dict(id=entry['id'],label=entry['label'],route=route,state='complete',brier=float(hourly.mean()),
            q95_brier=float(hourly[:,0].mean()),q99_brier=float(hourly[:,1].mean()),crps_q95=float((w@crps).mean()),
            epoch=complete.get('selected',{}).get('epoch'),updates=complete.get('updates'),lead_bands=bands,by_threshold={})
        wt=np.broadcast_to(w[:,None]/24,y.shape)
        for j,label in enumerate(['Q95','Q99']):
            positive=obs[:,:,j];negative=~positive
            record['by_threshold'][label]=dict(positive_brier=float((wt*brier[:,:,j]*positive).sum()/(wt*positive).sum()),
                negative_brier=float((wt*brier[:,:,j]*negative).sum()/(wt*negative).sum()),
                mean_probability_on_positive=float((wt*prob[:,:,j]*positive).sum()/(wt*positive).sum()),
                mean_probability=float((wt*prob[:,:,j]).sum()),prevalence=float((wt*positive).sum()))
        for station,event in sorted({(r['canonical_station'],r['event']) for r in rows}):
            ii=[i for i,r in enumerate(rows) if (r['canonical_station'],r['event'])==(station,event)]
            events.append(dict(id=entry['id'],route=route,station=station,event=event,brier=float(brier[ii].mean()),crps_q95=float(crps[ii].mean())))
        metrics.append(record);metadata[entry['id']]=rows;histories[entry['id']]=read(folder/'epochs.json',[])
        history=histories[entry['id']]
        selected_epoch=record['epoch']
        at_selected=next((r for r in history if r['epoch']==selected_epoch),None)
        last=history[-1] if history else None
        if history:
            if at_selected is None:raise ValueError('selected epoch missing from saved learning curve')
            definitions={r.get('train_loss_definition') for r in history}
            if len(definitions)>1:raise ValueError('training loss definition changed within a run')
            model_config=read(folder/'model_config.json',{})
            loss_definition=model_config.get('loss_definition')
            if any(any(r.get(key)!=loss_definition for key in ('train_loss_definition','fit_loss_definition','select_loss_definition')) for r in history):
                raise ValueError('train/fit/select loss definition differs from saved model config')
            best_selection=min(history,key=lambda r:r['selection_metric'])
            if best_selection['epoch']!=selected_epoch:raise ValueError('selected checkpoint differs from saved selection curve')
            record['learning_curve_summary']=dict(loss_definition=next(iter(definitions)),
                selected_epoch=selected_epoch,best_selection_epoch=best_selection['epoch'],
                selected=dict(train_loss=at_selected.get('train_loss'),fit_loss=at_selected.get('fit_loss'),
                    select_loss=at_selected.get('select_loss'),selection_brier=at_selected['selection_metric']),
                last=dict(epoch=last['epoch'],train_loss=last.get('train_loss'),fit_loss=last.get('fit_loss'),
                    select_loss=last.get('select_loss'),selection_brier=last.get('selection_metric')),
                interpretation='记录曲线形态供解释；训练/选模差异本身不能证明过拟合')
        case_rows=[]
        bundle=read(folder/'report_bundle.json',{})
        bundle_cases={r.get('case_id'):r for r in bundle.get('cases',[])}
        sample_lookup={r['sample_id']:i for i,r in enumerate(rows)}
        for sid in cfg.get('case_ids',[]):
            case=bundle_cases.get(sid)
            if case is None or sid not in sample_lookup or len(case.get('truth',[]))!=y.shape[1]:
                case_rows.append(dict(case_id=sid,state='unknown',reason='fixed_case_truth_not_in_training_report_bundle'))
                continue
            i=sample_lookup[sid];lead=int(np.argmax(case['truth']));scale=float(rows[i]['q_scale'])
            if not np.allclose(np.asarray(case['truth'],dtype=float),y[i]*scale,rtol=1e-10,atol=1e-8):
                raise ValueError('fixed case truth differs from saved selected prediction identity')
            observed=float(case['truth'][lead]);sample_values=draws[i,:,lead]*scale
            center=float(sample_values.mean());median=float(np.median(sample_values))
            case_rows.append(dict(case_id=sid,case_kind=case.get('case_kind'),state='observed',
                observed_peak_lead=lead+1,observed_peak_flow_m3_s=observed,
                final_sample_mean_at_observed_peak_m3_s=center,median_at_observed_peak_m3_s=median,
                center_underpredicts=bool(center<observed),median_underpredicts=bool(median<observed),
                fraction_final_samples_below_observed_peak=float((sample_values<observed).mean()),
                all_final_samples_below_observed_peak=bool((sample_values<observed).all()),
                interpretation='该提前量的边际样本比较；不推断24小时联合峰值分布'))
        record['fixed_case_peak_leads']=case_rows
        predictions[entry['id']]=dict(probability=prob,truth=y,median=np.median(draws,axis=1),
            lower=np.quantile(draws,.05,axis=1),upper=np.quantile(draws,.95,axis=1),hourly_brier=hourly)
        del draws,order
    baseline=cfg['baseline'];variants=cfg['variants'];pairs=[]
    by_id={r['id']:r for r in metrics if r.get('state')=='complete'}
    baseline_metrics=by_id.get(baseline);state_metrics=by_id.get('state_pq')
    full_state_reference=cfg.get('full_state_reference');full_state_metrics=by_id.get(full_state_reference)
    for record in by_id.values():
        if baseline_metrics:
            record['relative_primary_brier_percent_vs_diffpts']=100*(record['brier']-baseline_metrics['brier'])/baseline_metrics['brier'] if baseline_metrics['brier'] else None
            record['lead_brier_delta_vs_diffpts']={band:record['lead_bands'][band]['mean_q95_q99_brier']-baseline_metrics['lead_bands'][band]['mean_q95_q99_brier'] for band in record['lead_bands']}
        if state_metrics:
            record['lead_brier_delta_vs_state_pq']={band:record['lead_bands'][band]['mean_q95_q99_brier']-state_metrics['lead_bands'][band]['mean_q95_q99_brier'] for band in record['lead_bands']}
            if baseline_metrics:
                numerator=baseline_metrics['lead_bands']['1-3']['mean_q95_q99_brier']-record['lead_bands']['1-3']['mean_q95_q99_brier']
                denominator=baseline_metrics['lead_bands']['1-3']['mean_q95_q99_brier']-state_metrics['lead_bands']['1-3']['mean_q95_q99_brier']
                record['short_gain_retention_vs_state_pq']=numerator/denominator if denominator else None
                record['short_gain_retention_state']='descriptive_only; not a selection metric or threshold' if denominator else 'not_evaluable_zero_prior_state_pq_gain'
        if full_state_metrics:
            record['full_state_reference_id']=full_state_reference
            record['primary_brier_delta_vs_full_state_reference']=record['brier']-full_state_metrics['brier']
            record['lead_brier_delta_vs_full_state_reference']={band:record['lead_bands'][band]['mean_q95_q99_brier']-full_state_metrics['lead_bands'][band]['mean_q95_q99_brier'] for band in record['lead_bands']}
    lookup={(r['id'],r['station'],r['event']):r for r in events}
    for row in events:
        key=(baseline,row['station'],row['event'])
        if row['id'] in variants and row['route']=='pq' and key in lookup:
            pairs.append(dict(id=row['id'],station=row['station'],event=row['event'],delta_brier=row['brier']-lookup[key]['brier'],delta_crps=row['crps_q95']-lookup[key]['crps_q95']))
    result=dict(metrics=metrics,events=events,paired_deltas_vs_baseline=pairs,
        checks='saved probability, hourly Brier and CRPS independently recomputed; identical within-route China origins',
        scope='single-seed select development comparison; partial configurations excluded from ranking; no reserved observations')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    available=[key for key in [baseline]+variants if key in predictions]
    if available:
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        labels={r['id']:r['label'] for r in metrics if r['state']=='complete'};colors=['#444444','#2474a6','#7544ac','#e49b20','#1b9e77']
        with PdfPages(out/'state_comparison.pdf') as pdf:
            fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
            for key,color in zip(available,colors):
                for j in range(2):axes[j].plot(range(1,25),predictions[key]['hourly_brier'][:,j],label=labels[key],color=color)
                history=histories[key]
                if history:axes[2].plot([r['epoch'] for r in history],[r['selection_metric'] for r in history],label=labels[key],color=color)
            for ax,title in zip(axes,['Q95 risk error','Q99 risk error','Checkpoint selection']):
                ax.set(title=title,xlabel='Lead (hours)' if ax!=axes[2] else 'Epoch',ylabel='Brier (lower is better)');ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.suptitle('China select | same P+Q sample pool | seed 2')
            fig.savefig(out/'01_state_comparison.png',dpi=160);pdf.savefig(fig);plt.close(fig)
            case_ids=cfg.get('case_ids',[])
            if case_ids and baseline in predictions:
                fig,axes=plt.subplots(2,len(case_ids),figsize=(6*len(case_ids),7),squeeze=False,constrained_layout=True)
                rows=metadata[baseline];lookup_ids={r['sample_id']:i for i,r in enumerate(rows)}
                for col,sid in enumerate(case_ids):
                    index=lookup_ids[sid];row=rows[index];scale=row['q_scale'];truth=predictions[baseline]['truth'][index]*scale
                    axes[0,col].plot(range(1,25),truth,color='black',label='Observed',lw=2)
                    for key,color in zip(available,colors):
                        axes[0,col].plot(range(1,25),predictions[key]['median'][index]*scale,color=color,label=labels[key])
                        axes[1,col].plot(range(1,25),predictions[key]['probability'][index,:,0],color=color,label=labels[key])
                    axes[0,col].axhline(row['q95'],ls='--',color='gray',label='Fit Q95')
                    axes[1,col].step(range(1,25),(truth>row['q95']).astype(int),color='black',where='mid',label='Observed exceedance')
                    axes[0,col].set(title=row['station']+' / event '+row['event'].rsplit('_',1)[-1],ylabel='Flow (m3/s)')
                    axes[1,col].set(xlabel='Lead (hours)',ylabel='Q95 exceedance probability',ylim=(-.03,1.03))
                    for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
                fig.suptitle('Previously selected difficult examples | fixed before state experiment | descriptive cases')
                fig.savefig(out/'02_state_cases.png',dpi=160);pdf.savefig(fig);plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)
    return result


def run_review(index,out):
    return main(Path(out).parent/'review_index.json',out)


if __name__=='__main__':main()
