"""CPU-only aggregation of the immutable 18-run experiment; never loads weights."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from .data import save,table,digest

PAIRS=[('P2','P1'),('P4','P3'),('P3','P1'),('P4','P2'),('P5','P4')]
METRICS=['brier','ap','auc','Q6_mae','Q6_mae_m3_s','process6_mae','process12_mae',
         'process12_crps','peak6_crps','peak12_crps','peak6_mae','peak12_mae',
         'peak6_mae_m3_s','peak12_mae_m3_s','peak6_bias_m3_s','peak12_bias_m3_s',
         'peak6_timing_mae','peak12_timing_mae','process90_coverage','process90_width',
         'peak6_90_coverage','peak6_90_width','zero_mass']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csvread(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def number(value):
    return None if value in ('',None) else float(value)


def one(rows,**filters):
    chosen=[r for r in rows if all(r.get(k)==v for k,v in filters.items())]
    assert len(chosen)==1,(filters,len(chosen))
    return chosen[0]


def delta(a,b):
    return a-b if a is not None and b is not None else None


def summarize(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for identity,group in sorted(groups.items()):
        out=dict(zip(keys,identity));out['runs']=len(group)
        for metric in metrics:
            values=[r[metric] for r in group if r.get(metric) is not None]
            out[metric+'_n']=len(values)
            out[metric+'_mean']=statistics.mean(values) if values else None
            out[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
        result.append(out)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    record=read(source/'run_record.json');assert record['state']=='complete' and record['exit_code']==0
    assert record['head']=='48d2929b483476021c435286651f369899d34cd4'
    root=source/'artifacts';execution=read(root/'execution.json');config=execution['config']
    assert config['arms']==['P0','P1','P2','P3','P4','P5'] and config['seeds']==[2,17,42]
    ledger=read(Path(execution['budget_ledger']));assert ledger['active'] is None
    runs=[];optim=[];summaries=[];reliabilities=[];alarms=[];discoveries=[];cases=[];index=[]
    lookup={};slot_hashes={}
    for seed in config['seeds']:
        for arm in config['arms']:
            folder=root/f'seed_{seed}'/arm;dev=folder/'development'
            complete=read(folder/'completion.json');selected=read(folder/'selection.json');definition=read(folder/'model_definition.json')
            history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
            assert complete['state']=='complete' and complete['updates']==22600 and len(history)==100
            slot_hashes.setdefault(seed,set()).add(definition['slot_digest'])
            last=history[-1];tail=history[-10:];ys=[r['selection_score'] for r in tail]
            # Slope is descriptive optimization evidence, never a new selection rule.
            meanx=statistics.mean([r['epoch'] for r in tail]);meany=statistics.mean(ys)
            slope=sum((r['epoch']-meanx)*(r['selection_score']-meany) for r in tail)/sum((r['epoch']-meanx)**2 for r in tail)
            opt=dict(seed=seed,arm=arm,epochs=len(history),updates=complete['updates'],best_epoch=selected['epoch'],
                best_select=selected['score'],last_select=last['selection_score'],last_minus_best=last['selection_score']-selected['score'],
                last10_first=ys[0],last10_last=ys[-1],last10_change=ys[-1]-ys[0],last10_mean=meany,
                last10_min=min(ys),last10_max=max(ys),last10_slope_per_epoch=slope,
                tail_batches=complete['tail_batches'],all_zero_tail_examples=complete['all_zero_tail_examples'],
                tail_example_denominator=complete['tail_denominator'],active_parameters=definition['active'],total_parameters=definition['total'],
                model_gpu_seconds=ledger['models'][f'{seed}/{arm}'])
            assert abs(selected['score']-min(r['selection_score'] for r in history))<1e-12
            optim.append(opt)
            rows=csvread(dev/'summary.csv');lookup[seed,arm]=rows
            primary=one(rows,level='overall',stratum='current_not_high',horizon='6',probability='calibrated')
            overall=one(rows,level='overall',stratum='all',horizon='6',probability='calibrated')
            raw=one(rows,level='overall',stratum='current_not_high',horizon='6',probability='raw')
            baseline=one(rows,level='overall',stratum='current_not_high',horizon='6',probability='fit_constant')
            assert int(float(overall['n']))==1356
            line=one(csvread(dev/'alarms.csv'),horizon='6',limit='0.05')
            detected=csvread(dev/'event_discovery_h6.csv');eligible=[r for r in detected if r['state']=='evaluable']
            row=dict(seed=seed,arm=arm,state=complete['state'],development_windows=1356,primary_windows=int(float(primary['n'])),
                **{'primary_'+k:number(primary.get(k)) for k in METRICS},
                **{'all_'+k:number(overall.get(k)) for k in METRICS},
                raw_primary_brier=number(raw['brier']),raw_primary_ap=number(raw['ap']),
                fit_constant_primary_brier=number(baseline['brier']),primary_prevalence=number(primary['prevalence']),
                fpr_at_cal5=number(line['fpr']),recall_at_cal5=number(line['recall']),precision_at_cal5=number(line['precision']),
                source_event_labels=len(detected),evaluable_discovery_events=len(eligible),
                not_evaluable_discovery_events=len(detected)-len(eligible),discovered_events=sum(r['discovered']=='True' for r in eligible),
                **{k:v for k,v in opt.items() if k not in ['seed','arm']})
            runs.append(row)
            for role in ['calibration','development']:
                for filename,sink in [('summary.csv',summaries),('reliability.csv',reliabilities),('alarms.csv',alarms),('event_discovery_h6.csv',discoveries)]:
                    sink.extend(dict(seed=seed,arm=arm,source_role=role,**r) for r in csvread(folder/role/filename))
            for case in read(dev/'difficult_peak_cases.json'):
                truth=case['true_future_Q'];point=case['median_future_Q'];peak=case['trajectory_peak6_quantiles']
                cases.append(dict(seed=seed,arm=arm,station=case['station'],event=case['event'],window_id=case['window_id'],
                    origin_time=case['origin_time'],true_Q6=truth[5],predicted_Q6=point[5],true_peak6=max(truth[:6]),
                    predicted_peak6_median=peak[1],peak6_bias=peak[1]-max(truth[:6]),peak6_absolute_error=abs(peak[1]-max(truth[:6])),
                    peak6_q05=peak[0] if arm!='P0' else None,peak6_q95=peak[2] if arm!='P0' else None,
                    calibrated_risk_h6=case['calibrated_risk_h6'],forecast_point='median curve Q6; peak median computed from complete paths'))
    assert all(len(hashes)==1 for hashes in slot_hashes.values())
    comparisons=[];eventpairs=[]
    comparison_metrics=['primary_brier','primary_ap','primary_auc','primary_peak6_crps','primary_peak6_mae',
        'all_brier','all_peak6_crps','all_peak6_mae','all_Q6_mae','all_process12_mae','all_peak6_timing_mae',
        'fpr_at_cal5','recall_at_cal5','precision_at_cal5','discovered_events','model_gpu_seconds']
    for a,b in PAIRS:
        for seed in config['seeds']:
            ar=next(r for r in runs if r['seed']==seed and r['arm']==a);br=next(r for r in runs if r['seed']==seed and r['arm']==b)
            comparisons.append(dict(comparison=a+'-'+b,seed=seed,**{k:delta(ar.get(k),br.get(k)) for k in comparison_metrics}))
            left={tuple(r[k] for k in ['level','station','event','stratum']):r for r in lookup[seed,a] if r['horizon']=='6' and r['probability']=='calibrated' and r['level'] in ['station','event']}
            right={tuple(r[k] for k in ['level','station','event','stratum']):r for r in lookup[seed,b] if r['horizon']=='6' and r['probability']=='calibrated' and r['level'] in ['station','event']}
            assert left.keys()==right.keys()
            for key,r in left.items():
                s=right[key];eventpairs.append(dict(comparison=a+'-'+b,seed=seed,level=key[0],station=key[1],event=key[2],stratum=key[3],
                    candidate_n=number(r['n']),control_n=number(s['n']),
                    **{k:delta(number(r.get(k)),number(s.get(k))) for k in METRICS}))
    table(out/'all_18_runs.csv',runs);table(out/'optimization_all_18.csv',optim)
    table(out/'prespecified_contrasts_by_seed.csv',comparisons)
    table(out/'prespecified_contrasts_summary.csv',summarize(comparisons,['comparison'],comparison_metrics))
    main_metrics=['primary_brier','primary_ap','raw_primary_ap','all_brier','all_peak6_mae','all_Q6_mae','all_process12_mae',
                  'fpr_at_cal5','recall_at_cal5','discovered_events','model_gpu_seconds']
    table(out/'arm_summary.csv',summarize(runs,['arm'],main_metrics))
    table(out/'station_event_paired_differences.csv',eventpairs);table(out/'all_saved_summaries.csv',summaries)
    table(out/'all_saved_reliability.csv',reliabilities);table(out/'all_saved_alarms.csv',alarms)
    table(out/'all_event_discovery.csv',discoveries);table(out/'all_difficult_cases.csv',cases)
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        large=path.suffix in ['.pt','.npz'] or path.name=='window_scores.json'
        index.append(dict(relative_path=str(path.relative_to(source)),server_path=str(path),bytes=path.stat().st_size,
                          sha256=digest(path),storage='server_only' if large else 'local_small_file_mirror'))
    index.append(dict(relative_path='shared_budget_final_snapshot.json',server_path=execution['budget_ledger'],bytes=Path(execution['budget_ledger']).stat().st_size,
                      sha256=digest(execution['budget_ledger']),storage='local_small_file_mirror'))
    table(out/'evidence_index.csv',index);save(out/'shared_budget_final_snapshot.json',ledger)
    save(out/'report_manifest.json',dict(source=str(source),source_sha=record['head'],source_run_state='complete',runs=18,
        updates_per_run=22600,epochs_per_run=100,config=config,data_reference=read(root/'data_reference.json'),
        total_gpu_seconds=ledger['total_gpu_seconds'],evaluation_gpu_seconds=ledger['evaluation_gpu_seconds'],
        data_and_smoke_acceptance='Separate parent acceptance; this report only aggregates saved metrics, not trajectory recomputation',
        comparison_direction='candidate minus control; negative error deltas improve, positive AP/recall deltas improve; FPR differences remain visible',
        limitations=['Development repeatedly used historically; exploratory, no test access.',
            'Source event labels and overlapping windows are not independent storms.',
            'Qingyangcha fit has eight windows/two source labels/54 Q rows; three difficult stations lack positive high6 calibration windows.',
            'Fixed 100 epochs; last-ten score changes are descriptive, not a convergence guarantee.',
            'P0 has no forecast distribution interval; calibrated risk does not calibrate flow/timing distributions.']))
    print(json.dumps(dict(runs=18,source=str(source),out=str(out),gpu_seconds=ledger['total_gpu_seconds'],new_training=False),ensure_ascii=True))


if __name__=='__main__':main()
