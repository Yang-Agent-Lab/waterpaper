"""Server-only read-only W/E report from saved predictions; no model execution."""
import argparse
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

COMMIT = '983710c3c727cc1929b14610d4ad8db95cfac677'
ARMS = ('D', 'W', 'E')
ID = re.compile(r'"window_id"\s*:\s*("(?:\\.|[^"\\])*")')
CHINA = re.compile(r'"country"\s*:\s*"China"')
PRIMARY = re.compile(r'"primary_origin"\s*:\s*true')


def read(path):
    return json.loads(path.read_text())


def china_primary(path, expected):
    found = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if not CHINA.search(line) or not PRIMARY.search(line):
                continue
            row = json.loads(line)
            if row['window_id'] in found:
                raise ValueError('Duplicate China primary window')
            found[row['window_id']] = row
            if len(found) == expected:
                break
    if len(found) != expected:
        raise ValueError(f'{path}: expected {expected} China primary rows, got {len(found)}')
    return found


def selected_rows(path, wanted):
    found = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if not CHINA.search(line):
                continue
            match = ID.search(line)
            if match and json.loads(match.group(1)) in wanted:
                row = json.loads(line)
                found[row['window_id']] = row
                if set(found) == wanted:
                    break
    if set(found) != wanted:
        raise ValueError(f'Missing selected cases: {wanted-set(found)}')
    return found


def metrics(truth, prediction):
    return dict(window_peak_error=abs(max(prediction)-max(truth)),
                sequence_mae=statistics.mean(abs(a-b) for a,b in zip(truth,prediction)))


def stratified_experts(records):
    predicates = {
        'all_China_primary': lambda r: True,
        'future_high': lambda r: r['future_high'],
        'prethreshold_future_high': lambda r: r['future_high'] and r['q_now_normalized'] <= r['train_q95_normalized'],
        'already_high_future_high': lambda r: r['future_high'] and r['q_now_normalized'] > r['train_q95_normalized'],
        'non_high_future': lambda r: not r['future_high'],
        'retreat_or_low_response': lambda r: max(r['observed_normalized_q']) <= r['q_now_normalized'],
    }
    output = {}
    for name, predicate in predicates.items():
        chosen = [r for r in records if predicate(r)]
        routes = {}
        for route in ('regular', 'high', 'mixture', 'oracle'):
            events = defaultdict(list)
            for r in chosen:
                selected_route = ('high' if r['future_high'] else 'regular') if route == 'oracle' else route
                events[r['station_uid'],r['group_id']].append(metrics(r['observed_normalized_q'],r[selected_route+'_normalized_q']))
            event_rows, stations = [], defaultdict(list)
            for (station,group), values in events.items():
                row = dict(station_uid=station,group_id=group,origins=len(values),
                           **{key:statistics.mean(v[key] for v in values) for key in ('window_peak_error','sequence_mae')})
                event_rows.append(row);stations[station].append(row)
            station_rows = [dict(station_uid=station,events=len(values),origins=sum(v['origins'] for v in values),
                                 **{key:statistics.mean(v[key] for v in values) for key in ('window_peak_error','sequence_mae')})
                            for station,values in stations.items()]
            routes[route] = dict(origins=len(chosen),events=len(event_rows),stations=len(station_rows),
                                 **{key:statistics.mean(v[key] for v in station_rows) if station_rows else None for key in ('window_peak_error','sequence_mae')},
                                 event_rows=event_rows,station_rows=station_rows,
                                 state='EVALUABLE' if chosen else 'NOT_EVALUABLE')
        output[name] = dict(denominator_origins=len(chosen),routes=routes)
    return dict(strata=output,weighting='within original source event mean, then within station mean, then station equal',
                units='normalized Q errors, dimensionless',oracle='truth-assigned expert diagnostic only; not operational or ranked')


def gate_counts(records):
    output = {}
    groups = dict(already_high_future_high=[r for r in records if r['future_high'] and r['q_now_normalized']>r['train_q95_normalized']],
                  prethreshold_future_high=[r for r in records if r['future_high'] and r['q_now_normalized']<=r['train_q95_normalized']],
                  future_high=[r for r in records if r['future_high']],non_high_future=[r for r in records if not r['future_high']])
    for name, rows in groups.items():
        count = sum(r['gate'] >= .5 for r in rows)
        output[name] = dict(origins=len(rows),gate_at_least_half_count=count,
                            percent=100*count/len(rows) if rows else None,
                            gate_mean=statistics.mean(r['gate'] for r in rows) if rows else None)
    return output


def collect(run_dir):
    record = read(run_dir/'run_record.json')
    if record['head'] != COMMIT or record['exit_code'] != 0:
        raise ValueError('Require completed formal execution 983710c')
    artifacts = run_dir/'artifacts';cfg = read(artifacts/'phase_config.json');seeds = cfg['seeds']
    if set(seeds) != {2,17,42} or cfg['smoke_run']:
        raise ValueError('Require the approved three-seed formal run')
    sources = [str(run_dir/'run_record.json'),str(artifacts/'phase_config.json')]
    country_rows = read(artifacts/'country_seed_summary.json')
    for country in {r['country'] for r in country_rows}:
        for arm in ARMS:
            if sorted(r['seed'] for r in country_rows if r['country']==country and r['arm']==arm) != sorted(seeds):
                raise ValueError('Incomplete country/arm seed summary')
    means = [dict(country=country,arm=arm,seeds=len(seeds),
                  **{key:statistics.mean(r[key] for r in country_rows if r['country']==country and r['arm']==arm)
                     for key in ('window_peak_error','sequence_mae','timing_abs_hours')})
             for country in sorted({r['country'] for r in country_rows}) for arm in ARMS]
    china, expert_errors, gates, false_alarms, events, old_absolute = {}, {}, {}, [], {}, []
    for seed in seeds:
        e_dir = artifacts/f'seed_{seed}/E'
        primary = read(e_dir/'primary_metrics.json')
        expected = sum(r['origins'] for r in primary['events'] if r['country']=='China')
        preds_path = e_dir/'validation_predictions.jsonl.gz';side_path = e_dir/'expert_gate_sidecar.jsonl.gz'
        predictions = china_primary(preds_path,expected);sides = china_primary(side_path,expected)
        if set(predictions) != set(sides):
            raise ValueError('China mixture/sidecar IDs do not match')
        rows = []
        for wid,p in predictions.items():
            s = sides[wid]
            if p['station_uid'] != s['station_uid'] or p['group_id'] != s['group_id'] or p['qscale'] != s['qscale']:
                raise ValueError('Prediction/sidecar identity mismatch')
            if s['future_high'] != any(s['high95_hours']):
                raise ValueError('Authoritative future-high flag mismatch')
            vectors = [p['observed_normalized_q'],s['regular_normalized_q'],s['high_normalized_q'],s['mixture_normalized_q']]
            if any(len(v)!=12 or not all(math.isfinite(x) for x in v) for v in vectors):
                raise ValueError('Require 12 finite values for every curve')
            if not all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7) for a,b in zip(p['predicted_normalized_q'],s['mixture_normalized_q'])):
                raise ValueError('Saved final mixture disagrees with sidecar')
            rows.append(dict(window_id=wid,station_uid=p['station_uid'],group_id=p['group_id'],origin_time=p['origin_time'],
                             observed_normalized_q=p['observed_normalized_q'],observed_q=p['observed_q'],mixture_q=p['predicted_q'],
                             qscale=p['qscale'],q_now_normalized=p['q_now_normalized'],train_q95_normalized=s['train_q95_normalized'],
                             high95_hours=s['high95_hours'],future_high=s['future_high'],gate=s['gate'],gate_logit=s['gate_logit'],
                             **{k:s[k] for k in ('regular_normalized_q','high_normalized_q','mixture_normalized_q')}))
        china[str(seed)] = rows;expert_errors[str(seed)] = stratified_experts(rows);gates[str(seed)] = gate_counts(rows)
        for arm in ARMS:
            folder = artifacts/f'seed_{seed}/{arm}'
            strata_path = folder/'stratified_metrics.json'
            strata = [r for r in read(strata_path)['rows'] if r['level']=='country' and r['country']=='China' and not r['future_high']]
            denom = sum(r['nonhigh_denominator'] for r in strata);count = sum(r['high_false_positive_count'] for r in strata)
            if denom != gates[str(seed)]['non_high_future']['origins']:
                raise ValueError('Non-high denominator differs between saved strata and China labels')
            false_alarms.append(dict(seed=seed,arm=arm,nonhigh_denominator=denom,predicted_above_q95_count=count,percent=100*count/denom if denom else None))
            peak_path = folder/'event_peak_scores.json';peaks = [r for r in read(peak_path) if r['lead']==6]
            events[seed,arm] = {(r['station_uid'],r['event_group']):r for r in peaks if r['country']=='China'}
            station_cells = defaultdict(list)
            for r in peaks:station_cells[r['country'],r['station_uid']].append(r)
            for country in sorted({r['country'] for r in peaks}):
                values = [rs for (c,_),rs in station_cells.items() if c==country]
                old_absolute.append(dict(seed=seed,arm=arm,country=country,stations=len(values),events=sum(len(rs) for rs in values),
                                         weighting='event_mean_then_station_median',
                                         ape=statistics.median(statistics.mean(r['ape'] for r in rs) for rs in values),
                                         timing_abs_hours=statistics.median(statistics.mean(r['timing_abs_hours'] for r in rs) for rs in values)))
            sources.extend((str(strata_path),str(peak_path)))
        sources.extend((str(preds_path),str(side_path),str(e_dir/'primary_metrics.json')))
    if any({r['window_id'] for r in rows} != {r['window_id'] for r in china[str(seeds[0])]} for rows in china.values()):
        raise ValueError('China primary IDs differ between seeds')
    indexed = {int(seed):{r['window_id']:r for r in rows} for seed,rows in china.items()}
    common = set.intersection(*(set(rows) for rows in events.values()))
    candidates = []
    for station,event in sorted(common):
        ids = {events[s,a][station,event]['window_id'] for s in seeds for a in ARMS}
        if len(ids)!=1:
            raise ValueError('Old lead6 window mismatch')
        wid = ids.pop();gaps=[];peak_differences=[];ew=[]
        for seed in seeds:
            r=indexed[seed][wid];truth=events[seed,'E'][station,event]['observed_event_peak_q']/r['qscale']
            mixed=max(r['mixture_normalized_q']);high=max(r['high_normalized_q'])
            gaps.append(abs(mixed-truth)-abs(high-truth));peak_differences.append(mixed-high)
            ew.append(events[seed,'E'][station,event]['ape']-events[seed,'W'][station,event]['ape'])
        candidates.append(dict(station_uid=station,event_group=event,window_id=wid,
                               mean_mixture_minus_high_peak_error=statistics.mean(gaps),seed_mixture_minus_high_peak_error=gaps,
                               mean_mixture_minus_high_peak=statistics.mean(peak_differences),
                               mean_E_minus_W_ape_pp=statistics.mean(ew),seed_E_minus_W_ape_pp=ew))
    fixed = [r for r in candidates if r['station_uid']=='da_tl_cn:Liqingdian' and re.search(r'洪水[_\s]*6$',r['event_group'])]
    if len(fixed)!=1 or len(candidates)<3:
        raise ValueError('Require unique Liqingdian flood6 and at least three paired lead6 events')
    first=dict(fixed[0],case='固定极端失败案例')
    remaining=[r for r in candidates if r['window_id']!=first['window_id']]
    chosen=max(remaining,key=lambda r:(r['mean_mixture_minus_high_peak_error'],r['window_id']))
    second=dict(chosen,case='混合较高专家误差更大' if chosen['mean_mixture_minus_high_peak_error']>0 else '差距最大，但未见混合更差',
                positive_gap_signal=chosen['mean_mixture_minus_high_peak_error']>0,
                possible_peak_suppression=chosen['mean_mixture_minus_high_peak_error']>0 and chosen['mean_mixture_minus_high_peak']<0)
    remaining=[r for r in remaining if r['window_id']!=second['window_id']]
    chosen=max(remaining,key=lambda r:(r['mean_E_minus_W_ape_pp'],r['window_id']))
    third=dict(chosen,case='E相对W退步最大' if chosen['mean_E_minus_W_ape_pp']>0 else '余下E−W差值最大，未见退步')
    cases=[first,second,third];wanted={r['window_id'] for r in cases}
    reference=read(artifacts/'seed_2/D/reference.json')
    d_path=Path(reference['predictions_retained_at_source']);w_path=artifacts/'seed_2/W/validation_predictions.jsonl.gz'
    curves={a:selected_rows(p,wanted) for a,p in dict(D=d_path,W=w_path).items()}
    for case in cases:
        e=indexed[2][case['window_id']]
        physical={a:rows[case['window_id']]['predicted_q'] for a,rows in curves.items()}
        for arm in curves:
            r=curves[arm][case['window_id']]
            if r['observed_q']!=e['observed_q'] or r['qscale']!=e['qscale']:
                raise ValueError('Case truth differs across D/W/E')
        physical['E']=e['mixture_q']
        physical.update({a:[v*e['qscale'] for v in e[key]] for a,key in (('高专家','high_normalized_q'),('常规专家','regular_normalized_q'))})
        case['series']=dict(observed_q=e['observed_q'],q_now=e['q_now_normalized']*e['qscale'],predicted_q=physical,
                            gate=e['gate'],qscale=e['qscale'],curve_seed=2,truth_event_peak_q=events[2,'E'][case['station_uid'],case['event_group']]['observed_event_peak_q'])
    composition_path=artifacts/'train_composition.json';old_path=artifacts/'paired_old_peak_summary.json'
    sources.extend((str(d_path),str(w_path),str(artifacts/'country_seed_summary.json'),str(composition_path),str(old_path)))
    return dict(execution_commit=COMMIT,source_run=str(run_dir),seeds=seeds,China_primary_predictions=china,
                China_expert_stratified_metrics=expert_errors,China_gate_counts=gates,China_nonhigh_false_alarms=false_alarms,
                country_seed_summary=country_rows,country_seed_means=means,
                old6_paired_country_summary=[r for r in read(old_path) if r['lead']==6],old6_absolute_country_summary=old_absolute,
                train_composition=read(composition_path),case_candidates=candidates,cases=cases,source_files=sources,
                units=dict(Q='m³/s',normalized_Q_error='dimensionless',timing='hours',APE_difference='percentage points'),
                scope='saved development validation; China event fragments, not year-round operational false alarms',
                gate_interpretation='mixture weight, not calibrated flood probability; gate>=0.5 means high-expert weight at least one half',
                case_selection='posthoc old peak-minus-6h origins; fixed flood6, largest remaining mixture-vs-high error gap, largest remaining E-W APE difference',
                curve_interpretation='fixed seed2 individual-model curves, no seed averaging or operational trigger',
                calculation_precision='Python arithmetic on saved float values; no original role-file reread')


def draw(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    fonts=[f for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if not fonts:
        raise RuntimeError('Require server Noto Sans CJK font')
    plt.rcParams.update({'font.family':fonts[0].name,'font.size':10,'axes.unicode_minus':False})
    colors={'D':'#777777','W':'#0072B2','E':'#009E73','高专家':'#D55E00','常规专家':'#CC79A7'}
    figures=[];fig,axes=plt.subplots(2,2,figsize=(12,9))
    fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.87,wspace=.33,hspace=.48)
    for ax,key,title,unit in ((axes[0,0],'window_peak_error','中国窗口峰值误差','归一化绝对误差（无量纲，越低越好）'),
                              (axes[0,1],'timing_abs_hours','中国窗口峰时误差','绝对误差（小时，越低越好）')):
        for i,arm in enumerate(ARMS):
            values=[next(r[key] for r in data['country_seed_summary'] if r['country']=='China' and r['arm']==arm and r['seed']==s) for s in data['seeds']]
            ax.scatter([i+j for j in (-.12,0,.12)],values,color=colors[arm],s=32,zorder=3)
            ax.plot([i-.25,i+.25],[statistics.mean(values)]*2,color=colors[arm],linewidth=2.5)
        ax.set_xticks(range(3),ARMS);ax.set(title=title,ylabel=unit);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    for i,(name,label) in enumerate((('already_high_future_high','当前已高且未来高'),('prethreshold_future_high','当前未高且未来高'))):
        rows=[data['China_gate_counts'][str(s)][name] for s in data['seeds']]
        if len({r['origins'] for r in rows})!=1 or any(r['percent'] is None for r in rows):
            raise ValueError('Gate panel requires consistent nonempty state denominators')
        values=[r['percent'] for r in rows]
        ax.scatter([i+j for j in (-.12,0,.12)],values,color=colors['E'],s=32)
        ax.plot([i-.25,i+.25],[statistics.mean(values)]*2,color=colors['E'],linewidth=2.5)
        ax.text(i,-.19,f"{label}\nn={rows[0]['origins']}",ha='center',va='top',transform=ax.get_xaxis_transform())
    ax.set_xticks([]);ax.set(xlim=(-.5,1.5),ylim=(0,105),title='E：高专家权重≥一半的起点比例',ylabel='g≥0.5 的比例（%）');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1]
    for i,arm in enumerate(ARMS):
        rows=[next(r for r in data['China_nonhigh_false_alarms'] if r['seed']==s and r['arm']==arm) for s in data['seeds']]
        values=[r['percent'] for r in rows]
        ax.scatter([i+j for j in (-.12,0,.12)],values,color=colors[arm],s=32)
        ax.plot([i-.25,i+.25],[statistics.mean(values)]*2,color=colors[arm],linewidth=2.5)
    denominators={r['nonhigh_denominator'] for r in data['China_nonhigh_false_alarms']}
    if len(denominators)!=1:
        raise ValueError('False-alarm denominators differ')
    ax.set_xticks(range(3),ARMS);ax.set(title=f'非高目标窗口预测越过Q95（每seed n={denominators.pop()}）',ylabel='误报比例（%）');ax.grid(axis='y',alpha=.2)
    fig.suptitle('高流量关注与专家分工：峰值、峰时及误报',fontsize=16,y=.97)
    fig.text(.5,.925,'点为三个种子，横线为均值；主误差按原事件→站点等权；门控与误报比例按起点计数',ha='center',fontsize=10)
    fig.text(.5,.055,'中国资料为已保存洪水事件片段；非高目标不等于全年普通流量，误报比例不是业务全年误报率。',ha='center',fontsize=10)
    fig.text(.5,.024,f'来源：开发validation保存预测/sidecar/分层表；Q95仅来自训练；执行 {COMMIT[:7]}',ha='center',fontsize=9)
    figures.append(('page_1_summary',fig))
    fig,axes=plt.subplots(3,1,figsize=(12,11))
    fig.subplots_adjust(left=.10,right=.97,bottom=.12,top=.80,hspace=.70)
    for ax,case in zip(axes,data['cases']):
        s=case['series'];now=s['q_now'];ax.plot(range(13),[now,*s['observed_q']],color='black',linewidth=2,label='实测')
        for arm in ('D','W','E','高专家','常规专家'):
            ax.plot(range(13),[now,*s['predicted_q'][arm]],color=colors[arm],linewidth=1.8 if arm=='E' else 1.2,
                    linestyle='--' if arm in ('高专家','常规专家') else '-',label=arm)
        ax.axvline(0,color='#999999',linestyle=':');ax.set_xticks(range(13));ax.grid(alpha=.2)
        event=case['event_group'].rsplit(':',1)[-1];station=case['station_uid'].split(':',1)[-1]
        ax.set(xlim=(-.3,12.3),xlabel='相对起报时间（小时；0为起报Q）',ylabel='流量（m³/s）',
               title=f"{case['case']}｜{station} / {event}\ng={s['gate']:.3f} 是混合权重，不是洪水概率；曲线固定 seed2")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.5,.895),ncol=6,fontsize=10)
    fig.suptitle('真实旧峰前6小时案例：保留极端失败与退步',fontsize=16,y=.98)
    fig.text(.5,.94,'事后按三seed差值选例；混合与两专家均为已保存的单个seed2预测；不使用平均曲线或oracle作预测',ha='center',fontsize=10)
    fig.text(.5,.06,'第二例以混合峰误差−高专家峰误差排序；仅差值为正且混合峰更低时，才列为可能压低候选。',ha='center',fontsize=10)
    fig.text(.5,.025,f'来源：开发validation旧6h起点、原实测与保存预测；Q为m³/s；执行 {COMMIT[:7]}；不是业务触发',ha='center',fontsize=9)
    figures.append(('page_2_cases',fig))
    with PdfPages(out/'peak_specialization_report.pdf') as pdf:
        for name,figure in figures:
            figure.savefig(out/(name+'.png'),dpi=160);figure.savefig(out/(name+'.pdf'));pdf.savefig(figure);plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run_dir=args.run_dir.resolve();out=args.out.resolve()
    if out==run_dir or out.is_relative_to(run_dir):
        raise ValueError('Use a new output directory outside the original run')
    if out.exists():
        raise FileExistsError(out)
    data=collect(run_dir);out.mkdir(parents=True,exist_ok=False)
    with (out/'plot_source.json').open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
    draw(data,out)


if __name__=='__main__':main()
