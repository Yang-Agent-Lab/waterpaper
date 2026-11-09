"""Illustrate three real China train windows; no models, dev/test, or future P."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

MANIFEST = Path('./external-home/pooled_esdf_d781a3f_v2/results/experiments/esdf_projection_full_d781a3f_20261005/artifacts/model_manifest.json')


def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def labels(row):
    now = row['x_q'][-1]
    peak6 = max(row['y_q'][:6])
    peak12 = max(row['y_q'])
    return dict(Qnow=now, Q6=row['y_q'][5], Qpeak6=peak6,
                signed_peak_change6=peak6-now, rise6=max(0., peak6-now), Qpeak12=peak12,
                peak6_lead_hour=int(np.argmax(row['y_q'][:6])+1),
                peak12_lead_hour=int(np.argmax(row['y_q'])+1))


def choose(rows):
    observed = {}
    for r in rows:
        for source_row, q in zip(r['source_rows'], r['x_q']+r['y_q']):
            if source_row in observed:
                assert np.isclose(q, observed[source_row], rtol=0, atol=1e-7)
            observed[source_row] = q
    peak = max(observed.values())
    peak_row = min(i for i, q in observed.items() if q == peak)
    amplitude = peak-min(observed.values())
    small_history_change = .05*amplitude
    before, rising, falling = [], [], []
    for r in rows:
        tag = labels(r)
        d3 = r['x_q'][-1]-r['x_q'][-4]
        if r['origin_source_row'] < peak_row and tag['rise6'] > 0:
            if abs(d3) <= small_history_change:
                before.append(r)
            if d3 > small_history_change:
                rising.append(r)
        if r['origin_source_row'] > peak_row and d3 < 0 and tag['rise6'] == 0:
            falling.append(r)
    selected = [min(before, key=lambda r:(labels(r)['Qnow'],r['origin_source_row'])) if before else None,
                max(rising, key=lambda r:(labels(r)['rise6'],-r['origin_source_row'])) if rising else None,
                max(falling, key=lambda r:(labels(r)['Qnow'],-r['origin_source_row'])) if falling else None]
    return selected, dict(available_event_peak_normalized=peak, available_peak_source_row=peak_row,
                          history_3h_change_small_cutoff_normalized=small_history_change,
                          eligible_phase_counts=dict(before=len(before),rising=len(rising),falling=len(falling)))


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    # run_versioned creates the enclosing run directory before launching us.
    out.mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['source_data_commit'] == 'e50e3cfb1413338f876ba3475771a2116c25fe83'
    groups, sources = defaultdict(list), []
    for station in manifest['stations']:
        if station['country'] != 'China':
            continue
        path = Path(station['files']['train']['path'])
        assert path.name == 'train_windows.jsonl.gz'
        assert path.parent.name == station['station_uid'].replace(':','__')
        assert path.is_relative_to(Path(manifest['stageB_root'])/'stations')
        scales = station['scales']
        sources.append(dict(station=station['station_uid'],path=str(path),sha256=digest(path),
                            qscale=scales['qscale'],pscale=scales['pscale']))
        with gzip.open(path,'rt') as f:
            for line in f:
                raw = json.loads(line)
                # Future rainfall is never retrieved, plotted, or used to select examples.
                row = {k:raw[k] for k in ('window_id','station_uid','country','role','source_event_id',
                    'overlap_group_id','origin_time','origin_source_row','source_rows','source_times','x_p','x_q','y_q')}
                assert row['role']=='train' and row['country']=='China'
                assert len(row['x_p'])==len(row['x_q'])==len(row['y_q'])==12
                assert len(row['source_rows'])==len(row['source_times'])==24
                assert np.isfinite(row['x_p']+row['x_q']+row['y_q']).all()
                assert row['source_rows'][11]==row['origin_source_row']
                times=np.array(row['source_times'],dtype='datetime64[s]')
                assert np.all(np.diff(times)==np.timedelta64(1,'h'))
                row.update(qscale=scales['qscale'],pscale=scales['pscale'])
                groups[row['overlap_group_id']].append(row)
    assert sum(map(len,groups.values()))==8557
    order=sorted(groups,key=lambda e:('Xiagushan' not in e,
        -max(max(r['x_q']+r['y_q'])*r['qscale'] for r in groups[e]),e))
    attempts=[]
    for event in order:
        selected, details=choose(groups[event])
        attempts.append(dict(event=event,**details))
        if all(r is not None for r in selected):
            break
    else:
        event=order[0];selected,details=choose(groups[event])
    phases=['上涨前','上涨中','退水']
    example_records=[];source_rows=[]
    for phase,row in zip(phases,selected):
        if row is None:
            example_records.append(dict(phase=phase,state='NOT_EVALUABLE',reason='同事件没有满足本次阶段选择规则的合法窗口'))
            continue
        normalized=labels(row)
        physical={k:(float(v*row['qscale']) if not k.endswith('lead_hour') else v) for k,v in normalized.items()}
        assert np.isclose(physical['rise6'],max(0,physical['Qpeak6']-physical['Qnow']),rtol=0,atol=1e-9)
        assert np.isclose(physical['Q6'],row['y_q'][5]*row['qscale'])
        history_explanation={f'P_last_{h}_cells_sum_mm':float(sum(row['x_p'][-h:])*row['pscale']) for h in (1,3,6,12)}
        history_explanation.update(Qnow_m3_s=physical['Qnow'],
            Qnow_minus_1h_ago_m3_s=(row['x_q'][-1]-row['x_q'][-2])*row['qscale'],
            Qnow_minus_3h_ago_m3_s=(row['x_q'][-1]-row['x_q'][-4])*row['qscale'])
        record=dict(phase=phase,state='evaluable',**row,labels_normalized=normalized,labels_physical=physical,
                    history_explanation=history_explanation,
                    history_P_mm=(np.array(row['x_p'])*row['pscale']).tolist(),
                    history_Q_m3_s=(np.array(row['x_q'])*row['qscale']).tolist(),
                    future_Q_m3_s=(np.array(row['y_q'])*row['qscale']).tolist())
        example_records.append(record)
        for j,h in enumerate(range(-11,13)):
            normalized_q=(row['x_q']+row['y_q'])[j]
            source_rows.append(dict(phase=phase,window_id=row['window_id'],event_id=event,role='train',
                origin_time=row['origin_time'],relative_hour=h,source_row=row['source_rows'][j],source_time=row['source_times'][j],
                data_role='historical_input' if h<=0 else 'future_observed_label_not_prediction',
                q_normalized=normalized_q,q_m3_s=normalized_q*row['qscale'],
                p_normalized=row['x_p'][j] if h<=0 else None,
                p_mm=row['x_p'][j]*row['pscale'] if h<=0 else None,
                qscale_m3_s=row['qscale'],pscale_mm=row['pscale'],
                signed_peak_change6_normalized=normalized['signed_peak_change6'],rise6_normalized=normalized['rise6'],
                signed_peak_change6_m3_s=physical['signed_peak_change6'],rise6_m3_s=physical['rise6'],
                P_interval_semantics='unknown; sums follow recorded hourly-cell convention only',**history_explanation))
    dump(out/'selected_windows.json',dict(event_id=event,selection_details=details,windows=example_records))
    with (out/'window_source.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(source_rows[0]));writer.writeheader();writer.writerows(source_rows)
    dump(out/'input_manifest.json',dict(manifest=str(MANIFEST),manifest_sha256=digest(MANIFEST),
        source_data_commit=manifest['source_data_commit'],source_files=sources,train_windows=8557,
        selection_attempts=attempts,units=dict(Q='m3/s',P='mm per recorded hourly cell; source interval convention unknown'),
        normalization='Original train-normalized arrays multiplied once by their original train qscale/pscale',
        selection_rule='Prefer Xiagushan largest available train-event Q peak, then descending peak in other China train events; require all three phases if available. Before: before event peak, positive future6 rise, absolute historical3h change <=5% event Q range, lowest Qnow. Rising: before peak, historical3h rise >5% event range, largest future6 rise. Recession: after event peak, negative historical3h change, no future6 exceedance, largest Qnow. Ties earliest source row.',
        limitations=['Posthoc selection uses train Q answers solely for teaching, not forecast skill or generalization.',
                     'Event peak is maximum visible in legal training windows, not independently established whole-flood main peak.',
                     'Station clock and source hourly P interval convention remain unknown.',
                     'No development/test files, model weights, fitting, GPU, or future P access.']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.exists():font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    assert font.exists(),'Chinese font unavailable'
    plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(17,7.8),height_ratios=[3,1.2],sharex=True,constrained_layout=True)
    for col,(phase,record) in enumerate(zip(phases,example_records)):
        ax,rain=axes[:,col]
        if record['state']!='evaluable':
            ax.text(.5,.5,f'{phase}：缺少合法窗口',ha='center',transform=ax.transAxes);continue
        hp=np.array(record['history_P_mm']);hq=np.array(record['history_Q_m3_s']);fq=np.array(record['future_Q_m3_s'])
        ax.plot(range(-11,1),hq,'o-',ms=3,color='#2878b5',label='已知历史实测Q')
        ax.plot(range(0,7),np.r_[hq[-1],fq[:6]],'o-',ms=3,color='#e77822',label='未来1—6h实测标签')
        ax.plot(range(6,13),fq[5:],'o-',ms=3,color='#e77822',alpha=.35,label='未来7—12h实测标签')
        ax.scatter([6],[fq[5]],s=65,marker='D',color='#e77822',zorder=4)
        peaklead=record['labels_physical']['peak6_lead_hour']
        ax.scatter([peaklead],[record['labels_physical']['Qpeak6']],s=100,marker='*',color='#b44e00',zorder=5)
        lab=record['labels_physical']
        ax.set_title(f"{phase}｜起报 {record['origin_time'].replace('T',' ')}\nQnow {lab['Qnow']:.1f}   Q6 {lab['Q6']:.1f}   peak6 {lab['Qpeak6']:.1f}\npeak6−Qnow {lab['signed_peak_change6']:.1f}；rise6 {lab['rise6']:.1f} m³/s",fontsize=10)
        ax.set_ylabel('流量 Q（m³/s）');ax.legend(loc='upper left',fontsize=8)
        rain.bar(range(-11,1),hp,width=.8,color='#2878b5',alpha=.8)
        rain.set_ylabel('历史雨量 P（mm）');rain.set_xlabel('相对起报时间（h）')
        rain.text(.98,.94,'未来雨量不作为输入，也不展示',ha='right',va='top',transform=rain.transAxes,fontsize=8)
        for a in (ax,rain):
            a.axvline(0,color='#444444',lw=1);a.axvline(6,color='#999999',ls='--',lw=1)
            a.axvspan(6,12,color='#eeeeee',alpha=.5,zorder=0)
            a.set_xlim(-11.7,12.7);a.set_xticks([-11,-6,0,3,6,9,12])
    fig.suptitle(f"中国训练事件 {event.replace('da_tl_cn:','')}｜蓝色为模型可见历史，橙色为未来实测标签，不是模型预测\nQ6是第6小时流量；peak6是未来6小时最大流量；rise6是相对起报时的最大上涨量",fontsize=13)
    fig.savefig(out/'figures'/'prediction_task_three_phases.png',dpi=160)
    plt.close(fig)
    dump(out/'completion.json',dict(state='complete',wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
        event=event,available_phases=sum(r is not None for r in selected),source_table_rows=len(source_rows),
        train_only=True,model_loaded=False,fitting=False,gpu=False,future_P_access=False))


if __name__=='__main__':main()
