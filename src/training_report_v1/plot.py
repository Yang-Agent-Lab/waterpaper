"""Three prediction pages by default; saved numerical diagnostics remain intact."""
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from .schema import validate_bundle

COLORS = {'current_s2':'#2563eb', 'direct_lstm':'#e67e22', 'direct_tcn':'#159777',
          'persistence':'#777777','native_lstm':'#2563eb','native_tcn':'#e67e22',
          'native_nhits':'#159777','native_nbeatsx':'#be3455','native_tft':'#7950a3',
          'A':'#2563eb', 'B':'#e67e22', 'C':'#159777'}
NAMES = ['01_learning.png','02_predictions.png','03_risk.png','04_modules.png']
MODEL_LABELS={'current_s2':'当前模型','direct_lstm':'概率LSTM','direct_tcn':'时序卷积'}


def learning_series(bundle, field):
    """Select saved scalar fields with the role defined by that field."""
    role={'fit_loss':'fit_probe','select_loss':'select','selection_metric':'select'}.get(field,'fit')
    aggregate=field.startswith('train_')
    selection_country=bundle['metadata'].get('selection_country','China')
    result=[]
    for model in bundle['models']:
        unique={}
        for row in bundle['epochs']:
            if row['model']!=model['id'] or row.get(field) is None:continue
            legacy_china_selection=field=='selection_metric' and row['country']=='China+Germany+Iceland' and bundle['metadata'].get('selection_label','').startswith('中国')
            if not aggregate and row['country']!=selection_country and not legacy_china_selection:continue
            identity=(row['epoch'],row['step'])
            if identity in unique and unique[identity]['value']!=row[field]:raise ValueError(f'Conflicting duplicated {field}')
            selected=row.get('selected',False) or unique.get(identity,{}).get('selected',False)
            unique[identity]=dict(epoch=row['epoch'],step=row['step'],value=row[field],selected=selected)
        if unique:result.append(dict(model=model['id'],field=field,role=role,country=bundle['metadata'].get('training_countries','China+Germany+Iceland') if aggregate else selection_country,rows=sorted(unique.values(),key=lambda row:row['epoch'])))
    return result


def time_segments(x):
    starts=[0]+[i for i in range(1,len(x)) if x[i]-x[i-1]>timedelta(hours=1.01)]
    return list(zip(starts,starts[1:]+[len(x)]))


def intervention_panels(bundle):
    """Compare the same country, role, checkpoint, metric, and intervention."""
    checkpoints={model['id']:{'selected'} for model in bundle['models']}
    for row in bundle['epochs']:
        if row.get('selected'):checkpoints[row['model']].add(f"epoch_{row['epoch']}")
    rows=[r for r in bundle['interventions'] if r.get('country')=='China' and r.get('role')=='select' and r.get('checkpoint') in checkpoints[r['model']] and r.get('horizon')==6 and r.get('delta') is not None]
    common=[r for r in rows if r.get('module') in ('input_rain_zero','input_flow_zero') and r.get('metric')=='crps']
    dedicated=[r for r in rows if r.get('module') in ('adapter_zero','encoder_zero','rain_off','flow_off','adapter_off')]
    dedicated_metric='crps' if any(r.get('metric')=='crps' for r in dedicated) else 'peak_mae_normalized'
    dedicated=[r for r in dedicated if r.get('metric')==dedicated_metric]
    return common,dedicated


def pointwise_risk_page(bundle, plt, labels, color, absent):
    """Draw saved per-lead scores; never accumulate or recompute probabilities."""
    rows=bundle['risk']
    countries=list(dict.fromkeys(r['country'] for r in rows))
    fig,axs=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    missing=[]
    for ax,country in zip(axs.flat,countries):
        rr=[r for r in rows if r['country']==country]
        seen=False
        for role,target in sorted({(r['role'],r['target']) for r in rr}):
            group=sorted([r for r in rr if r['role']==role and r['target']==target],key=lambda r:r['horizon'])
            ax.plot([r['horizon'] for r in group],[float('nan') if r['brier'] is None else r['brier'] for r in group],marker='o',markersize=3,label=f'{target.rsplit("_",1)[-1]} / {role}')
            seen=seen or any(r['brier'] is not None for r in group)
        ax.set(title=country,xlabel='提前量（h）',ylabel='逐时 Brier（越低越好）')
        if seen:ax.legend(fontsize=8)
        else:absent(ax,'未记录 / NOT_EVALUABLE')
        if any(r['brier'] is None for r in rr):missing.append(country+' Brier');ax.text(.02,.98,'部分未记录 / 不可评价',transform=ax.transAxes,va='top',fontsize=8)
    for ax in list(axs.flat)[len(countries):5]:ax.set_axis_off()
    ax=axs[1,2];seen=False
    # One saved lead only: combining reliability bins across leads changes their meaning.
    horizon=bundle['metadata'].get('reliability_horizon',1)
    target=bundle['metadata'].get('reliability_target','pointwise_exceedance_Q95')
    for row in rows:
        if row['horizon']!=horizon or row['target']!=target:continue
        bins=[v for v in row['reliability_bins'] if v.get('mean_probability') is not None and v.get('observed_fraction') is not None]
        if bins:ax.plot([v['mean_probability'] for v in bins],[v['observed_fraction'] for v in bins],'o-',label=row['country']+' / '+row['role']);seen=True
    ax.plot([0,1],[0,1],':',color='gray')
    ax.set(title=f'H{horizon} {target.rsplit("_",1)[-1]} 逐时可靠性',xlabel='预测概率',ylabel='实测发生比例',xlim=(0,1),ylim=(0,1))
    if seen:ax.legend(fontsize=7)
    else:absent(ax,'未记录 / NOT_EVALUABLE');missing.append('reliability')
    return fig,missing

def extended_interventions(bundle,metric):
    return [r for r in bundle['interventions'] if r.get('country')=='China' and r.get('role')=='select' and r.get('checkpoint')=='selected' and r.get('horizon')==6 and r.get('module') in ('input_rain_zero','input_flow_zero') and r.get('metric')==metric]


def render_report(bundle, out, *, include_diagnostics=False):
    validate_bundle(bundle)
    pointwise=bundle.get('target')=='pointwise_exceedance'
    risk_only=pointwise and bundle['metadata'].get('risk_only_report',False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    import numpy as np
    font_path=os.environ.get('TRAINING_REPORT_FONT')
    if not font_path:
        for name in ('NotoReportSubset.ttf','NotoSansCJKsc-Regular.otf'):
            candidate=Path.home()/'.local/share/fonts'/name
            if candidate.exists():font_path=str(candidate);break
    if font_path:font_manager.fontManager.addfont(font_path)
    fonts = [f.name for f in font_manager.fontManager.ttflist]
    chinese = next((f for f in ['Noto Sans SC','Noto Sans CJK JP','Noto Sans CJK SC','WenQuanYi Micro Hei','SimHei'] if f in fonts), None)
    if not chinese:
        raise RuntimeError('Chinese font unavailable; install server font before rendering')
    plt.rcParams.update({'font.family':chinese,'axes.unicode_minus':False,'font.size':10})
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    models=bundle['models']; labels={m['id']:MODEL_LABELS.get(m['id'],m['label']) for m in models}
    def color(model): return COLORS.get(model,'#7950a3')
    def absent(ax, text='未记录'):
        ax.text(.5,.5,text,transform=ax.transAxes,ha='center',va='center',color='#777777')
    pages=[]
    smoke=bundle['metadata'].get('runtime',{}).get('smoke') or bundle['metadata'].get('scientific_state')=='engineering_smoke_not_scientific'
    display_note='工程小试 · fit/select替身 · 不作科学结论' if smoke else bundle.get('metadata',{}).get('display_note','开发评价；单次结果')
    with PdfPages(out/'training_report.pdf') as pdf:
        def save(fig,index,sections,missing):
            titles=['学习与选模','固定案例风险曲线' if risk_only else '固定案例预测','风险与国别','模块与干预']
            fig.suptitle(titles[index],fontsize=17)
            fig.text(.02,.015,'保存结果 · '+display_note,fontsize=8,color='#555555')
            fig.savefig(out/NAMES[index],dpi=150);pdf.savefig(fig);plt.close(fig)
            source_tokens=[('epochs','selection'),('development_final','development_rows','point_windows'),('summary.csv','reliability.csv'),('health.jsonl','selected_diagnostic')][index]
            source_inputs={p:h for p,h in bundle['metadata'].get('sources',{}).items() if any(token in p for token in source_tokens)}
            pages.append(dict(file=NAMES[index],sections=sections,missing=missing,source_inputs=source_inputs))
        fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
        frozen=any(r.get('state')=='NOT_APPLICABLE' for r in bundle['health'])
        components=bundle['metadata'].get('loss_components',[('train_nll','--','训练NLL'),('train_crps',':','训练CRPS')])
        scope=bundle['metadata'].get('selection_country','China')
        panels=[([('train_loss','-','训练batch均值')],bundle['metadata'].get('loss_label','训练总损失')),([('fit_loss','--','fit_probe/'+scope),('select_loss','-','select/'+scope)],scope+'训练小组与选择集损失'),(components,'训练损失分项'),([('selection_metric','-','select/'+scope)],scope+'选模指标')]
        missing=[]
        for ax,(fields,title) in zip(axs.flat,panels):
            seen=False;panel_missing=[]
            for key,style,scope_label in fields:
                entries=learning_series(bundle,key)
                if not entries:missing.append(key);panel_missing.append(key)
                for entry in entries:
                    rr=entry['rows'];model=entry['model']
                    ax.plot([r['epoch'] for r in rr],[r['value'] for r in rr],style,color=color(model),label=f'{labels[model]} {scope_label}')
                    for r in rr:
                        if r['selected']:ax.scatter(r['epoch'],r['value'],marker='*',s=100,color=color(model),zorder=5)
                    seen=True
            ax.set(title=title,xlabel='训练轮次',ylabel=bundle['metadata'].get('selection_label','选模指标') if fields and fields[0][0]=='selection_metric' else '损失')
            if seen:ax.legend(fontsize=7)
            else:absent(ax,bundle['metadata'].get('loss_components_note','未记录') if not fields else '未记录')
            if seen and panel_missing:ax.text(.98,.02,' / '.join(panel_missing)+' 未记录',transform=ax.transAxes,ha='right',fontsize=8)
        save(fig,0,['epochs'],missing)
        if risk_only:
            risk_case_curves=bundle['metadata']['risk_case_curves']
            preferred=[dict(station=r['station'],role=r['role'],case_kind='risk_only',case_id=r['case_id']) for r in risk_case_curves]
        else:
            preferred=bundle['metadata'].get('preferred_cases',[dict(station=s,role='development',case_kind=k) for s,k in [('Liqingdian','difficult'),('Luanchuan','difficult'),('Luxi','ordinary'),('Zhuxipo','difficult')]])
        if pointwise and not risk_only and 'preferred_cases' not in bundle['metadata']:
            preferred=[dict(station=r['station'],role=r['role'],case_kind=r['case_kind'],case_id=r['case_id']) for r in bundle['cases'][:4]]
        if len(preferred)>4:raise ValueError('Fixed page supports at most four preferred cases')
        fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
        case_missing=[]
        for ax,slot in zip(axs.flat,preferred):
            if risk_only:
                row=next(r for r in risk_case_curves if r['case_id']==slot['case_id'])
                lead=row['lead_hours'];label=labels[models[0]['id']]
                ax.plot(lead,row['p95'],marker='o',markersize=2,color=color(models[0]['id']),label=label+' · P(Q>fit Q95)')
                ax.plot(lead,row['p99'],marker='o',markersize=2,color='#d95f02',label=label+' · P(Q>fit Q99)')
                ax.step(lead,row['observed_q95'],where='mid',linestyle='--',color='black',label='实测超过Q95')
                ax.step(lead,row['observed_q99'],where='mid',linestyle=':',color='#555555',label='实测超过Q99')
                origin=' · '+row['origin_time'][:16].replace('T',' ') if row.get('origin_time') else ''
                event=' · '+str(row.get('event','')) if row.get('event') else ''
                ax.set(title=f"{row['station'].split(':')[-1]}{event}{origin}",xlabel='提前量（h）',ylabel='逐时超阈概率',xlim=(1,bundle['metadata']['forecast_hours']),ylim=(-.03,1.03))
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True));ax.grid(alpha=.2);ax.legend(fontsize=7)
                continue
            rr=[r for r in bundle['cases'] if r.get('station','').split(':')[-1]==slot['station'].split(':')[-1] and r['role']==slot['role'] and r.get('case_kind')==slot['case_kind']]
            if slot.get('case_id'):rr=[r for r in rr if r['case_id']==slot['case_id']]
            if rr and any((r['case_id'],r['x_kind'],r['x'],r.get('origin_time'),r['truth']) != (rr[0]['case_id'],rr[0]['x_kind'],rr[0]['x'],rr[0].get('origin_time'),rr[0]['truth']) for r in rr):
                raise ValueError('case overlays must share case identity, origin, x and truth')
            if not rr:
                absent(ax,'未记录 / 不可评价');ax.set_title(slot['station']+' · '+slot['case_kind']);case_missing.append(slot);continue
            r=rr[0];caseid=r['case_id']
            x=r['x']
            is_dates=r['x_kind'] in ('origin_time_h6_peak','valid_time_pointwise') and x and isinstance(x[0],str)
            if is_dates:x=[datetime.fromisoformat(v) for v in x]
            segments=time_segments(x) if is_dates else [(0,len(x))]
            for j,(start,end) in enumerate(segments):ax.plot(x[start:end],r['truth'][start:end],color='black',linewidth=2,label=('实测逐时流量' if pointwise else '实测窗口峰') if j==0 else None)
            for r in rr:
                if any(v is not None for v in r['median']):
                    for j,(start,end) in enumerate(segments):
                        ax.plot(x[start:end],[np.nan if v is None else v for v in r['median'][start:end]],color=color(r['model']),label=labels[r['model']] if j==0 else None)
                        ax.fill_between(x[start:end],[np.nan if v is None else v for v in r['lower'][start:end]],[np.nan if v is None else v for v in r['upper'][start:end]],color=color(r['model']),alpha=.10)
            short_label={'ordinary':'普通时段','difficult':'困难案例','fixed_source_station_earliest_qualified_origin':'预选固定案例'}.get(slot['case_kind'],slot['case_kind'])
            origin=' · '+r['origin_time'][:16].replace('T',' ') if r.get('origin_time') else ''
            ax.set(title=f"{slot['station'].split(':')[-1]} · {short_label}\n{r['country']} / {r['role']}{origin}",xlabel='起报时间' if r['x_kind']=='origin_time_h6_peak' else '未来窗口长度（h）',ylabel='未来6h最高流量（m³/s）' if r['x_kind']=='origin_time_h6_peak' else '窗口最高流量（m³/s）')
            if pointwise:ax.set(xlabel='有效时间' if is_dates else '提前量（h）',ylabel='逐时流量（m³/s）')
            if r['x_kind']=='origin_time_h6_peak' or is_dates:
                ax.tick_params(axis='x',rotation=20)
                if is_dates:
                    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
                    locator=AutoDateLocator(minticks=3,maxticks=5);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
                else:ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            elif pointwise:ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            else:ax.set_xticks([1,3,6,12])
            ax.legend(fontsize=8)
        for ax in list(axs.flat)[len(preferred):]:absent(ax,'固定案例未记录');ax.set_axis_off()
        save(fig,1,['risk_case_curves'] if risk_only else ['cases'],case_missing)
        if pointwise:
            fig,risk_missing=pointwise_risk_page(bundle,plt,labels,color,absent)
            save(fig,2,['risk'],risk_missing)
        else:
            fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
            selected=[r for r in bundle['risk'] if r.get('horizon')==6 and ((r.get('country')=='China' and r.get('target')=='combined_first_q99') or (r.get('country')!='China' and r.get('target')=='window_exceedance_Q99'))]
            for ax,country in zip(axs.flat,['China','Germany','Iceland']):
                rr=[r for r in selected if r['country']==country]; seen=False
                for r in rr:
                    bins=[v for v in r.get('reliability_bins',[]) if v.get('mean_probability') is not None and v.get('observed_fraction') is not None]
                    if bins:ax.plot([v['mean_probability'] for v in bins],[v['observed_fraction'] for v in bins],'o-',color=color(r['model']),label=f"{labels[r['model']]} {r['role']}");seen=True
                ax.plot([0,1],[0,1],':',color='gray');ax.set(title=country+(' · H6 合并首报Q99' if country=='China' else ' · H6 原始窗口Q99'),xlabel='预测概率',ylabel='实测发生比例',xlim=(0,1),ylim=(0,1))
                if seen:ax.legend(fontsize=8)
                else:absent(ax,'未记录 / NOT_EVALUABLE')
            ax=axs[1,1];ax.set_axis_off()
            rows=[]
            for r in selected:
                fmt=lambda k: ('不适用' if r['country']!='China' and k in ('detected','events','false_alarm_segments','lead_hours') else '未记录') if r.get(k) is None else f'{r[k]:.4g}' if isinstance(r[k],(float,int)) else str(r[k])
                rows.append([labels[r['model']]+' / '+r['country']+' / '+r['role'],fmt('brier'),fmt('crps'),fmt('detected')+'/'+fmt('events'),fmt('false_alarm_segments'),fmt('lead_hours')])
            if rows:
                table=ax.table(cellText=rows,colLabels=['模型 / 国家 / 角色','Brier',bundle['metadata'].get('crps_label','CRPS'),'检出','误报段','提前h'],loc='center',cellLoc='center',colWidths=[.42,.12,.12,.12,.12,.1]);table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.5)
            else:absent(ax)
            save(fig,2,['risk'],[] if selected else ['risk'])
        if include_diagnostics:
            extended=any('update_norm' in r for r in bundle['health']) or any(r.get('metric') in ('brier_q95','brier_q99') for r in bundle['interventions'])
            if extended:
                fig=plt.figure(figsize=(16,11));fig.subplots_adjust(left=.1,right=.98,bottom=.13,top=.9,wspace=.9,hspace=.65);grid=fig.add_gridspec(3,12)
                for index,(key,title) in enumerate([('gradient_norm','模块梯度'),('output_variance','输出方差'),('nonfinite_fraction','非有限比例'),('parameter_norm','更新前参数范数'),('update_norm','单步更新绝对范数'),('relative_update_norm','单步更新相对范数')]):
                    ax=fig.add_subplot(grid[index//3,(index%3)*4:(index%3+1)*4]);seen=False
                    for model,module in sorted({(r['model'],r['module']) for r in bundle['health']}):
                        rr=[r for r in bundle['health'] if r['model']==model and r['module']==module and r.get(key) is not None]
                        if rr:ax.plot([r['step'] for r in rr],[r[key] for r in rr],marker='o',markersize=2,color=color(model),label=f'{labels[model]}:{module}');seen=True
                    ax.set(title=title,xlabel='更新步',ylabel='L2' if key in ('parameter_norm','update_norm') else '更新/参数' if key=='relative_update_norm' else title);ax.tick_params(labelsize=8);ax.title.set_fontsize(9)
                    if seen:ax.legend(fontsize=5,ncol=2)
                    else:absent(ax,'未记录' if not any(key in r for r in bundle['health']) else '不适用 / 不可评价')
                for index,(metric,title) in enumerate([('crps','H6 CRPS/Q95'),('brier_q95','H6 Q95 Brier'),('brier_q99','H6 Q99 Brier'),('peak_mae','H6 中位峰MAE/Q95')]):
                    ax=fig.add_subplot(grid[2,index*3:(index+1)*3]);rr=extended_interventions(bundle,metric);valid=[r for r in rr if r.get('delta') is not None]
                    for j,r in enumerate(valid):ax.barh(j,r['delta'],color=color(r['model']))
                    if valid:ax.set_yticks(range(len(valid)),[f"{labels[r['model']]}:{'P' if r['module']=='input_rain_zero' else 'Q'} n={r['n']}" for r in valid],fontsize=6)
                    else:absent(ax,'不可评价（既有fit阈值缺失）' if rr else '未记录')
                    if valid and len(valid)<len(rr):ax.text(.02,.98,'部分不可评价',transform=ax.transAxes,va='top',fontsize=7)
                    ax.axvline(0,color='gray',linewidth=.7);ax.set(title=title,xlabel='关闭后−完整');ax.tick_params(labelsize=7);ax.title.set_fontsize(9)
                fig.text(.02,.075,'更新量：单次采样步的更新（正式每50步采样）；不是50步累计',fontsize=8)
                fig.text(.02,.055,'下排：中国select固定样本全部起报；非正式预警评分；正差为关闭后变差',fontsize=8)
                fig.text(.02,.035,'未定义的模块关闭：不可评价；临时输入关闭不是去模块重训贡献',fontsize=8)
                save(fig,3,['health','interventions','parameter_updates','probability_peak_interventions'],[])
            if not extended:
                fig=plt.figure(figsize=(14,9),constrained_layout=True)
                grid=fig.add_gridspec(2,6)
                health_axes=[fig.add_subplot(grid[0,i:i+2]) for i in (0,2,4)]
                common_ax=fig.add_subplot(grid[1,:3]);dedicated_ax=fig.add_subplot(grid[1,3:])
                for ax,key,title in zip(health_axes,('gradient_norm','output_variance','nonfinite_fraction'),('模块梯度','输出方差','非有限比例')):
                    seen=False
                    groups=sorted({(r['model'],r['module'],r['role'],r['country']) for r in bundle['health']})
                    for model,module,role,country in groups:
                        rr=sorted([r for r in bundle['health'] if (r['model'],r['module'],r['role'],r['country'])==(model,module,role,country) and r.get(key) is not None],key=lambda r:r['step'])
                        if rr:ax.plot([r['step'] for r in rr],[r[key] for r in rr],color=color(model),alpha=.65,label=f'{labels[model]}:{module}');seen=True
                    ax.set(title=title,xlabel='更新步',ylabel=title)
                    if seen:ax.legend(fontsize=5,ncol=2,loc='upper left')
                    else:absent(ax)
                    if frozen and key=='gradient_norm':ax.set_title('模块梯度（冻结P0不适用）')
                common,dedicated=intervention_panels(bundle)
                for ax,rr,title in [(common_ax,common,'中国select · 共同输入临时关闭'),(dedicated_ax,dedicated,'中国select · 模型专属临时关闭')]:
                    metrics={r['metric'] for r in rr}
                    if len(metrics)>1:raise ValueError('Do not mix intervention metrics in one panel')
                    for j,r in enumerate(rr):ax.barh(j,r['delta'],color=color(r['model']))
                    if rr:ax.set_yticks(range(len(rr)),[f"{labels[r['model']]}:{r['module']} n={r['n']}" for r in rr],fontsize=7)
                    else:absent(ax,'未记录（旧版仅关闭新增分支）' if not common else '未记录')
                    metric_label='CRPS' if metrics=={'crps'} or not metrics else '窗口峰MAE/Q95'
                    ax.axvline(0,color='gray',linewidth=.7);ax.set(title=title,xlabel=f'H6 {metric_label}变化（关闭后−完整）')
                if any(r.get('kind')=='legacy_shadow_zero' for r in dedicated):
                    dedicated_ax.text(.02,.98,'旧新增支路关闭；P0仍保留输入',transform=dedicated_ax.transAxes,va='top',fontsize=8)
                else:dedicated_ax.text(.02,.98,'adapter与encoder含义不同，不作同模块归因',transform=dedicated_ax.transAxes,va='top',fontsize=8)
                save(fig,3,['health','interventions'],[] if bundle['health'] else ['health'])
    manifest=dict(schema_version=1,run_id=bundle['run_id'],pages=pages,colors={m['id']:color(m['id']) for m in models},model_labels=labels,metadata=bundle['metadata'],bundle_sha256=hashlib.sha256(json.dumps(bundle,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),renderer=dict(python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__,font=chinese,font_path=font_path,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'report_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
    (out/'interventions.json').write_text(json.dumps(bundle['interventions'],ensure_ascii=False,indent=2))
    (out/'report_notes.md').write_text('# 定义与来源\n\n'+json.dumps(bundle['metadata'],ensure_ascii=False,indent=2)+'\n\n四窗口峰不是逐小时流量。临时关闭模块后的变化不是去模块重训收益。未记录不代表零。\n')
    return manifest
