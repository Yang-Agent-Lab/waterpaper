"""One-shot, read-only status. No polling, process signals or automatic wakeups."""
import argparse
import json
import socket
from pathlib import Path


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def training_status(root,config):
    rows=[];failure=read(root/'failure.json',{});run=read(root.parent/'run_record.json',{})
    runtime=read(root/'runtime.json',{});live=None
    if runtime.get('host')==socket.gethostname():
        try:
            ticks=Path('/proc/%s/stat'%runtime['pid']).read_text().rsplit(')',1)[1].split()[19]
            live=ticks==runtime['process_start_ticks']
        except OSError:live=False
    global_state=failure.get('state',read(root/'completion.json',{}).get('state',run.get('state','unknown')))
    plan=config.get('execution_plan') or [dict(id=name+'_'+route,model=name,route=route) for route in ('pq','q_only') for name in config['models'][route]+['persistence']]
    for item in plan:
            name,route=item['model'],item['route'];key=item['id'];folder=root/key
            reuse=read(folder/'reuse.json',{})
            if reuse:folder=Path(reuse['source'])
            complete=read(folder/'completion.json',{})
            progress=read(folder/'progress.json',{});epochs=read(folder/'epochs.json',[])
            selection=read(folder/'selection.json',{})
            scores=read(folder/'scores.json',[])
            china=[r['brier'] for r in scores if r['country']=='CN' and r['role']=='select']
            final_valid=len(china)==config['forecast_hours']*2
            state=complete.get('state',progress.get('state','partial' if folder.exists() else 'pending'))
            local_failure=read(folder/'failure.json',{})
            if complete.get('state')!='complete':
                if local_failure:state='failed'
                elif failure and live is not True:state='failed_stopped' if folder.exists() else 'not_started_after_failure'
                elif run.get('state')=='failed':state='failed_stopped' if folder.exists() else 'not_started_after_failure'
                elif live is False:state='stopped_incomplete' if folder.exists() else 'pending'
                elif state=='training' and live is None:state='training_recorded_process_unverified'
            elif reuse:state='reused_complete'
            last=epochs[-1] if epochs else {}
            done=bool(complete.get('training_ended'))
            epoch_done=len(epochs)
            if name!='persistence' and epoch_done==config['epochs'] and last.get('step')==config['epochs']*config['updates_per_epoch']:
                done=True
            rows.append(dict(model=name,route=route,configuration_id=key,state=state,global_run_state=global_state,process_alive=live,
                failed_attempted_update=local_failure.get('failed_attempted_update'),reused_source=reuse.get('source'),
                epochs_completed=epoch_done,epoch_current=progress.get('epoch',epoch_done),
                epoch_limit=0 if name=='persistence' else config['epochs'],
                updates_recorded=complete.get('updates',progress.get('updates',last.get('step',0))),
                update_limit=0 if name=='persistence' else config['epochs']*config['updates_per_epoch'],
                training_ended=done,selection_ended=bool(complete.get('selection_ended')) or bool(done and selection),
                final_scoring_ended=bool(complete.get('final_scoring_ended')) or final_valid,
                china_brier=sum(china)/len(china) if final_valid else selection.get('score'),
                china_brier_stage=f"final_{config['final_draws']}_draws" if final_valid else f"best_selection_{config['select_draws']}_draws" if selection else 'not_available',
                china_brier_unit='dimensionless',china_brier_direction='lower_is_better',
                elapsed_seconds=complete.get('elapsed_seconds',progress.get('elapsed_seconds')),
                elapsed_is_last_recorded=not bool(complete),
                training_kind='no_optimization_empirical_fit_reference' if name=='persistence' else 'neural_training'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config')
    p.add_argument('--pack',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    root=Path(a.root)
    if a.pack:
        completion=read(root/'completion.json',{})
        rows=[dict(read(root/route/'progress.json',dict(state='planning_or_pending')),route=route,
            exposure=read(root/route/'exposure_manifest.json',{})) for route in ('pq','q_only')]
        print(json.dumps(dict(state=completion.get('state','running_or_pending'),routes=rows),ensure_ascii=False,indent=2));return
    config=read(a.config) if a.config else read(root/'condition.json')
    if not config:raise ValueError('provide a committed --config or an existing run condition.json')
    rows=training_status(root,config)
    if a.json:print(json.dumps(rows,ensure_ascii=False,indent=2));return
    print('|模型 / 路线|已完成epoch / 上限|记录updates / 上限|训练结束|选模结束|最终评分结束|China Brier ↓（无单位）|记录耗时（秒）|')
    print('|---|---:|---:|---|---|---|---:|---:|')
    for row in rows:
        metric='未记录' if row['china_brier'] is None else f"{row['china_brier']:.6f}"
        elapsed='未记录' if row['elapsed_seconds'] is None else f"{row['elapsed_seconds']:.1f}"
        print('|'+f"{row['configuration_id']} [{row['state']}]|{row['epochs_completed']} / {row['epoch_limit']}|{row['updates_recorded']} / {row['update_limit']}|"+
            '|'.join('是' if row[k] else '否' for k in ('training_ended','selection_ended','final_scoring_ended'))+f'|{metric}|{elapsed}|')
    print('未完成项显示最近已记录的更新/耗时；JSON另列Brier为选模最佳值或最终评分值。持续性没有神经网络更新。')


if __name__=='__main__':main()
