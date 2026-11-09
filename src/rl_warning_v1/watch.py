"""Standard-library progress display and low-frequency local status writer."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

def utc():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)

class LiveStatus:
    """Local writes at most once per two seconds, plus explicit phase events."""
    def __init__(self,out,total,cpu_budget):
        self.out=Path(out);self.wall=time.monotonic();self.cpu=time.process_time();self.last=-float('inf')
        self.data=dict(schema_version=1,state='running',phase='preparation',sampled_transitions=0,
            optimizer_trained_transitions=0,total_transitions=total,cpu_budget_seconds=cpu_budget,
            pid=os.getpid(),started_at_utc=utc(),stage_timings={},events=[])

    def update(self,force=False,**values):
        self.data.update(values)
        now=time.monotonic()
        if not force and now-self.last<2:return
        self.data.update(updated_at_utc=utc(),cpu_seconds=time.process_time()-self.cpu,
            process_cpu_seconds=time.process_time(),wall_seconds=now-self.wall)
        atomic_json(self.out/'live_status.json',self.data);self.last=now

    def event(self,name,**values):
        self.data['events'].append(dict(at_utc=utc(),name=name,**values))
        self.update(force=True)

    @contextmanager
    def stage(self,name):
        before=time.monotonic();cpu_before=time.process_time();previous=self.data['phase']
        self.update(force=True,phase=name)
        try:yield
        finally:
            item=self.data['stage_timings'].setdefault(name,dict(cpu_seconds=0.,wall_seconds=0.,calls=0))
            item['cpu_seconds']+=time.process_time()-cpu_before;item['wall_seconds']+=time.monotonic()-before;item['calls']+=1
            self.update(force=True,phase=previous)

class StatusReader:
    def __init__(self):self.cache={}
    def read(self,path,warnings):
        try:
            value=json.loads(path.read_text())
            expected=list if path.name=='pilot_progress.json' else dict
            if not isinstance(value,expected):raise ValueError('unexpected status structure')
            self.cache[str(path)]=value
            return value
        except FileNotFoundError:return None
        except (OSError,ValueError) as error:
            warnings.append(f'{path.name} 读取失败/不完整JSON，保留上次完整值 ({type(error).__name__})')
            return self.cache.get(str(path))

    def snapshot(self,root,total=29696,budget=900,stale_after=15):
        root=Path(root);warnings=[]
        live=self.read(root/'live_status.json',warnings) or {}
        history=self.read(root/'pilot_progress.json',warnings) or []
        completion=self.read(root/'pilot_completion.json',warnings) or self.read(root/'completion.json',warnings) or {}
        record=self.read(root/'run_record.json',warnings) or {}
        last=history[-1] if isinstance(history,list) and history else {}
        valid=live.get('last_validation') or last
        if not valid and completion.get('result'):valid=dict(transitions=completion.get('transitions'),summary=completion['result'])
        state=completion.get('state',record.get('state',live.get('state','WAITING')))
        terminal=str(state).lower() in ('complete','completed','failed','partial','not_run','cancelled')
        sampled=completion.get('transitions',live.get('sampled_transitions',last.get('transitions',0)))
        trained=completion.get('optimizer_trained_transitions',live.get('optimizer_trained_transitions',last.get('optimizer_trained_transitions',0)))
        source=next((root/name for name in ['live_status.json','pilot_completion.json','completion.json','pilot_progress.json','run_record.json'] if (root/name).exists()),None)
        if source and not terminal and time.time()-source.stat().st_mtime>stale_after:warnings.append('STALE：进度已超过阈值未更新，不能据此判断进程已停止')
        if not source:warnings.append('等待状态文件；未启动任何作业')
        if not live and not terminal:warnings.append('无逐步状态；步数/CPU仅为最近已保存检查点')
        stamp=valid.get('at_utc') or valid.get('updated_at_utc')
        if not stamp and (root/'pilot_progress.json').exists():stamp=datetime.fromtimestamp((root/'pilot_progress.json').stat().st_mtime,timezone.utc).isoformat()+' (文件时间)'
        return dict(state=state,terminal=terminal,phase=live.get('phase','已结束' if terminal else '历史检查点/等待'),
            sampled=sampled,trained=trained,total=live.get('total_transitions',total),budget=live.get('cpu_budget_seconds',budget),
            cpu=completion.get('process_cpu_seconds',live.get('process_cpu_seconds',last.get('cpu_seconds'))),
            wall=completion.get('elapsed_seconds',live.get('wall_seconds',last.get('elapsed_seconds'))),
            validation=valid,validation_time=stamp or 'unknown',warnings=warnings)

def render(s):
    ratio=min(1.,max(0.,s['sampled']/s['total'])) if s['total'] else 0.
    if str(s['state']).lower() in ('complete','completed'):ratio=1.
    bar='#'*int(28*ratio)+'-'*(28-int(28*ratio))
    number=lambda x:'unknown' if x is None else f'{x:.1f}'
    lines=[f"{utc()}  状态 {s['state']} | 阶段 {s['phase']}",
        f"[{bar}] {ratio*100:5.1f}%  采样 {s['sampled']}/{s['total']} | 已优化 {s['trained']}",
        f"CPU(进程快照) {number(s['cpu'])}/{s['budget']} 秒 | 墙钟(记录阶段) {number(s['wall'])} 秒"]
    v=s['validation'];m=v.get('summary') or v.get('result')
    if m:
        hit=m.get('hits',['?','?']);count=m.get('onset_count',['?','?'])
        utility=m.get('raw_reward')
        utility=f'{utility:.2f}' if isinstance(utility,(int,float)) else '?'
        def hours(key):
            values=m.get(key,['?','?'])
            return f'{values[0]} / {values[1]}小时'
        lines.extend([f"最近验证：{s['validation_time']} | 采样 {v.get('transitions','?')}（验证值保持至下次验证）",
            f"效用↑ {utility} | 及时95↑ {hit[0]}/{count[0]} | 及时99↑ {hit[1]}/{count[1]}",
            f"误报↓(等级1/2) {hours('false_alarm_hours')} | 未覆盖↓(95/99) {hours('uncovered_hours')}"])
    else:lines.append('最近验证：尚无结果')
    lines.extend(s['warnings'])
    if s['terminal']:lines.append('作业已结束；仅持续查看，不会重启。Ctrl-C退出。')
    return '\n'.join(lines)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--once',action='store_true');p.add_argument('--plain',action='store_true')
    p.add_argument('--interval',type=float,default=2.);p.add_argument('--stale-after',type=float,default=15.)
    p.add_argument('--total',type=int,default=29696);p.add_argument('--cpu-budget',type=float,default=900.)
    a=p.parse_args(argv)
    if a.interval<=0:p.error('--interval must be positive')
    reader=StatusReader()
    try:
        while True:
            s=reader.snapshot(a.run_dir,a.total,a.cpu_budget,a.stale_after)
            if sys.stdout.isatty() and not a.plain:sys.stdout.write('\033[2J\033[H')
            print(render(s),flush=True)
            if a.once:return 0
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print('\n监视已停止，未修改作业。',flush=True);return 0

if __name__=='__main__':sys.exit(main())
