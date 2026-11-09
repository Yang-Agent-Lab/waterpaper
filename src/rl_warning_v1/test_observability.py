"""Engineering-only real subprocess refresh fixtures; no scientific rerun."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .watch import LiveStatus,atomic_json,render

def test_render_format():
    snapshot=dict(state='completed',phase='finished',sampled=29696,trained=29696,total=29696,
        cpu=247.4,budget=900,wall=253.6,terminal=True,warnings=[],validation_time='fixture',
        validation=dict(transitions=29696,summary=dict(raw_reward=12.279999999999998,
            hits=[5,4],onset_count=[5,4],false_alarm_hours=[1,77],uncovered_hours=[8,0])))
    text=render(snapshot)
    for expected in ['效用↑ 12.28','1 / 77小时','8 / 0小时','CPU(进程快照)','墙钟(记录阶段)']:
        assert expected in text,expected
    assert '12.279999' not in text and '[1, 77]' not in text and '[8, 0]' not in text
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--format-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();cpu=time.process_time()
    if a.format_only:
        text=test_render_format();(a.output/'render_format_stdout.txt').write_text(text+'\n')
        receipt=dict(state='PASS',checks=['render_format'],training_steps=0,
            process_cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-start)
        (a.output/'test_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    script=Path(__file__).with_name('watch.py');checks=[]
    def once(folder,*extra):
        return subprocess.check_output([sys.executable,str(script),'--run-dir',str(folder),'--plain','--once',*extra],text=True)
    empty=a.output/'empty';empty.mkdir()
    assert '等待状态文件' in once(empty);checks.append('missing_files')
    broken=a.output/'broken';broken.mkdir();(broken/'live_status.json').write_text('{"state":')
    assert '不完整JSON' in once(broken);checks.append('partial_json_once')
    wrong=a.output/'wrong_type';wrong.mkdir();(wrong/'live_status.json').write_text('[1]')
    assert '不完整JSON' in once(wrong);checks.append('wrong_json_structure')
    stale=a.output/'stale';stale.mkdir();atomic_json(stale/'live_status.json',dict(state='running',sampled_transitions=32))
    os.utime(stale/'live_status.json',(time.time()-60,time.time()-60))
    assert 'STALE' in once(stale);checks.append('stale_not_failed')
    failed=a.output/'failed';failed.mkdir();atomic_json(failed/'pilot_completion.json',dict(state='failed',transitions=64))
    assert '状态 failed' in once(failed) and '作业已结束' in once(failed);checks.append('failed_terminal')
    fixture=a.output/'refresh';fixture.mkdir()
    validation=dict(transitions=0,at_utc='2026-10-19T00:00:00+00:00',summary=dict(raw_reward=-1,hits=[0,0],onset_count=[1,1],false_alarm_hours=[2,3],uncovered_hours=[4,5]))
    data=dict(state='running',phase='training',sampled_transitions=0,optimizer_trained_transitions=0,
        total_transitions=256,cpu_budget_seconds=900,last_validation=validation)
    atomic_json(fixture/'live_status.json',data)
    child=subprocess.Popen([sys.executable,str(script),'--run-dir',str(fixture),'--plain','--interval','0.1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        time.sleep(.25)
        data.update(sampled_transitions=128,optimizer_trained_transitions=64);atomic_json(fixture/'live_status.json',data)
        time.sleep(.25)
        (fixture/'live_status.json').write_text('{')
        time.sleep(.25)
        data.update(sampled_transitions=256,optimizer_trained_transitions=256,state='completed',phase='finished')
        atomic_json(fixture/'live_status.json',data);time.sleep(.25)
        assert child.poll() is None,'watch must keep displaying a finished run until Ctrl-C'
        child.send_signal(signal.SIGINT);stdout,stderr=child.communicate(timeout=5)
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
    (a.output/'continuous_stdout.txt').write_text(stdout);(a.output/'continuous_stderr.txt').write_text(stderr)
    assert child.returncode==0 and not stderr
    for text in ['采样 0/256','采样 128/256','已优化 64','不完整JSON','100.0%','采样 256/256','作业已结束','监视已停止','采样 0（验证值保持至下次验证）']:
        assert text in stdout,text
    checks.extend(['real_continuous_refresh','atomic_updates','retain_last_valid_json','validation_lag','finished_100_percent','finished_does_not_exit_or_restart','ctrl_c_exit0'])
    writer=LiveStatus(a.output/'writer',256,900);writer.update(sampled_transitions=0)
    snapshot=(a.output/'writer/live_status.json').read_text();writer.update(sampled_transitions=1)
    assert (a.output/'writer/live_status.json').read_text()==snapshot
    writer.last-=2.1;writer.update(sampled_transitions=2)
    assert json.loads((a.output/'writer/live_status.json').read_text())['sampled_transitions']==2
    with writer.stage('fixture_timing'):time.sleep(.01)
    status=json.loads((a.output/'writer/live_status.json').read_text())
    assert status['stage_timings']['fixture_timing']['wall_seconds']>0
    checks.extend(['writer_throttled_two_seconds','stage_timing','local_atomic_status'])
    receipt=dict(state='PASS',checks=checks,training_steps=0,installation_count=0,
        process_cpu_seconds=time.process_time(),parent_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=os.times().children_user+os.times().children_system,wall_seconds=time.monotonic()-start)
    (a.output/'test_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
