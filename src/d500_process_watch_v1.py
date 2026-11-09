"""User-invoked terminal view; never launches or modifies an experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def seconds(value):
    if not isinstance(value, (int, float)):
        return '未记录'
    return f'{value:.0f}s'


def render(queue_path):
    queue = read(queue_path)
    if not queue:
        return '队列状态尚未写入。'
    lines = [f"D500过程适配器 | {queue['state']} | 更新UTC {queue.get('updated_at_utc', '未记录')}",
             'D500：复用既有权重/结果；以下为TCN适配器T、同构MoE M、完整候选A。',
             '任务       状态        GPU   更新进度               阶段       墙钟/CPU        最近验证Brier↓']
    entries = queue['jobs'] + ([queue['finalize']] if queue.get('finalize') else [])
    repo = Path(queue['repo'])
    for entry in entries:
        folder = repo / entry['run_dir']
        candidates = [folder / 'progress.json', folder / 'artifacts' / 'progress.json']
        progress = next((read(p) for p in candidates if p.exists()), {})
        updates = progress.get('updates', progress.get('completed_updates', 0))
        limit = progress.get('update_limit', progress.get('maximum_updates', 6000 if entry.get('variant') else 0))
        ratio = min(1., updates / limit) if limit else 0
        filled = int(ratio * 10)
        bar = '[' + '#' * filled + '-' * (10 - filled) + ']'
        phase = progress.get('phase', progress.get('stage', '—'))
        wall = (entry.get('wall_seconds') if entry['state'] in ('complete', 'failed')
                else progress.get('wall_seconds', progress.get('elapsed_seconds')))
        if wall is None and entry.get('started_unix'):
            wall = time.time() - entry['started_unix']
        cpu = progress.get('process_cpu_seconds', progress.get('cpu_seconds'))
        val = progress.get('last_validation', progress.get('last_validation_metrics', {}))
        if isinstance(val, dict):
            metrics = val.get('constraints', val.get('metrics', val))
            brier = metrics.get('brier', val.get('selection_score', val.get('brier'))) if isinstance(metrics, dict) else None
            stamp = val.get('at_utc', val.get('updated_at_utc', ''))
        else:
            brier, stamp = val, ''
        shown = f'{brier:.7f}' if isinstance(brier, (int, float)) else '未评价'
        if stamp:
            shown += ' @' + stamp
        lines.append(f"{entry['id']:<10} {entry['state']:<11} {str(entry.get('physical_gpu') or '—'):<4} "
                     f'{bar} {updates}/{limit:<6} {str(phase):<10} {seconds(wall)}/{seconds(cpu)} {shown}')
        train_loss = progress.get('last_train_loss', progress.get('train_loss'))
        if isinstance(train_loss, (int, float)):
            lines.append(f"  当前训练loss={train_loss:.6g}（同定义越低越好）；分项见原日志。")
    lines.append('Brier为无量纲概率平方误差；越低越好。未评价/未记录保持原状态；不是最终性能表。')
    if queue.get('active_jobs_preserved'):
        lines.append('队列退出时保留的任务仍可能运行：' + ', '.join(queue['active_jobs_preserved'])
                     + '。退出本查看命令不会停止它们。')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue-status', required=True)
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=3.)
    a = p.parse_args()
    path = Path(a.queue_status)
    while True:
        if not a.once:
            print('\033[2J\033[H', end='')
        print(render(path), flush=True)
        state = read(path).get('state')
        if a.once or state in ('complete', 'complete_with_failures', 'interrupted', 'queue_failed'):
            break
        time.sleep(max(1., a.interval))


if __name__ == '__main__':
    main()
