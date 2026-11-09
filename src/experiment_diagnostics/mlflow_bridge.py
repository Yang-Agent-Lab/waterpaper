"""Incremental local-log -> MLflow bridge. No model, dataset or credential reads.

The source files remain authoritative. Historical imports explicitly use import
timestamps; the original step/epoch is preserved. This bridge never starts training.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_records(path, live=False):
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index + 1 < len(lines):
                marker = json.loads(lines[index + 1])
                if marker.get('recovered_torn_sha256') == hashlib.sha256(line.encode()).hexdigest():
                    continue
            if not live or index != len(lines) - 1:
                raise
    return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TrackingAPI:
    def __init__(self, origin):
        p = urllib.parse.urlsplit(origin)
        if p.scheme != 'http' or p.hostname not in ('127.0.0.1', 'localhost') or p.username or p.password or p.path not in ('', '/') or p.query or p.fragment:
            raise ValueError('Use the existing loopback tracking server or an SSH forward')
        self.origin = origin.rstrip('/')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, endpoint, payload=None, method=None, raw=None):
        data = raw if raw is not None else json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        req = urllib.request.Request(self.origin + '/api/2.0/' + endpoint, data=data,
            method=method, headers={'Content-Type': 'application/octet-stream' if raw is not None else 'application/json'})
        with self.opener.open(req, timeout=8) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def artifact(self, run, name, content):
        uri = run['info']['artifact_uri']
        if not uri.startswith('mlflow-artifacts:/'):
            raise ValueError('Tracking server does not expose the existing artifact proxy')
        path = uri[len('mlflow-artifacts:/'):].strip('/') + '/' + name
        self.request('mlflow-artifacts/artifacts/' + urllib.parse.quote(path, safe='/'), method='PUT', raw=content)


def metric_items(row):
    """Waterpaper adapter; other experiments can emit the same numeric key/value rows."""
    m = row.get('metrics', {})
    kind = row.get('kind')
    if kind == 'epoch':
        for key in ('risk_nll', 'elapsed_seconds', 'gpu_memory_bytes', 'clipped_updates'):
            yield 'training/' + key, m.get(key)
        yield 'epoch', row.get('epoch')
        for role in ('fit', 'select'):
            for h, value in zip((1, 3, 6), m.get(role + '_brier') or []):
                yield f'{role}/brier_{h}h', value
    elif kind == 'health':
        for key in ('nonfinite_fraction', 'saturation_fraction', 'pre_clip_gradient_norm'):
            yield 'health/' + key, m.get(key)
        for module, values in m.get('module_health', {}).items():
            for key in ('gradient_norm', 'output_variance', 'nonfinite_fraction'):
                yield f'health/{module}/{key}', values.get(key)
    elif kind in ('shadow_only', 'selected_diagnostic') or 'diagnostic' in row:
        result = row.get('diagnostic', {}).get('result', {})
        stage = 'selected_shadow' if kind in ('selected_diagnostic', 'selected_checkpoint') or row.get('selected_checkpoint') else 'shadow'
        for role, values in result.get('roles', {}).items():
            if role not in ('fit', 'select'):
                continue
            for module, shadow in values.get('shadows', {}).items():
                for h, value in zip((1, 3, 6), shadow.get('brier_shadow_minus_full') or []):
                    yield f'{stage}/{role}/{module}/brier_delta_{h}h', value


def sync_run(source: Path, receipt_dir: Path, tracking_uri: str, record_key: str,
             diagnostic_dir: Path | None = None, live: bool = False) -> dict:
    source, receipt_dir = Path(source), Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    with (receipt_dir / 'mlflow-sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _sync(source, receipt_dir, tracking_uri, record_key, diagnostic_dir, live)


def _sync(source, receipt_dir, tracking_uri, record_key, diagnostic_dir, live):
    api = TrackingAPI(tracking_uri)
    config = load(source / 'effective_config.json', {})
    if not config.get('run_id'):
        raise ValueError('Missing source run identity')
    receipt_path = receipt_dir / 'mlflow_links.json'
    state = load(receipt_path, {'timestamp_ms': int(time.time() * 1000), 'runs': {}})
    identity = {'run_id': config['run_id'], 'code_version': config.get('code_version'),
                'data_role_id': config.get('data_role_id'), 'record_key': record_key,
                'tracking_uri': tracking_uri}
    if state.get('source_identity', identity) != identity:
        raise ValueError('Receipt belongs to a different source or tracking server')
    state['source_identity'] = identity
    save(receipt_path, state)
    name = 'waterpaper-diagnostics'
    found = api.request('mlflow/experiments/search', {'filter': f"name = '{name}'", 'max_results': 10})
    experiments = found.get('experiments', [])
    eid = experiments[0]['experiment_id'] if experiments else api.request('mlflow/experiments/create', {'name': name})['experiment_id']
    rows = read_records(source / 'metrics.jsonl', live)
    if diagnostic_dir:
        supplemental = Path(diagnostic_dir).parent / 'metrics.jsonl'
        if supplemental.exists() and supplemental.resolve() != (source / 'metrics.jsonl').resolve():
            rows.extend(r for r in read_records(supplemental)
                        if r.get('kind') == 'selected_checkpoint')
    final = load(source / 'final_summary.json', {})
    completion = load(source / 'completion.json', {})
    arms = sorted({r['arm'] for r in rows if r.get('arm')})
    for arm in arms:
        key = digest({**identity, 'arm': arm})
        cached = state['runs'].setdefault(arm, {'sent': {}, 'artifacts': {}})
        matches = api.request('mlflow/runs/search', {'experiment_ids': [eid], 'filter': f"tags.bridge_key = '{key}'", 'max_results': 2}).get('runs', [])
        if len(matches) > 1:
            raise ValueError('Duplicate tracking runs; do not silently select one')
        run = matches[0] if matches else api.request('mlflow/runs/create', {'experiment_id': eid, 'start_time': state['timestamp_ms'], 'tags': [
            {'key': 'mlflow.runName', 'value': f'{record_key} {arm}'},
            {'key': 'bridge_key', 'value': key}, {'key': 'project_slug', 'value': 'waterpaper'},
            {'key': 'research_record_key', 'value': record_key},
            {'key': 'record_mode', 'value': 'live_log_sync' if live else 'historical_import'},
            {'key': 'metric_time_semantics', 'value': 'sync time; source step/epoch preserved; not original measurement time'},
            {'key': 'source_training_commit', 'value': str(config.get('code_version', 'unknown'))},
            {'key': 'source_run_id', 'value': config['run_id']},
        ]})['run']
        rid = run['info']['run_id']
        cached.update(run_id=rid, experiment_id=eid, artifact_uri=run['info']['artifact_uri'],
                      ui_path=f'/#/experiments/{eid}/runs/{rid}')
        save(receipt_path, state)
        # Recover cursors from a tracking run without manufacturing a second run.
        histories = {}
        def history_seen(metric):
            if metric not in histories:
                hist = api.request('mlflow/metrics/get-history?' + urllib.parse.urlencode({'run_id': rid, 'metric_key': metric})).get('metrics', [])
                histories[metric] = {(x['step'], x['value']) for x in hist}
            return histories[metric]
        def publish_record(row):
            rec_id = row['record_id']
            fingerprint = digest(row)
            if rec_id in cached['sent']:
                if cached['sent'][rec_id] != fingerprint:
                    raise ValueError('Previously synchronized source record changed')
                return
            points = []
            for metric, value in metric_items(row):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    continue
                step = int(row.get('step', 0))
                if (step, value) not in history_seen(metric):
                    points.append({'key': metric, 'value': value, 'step': step, 'timestamp': state['timestamp_ms']})
                    histories[metric].add((step, value))
            if points:
                api.request('mlflow/runs/log-batch', {'run_id': rid, 'metrics': points})
            cached['sent'][rec_id] = fingerprint
            save(receipt_path, state)
        for row in rows:
            if row.get('arm') == arm:
                publish_record(row)
        best = load(source / arm / 'best.json', {})
        note = (f'研究记录：{record_key}\n\n本次：保护原流量，学习未来1/3/6小时首次越线风险；不能解释为峰幅改善。'
                f'\n\n候选：{arm}；选中轮次：{best.get("epoch", "待定")}。'
                '\n\n先看 select/brier_6h，再看 fit；随后查看 health 与 shadow。'
                'shadow差值=临时关闭后误差−完整模型误差，正值表示关闭后变差，负值表示关闭后变好；它是线索，不是正式模块贡献。'
                '\n\nArtifacts 中 diagnostics_summary 是诊断说明，guide 是长期阅读指南。'
                '\n\n指标时间戳为同步时间，横轴请使用 step。历史导入只补记已有实验；不会重新训练。')
        context = load(Path(diagnostic_dir)/'research_context.json', {}) if diagnostic_dir else {}
        if context:
            if context.get('record_key') != record_key:
                raise ValueError('Research context belongs to another experiment')
            note += '\n\nScience研究摘要（快照）：\n' + '\n\n'.join(
                f'{label}：{context[key]}' for key, label in (
                    ('research_question','研究问题'),('baseline','对照'),('conclusion','已有结论'),('next_step','下一决定'))
                if context.get(key))
        tags = [{'key': 'mlflow.note.content', 'value': note},
                {'key': 'selected_epoch', 'value': str(best.get('epoch', 'unknown'))},
                {'key': 'scientific_status', 'value': 'exploratory_mixed' if completion.get('state') == 'complete' else 'pending'},
                {'key': 'science_project_ref', 'value': 'hzwwxlaxyxnwsxwpjwpt'}]
        params = [{'key': key, 'value': str(config[key])} for key in ('seed', 'batch_size', 'learning_rate', 'max_updates') if key in config]
        # Final metrics are separate from training-time metrics, including data roles.
        values = []
        for role, report in final.get(arm, {}).items():
            if role not in ('fit', 'select', 'calibration', 'development') or not isinstance(report, dict):
                continue
            for h, v in zip((1, 3, 6), report.get('metrics', {}).get('brier') or []):
                if isinstance(v, (int, float)) and math.isfinite(v):
                    values.append({'key': f'final/{role}/brier_{h}h', 'value': v, 'step': best.get('step', 0), 'timestamp': state['timestamp_ms']})
        signature = digest(values)
        payload = {'run_id': rid, 'tags': tags, 'params': params}
        if cached.get('final_signature') != signature:
            payload['metrics'] = [v for v in values if (v['step'], v['value']) not in history_seen(v['key'])]
        api.request('mlflow/runs/log-batch', payload)
        cached['final_signature'] = signature
        if diagnostic_dir:
            for path in sorted(Path(diagnostic_dir).glob('*')):
                if not path.is_file() or path.suffix not in ('.md', '.json', '.png') or path.stat().st_size > 2_000_000:
                    continue
                content = path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                name = 'diagnostics/' + path.name
                if cached['artifacts'].get(name) != sha:
                    api.artifact(run, name, content)
                    cached['artifacts'][name] = sha
        status = 'FINISHED' if completion.get('state') == 'complete' else 'FAILED' if completion.get('state') == 'failed' else 'RUNNING'
        if run['info'].get('status') != status:
            update = {'run_id': rid, 'status': status}
            if status != 'RUNNING':
                update['end_time'] = int(time.time() * 1000)
            api.request('mlflow/runs/update', update)
        save(receipt_path, state)
    state['state'] = 'synced'
    state['last_sync_ms'] = int(time.time() * 1000)
    save(receipt_path, state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tracking-uri', required=True)
    p.add_argument('--record-key', required=True)
    p.add_argument('--diagnostic-dir', type=Path)
    args = p.parse_args()
    result = sync_run(args.source, args.output, args.tracking_uri, args.record_key, args.diagnostic_dir)
    print(json.dumps({'state': result['state'], 'runs': {a: r['run_id'] for a, r in result['runs'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
