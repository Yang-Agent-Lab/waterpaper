"""Summarize sampled training evidence without inferring module efficacy."""
import argparse
import json
import math
from pathlib import Path


DEFAULT_RULES = dict(consecutive_samples=3, saturation_fraction=.95,
                     near_zero_gradient_by_module={}, fit_score_key='fit_brier',
                     select_score_key='select_brier', score_index=2,
                     score_name='H6 Brier', score_unit='无量纲', score_direction='lower')


def load_records(path):
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Recorder retains damaged lines and appends a recovery marker.
            records.append(dict(kind='unreadable', line=number))
            continue
        row['_line'] = number
        records.append(row)
    return records


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize(training, rules=None):
    training = Path(training).resolve()
    rule = dict(DEFAULT_RULES, **(rules or {}))
    count = rule['consecutive_samples']
    if not isinstance(count, int) or count < 2:
        raise ValueError('consecutive_samples must be an integer >=2')
    if not 0 <= rule['saturation_fraction'] <= 1 or rule['score_direction'] not in ('lower', 'higher'):
        raise ValueError('invalid diagnostic rules')
    for threshold in rule['near_zero_gradient_by_module'].values():
        if not finite(threshold) or threshold < 0:
            raise ValueError('module gradient thresholds must be finite and nonnegative')
    metrics_path = training/'metrics.jsonl'
    records = load_records(metrics_path) if metrics_path.exists() else []
    arms = sorted({r['arm'] for r in records if r.get('arm')} | {p.parent.name for p in training.glob('*/best.json')})
    observations = []
    recommendations = []

    def observe(arm, code, message, evidence, **details):
        observations.append(dict(arm=arm, code=code, message=message, evidence=evidence,
                                 evidence_strength='sampled_diagnostic', **details))

    for arm in arms:
        rows = [r for r in records if r.get('arm') == arm]
        health = [r for r in rows if r.get('kind') == 'health']
        evidence = lambda selected: [str(metrics_path)+':'+str(r['_line']) for r in selected]
        names = sorted({n for r in health for n in r.get('metrics', {}).get('module_health', {})})
        for name in names:
            samples = [(r, r['metrics']['module_health'].get(name)) for r in health]
            samples = [(r, m) for r, m in samples if m and m.get('state') != 'NOT_APPLICABLE']
            tail = samples[-count:]
            if len(tail) < count:
                continue
            gradients = [m.get('gradient_norm') for _, m in tail]
            if all(g is None for g in gradients):
                observe(arm, 'missing_gradient', name+'连续采样没有梯度；先检查是否接入损失及是否有有效样本。', evidence([r for r, _ in tail]), module=name, sampled_steps=[r.get('step') for r, _ in tail])
            elif all(finite(g) and g == 0 for g in gradients):
                observe(arm, 'zero_gradient', name+'连续采样梯度精确为零；这是采样线索，不能推断整个训练均未学习。', evidence([r for r, _ in tail]), module=name)
            threshold = rule['near_zero_gradient_by_module'].get(name)
            if threshold is not None and all(finite(g) and 0 < g <= threshold for g in gradients):
                observe(arm, 'configured_small_gradient', name+'连续采样低于该模块专用阈值；需结合尺度及变化判断。', evidence([r for r, _ in tail]), module=name, threshold=threshold)
        bad = [r for r in health if any(finite(m.get('nonfinite_fraction')) and m['nonfinite_fraction'] > 0 for m in [r.get('metrics', {})]+list(r.get('metrics', {}).get('module_health', {}).values()))]
        if bad:
            observe(arm, 'nonfinite', '记录中存在非有限输出；核查对应步骤和失败状态。', evidence(bad))
        tail = health[-count:]
        if len(tail) == count and all(finite(r['metrics'].get('saturation_fraction')) and r['metrics']['saturation_fraction'] >= rule['saturation_fraction'] for r in tail):
            observe(arm, 'saturation', '连续采样概率大量接近0或1；结合标签比例与可靠性检查，不能直接判模型失效。', evidence(tail), threshold=rule['saturation_fraction'])
        epochs = [r for r in rows if r.get('kind') == 'epoch'][-(count+1):]
        def score(row, key):
            value = row.get('metrics', {}).get(key)
            if isinstance(value, list):
                index = rule['score_index']
                return value[index] if isinstance(index, int) and 0 <= index < len(value) else None
            return value
        if len(epochs) == count+1:
            pairs = [(score(r, rule['fit_score_key']), score(r, rule['select_score_key'])) for r in epochs]
            sign = 1 if rule['score_direction'] == 'lower' else -1
            if all(finite(v) for pair in pairs for v in pair) and all(sign*(b[0]-a[0]) < 0 and sign*(b[1]-a[1]) > 0 for a, b in zip(pairs, pairs[1:])):
                observe(arm, 'fit_select_divergence', '训练池分数连续改善、选模池连续退步；更符合泛化恶化线索，不能仅据此认定过拟合机制。', evidence(epochs), metric=rule['score_name'])
        best_path = training/arm/'best.json'
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        diagnostics = [r for r in rows if r.get('kind') in ('shadow_only', 'selected_checkpoint')]
        same_epoch = [r for r in diagnostics if best and r.get('epoch') == best.get('epoch')]
        expected_hash=best.get('checkpoint_sha256') if best else None
        matched = [r for r in same_epoch if expected_hash and r.get('identity',{}).get('checkpoint_sha256')==expected_hash]
        if not best:
            observe(arm, 'best_missing', '缺少选中checkpoint身份，无法判断诊断是否对应最终模型。', [str(best_path)])
        elif not expected_hash and same_epoch:
            observe(arm, 'best_identity_unverified', '只有选中轮次相同，缺少可核对的权重hash；不能认定最终模型已覆盖诊断。', [str(best_path)]+evidence(same_epoch), selected_checkpoint=best,identity_state='unknown_legacy')
        elif not matched:
            observe(arm, 'best_diagnostic_missing', '现有固定诊断没有覆盖选中轮次；只补该权重的fit/select固定样本诊断。', [str(best_path)]+evidence(diagnostics), selected_checkpoint=best)
        else:
            observe(arm, 'best_diagnostic_coverage', '已有选中轮次诊断；完整hash身份是否匹配见identity_state。', [str(best_path)]+evidence(matched), selected_checkpoint=best,
                    identity_state='content_hash_verified')
        selected_keys={r.get('diagnostic',{}).get('key') for r in diagnostics if r.get('kind')=='selected_checkpoint'}
        for row in diagnostics:
            if row.get('kind')=='shadow_only' and row.get('diagnostic',{}).get('key') in selected_keys:
                continue
            for role, values in row.get('diagnostic', {}).get('result', {}).get('roles', {}).items():
                for module, shadow in values.get('shadows', {}).items():
                    deltas = shadow.get('brier_shadow_minus_full')
                    if deltas and finite(deltas[-1]) and deltas[-1] < 0:
                        observe(arm, 'shadow_better', module+'临时干预在固定小组H6 Brier更低；只提示敏感性，不是重训消融贡献。', evidence([row]), data_role=role, module=module, delta=deltas[-1], epoch=row.get('epoch'), diagnostic_scope='selected' if row.get('kind')=='selected_checkpoint' else 'during_training')
    for item in observations:
        if item['code'] != 'best_diagnostic_coverage':
            recommendations.append(dict(arm=item['arm'], observation_code=item['code'], action=item['message'], automatic_action=False))
    tracking_path = training/'tracking_map.json'
    tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else None
    return dict(schema_version=1, source=str(training), rules=rule, observations=observations, recommendations=recommendations,
                metrics=dict(gradient_norm=dict(unit='参数梯度的L2范数，受参数化/裁剪影响', direction='无统一越大越好方向'), saturation_fraction=dict(unit='比例0–1', direction='需结合标签判断'), score=dict(name=rule['score_name'],unit=rule['score_unit'],direction=rule['score_direction'])),
                data_roles=['fit','select'], tracking=tracking, missing=dict(metrics=not metrics_path.exists(), health=not any(r.get('kind')=='health' for r in records), tracking=tracking is None),
                limitations=['健康是定步长采样，不代表每个更新。','影子干预不是独立重训贡献；没有确认模块效能。','缺少模块专用阈值时不判断近零梯度。','历史仅轮次匹配不等于完整权重身份确认。'])


def write_summary(training, output, rules=None):
    output = Path(output).resolve()
    source = Path(training).resolve()
    if output == source:
        raise ValueError('summary output must be a separate directory')
    report = summarize(source, rules)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    lines = ['# 诊断摘要', '只读现有记录，不重新推断。来源：'+str(source), '数据角色：fit/select；健康为采样，临时干预不是正式模块贡献。',
             '下表差值＝临时干预后Brier−完整模型Brier；负值表示干预后误差更低。不要把训练途中诊断当成最终模型诊断。',
             '|方案|模型状态|数据角色|模块干预|差值（无量纲）|', '|---|---|---|---|---:|']
    for r in report['observations']:
        if r['code']=='shadow_better':
            phase='最终选中' if r['diagnostic_scope']=='selected' else '训练途中'
            lines.append(f"|{r['arm']}|{phase}第{r['epoch']}轮|{r['data_role']}|{r['module']}|{r['delta']:+.6f}|")
    lines.append('\n## 覆盖与健康提示\n')
    for r in report['observations']:
        if r['code']=='shadow_better':continue
        detail=''
        if r.get('selected_checkpoint'):
            detail=f" 选中第{r['selected_checkpoint'].get('epoch')}轮；身份状态：{r.get('identity_state','missing')}。"
        lines.append(f"- {r['arm']} [{r['code']}] {r['message']}{detail}")
    lines.append('\n逐项证据路径、权重身份与规则阈值见同目录 summary.json。未触发提示只表示本次采样未发现对应规则异常。')
    lines += ['指标：'+json.dumps(report['metrics'], ensure_ascii=False), '限制：'+'；'.join(report['limitations'])]
    (output/'summary.md').write_text('\n\n'.join(lines)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rules', help='JSON rule overrides, including per-module gradient thresholds')
    args = parser.parse_args()
    write_summary(args.training, args.output, json.loads(Path(args.rules).read_text()) if args.rules else None)


if __name__ == '__main__':
    main()
