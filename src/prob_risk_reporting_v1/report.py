"""Read only saved small tables. Never import training or load model tensors."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from src.multiwindow_peak_v1.interpret import completed_source

ROLES = ('fit', 'select', 'development')
ARMS = ('S0', 'S1', 'S2', 'S3')
SEEDS = (2, 17, 42)
SOURCE_VERSION = '97c39309ac38bfe43f228f3ae790986647c662af'
SOURCE_RUN = 'multiwindow_formal_97c3930_20261011'
HORIZONS = (1, 3, 6, 12)
SOURCE_ROLES = ROLES + ('calibration',)


def validate_source(source):
    config = completed_source(source)
    record = json.loads((source.parent/'run_record.json').read_text())
    if (config.get('code_version') != SOURCE_VERSION or record.get('head') != SOURCE_VERSION
            or config.get('run_id') != SOURCE_RUN or source.parent.name != SOURCE_RUN
            or config.get('change_id') != 'WP-20261011-MULTIWINDOW-PEAK-V1'
            or record.get('state') != 'complete' or record.get('exit_code') != 0):
        raise ValueError('source is not the approved completed formal batch')
    return config['code_version']


def validate_rows(rows, arm, seed, strata=False):
    seen = set()
    for row in rows:
        role, horizon = row['role'], int(row['horizon'])
        if row['arm'] != arm or int(row['seed']) != seed or role not in SOURCE_ROLES or horizon not in HORIZONS:
            raise ValueError('saved summary/stratum identity mismatch')
        key = (role, horizon, row['stratum']) if strata else (role, horizon)
        if key in seen: raise ValueError('duplicate saved summary/stratum')
        seen.add(key)
    if not strata and seen != {(role,h) for role in SOURCE_ROLES for h in HORIZONS}:
        raise ValueError('incomplete saved summary')


def input_identity(station):
    if station.startswith('da_tl_cn:'):
        return 'historical_author_P_Q; P_timezone_and_accumulation_unknown; no_future_rain'
    if station.startswith('camels_de'):
        return 'historical_observed_Q_and_author_filled_P; no_future_rain'
    if station.startswith('lamah_ice:'):
        return 'historical_observed_Q_and_ERA5_Land_reanalysis_P; no_future_rain'
    return 'historical_P_Q; precipitation_provenance_unknown; no_future_rain'


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def standard_score(truth, prediction, eligible):
    """GB/T22482 6.5.4a/6.5.5a: strict, never discard model failures."""
    if eligible is not True:
        return dict(standard_state='NOT_EVALUABLE', standard_reason='mainpeak_eligibility_unconfirmed', absolute_relative_error=None, qualified=None, boundary20=False)
    q, pred = number(truth), number(prediction)
    if q is None or q <= 0:
        return dict(standard_state='NOT_EVALUABLE', standard_reason='truth_missing_or_nonpositive', absolute_relative_error=None, qualified=None, boundary20=False)
    if pred is None:
        return dict(standard_state='MODEL_FAILED', standard_reason='prediction_nonfinite_or_missing', absolute_relative_error=None, qualified=False, boundary20=False)
    # Compare error with allowed magnitude directly; do not round percentages.
    error = abs(pred - q)
    decimal_error = abs(Decimal(str(pred)) - Decimal(str(q)))
    allowed = Decimal("0.2") * Decimal(str(q))
    boundary = decimal_error == allowed
    return dict(standard_state='EVALUABLE', standard_reason='strict_absolute_relative_error_lt_20_percent', absolute_relative_error=error / q, qualified=decimal_error < allowed, boundary20=boundary)


def risk_target(current_state):
    return {'current_low': 'future_window_exceedance', 'current_high': 'maintenance_or_reexceedance'}.get(current_state, 'future_window_exceedance_current_state_unknown')


class Macro:
    """Window mean -> source event mean -> station mean. Keep denominator counts."""
    def __init__(self):
        self.values = defaultdict(lambda: [0., 0])

    def add(self, station, event, value):
        if value is not None:
            v = self.values[station, event]
            v[0] += value
            v[1] += 1

    def result(self):
        stations = defaultdict(list)
        for (station, _), (total, n) in self.values.items():
            stations[station].append(total / n)
        return dict(value=sum(sum(v) / len(v) for v in stations.values()) / len(stations) if stations else None,
                    n_windows=sum(v[1] for v in self.values.values()), n_events=len(self.values), n_stations=len(stations))


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    rows = list(rows)
    if rows:
        with path.open('x', newline='') as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def convert_row(r, source, line, model_version):
    q, p, threshold = number(r['truth_m3_s']), number(r['point_m3_s']), number(r['qscale'])
    deterministic = r['arm'] == 'S0'
    prob = None if deterministic else number(r['probability'])
    probability_state = 'NOT_APPLICABLE' if deterministic else 'EVALUABLE' if prob is not None and 0 <= prob <= 1 else 'MODEL_FAILED'
    if probability_state != 'EVALUABLE': prob = None
    low = None if deterministic else number(r['q05_m3_s'])
    high = None if deterministic else number(r['q95_m3_s'])
    interval_state = 'NOT_APPLICABLE' if deterministic else 'EVALUABLE' if low is not None and high is not None and low <= high else 'MODEL_FAILED'
    label = int(q > threshold) if q is not None and threshold is not None else None
    legal = r.get('legal_mainpeak_minus6') == 'True' and int(r['horizon']) == 6
    row = dict(model=r['arm'], model_version=model_version, seed=int(r['seed']), data_role=r['role'],
               station=r['station'], source_event_id=r.get('source_event_id') or r['event'], event_unit='source_event' if r.get('source_event_id') else 'saved_temporal_block_not_independent_flood',
               window_id=r['window_id'], origin_time=r['origin_time'], origin_identity='retrospective_mainpeak_minus6' if legal else 'ordinary_saved_window', horizon_hours=int(r['horizon']),
               input_information=input_identity(r['station']), static_information='retrospective_basin_background; as_issued_availability_not_verified', threshold_m3_s=threshold, threshold_source='fit_Q95_saved_qscale; research_threshold_not_warning_level',
               probability=prob, probability_state=probability_state, probability_identity='raw_saved_scenario_exceedance_frequency; not_probability_calibrated',
               peak_median_m3_s=p, peak_q05_m3_s=low, peak_q95_m3_s=high, interval_state=interval_state, truth_peak_m3_s=q,
               current_state='UNKNOWN', current_state_reason='current_Q_not_saved_in_window_table', risk_target=risk_target('UNKNOWN'),
               escalation_state='PENDING_TARGET_DEFINITION', exact_peak_time_state='NOT_APPLICABLE' if r['arm'] in ('S2','S3') else 'NOT_EVALUABLE',
               peak_time_reason='no_hourly_peak_time_saved_in_input_table',
               brier=(prob-label)**2 if prob is not None and label is not None else None,
               coverage90=int(low <= q <= high) if interval_state == 'EVALUABLE' and q is not None else None,
               width90_m3_s=high-low if interval_state == 'EVALUABLE' else None,
               observed_exceedance=label, source_table=str(source), source_csv_line=line)
    row.update(standard_score(q, p, legal))
    return row


def summarize(rows):
    """One model/seed/role/horizon; failure-inclusive standard rate is explicit."""
    metrics = {key: Macro() for key in ('brier','coverage90','width90_m3_s','standard_failure_inclusive','standard_finite_only')}
    counts = Counter()
    for r in rows:
        counts['n_rows'] += 1; counts[r['standard_state']] += 1
        counts['qualified'] += int(r['qualified'] is True); counts['boundary20'] += int(r['boundary20'])
        counts['probability_'+r['probability_state']] += 1
        for key in ('brier','coverage90','width90_m3_s'): metrics[key].add(r['station'],r['source_event_id'],r[key])
        if r['standard_state'] in ('EVALUABLE','MODEL_FAILED'):
            metrics['standard_failure_inclusive'].add(r['station'],r['source_event_id'],float(r['qualified']))
        if r['standard_state'] == 'EVALUABLE':
            metrics['standard_finite_only'].add(r['station'],r['source_event_id'],float(r['qualified']))
    return dict(counts=dict(counts), metrics={k:v.result() for k,v in metrics.items()},
                standard_denominator=counts['EVALUABLE']+counts['MODEL_FAILED'],
                raw_standard_rate=counts['qualified']/(counts['EVALUABLE']+counts['MODEL_FAILED']) if counts['EVALUABLE']+counts['MODEL_FAILED'] else None)


def run(source, out):
    model_version = validate_source(source)
    out.mkdir(parents=True, exist_ok=False)
    provenance = []; summaries = []; comparison = []; strata = []; historical = []
    def remember(path):
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return path
    for seed in SEEDS:
        for arm in ARMS:
            folder = source / f'seed_{seed}' / arm
            path = remember(folder/'window_metrics.csv')
            groups = defaultdict(list); seen_windows = set()
            dest = out/f'{arm}_seed{seed}_risk_windows.csv'
            with path.open(newline='') as f, dest.open('x',newline='') as output:
                writer = None
                for line, raw in enumerate(csv.DictReader(f), 2):
                    if raw['role'] not in SOURCE_ROLES or int(raw['horizon']) not in HORIZONS:
                        raise ValueError('unexpected saved role or horizon')
                    if raw['role'] not in ROLES: continue
                    if raw['arm'] != arm or int(raw['seed']) != seed: raise ValueError('arm/seed mismatch')
                    identity = (raw['role'],raw['horizon'],raw['window_id'])
                    if identity in seen_windows: raise ValueError('duplicate saved window')
                    seen_windows.add(identity)
                    r = convert_row(raw,path,line,model_version)
                    if writer is None: writer=csv.DictWriter(output,list(r)); writer.writeheader()
                    writer.writerow(r); groups[r['data_role'],r['horizon_hours']].append(r)
            saved = json.loads(remember(folder/'final_summary.json').read_text())
            validate_rows(saved,arm,seed)
            lines = json.loads(remember(folder/'alarm_lines.json').read_text())
            for key, rows in sorted(groups.items()):
                role, h = key
                s = dict(model=arm,seed=seed,data_role=role,horizon_hours=h,**summarize(rows)); summaries.append(s)
                old = next(r for r in saved if r['role']==role and r['horizon']==h)
                for metric, old_value in [('brier',(old.get('probability') or {}).get('brier')),('coverage90',old.get('coverage90')),('width90_m3_s',old.get('width90_m3_s'))]:
                    new = s['metrics'][metric]['value']
                    comparison.append(dict(model=arm,seed=seed,data_role=role,horizon_hours=h,metric=metric,saved=old_value,recomputed=new,absolute_difference=abs(new-old_value) if new is not None and old_value is not None else None))
                historical.append(dict(model=arm,seed=seed,data_role=role,horizon_hours=h,saved_summary=old,historical_alarm_line=lines.get(str(h)),source=str(folder/'final_summary.json'),
                    calibration_identity='historical current-low 5% FPR target; calibration is alarm threshold selection, not probability recalibration', event_discovery_state='partial: not saved in source summary'))
            with remember(folder/'strata_metrics.csv').open() as f:
                saved_strata = list(csv.DictReader(f))
            validate_rows(saved_strata,arm,seed,strata=True)
            for r in saved_strata:
                if r['role'] not in ROLES: continue
                strata.append(dict(**r, reporting_label='PENDING_TARGET_DEFINITION' if r['stratum']=='current_high_upgrade' else 'maintenance_or_reexceedance' if r['stratum']=='current_high' else r['stratum'],
                    provenance='saved aggregate only; current-high-upgrade legacy label does not define a risk escalation target'))
    dump(out/'summary.json',dict(state='partial',scope='saved fit/select/development only; no test/inference/training',unit='window -> source event or saved temporal block -> station equal; roles/seeds separate',summaries=summaries))
    dump(out/'historical_metrics.json',historical)
    dump(out/'provenance.json',dict(inputs=provenance,mapping={'truth_peak_m3_s':'truth_m3_s','peak_median_m3_s':'point_m3_s','threshold_m3_s':'qscale','probability':'probability; S0 null','interval':'q05_m3_s/q95_m3_s; S0 null','standard_eligibility':'legal_mainpeak_minus6=True AND horizon=6'},gaps=['current Q absent: per-window current status UNKNOWN; saved current-low/high strata retained separately','precise peak time absent; GB/T22482 6.5.4b not scored','new escalation definition pending','event discovery and lead time unavailable in saved small tables','no business warning thresholds/impact plans; no government response grade']))
    csv_write(out/'historical_consistency.csv',comparison); csv_write(out/'saved_strata.csv',strata)
    (out/'README.md').write_text('''# 概率风险附加报告\n\n本报告只转换历史保存的 fit/select/development 表，S0–S3各三seed分开；不读取test、不重推断、不抽样、不训练。JSON为分层汇总，CSV为逐窗记录与原字段位置。\n\n概率表示未来1/3/6/12小时窗口峰超过fit Q95的概率，是原始情景频率，未必经过概率校准。历史报警线来自calibration角色选择，不代表概率本身已校准。Q95是研究阈值，不是业务警戒线。当前Q未保存，因此逐窗状态UNKNOWN；当前未高/已高的保存分层另列，已高只能称维持/再超过。严重升级定义待定。S0概率和区间NOT_APPLICABLE；S2/S3无精确峰时。\n\nGB/T22482—2008 6.5.4a、6.5.5a仅对已保存真实主峰前6小时且horizon=6的起报标记合格，绝对相对误差严格小于20%，等于20%不合格并单列。真值缺失/非正、资格不足不可评；模型失败进入失败计数及含失败的分母，另报有限预测分母。峰时条款6.5.4b未评价；许可时差为max(0.3*T,3小时,计算时段)。\n\n统计先窗口内平均，再来源事件/保存时间块，最后站点等权；时间块不能当独立洪水。提供原始行数、事件/站点数和各分母。片段资料不能推断全年误报，development已用于方向选择，旧test曾被早期使用，本批不提供新的独立测试证据。无业务预案/影响资料，不给政府响应等级或应急指令。\n''')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); run(args.source,args.out)

if __name__ == '__main__': main()
