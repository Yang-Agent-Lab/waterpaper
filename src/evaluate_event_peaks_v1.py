"""Validation-only, saved-prediction evaluation of author event-record peaks."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import time

MODELS = ('baseline', 'none', 'unconditional', 'high95', 'persistence')
PROTOCOLS = [('retrospective_peak_anchored', h) for h in (1, 3, 6, 12)] + [('earliest_available_origin', 0)]
METRICS = ('peak_absolute_error_m3_s', 'peak_signed_error_m3_s', 'peak_absolute_error_percent',
           'peak_signed_error_percent', 'timing_absolute_error_hours', 'timing_signed_error_hours')
COMPARISONS = (('none', 'baseline'), ('none', 'unconditional'), ('high95', 'none'))


def read_jsonl(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def raw(values, scale):
    return [v * scale['scale'] + scale['mean'] for v in values]


def reconstruct(rows, scale):
    """Union 24h source rows, assigning each its offset from origin hour 11."""
    points, by_source, windows, issues = {}, {}, {}, set()
    if not finite([scale['mean'], scale['scale']]) or scale['scale'] <= 0:
        raise ValueError('Invalid training Q scale')
    for row in rows:
        if len(row['x_q']) != 12 or len(row['y_q']) != 12 or len(row['source_rows']) != 24:
            raise ValueError('Expected complete 12h history and 12h target')
        if len(set(row['source_rows'])) != 24 or not finite(row['x_q'] + row['y_q']):
            raise ValueError('Invalid source rows or nonfinite observed Q')
        origin = datetime.fromisoformat(row['origin_source_time'])
        if origin in windows:
            raise ValueError('Duplicate event/origin window')
        windows[origin] = row
        for offset, (source, q) in enumerate(zip(row['source_rows'], raw(row['x_q'] + row['y_q'], scale))):
            stamp = origin + timedelta(hours=offset-11)
            source = str(source)
            if source in by_source:
                old_time, old_q = by_source[source]
                if old_time != stamp:
                    issues.add('duplicate_source_time_conflict')
                if not math.isclose(old_q, q, rel_tol=1e-12, abs_tol=1e-9):
                    issues.add('duplicate_q_conflict')
            if stamp in points and points[stamp][0] != source:
                issues.add('duplicate_time_source_conflict')
            by_source.setdefault(source, (stamp, q))
            points.setdefault(stamp, (source, q))
    return points, windows, sorted(issues)


def record_info(points, catalog, issues):
    reasons = list(issues)
    expected_rows = int(catalog['rows'])
    start, end = (datetime.fromisoformat(catalog[k]) for k in ('source_start', 'source_end'))
    times = sorted(points)
    if not times:
        return dict(record_covered=False, reasons=sorted(set(reasons + ['no_validation_windows'])),
                    physical_event_completeness='unknown', union_rows=0, expected_rows=expected_rows,
                    observed_peak_m3_s=None, observed_peak_time=None, high95=False)
    unique_sources = len({v[0] for v in points.values()})
    endpoints_match = times[0] == start and times[-1] == end
    count_match = len(times) == unique_sources == expected_rows
    grid_match = (end-start).total_seconds() == (expected_rows-1)*3600
    gaps = sum((b-a).total_seconds() != 3600 for a,b in zip(times, times[1:]))
    if not count_match or not endpoints_match or not grid_match:
        reasons.append('record_union_incomplete')
    if gaps:
        reasons.append('record_time_gap')
    source_gap = any(int(points[b][0])-int(points[a][0]) != 1 for a,b in zip(times, times[1:]))
    if source_gap:
        reasons.append('source_row_gap')
    if int(catalog['missing_grid_hours']): reasons.append('catalog_missing_hours')
    if int(catalog['conflict_rows']): reasons.append('catalog_conflicts')
    if (int(catalog['pq_valid_rows']) != expected_rows or int(catalog['longest_pq_run']) != expected_rows
            or int(catalog['valid_run_count']) != 1):
        reasons.append('catalog_incomplete_continuous_record')
    covered = not reasons
    peak = max(points[t][1] for t in times)
    maxima = [t for t in times if points[t][1] == peak]
    boundary = times[0] in maxima or times[-1] in maxima
    segments = 1 + sum((b-a).total_seconds() != 3600 for a,b in zip(maxima, maxima[1:]))
    if boundary: reasons.append('observed_peak_at_record_boundary')
    return dict(record_covered=covered, reasons=sorted(set(reasons)), union_rows=len(times),
        unique_source_rows=unique_sources, expected_rows=expected_rows,
        union_start=times[0].isoformat(), union_end=times[-1].isoformat(),
        catalog_start=start.isoformat(), catalog_end=end.isoformat(),
        hourly_gaps=gaps, source_row_gap=source_gap, endpoints_match=endpoints_match,
        count_match=count_match, grid_match=grid_match,
        observed_peak_m3_s=peak, observed_peak_time=maxima[0].isoformat(),
        observed_peak_tied_hours=len(maxima), observed_peak_separated_segments=segments,
        observed_peak_at_record_boundary=boundary, physical_event_completeness='unknown')


def select_origin(protocol, lead, windows, info, catalog):
    """Earliest origin selection uses only window keys, never observed values."""
    if protocol == 'earliest_available_origin':
        origin = min(windows) if windows else None
    else:
        peak_time = info.get('observed_peak_time')
        origin = datetime.fromisoformat(peak_time)-timedelta(hours=lead) if peak_time else None
    reasons = []
    if origin is None:
        reasons.append('no_validation_windows')
    elif origin not in windows:
        start, end = (datetime.fromisoformat(catalog[k]) for k in ('source_start', 'source_end'))
        if origin-timedelta(hours=11) < start: reasons.append('insufficient_history_12h')
        if origin+timedelta(hours=12) > end: reasons.append('insufficient_future_record_12h')
        if not reasons: reasons.append('origin_missing_other')
    elif info.get('observed_peak_time'):
        peak_time = datetime.fromisoformat(info['observed_peak_time'])
        if peak_time <= origin: reasons.append('peak_already_occurred')
        elif peak_time > origin+timedelta(hours=12): reasons.append('peak_beyond_12h_range')
    return origin, reasons


def validate_predictions(records, rows, scale):
    if len(records) != len(rows):
        raise ValueError('Saved prediction count differs from validation')
    expected = {(r['event_uid'], r['origin_source_time']): r for r in rows}
    if len(expected) != len(rows): raise ValueError('Duplicate validation event/origin')
    seen, max_difference = set(), 0.
    for record in records:
        key = (record['event_uid'], record['origin_source_time'])
        if key not in expected or key in seen:
            raise ValueError('Saved prediction event/origin mismatch or duplicate')
        seen.add(key)
        row = expected[key]
        if record['source_rows'] != row['source_rows']:
            raise ValueError('Saved prediction source_rows mismatch')
        truth, prediction = record['observed_q_m3_s'], record['predicted_q_m3_s']
        if len(truth) != 12 or len(prediction) != 12 or not finite(truth + prediction):
            raise ValueError('Saved predictions must contain finite complete 12h values')
        for a,b in zip(truth, raw(row['y_q'], scale)):
            max_difference = max(max_difference, abs(a-b))
            if not math.isclose(a,b, rel_tol=1e-6, abs_tol=1e-3):
                raise ValueError(f'Saved observed Q differs from validation: {key}, {a}, {b}')
    return dict(n_windows=len(records), max_observation_absolute_difference_m3_s=max_difference,
                observation_rtol=1e-6, observation_atol_m3_s=1e-3,
                provenance_fields=['event_uid', 'origin_source_time', 'source_rows'])


def score(predicted, origin, info):
    maximum = max(predicted)
    maxima = [i for i,v in enumerate(predicted) if v == maximum]
    when = origin+timedelta(hours=maxima[0]+1)
    difference = maximum-info['observed_peak_m3_s']
    timing = (when-datetime.fromisoformat(info['observed_peak_time'])).total_seconds()/3600
    ratio = 100*difference/info['observed_peak_m3_s'] if info['observed_peak_m3_s'] > 0 else None
    return dict(predicted_peak_m3_s=maximum, predicted_peak_time=when.isoformat(),
        peak_signed_error_m3_s=difference, peak_absolute_error_m3_s=abs(difference),
        peak_signed_error_percent=ratio, peak_absolute_error_percent=abs(ratio) if ratio is not None else None,
        timing_signed_error_hours=timing, timing_absolute_error_hours=abs(timing),
        ratio_status='evaluable' if ratio is not None else 'NOT_EVALUABLE',
        predicted_peak_tied_hours=len(maxima), predicted_peak_at_output_edge=0 in maxima or 11 in maxima,
        predicted_peak_first_at_output_edge=maxima[0] in (0,11),
        predicted_peak_separated_segments=1+sum(b-a != 1 for a,b in zip(maxima,maxima[1:])))


def mean_present(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(values) if values else None


def summarize(scores, eligibility):
    targets = sorted({r['target'] for r in scores + eligibility})
    target_rows, aggregate, paired, paired_aggregate = [], [], [], []
    for protocol, lead in PROTOCOLS:
        for subset in ('all', 'high95'):
            for model in MODELS:
                summaries = []
                for target in targets:
                    rows = [r for r in scores if r['protocol']==protocol and r['lead_hours']==lead
                        and r['model']==model and r['target']==target and (subset=='all' or r['high95'])]
                    total = [r for r in eligibility if r['protocol']==protocol and r['lead_hours']==lead
                        and r['target']==target and (subset=='all' or r['high95'])]
                    item = dict(protocol=protocol, lead_hours=lead, subset=subset, model=model, target=target,
                        status='evaluable' if rows else 'NOT_EVALUABLE', n_events=len(rows), n_total_events=len(total),
                        n_ratio_evaluable=sum(r['ratio_status']=='evaluable' for r in rows),
                        predicted_output_edge_events=sum(r['predicted_peak_at_output_edge'] for r in rows),
                        predicted_tied_peak_events=sum(r['predicted_peak_tied_hours']>1 for r in rows),
                        **{k:mean_present(rows,k) for k in METRICS})
                    target_rows.append(item)
                    if rows: summaries.append(item)
                cross = dict(protocol=protocol, lead_hours=lead, subset=subset, model=model,
                    status='evaluable' if summaries else 'NOT_EVALUABLE', n_targets=len(summaries),
                    n_events=sum(r['n_events'] for r in summaries),
                    n_total_events=sum(r['n_total_events'] for r in target_rows if r['protocol']==protocol
                        and r['lead_hours']==lead and r['subset']==subset and r['model']==model),
                    predicted_output_edge_events=sum(r['predicted_output_edge_events'] for r in summaries),
                    predicted_tied_peak_events=sum(r['predicted_tied_peak_events'] for r in summaries))
                for k in METRICS:
                    if k.endswith('m3_s'): continue
                    values = [r[k] for r in summaries if r[k] is not None]
                    cross[k+'_target_median'] = statistics.median(values) if values else None
                    cross[k+'_n_targets'] = len(values)
                aggregate.append(cross)
            for left, right in COMPARISONS:
                indexed = {(r['target'],r['event_uid'],r['model']):r for r in scores
                    if r['protocol']==protocol and r['lead_hours']==lead and (subset=='all' or r['high95'])}
                comparison_rows = []
                for target,event,model in sorted(indexed):
                    if model != left: continue
                    a,b = indexed[(target,event,left)], indexed[(target,event,right)]
                    entry = dict(protocol=protocol, lead_hours=lead, subset=subset, target=target,
                        event_uid=event, comparison=left+'_vs_'+right,
                        origin_source_time=a['origin_source_time'])
                    for k in METRICS:
                        entry[k+'_left'] = a[k]; entry[k+'_right'] = b[k]
                        entry[k+'_delta'] = a[k]-b[k] if a[k] is not None and b[k] is not None else None
                    paired.append(entry); comparison_rows.append(entry)
                cross = dict(protocol=protocol, lead_hours=lead, subset=subset, comparison=left+'_vs_'+right,
                    n_events=len(comparison_rows), n_targets=len({r['target'] for r in comparison_rows}),
                    status='evaluable' if comparison_rows else 'NOT_EVALUABLE')
                for k in METRICS:
                    deltas, relative = [], []
                    for target in targets:
                        rows = [r for r in comparison_rows if r['target']==target]
                        delta = mean_present(rows,k+'_delta')
                        if delta is not None: deltas.append(delta)
                        av,bv = mean_present(rows,k+'_left'), mean_present(rows,k+'_right')
                        if k.startswith(('peak_absolute','timing_absolute')) and av is not None and bv:
                            relative.append(100*(av-bv)/bv)
                    if not k.endswith('m3_s'):
                        cross[k+'_delta_target_median'] = statistics.median(deltas) if deltas else None
                    if k.startswith(('peak_absolute','timing_absolute')):
                        cross[k+'_relative_percent_target_median'] = statistics.median(relative) if relative else None
                        cross[k+'_relative_n_targets'] = len(relative)
                        cross[k+'_improved_targets'] = sum(d < 0 for d in deltas)
                paired_aggregate.append(cross)
    return target_rows, aggregate, paired, paired_aggregate


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:';'.join(v) if isinstance(v,list) else v for k,v in r.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'catalog', 'split-manifest', 'baseline-root', 'alignment-root', 'out-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--walltime-seconds', type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.walltime_seconds <= 600: parser.error('CPU budget must be 1..600 seconds')
    started = time.monotonic()
    def budget_exceeded(*unused): raise TimeoutError('600s or selected CPU wall budget exceeded')
    signal.signal(signal.SIGALRM, budget_exceeded); signal.alarm(args.walltime_seconds)
    inputs = []
    def remember(path):
        inputs.append(dict(path=str(path.resolve()), bytes=path.stat().st_size))
        return path
    catalogs = {r['event_uid']:r for r in csv.DictReader(remember(args.catalog).open())}
    splits = list(csv.DictReader(remember(args.split_manifest).open()))
    validation_events = {r['event_uid'] for r in splits if r['candidate_split']=='validation'}
    targets = sorted(p.name for p in args.data_root.iterdir() if p.is_dir() and (p/'training_transforms.json').is_file())
    if len(targets) != 9 or len(validation_events) != 21:
        raise ValueError('Expected fixed nine targets and 21 validation events')
    out = args.out_dir.resolve()
    for root in (args.data_root, args.baseline_root, args.alignment_root):
        if out.is_relative_to(root.resolve()) or root.resolve().is_relative_to(out):
            raise ValueError('Output overlaps input tree')
    out.mkdir(parents=True, exist_ok=False)
    records, eligibility, scores, checks = [], [], [], {}
    for target in targets:
        folder = args.data_root/target
        transform = json.loads(remember(folder/'training_transforms.json').read_text())
        scale, q95 = transform['scales']['q_m3_s'], transform['q95_m3_s']
        if not finite([q95]) or q95 < 0: raise ValueError('Invalid training Q95')
        rows = read_jsonl(remember(folder/'validation.jsonl.gz'))
        expected = {e for e in validation_events if e.startswith(transform['station_uid']+':')}
        if not expected or any(r['event_uid'] not in expected for r in rows):
            raise ValueError('Validation window/split/target mismatch')
        if expected & set(transform['fit_event_ids']): raise ValueError('Validation event is in training scale fit')
        predictions = {}
        checks[target] = {}
        for model in MODELS[:-1]:
            path = (args.baseline_root/target/'baseline' if model=='baseline' else args.alignment_root/target/model)/'validation_predictions.jsonl.gz'
            saved = read_jsonl(remember(path))
            checks[target][model] = validate_predictions(saved,rows,scale)
            predictions[model] = {(r['event_uid'],r['origin_source_time']):r['predicted_q_m3_s'] for r in saved}
        for event in sorted(expected):
            event_rows = [r for r in rows if r['event_uid']==event]
            points, windows, issues = reconstruct(event_rows,scale)
            cat = catalogs[event]
            if cat['station_uid'] != transform['station_uid']: raise ValueError('Catalog target mismatch')
            info = record_info(points,cat,issues)
            info.update(target=target,event_uid=event,timezone=transform.get('timezone','unknown'),
                high95=info['observed_peak_m3_s'] is not None and info['observed_peak_m3_s'] > q95,
                q95_m3_s=q95,n_validation_windows=len(event_rows),
                known_overlap_group=cat['known_overlap_group'],temporal_overlap_component=cat['temporal_overlap_component'])
            records.append(info)
            for protocol,lead in PROTOCOLS:
                origin, origin_reasons = select_origin(protocol,lead,windows,info,cat)
                reasons = sorted(set(info['reasons']+origin_reasons))
                peak_time = datetime.fromisoformat(info['observed_peak_time']) if info['observed_peak_time'] else None
                available = origin in windows if origin else False
                covered = bool(origin and peak_time and origin < peak_time <= origin+timedelta(hours=12))
                entry = dict(target=target,event_uid=event,protocol=protocol,lead_hours=lead,
                    origin_source_time=origin.isoformat() if origin else None,
                    status='evaluable' if not reasons else 'NOT_EVALUABLE',reasons=reasons,
                    origin_available=available,peak_in_12h_range=covered,record_covered=info['record_covered'],
                    actual_peak_lead_hours=(peak_time-origin).total_seconds()/3600 if origin and peak_time else None,
                    observed_peak_m3_s=info['observed_peak_m3_s'],observed_peak_time=info['observed_peak_time'],
                    high95=info['high95'],physical_event_completeness='unknown')
                eligibility.append(entry)
                if reasons: continue
                row = windows[origin]
                for model in MODELS:
                    forecast = ([raw(row['x_q'],scale)[-1]]*12 if model=='persistence'
                        else predictions[model][(event,row['origin_source_time'])])
                    scores.append(dict(entry,model=model,
                        origin_latest_observed_q_m3_s=raw(row['x_q'],scale)[-1],
                        observed_peak_tied_hours=info['observed_peak_tied_hours'],
                        observed_peak_separated_segments=info['observed_peak_separated_segments'],
                        **score(forecast,origin,info)))
    if {r['event_uid'] for r in records} != validation_events: raise ValueError('Validation event denominator mismatch')
    target_rows, aggregate, paired, paired_aggregate = summarize(scores,eligibility)
    for name,rows in (('event_records.csv',records),('eligibility.csv',eligibility),('event_scores.csv',scores),
                      ('target_summary.csv',target_rows),('paired_scores.csv',paired)):
        write_csv(out/name,rows)
    report = dict(state='complete',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        checked_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        scope='author_event_record_peak_validation_only',physical_event_completeness='unknown',
        test_read=False,training_read=False,weights_read=False,model_inference=False,gpu_used=False,
        n_validation_events=len(records),n_targets=len(targets),n_covered_event_records=sum(r['record_covered'] for r in records),
        n_main_eligible_records=sum(not r['reasons'] for r in records),
        protocol_denominators=[dict(protocol=p,lead_hours=h,n_total_events=len(records),
            n_evaluable_events=sum(r['status']=='evaluable' for r in eligibility if r['protocol']==p and r['lead_hours']==h),
            n_evaluable_targets=len({r['target'] for r in eligibility if r['protocol']==p and r['lead_hours']==h and r['status']=='evaluable'}),
            reasons=dict(Counter(reason for r in eligibility if r['protocol']==p and r['lead_hours']==h for reason in r['reasons']))) for p,h in PROTOCOLS],
        implementation_choices=dict(prediction_observation_rtol=1e-6,prediction_observation_atol_m3_s=1e-3,
            duplicate_rtol=1e-12,duplicate_atol_m3_s=1e-9,ties='exact maximum; first occurrence; any tied boundary maximum excludes observed record',
            lead_hours_zero='B uses a single earliest origin; actual_peak_lead_hours is reported',
            denominator='21 validation event labels, not independent physical floods',
            high95='record maximum strictly above fixed training Q95',
            prediction_scope='maximum over all 12 forecast hours, never truncated to lead h',
            magnitude_signed_error='predicted minus observed',timing_signed_error='predicted first maximum time minus observed first maximum time'),
        limitations=['A origins use retrospective observed peak time, not a real-time known peak',
            'B earliest validation origin is value-independent but author event boundaries were selected retrospectively',
            'Observed record coverage does not establish physical flood completeness or event independence',
            'Output ties and boundary maxima limit peak-time information; persistence ties all 12 hours',
            'Small validation sample, one training seed, repeated validation use; no final efficacy or window-level significance claim',
            'Target median percentage and hour metrics; no pooled across-target m3/s error'],
        source_manifest=inputs,prediction_checks=checks,aggregate=aggregate,paired_aggregate=paired_aggregate)
    with (out/'evaluation.json').open('x') as handle: json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    signal.alarm(0)
    print(json.dumps({k:report[k] for k in ('state','n_validation_events','n_targets','n_covered_event_records','n_main_eligible_records','elapsed_seconds','protocol_denominators')},ensure_ascii=False))


if __name__ == '__main__':
    main()
