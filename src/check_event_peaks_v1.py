"""Hand-calculated, validation-free checks for event-record peak evaluation."""
from copy import deepcopy
from datetime import datetime, timedelta
from evaluate_event_peaks_v1 import reconstruct, record_info, select_origin, score, summarize, validate_predictions

START = datetime(2000, 1, 1)
SCALE = dict(mean=0., scale=1.)


def fixture(values, origins=None):
    rows = []
    for at in (range(11, len(values)-12) if origins is None else origins):
        rows.append(dict(event_uid='fixture', origin_source_time=(START+timedelta(hours=at)).isoformat(),
            source_rows=[str(i) for i in range(at-11, at+13)],
            x_q=values[at-11:at+1], y_q=values[at+1:at+13]))
    catalog = dict(rows=str(len(values)), source_start=START.isoformat(),
        source_end=(START+timedelta(hours=len(values)-1)).isoformat(),
        missing_grid_hours='0', conflict_rows='0', pq_valid_rows=str(len(values)),
        longest_pq_run=str(len(values)), valid_run_count='1')
    return rows, catalog


def main():
    values = [float(i) if i <= 24 else float(48-i) for i in range(49)]
    rows, cat = fixture(values)
    points, windows, issues = reconstruct(rows, SCALE)
    info = record_info(points, cat, issues)
    assert len(points) == 49 and info['record_covered'] and not info['reasons']
    assert info['observed_peak_m3_s'] == 24 and info['observed_peak_time'] == '2000-01-02T00:00:00'
    for h in (1, 3, 6, 12):
        origin, reasons = select_origin('retrospective_peak_anchored', h, windows, info, cat)
        assert not reasons and origin == START+timedelta(hours=24-h)
        assert windows[origin]['y_q'][h-1] == 24
    earliest, reasons = select_origin('earliest_available_origin', 0, windows, info, cat)
    assert earliest == START+timedelta(hours=11) and reasons == ['peak_beyond_12h_range']
    altered = dict(info, observed_peak_time=(START+timedelta(hours=15)).isoformat())
    assert select_origin('earliest_available_origin', 0, windows, altered, cat)[0] == earliest
    assert select_origin('earliest_available_origin', 0, windows, altered, cat)[1] == []
    incomplete, cc = fixture(values, [12])
    pp, ww, ii = reconstruct(incomplete, SCALE)
    assert 'record_union_incomplete' in record_info(pp, cc, ii)['reasons']
    gap = dict(points); del gap[START+timedelta(hours=5)]
    assert 'record_time_gap' in record_info(gap, cat, [])['reasons']
    conflict = deepcopy(rows); conflict[1]['x_q'][0] += 1
    assert 'duplicate_q_conflict' in reconstruct(conflict, SCALE)[2]
    for offset in (0, 48):
        vv = [1.] * 49; vv[offset] = 10.
        rr, cc = fixture(vv); pp, _, ii = reconstruct(rr, SCALE)
        assert 'observed_peak_at_record_boundary' in record_info(pp, cc, ii)['reasons']
    vv = [1.] * 49; vv[24] = vv[25] = vv[28] = 10.
    rr, cc = fixture(vv); pp, _, ii = reconstruct(rr, SCALE); tie = record_info(pp, cc, ii)
    assert tie['observed_peak_tied_hours'] == 3 and tie['observed_peak_separated_segments'] == 2
    vv[0] = 10.; rr, cc = fixture(vv); pp, _, ii = reconstruct(rr, SCALE)
    assert 'observed_peak_at_record_boundary' in record_info(pp, cc, ii)['reasons']
    shortpeak = dict(info, observed_peak_time=(START+timedelta(hours=11)).isoformat())
    assert select_origin('retrospective_peak_anchored', 1, windows, shortpeak, cat)[1] == ['insufficient_history_12h']
    latepeak = dict(info, observed_peak_time=(START+timedelta(hours=47)).isoformat())
    assert select_origin('retrospective_peak_anchored', 1, windows, latepeak, cat)[1] == ['insufficient_future_record_12h']
    absent = dict(windows); del absent[START+timedelta(hours=23)]
    assert select_origin('retrospective_peak_anchored', 1, absent, info, cat)[1] == ['origin_missing_other']
    bad = deepcopy(rows); bad[0]['y_q'].pop()
    try: reconstruct(bad, SCALE)
    except ValueError: pass
    else: raise AssertionError('Incomplete 12h target accepted')
    hand = dict(observed_peak_m3_s=20., observed_peak_time=(START+timedelta(hours=15)).isoformat())
    pred = [2., 4., 18., 18., 3., 1., 2., 3., 4., 5., 6., 7.]
    got = score(pred, START+timedelta(hours=11), hand)
    assert got['peak_signed_error_m3_s'] == -2. and got['peak_absolute_error_percent'] == 10.
    assert got['peak_signed_error_percent'] == -10. and got['timing_signed_error_hours'] == -1.
    assert got['predicted_peak_tied_hours'] == 2 and not got['predicted_peak_at_output_edge']
    flat = score([5.] * 12, START+timedelta(hours=11), hand)
    assert flat['predicted_peak_tied_hours'] == 12 and flat['predicted_peak_at_output_edge']
    assert flat['timing_signed_error_hours'] == -3.
    assert score([0.] * 12, START+timedelta(hours=11), dict(hand, observed_peak_m3_s=0.))['ratio_status'] == 'NOT_EVALUABLE'
    preds = [dict(event_uid=r['event_uid'], origin_source_time=r['origin_source_time'],
        source_rows=r['source_rows'], observed_q_m3_s=list(r['y_q']), predicted_q_m3_s=list(r['y_q'])) for r in rows]
    assert validate_predictions(preds, rows, SCALE)['max_observation_absolute_difference_m3_s'] == 0
    preds[0]['observed_q_m3_s'][0] += 1
    try: validate_predictions(preds, rows, SCALE)
    except ValueError: pass
    else: raise AssertionError('Observation mismatch accepted')
    metric = dict(got, protocol='retrospective_peak_anchored', lead_hours=1, model='baseline', high95=True)
    scores = [dict(metric, target='a', event_uid=str(i), peak_absolute_error_percent=v,
        timing_absolute_error_hours=v) for i,v in enumerate((2., 4.))]
    scores.append(dict(metric, target='b', event_uid='x', peak_absolute_error_percent=10., timing_absolute_error_hours=10.))
    targets, aggregate, _, _ = summarize(scores, [])
    a = next(r for r in targets if r['target']=='a' and r['subset']=='all' and r['lead_hours']==1 and r['model']=='baseline')
    cross = next(r for r in aggregate if r['subset']=='all' and r['lead_hours']==1 and r['model']=='baseline')
    assert a['peak_absolute_error_percent'] == 3 and cross['peak_absolute_error_percent_target_median'] == 6.5
    assert cross['n_events'] == 3 and cross['n_targets'] == 2
    print('PASS: union/coverage/conflicts/boundaries/ties; fixed origin; 1/3/6/12h origins; 12h targets; hand scores; observation mismatch; event/target aggregation')


if __name__ == '__main__':
    main()
