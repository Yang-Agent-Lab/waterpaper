from collections import defaultdict
from datetime import datetime,timedelta
import numpy as np
from src.event_risk_v1.core import mean,weights,window_metrics,false_segments

STRATEGIES=('low_start_q99','upgrade_q99','combined_first_q99')
OBSERVATION_NE={'no_observed_exceedance','first_crossing_outside_trusted_predecessor_coverage','author_trusted_threshold_side_conflict','predecessor_already_high','author_start_already_high','author_gap_before_first_crossing','trusted_gap_before_first_crossing','trusted_start_already_high','source_time_or_Q_conflict','author_prepared_conflict','no_author_records','invalid_fit_threshold_metadata'}


def strategy_mask(strategy,low,high):
    low=np.asarray(low,bool);high=np.asarray(high,bool)
    if (low&high).any():raise ValueError('current-state masks overlap')
    return low if strategy=='low_start_q99' else high if strategy=='upgrade_q99' else low|high


def routed_lines(strategy,low,low_line,high_line):
    n=len(low)
    def value(x):return np.nan if x is None else x
    if strategy=='low_start_q99':return np.repeat(value(low_line),n)
    if strategy=='upgrade_q99':return np.repeat(value(high_line),n)
    return np.where(low,value(low_line),value(high_line))


def metrics(scores,truth,rows,lines,reference,probabilistic):
    scores=np.asarray(scores);truth=np.asarray(truth,bool);lines=np.asarray(lines)
    result=window_metrics(scores,truth,rows,None,reference,probabilistic)
    available=np.isfinite(lines);finite=np.isfinite(scores);alarm=available&finite&(scores>lines)
    result.update(state='partial' if len(rows) and not available.all() else result['state'],missing_alarm_line_windows=int((~available).sum()),
      false_alarm_origin_count=int((alarm&~truth).sum()),false_alarm_segments=false_segments(rows,alarm&~truth),
      tp=int((alarm&truth).sum()),fp=int((alarm&~truth).sum()),tn=int((~alarm&~truth&finite&available).sum()),fn=int((~alarm&truth&finite&available).sum()))
    if available.all():
        for mask,key in [(~truth,'fpr'),(truth,'recall')]:
            ix=np.flatnonzero(mask)
            if len(ix):result[key]=mean(alarm[ix],[rows[i] for i in ix])
        if alarm.any():
            w=weights(rows);result['precision']=float(w@(alarm&truth)/(w@alarm))
    return result


def score_events(rows,scores,low,high,strategy,h,low_line,high_line,bases):
    mask=strategy_mask(strategy,low,high);lines=routed_lines(strategy,low,low_line,high_line);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['station_uid'],r['source_event_id']].append(i)
    result=[]
    for key,ix in sorted(groups.items()):
        base=bases[key];reason=base['reason'] or None;stamp=base['first_crossing'];cross=datetime.fromisoformat(stamp) if stamp else None
        if reason and reason not in OBSERVATION_NE and reason not in ('no_legal_current_state_origin_covering_first_crossing','no_calibration_alarm_line'):
            raise ValueError('unrecognized old eligibility state: '+reason)
        observed_ok=reason not in OBSERVATION_NE and cross is not None
        candidates=[i for i in ix if mask[i] and datetime.fromisoformat(rows[i]['origin_time'])<cross<=datetime.fromisoformat(rows[i]['origin_time'])+timedelta(hours=h)] if observed_ok else []
        missing=[i for i in candidates if not np.isfinite(lines[i])]
        hits=[i for i in candidates if np.isfinite(lines[i]) and np.isfinite(scores[i]) and scores[i]>lines[i]]
        first=min(hits,key=lambda i:rows[i]['origin_time']) if hits else None
        if not observed_ok:state='NOT_EVALUABLE'
        elif not candidates:state='NOT_EVALUABLE';reason='no_legal_current_state_origin_covering_first_crossing'
        elif missing:state='partial';reason='candidate_state_missing_calibration_alarm_line'
        else:state='evaluable';reason=None
        issue=rows[first]['origin_time'] if first is not None else None
        result.append(dict(station=key[0],source_event_id=key[1],strategy=strategy,horizon=h,state=state,reason=reason,first_crossing=stamp or None,
          candidate_origins=len(candidates),low_candidate_origins=sum(bool(low[i]) for i in candidates),high_candidate_origins=sum(bool(high[i]) for i in candidates),
          missing_line_candidates=len(missing),failed_candidates=sum(not np.isfinite(scores[i]) for i in candidates),
          discovered=bool(hits) if state=='evaluable' else None,known_detected=bool(hits),first_correct_issue=issue if state=='evaluable' else None,
          first_known_issue=issue,first_issue_state=('low_start' if low[first] else 'already_high') if first is not None else None,
          lead_hours=(cross-datetime.fromisoformat(issue)).total_seconds()/3600 if issue and state=='evaluable' else None,
          observation_eligibility='verified_in_EVENT_RISK_V1' if observed_ok else 'NOT_EVALUABLE_preserved',prefix_source_status='unknown'))
    return result


def summarize_events(records):
    legal=[r for r in records if r['state']=='evaluable'];hits=[r for r in legal if r['discovered']];by=defaultdict(list)
    for r in legal:by[r['station']].append(float(r['discovered']))
    return dict(evaluable_events=len(legal),discovered_events=len(hits),partial_events=sum(r['state']=='partial' for r in records),
      not_evaluable_events=sum(r['state']=='NOT_EVALUABLE' for r in records),station_equal_event_recall=float(np.mean([np.mean(v) for v in by.values()])) if by else None,
      detected_mean_lead_hours=float(np.mean([r['lead_hours'] for r in hits])) if hits else None)
