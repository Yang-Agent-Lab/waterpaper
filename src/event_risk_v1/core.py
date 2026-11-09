"""Pure bounded scoring functions, with explicit eligibility and failure counts."""
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

HORIZONS=(1,3,6,12)
TARGETS=('first_q95','upgrade_q99')


def weights(rows):
    counts=defaultdict(int);events=defaultdict(set)
    for r in rows:
        key=(r['station_uid'],r['source_event_id']);counts[key]+=1;events[key[0]].add(key[1])
    return np.array([1/(len(events)*len(events[r['station_uid']])*counts[r['station_uid'],r['source_event_id']]) for r in rows])


def mean(values,rows):
    return float(weights(rows)@np.asarray(values)) if rows else None


def join_ids(expected, actual):
    actual=[str(x) for x in actual]
    if len(set(actual))!=len(actual) or set(expected)!=set(actual):raise ValueError('saved window membership mismatch')
    index={v:i for i,v in enumerate(actual)}
    return np.array([index[x] for x in expected],dtype=int)


def state_masks(now, ratio):
    valid=np.isfinite(ratio)&(ratio>1)&np.isfinite(now)
    return {'first_q95':valid&(now<=1),'upgrade_q99':valid&(now>1)&(now<=ratio),
            'already_above_q99':valid&(now>ratio)}


def choose_line(scores,truth,rows,limit):
    neg=(~truth)&np.isfinite(scores)
    ix=np.flatnonzero(neg)
    if not len(ix):return dict(state='NOT_EVALUABLE',line=None,reason='no_legal_calibration_negative',negative_windows=0)
    s=scores[ix];w=weights([rows[i] for i in ix]);values=np.unique(s)
    # Strict score > line, with earliest line meeting the fixed weighted FPR.
    mass=np.array([w[s==v].sum() for v in values]);tail=np.maximum(0,1-np.cumsum(mass))
    line=float(values[np.flatnonzero(tail<=limit+1e-12)[0]])
    return dict(state='evaluable',line=line,target_fpr=limit,actual_fpr=float(w@(s>line)),negative_windows=len(ix),
                negative_events=len({(rows[i]['station_uid'],rows[i]['source_event_id']) for i in ix}),
                failed_negative_windows=int(((~truth)&(~np.isfinite(scores))).sum()))


def window_metrics(scores,truth,rows,line,reference=None,probabilistic=True):
    scores=np.asarray(scores);truth=np.asarray(truth,bool);finite=np.isfinite(scores);n=len(rows)
    answer=dict(state='evaluable' if n else 'NOT_EVALUABLE',windows=n,events=len({(r['station_uid'],r['source_event_id']) for r in rows}),
      stations=len({r['station_uid'] for r in rows}),failed_windows=int((~finite).sum()),positives=int(truth.sum()),negatives=int((~truth).sum()),
      brier=None,brier_reference=None,brier_skill=None,brier_common_windows=0,prevalence=mean(truth,rows),
      fpr=None,recall=None,precision=None,tp=None,fp=None,tn=None,fn=None,failed_positive_windows=int((truth&~finite).sum()),failed_negative_windows=int((~truth&~finite).sum()))
    fi=np.flatnonzero(finite);sub=[rows[i] for i in fi]
    if probabilistic and len(fi):answer['brier']=mean((scores[fi]-truth[fi])**2,sub)
    if probabilistic and reference is not None:
        common=np.flatnonzero(finite&np.isfinite(reference));rs=[rows[i] for i in common]
        answer['brier_common_windows']=len(common)
        if len(common):
            b=mean((scores[common]-truth[common])**2,rs);ref=mean((reference[common]-truth[common])**2,rs)
            answer.update(brier_common=b,brier_reference=ref,brier_skill=1-b/ref if ref>0 else None)
    if line is None:return answer
    alarm=finite&(scores>line)
    answer.update(tp=int((alarm&truth).sum()),fp=int((alarm&~truth).sum()),tn=int((~alarm&~truth&finite).sum()),fn=int((~alarm&truth&finite).sum()))
    # Failed forecasts remain in observed-positive/negative denominators, named explicitly.
    for mask,key in [(~truth,'fpr'),(truth,'recall')]:
        ix=np.flatnonzero(mask)
        if len(ix):answer[key]=mean(alarm[ix],[rows[i] for i in ix])
    if alarm.any():
        w=weights(rows);answer['precision']=float(w@(alarm&truth)/(w@alarm))
    return answer


def reliability(scores,truth,rows):
    result=[];scores=np.asarray(scores)
    for b in range(10):
        left=b/10;right=(b+1)/10
        ix=np.flatnonzero(np.isfinite(scores)&(scores>=left)&((scores<right) if b<9 else (scores<=right)))
        sub=[rows[i] for i in ix]
        result.append(dict(bin=b,left=left,right=right,windows=len(ix),events=len({(r['station_uid'],r['source_event_id']) for r in sub}),
          predicted=mean(scores[ix],sub),observed=mean(np.asarray(truth)[ix],sub),state='evaluable' if len(ix) else 'NOT_EVALUABLE'))
    return result


def first_crossing(author, trusted, threshold):
    """Only a continuous trusted record from its retained start can establish first onset.

    Author points may invalidate an onset but never restore a missing trusted point.
    Original earlier exceedance cannot be replaced by a later trusted crossing.
    """
    if not author:return None,'no_author_records'
    if any(v.get('conflict') for v in author.values()) or any(v.get('conflict') for v in trusted.values()):return None,'source_time_or_Q_conflict'
    times=sorted(author);high=[t for t in times if author[t]['q']>threshold]
    if not high:return None,'no_observed_exceedance'
    t=high[0];previous=t-timedelta(hours=1)
    if t not in trusted or previous not in trusted:return t,'first_crossing_outside_trusted_predecessor_coverage'
    if trusted[t]['q']<=threshold:return t,'author_trusted_threshold_side_conflict'
    if trusted[previous]['q']>threshold:return t,'predecessor_already_high'
    if not trusted:return t,'no_trusted_coverage'
    start=min(trusted)
    # Any author high at beginning is left censored, even when prepared begins later.
    if author[min(times)]['q']>threshold:return t,'author_start_already_high'
    # Unknown prefix cannot supply T/T-1; earlier recorded non-high prefix is
    # retained as a limitation, not a veto of the record-defined first onset.
    prior=[v for v in times if v<=t]
    if any(b-a!=timedelta(hours=1) for a,b in zip(prior,prior[1:])):return t,'author_gap_before_first_crossing'
    if trusted[start]['q']>threshold:return t,'trusted_start_already_high'
    return t,None


def event_score(rows, xq, y, author_by_event, scores, thresholds, target, h, line):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['station_uid'],r['source_event_id']].append(i)
    records=[]
    for (station,event),ix in sorted(groups.items()):
        th=thresholds[station]
        if th.get('state','evaluable')!='evaluable':
            records.append(dict(station=station,source_event_id=event,target=target,horizon=h,state='NOT_EVALUABLE',reason='invalid_fit_threshold_metadata',first_crossing=None,candidate_origins=0,failed_candidates=0,discovered=None,first_correct_issue=None,lead_hours=None,prefix_source_status='unknown',physical_event_completeness='unknown'))
            continue
        u=th['q95'] if target=='first_q95' else th['q99'];trusted={};issues=[]
        for i in ix:
            r=rows[i]
            for ts,source,q in zip(r['source_times'],r['source_rows'],np.r_[xq[i],y[i]]*r['qscale']):
                t=datetime.fromisoformat(ts);p=dict(q=float(q),source=str(source),conflict=False)
                if t in trusted and (trusted[t]['source']!=p['source'] or not np.isclose(trusted[t]['q'],p['q'],rtol=2e-6,atol=1e-5)):p['conflict']=True
                if t in trusted:p['conflict']|=trusted[t]['conflict']
                trusted[t]=p
        author=author_by_event.get((station,event),{})
        for t,p in trusted.items():
            if t not in author or author[t]['source']!=p['source'] or not np.isclose(author[t]['q'],p['q'],rtol=2e-6,atol=1e-5):issues.append('author_prepared_conflict')
        crossing,reason=first_crossing(author,trusted,u)
        if issues:reason=issues[0]
        candidate=[]
        if crossing is not None:
            for i in ix:
                t=datetime.fromisoformat(rows[i]['origin_time']);now=xq[i,-1]
                ratio=th['q99']/rows[i]['qscale']
                state=bool(state_masks(np.array([now]),np.array([ratio]))[target][0])
                if state and t<crossing<=t+timedelta(hours=h):candidate.append(i)
        if not reason and not candidate:reason='no_legal_current_state_origin_covering_first_crossing'
        if not reason and line is None:reason='no_calibration_alarm_line'
        hits=[i for i in candidate if np.isfinite(scores[i]) and scores[i]>line] if line is not None else []
        issued=min((datetime.fromisoformat(rows[i]['origin_time']) for i in hits),default=None)
        record=dict(station=station,source_event_id=event,target=target,horizon=h,state='NOT_EVALUABLE' if reason else 'evaluable',reason=reason,
          first_crossing=crossing.isoformat() if crossing else None,candidate_origins=len(candidate),failed_candidates=sum(not np.isfinite(scores[i]) for i in candidate),
          discovered=bool(hits) if not reason else None,first_correct_issue=issued.isoformat() if issued and not reason else None,
          lead_hours=(crossing-issued).total_seconds()/3600 if issued and not reason else None,prefix_source_status='unknown',physical_event_completeness='unknown')
        records.append(record)
    return records


def false_segments(rows, false_alarm):
    groups=defaultdict(list)
    for r,flag in zip(rows,false_alarm):
        if flag:groups[r['station_uid'],r['source_event_id']].append(datetime.fromisoformat(r['origin_time']))
    return sum(sum(j==0 or t-ts[j-1]!=timedelta(hours=1) for j,t in enumerate(ts)) for ts in (sorted(set(v)) for v in groups.values()))
