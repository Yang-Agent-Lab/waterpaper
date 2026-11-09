"""Five permitted fit examples plus synthetic checks; no model, fitting or rescanning."""
import argparse
import copy
import hashlib
import json
import math
import socket
import subprocess
import sys
from datetime import datetime,timedelta
from pathlib import Path

import numpy as np
from src.five_country_model_data_v1.reader import Reader
from src.five_country_model_data_v1.prepare import hour
from .adapter import adapt,iter_samples,summarize_paths,context


def reject(call):
    try:call()
    except (ValueError,PermissionError):return
    raise AssertionError('invalid data contract accepted')


def self_test(reader):
    times=[datetime(2020,1,1)+timedelta(hours=i) for i in range(20)]
    records=[reader.record('camelsh_us_v6','00000001',t,2,i,role='fit',label='end',source_row=i) for i,t in enumerate(times)]
    proc={'process_id':'synthetic_censored','first_exceed_hour':hour(times[8]),'last_exceed_hour':hour(times[9]),'complete':False}
    ctx=dict(table={'roles':{'fit':{'valid_segments':[{'start_hour':hour(times[0]),'end_hour':hour(times[-1])}],
        'processes':[proc]}}},qscale=10.,pscale=2.,threshold_value=5.,threshold_eligible=True,
        threshold_source={'kind':'fit_research_quantile_proxy','quantile':'q95','value':5.,'unit':'m3/s','role':'fit',
                          'official_flood_threshold':False,'evidence':'synthetic_fixture'},data_version='synthetic_fixture')
    sample=adapt(records,times[8],ctx,route='pq',history_hours=5,forecast_hours=7)
    short=adapt(records,times[8],ctx,route='pq',history_hours=3,forecast_hours=7)
    assert len(sample['x'])==5 and len(short['x'])==3 and sample['x'][-1][1]==8
    assert sample['current_information_included'] and sample['lead_hours']==list(range(1,8))
    assert sample['y_q']==list(range(9,16)) and np.allclose(sample['y_q_normalized'],np.arange(9,16)/10)
    assert not sample['offline_metadata']['offline_event_evaluation_eligible'] and sample['continuous_pointwise_eligible']
    assert 'window_max_targets' not in sample and all(len(v)==2 for v in sample['x'])
    changed=copy.deepcopy(records)
    for r in changed[9:]:r['q']=500;r['p']=700
    future=adapt(changed,times[8],ctx,route='pq',history_hours=5,forecast_hours=7)
    assert sample['x']==future['x'] and sample['y_q']!=future['y_q']
    missing=copy.deepcopy(records);missing[8]['p_mask']=False
    reject(lambda:adapt(missing,times[8],ctx,route='pq',history_hours=5,forecast_hours=7))
    qonly=adapt(missing,times[8],ctx,route='q_only',history_hours=5,forecast_hours=7)
    assert all(math.isnan(x[0]) for x in qonly['x']) and all(not m[0] for m in qonly['mask'])
    assert qonly['y_q']==sample['y_q'] and qonly['current_state']==sample['current_state']
    boundary=copy.deepcopy(records)
    for r in boundary:r['role_bounds']=[None,times[12].isoformat()]
    reject(lambda:adapt(boundary,times[8],ctx,route='pq',history_hours=5,forecast_hours=7))
    reject(lambda:adapt(records[:8]+records[9:],times[8],ctx,route='pq',history_hours=5,forecast_hours=7))
    probability_sample=copy.deepcopy(sample)
    probability_sample.update(identity=['synthetic','synthetic_station','synthetic','fit'],origin_time=times[8].isoformat(),
        lead_hours=[1,2,3],target_times=[times[i].isoformat() for i in (9,10,11)],exceedance_label_mask=[True]*3)
    paths=np.array([[[6,0,0],[0,6,0],[0,6,6]]],float)
    probability=summarize_paths(paths,[probability_sample],model_version='synthetic_fixture',
        calibration_version='not_calibrated',prediction_kind='synthetic_check',include_window_aux=True)
    assert np.allclose(probability['pointwise_probability'],[[1/3,2/3,1/3]])
    assert probability['aux_window_ever_exceeded_probability']==[1.]
    assert probability['pointwise_probability'][0][2]<probability['pointwise_probability'][0][1]
    assert sample['current_state']['research_proxy_exceeded'] and sample['current_state']['semantics']=='future_still_or_again_exceeded_not_new_onset'
    reject(lambda:summarize_paths(np.zeros((1,0,3)),[probability_sample],model_version='s',calibration_version='none',prediction_kind='synthetic_check'))
    reject(lambda:list(iter_samples(reader,'camelsh_us_v6','05464500',role='evaluation_reserved',route='q_only',history_hours=5,forecast_hours=7,threshold='q95')))
    return dict(state='passed',checks=['history_3_and_5_not_fixed12','all_future_leads_1_through7','current_point_included',
        'future_observations_not_in_x','censored_valid_pointwise_window_retained','censored_whole_event_not_evaluable',
        'Q_only_P_mask','P_Q_missing_rejection','role_boundary','clock_gap','pointwise_can_fall',
        'pointwise_differs_from_window_ever','current_high_not_new_onset','reserved_role_forbidden'],
        synthetic_probability_example=probability,censored_example=dict(retained=True,offline_event_evaluation_eligible=False))


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


def run(args):
    reader=Reader(args.manifest,prepared_dir=args.prepared_dir)
    checks=self_test(reader)
    choices=[('da_tl_cn','Jinping','pq',None),('camels_de_1h','DE110000','pq',None),
        ('lamah_ice','1','pq',None),('waterbench_iowa_v2_7087806','1609','pq',None),
        ('jp_jma_maebashi_mlit_kurihashi_2015_2024','303031283301030','pq',None)]
    examples=[]
    for source,station,route,start in choices:
        samples=list(iter_samples(reader,source,station,role='fit',route=route,history_hours=5,
            forecast_hours=7,threshold='q95',start=start,record_limit=256,max_samples=1))
        if not samples:raise ValueError('no permitted fit pointwise example: '+source)
        sample=samples[0]
        assert len(sample['target_times'])==len(sample['y_q'])==7 and len(sample['x'])==5
        assert sample['current_information_included'] and sample['threshold_source']['role']=='fit'
        assert not sample['threshold_source']['official_flood_threshold']
        examples.append(sample)
        if source.startswith('jp_jma'):
            q_only=list(iter_samples(reader,source,station,role='fit',route='q_only',history_hours=5,
                forecast_hours=7,threshold='q95',start=sample['input_times'][0],record_limit=256,max_samples=1))[0]
            assert q_only['y_q']==sample['y_q'] and q_only['y_q_normalized']==sample['y_q_normalized']
            assert all(not m[0] for m in q_only['mask'])
            checks['actual_Kurihashi_both_routes_same_Q_threshold_scale']=True
    long_history=list(iter_samples(reader,'waterbench_iowa_v2_7087806','1609',role='fit',route='pq',
        history_hours=24,forecast_hours=13,threshold='q99',record_limit=256,max_samples=1))
    assert long_history and len(long_history[0]['x'])==24 and len(long_history[0]['y_q'])==13
    assert long_history[0]['lead_hours']==list(range(1,14)) and long_history[0]['threshold_source']['quantile']=='q99'
    # A real, already saved censored process supplies a valid pointwise example.
    source='jp_jma_maebashi_mlit_kurihashi_2015_2024';station='303031283301030'
    table=reader.prepared_station(source,station)
    real_censored=None
    for process in table['roles']['fit']['processes']:
        if process['complete']:continue
        candidate=None
        for segment in table['roles']['fit']['valid_segments']:
            low=max(segment['start_hour']+4,process['first_exceed_hour']-7)
            high=min(segment['end_hour']-7,process['last_exceed_hour']+4)
            if low<=high:candidate=low;break
        if candidate is None:continue
        origin=candidate
        from src.five_country_model_data_v1.prepare import iso
        samples=list(iter_samples(reader,source,station,role='fit',route='q_only',history_hours=5,forecast_hours=7,
            threshold='q95',start=iso(origin-4,'+09:00'),record_limit=32,max_samples=1))
        if samples and samples[0]['offline_metadata']['offline_event_evaluation_eligible'] is False:
            real_censored=samples[0];break
    if real_censored is None:raise ValueError('no actual permitted censored-process pointwise example found')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    receipt=dict(state='complete',change_id='WP-20261015-POINTWISE-RISK-CONTRACT-V1',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),host=socket.gethostname(),
        python=sys.executable,self_checks=checks,fit_examples=examples,actual_censored_fit_example=real_censored,
        actual_long_history_fit_example=long_history[0],
        read_paths=sorted(set(reader.access)),history_hours_checked=[3,5,24],forecast_hours_checked=[3,7,13],
        actual_probability_forecasts_generated=False,model_training=False,new_fit_scales=False,
        calibration_evaluation_test_values_read=False,
        limitation='Supervision/interface and synthetic path math only; actual model outputs and operational release times not supplied')
    with (out/'receipt.json').open('x') as f:json.dump(json_safe(receipt),f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(state='complete',fit_countries=5,fit_examples=len(examples),real_censored_retained=True,
        synthetic_pointwise_probability=checks['synthetic_probability_example']['pointwise_probability'],model_training=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/existing_data_integration_v2.json')
    p.add_argument('--prepared-dir',required=True);p.add_argument('--output',required=True);run(p.parse_args())
