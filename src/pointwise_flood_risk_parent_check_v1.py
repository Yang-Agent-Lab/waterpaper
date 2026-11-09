"""Independent saved-interface check; does not import the implementation or any model."""
import argparse
import json
import math
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--receipt', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
r = json.loads(Path(a.receipt).read_text())
assert r['state'] == 'complete'
assert not r['actual_probability_forecasts_generated'] and not r['model_training']
assert not r['new_fit_scales'] and not r['calibration_evaluation_test_values_read']
examples = r['fit_examples'] + [r['actual_censored_fit_example'], r['actual_long_history_fit_example']]
assert len(r['fit_examples']) == 5
time_checks = label_checks = 0
for sample in examples:
    origin = datetime.fromisoformat(sample['origin_time'])
    assert sample['identity'][3] == 'fit'
    assert sample['threshold_source']['role'] == 'fit'
    assert sample['threshold_source']['official_flood_threshold'] is False
    assert sample['lead_hours'] == list(range(1, sample['forecast_hours']+1))
    assert len(sample['x']) == sample['history_hours']
    assert datetime.fromisoformat(sample['input_times'][-1]) == origin
    assert sample['current_state']['q_m3_s'] == sample['x'][-1][1]
    threshold = sample['threshold_source']['value']
    assert sample['current_state']['research_proxy_exceeded'] == (sample['x'][-1][1] > threshold)
    for lead, time, q, target, mask in zip(sample['lead_hours'], sample['target_times'], sample['y_q'],
                                         sample['y_exceedance'], sample['exceedance_label_mask']):
        assert datetime.fromisoformat(time) == origin + timedelta(hours=lead)
        assert mask and target == (q > threshold)
        time_checks += 1
        label_checks += 1
    assert len(sample['y_q']) == len(sample['y_q_normalized']) == sample['forecast_hours']
    assert 'window_max_targets' not in sample
    assert sample['prediction_state'] == 'input_and_supervision_only_no_model_forecast'
    if sample['route'] == 'q_only':
        assert all(x[0] is None for x in sample['x']) and all(not mask[0] for mask in sample['mask'])
censored = r['actual_censored_fit_example']
assert censored['continuous_pointwise_eligible']
assert censored['offline_metadata']['censored_process_ids']
assert censored['offline_metadata']['offline_event_evaluation_eligible'] is False
long = r['actual_long_history_fit_example']
assert long['history_hours'] == 24 and long['forecast_hours'] == 13
assert long['lead_hours'] == list(range(1,14))
prob = r['self_checks']['synthetic_probability_example']
expected = [1/3, 2/3, 1/3]
assert all(math.isclose(x,y,rel_tol=0,abs_tol=1e-15) for x,y in zip(prob['pointwise_probability'][0], expected))
assert prob['aux_window_ever_exceeded_probability'] == [1.0]
assert not prob['actual_model_forecast'] and prob['prediction_kind'] == 'synthetic_check'
assert len(prob['rows']) == 3
frozen = ['src/five_country_model_data_v1/reader.py', 'src/five_country_model_data_v1/prepare.py',
          'configs/five_country_model_data_roles_v1.json']
assert not subprocess.check_output(['git','diff','b448d6b','HEAD','--',*frozen],text=True)
result = dict(state='passed', producer_commit=r['commit'], real_fit_examples_checked=len(examples),
              pointwise_time_checks=time_checks, threshold_label_checks=label_checks,
              synthetic_pointwise_probability=expected, synthetic_window_ever_probability=1.0,
              actual_censored_pointwise_window_retained=True, actual_history24_forecast13=True,
              accepted_data_code_and_policy_unchanged=True, actual_model_forecasts=False,
              scope='Saved fit samples, target semantics and synthetic path math; no model skill claim')
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
