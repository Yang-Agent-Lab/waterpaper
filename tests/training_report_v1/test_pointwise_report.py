"""Synthetic engineering fixtures only; no scientific data or model loading."""
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
import pytest


def synthetic_bundle():
    case=dict(model='synthetic',role='select',country='Japan',station='synthetic_station',
              case_id='synthetic_case',case_kind='ordinary',origin_time='2020-01-01T00:00:00',
              x_kind='lead_hour_pointwise',unit='m3/s',x=[1,2,3],truth=[10,30,15],
              median=[12,25,18],lower=[5,15,10],upper=[20,40,30])
    risk=[dict(model='synthetic',role='select',country=country,horizon=h,
               target='pointwise_exceedance_'+target,brier=.1*h,crps=2*h,
               reliability_bins=[dict(mean_probability=.2,observed_fraction=.1)])
          for country in ('China','Germany','Iceland','USA','Japan')
          for target in ('Q95','Q99') for h in range(1,4)]
    risk[1].update(brier=None,state='NOT_EVALUABLE',reason='synthetic missing score')
    return dict(schema_version=1,target='pointwise_exceedance',run_id='synthetic_only',
                models=[dict(id='synthetic',label='合成样例')],
                epochs=[dict(model='synthetic',role='select',country='Japan',epoch=1,step=1,
                             train_loss=2,fit_loss=2.1,select_loss=2.2,selection_metric=.1)],
                cases=[case],risk=risk,health=[],interventions=[],
                metadata=dict(input_group='q_only',selection_country='Japan',forecast_hours=3,
                              training_countries=['China','Germany','Iceland','USA','Japan'],
                              loss_definition='synthetic shared CRPS',loss_components=[],
                              scientific_state='engineering_smoke_not_scientific',
                              crps_label='CRPS (m3/s)'))


def test_pointwise_validation():
    from training_report_v1.schema import validate_bundle
    b=synthetic_bundle();validate_bundle(b)
    b['cases'][0]['x_kind']='valid_time_pointwise'
    origin=datetime.fromisoformat(b['cases'][0]['origin_time'])
    b['cases'][0]['x']=[(origin+timedelta(hours=h)).isoformat() for h in (1,2,3)]
    validate_bundle(b)
    for section,key,value in [('cases','x',[1,3,6]),('cases','unit','mm'),
                              ('risk','target','window_exceedance_Q99'),('risk','brier',-1)]:
        bad=synthetic_bundle();bad[section][0][key]=value
        with pytest.raises(ValueError):validate_bundle(bad)
    bad=synthetic_bundle();bad['risk'].pop(1)
    with pytest.raises(ValueError,match='contiguous'):validate_bundle(bad)
    bad=synthetic_bundle();bad['risk']=[r for r in bad['risk'] if r['horizon'] < 3]
    with pytest.raises(ValueError,match='contiguous'):validate_bundle(bad)
    bad=synthetic_bundle();bad['metadata']['forecast_hours']=4
    with pytest.raises(ValueError,match='length'):validate_bundle(bad)
    bad=synthetic_bundle();bad['epochs'][0]['select_loss_definition']='different'
    with pytest.raises(ValueError,match='definitions'):validate_bundle(bad)


def test_non_china_learning():
    from training_report_v1.plot import learning_series
    result=learning_series(synthetic_bundle(),'select_loss')
    assert result[0]['country']=='Japan' and result[0]['rows'][0]['value']==2.2


def test_pointwise_three_pages(tmp_path):
    from training_report_v1.plot import render_report
    from pypdf import PdfReader
    out=tmp_path/'pointwise';b=synthetic_bundle()
    manifest=render_report(b,out)
    assert len(PdfReader(out/'training_report.pdf').pages)==3
    assert len(list(out.glob('*.png')))==3 and not (out/'04_modules.png').exists()
    assert 'China Brier' in manifest['pages'][2]['missing']
    saved=json.loads((out/'report_bundle.json').read_text())
    assert saved==b and saved['risk'][1]['brier'] is None
    # Save one stable sample for independent inspection when the test is run explicitly.


if __name__=='__main__':
    import sys
    from training_report_v1.plot import render_report
    render_report(synthetic_bundle(),Path(sys.argv[1]))
