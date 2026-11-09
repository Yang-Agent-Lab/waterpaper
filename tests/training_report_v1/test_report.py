import copy
import json
import subprocess
import sys
import pytest


def bundle():
    return dict(schema_version=1,run_id="test",models=[dict(id="current_s2",label="当前")],epochs=[],cases=[],risk=[],health=[],interventions=[],metadata={})


def test_semantics_and_missing():
    from training_report_v1.schema import validate_bundle
    b=bundle()
    b["epochs"]=[dict(model="current_s2",role="select",country="China",epoch=1,step=2,train_loss=1,select_loss=None,state="missing",reason="not recorded")]
    assert validate_bundle(b)["epochs"][0]["select_loss"] is None
    for field,value in [("role","test"),("model","unknown")]:
        bad=copy.deepcopy(b);bad["epochs"][0][field]=value
        with pytest.raises(ValueError):validate_bundle(bad)
    b["cases"]=[dict(model="current_s2",role="development",country="China",case_id="x",x_kind="hourly_flow",x=[1,3,6,12],truth=[1]*4,median=[1]*4,lower=[0]*4,upper=[2]*4)]
    with pytest.raises(ValueError):validate_bundle(b)


def test_fixed_pages_and_missing_render(tmp_path):
    from training_report_v1.plot import render_report,COLORS
    b=bundle(); b["models"].append(dict(id="direct_tcn",label="卷积"))
    result=render_report(b,tmp_path/"report")
    assert [p.name for p in sorted((tmp_path/"report").glob("*.png"))]==["01_learning.png","02_predictions.png","03_risk.png"]
    from pypdf import PdfReader
    assert len(PdfReader(tmp_path/"report"/"training_report.pdf").pages)==3
    assert result["pages"][0]["missing"]
    assert COLORS["current_s2"]!=COLORS["direct_tcn"]
    assert json.loads((tmp_path/"report"/"report_manifest.json").read_text())["schema_version"]==1


def test_null_requires_reason_and_no_nan():
    from training_report_v1.schema import validate_bundle
    b=bundle();b['health']=[dict(model='current_s2',role='fit',country='China',module='P0',gradient_norm=None)]
    with pytest.raises(ValueError,match='null requires'):validate_bundle(b)
    b['health'][0].update(state='NOT_APPLICABLE',reason='frozen')
    validate_bundle(b)
    b['health'][0]['gradient_norm']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):validate_bundle(b)


def test_country_curves_are_not_stitched_and_invalid_case_lengths():
    from training_report_v1.schema import validate_bundle
    b=bundle();b['cases']=[dict(model='current_s2',role='development',country='China',case_id='x',x_kind='horizon_window_peak',x=[1,3,6,12],truth=[1]*4,median=[1]*3,lower=[0]*4,upper=[2]*4)]
    with pytest.raises(ValueError,match='length'):validate_bundle(b)


def test_interventions_do_not_mix_countries_roles_or_semantics():
    from training_report_v1.plot import intervention_panels
    b=bundle()
    for country in ('China','Germany','Iceland'):
        for role in ('fit','select'):
            for module in ('input_rain_zero','input_flow_zero','encoder_zero'):
                b['interventions'].append(dict(model='current_s2',country=country,role=role,module=module,kind=module,metric='crps',horizon=6,delta=1,n=12,checkpoint='selected',state='evaluable'))
    common,dedicated=intervention_panels(b)
    assert len(common)==2 and len(dedicated)==1
    b['interventions'].extend([dict(r,metric=metric) for r in list(b['interventions']) for metric in ('brier','peak_mae_normalized')])
    common,dedicated=intervention_panels(b)
    assert len(common)==2 and len(dedicated)==1
    assert all(r['country']=='China' and r['role']=='select' for r in common+dedicated)
    assert {r['module'] for r in common}=={'input_rain_zero','input_flow_zero'}
    assert dedicated[0]['module']=='encoder_zero'
    b['epochs']=[dict(model='current_s2',epoch=2,selected=True)]
    for r in b['interventions']:r['checkpoint']='epoch_2'
    common,dedicated=intervention_panels(b)
    assert len(common)==2 and len(dedicated)==1


def test_event_gaps_are_not_connected_as_continuous_flow():
    from datetime import datetime
    from training_report_v1.plot import time_segments
    x=[datetime.fromisoformat(s) for s in ['2020-01-01T00:00','2020-01-01T01:00','2020-03-01T00:00','2020-03-01T01:00']]
    assert time_segments(x)==[(0,2),(2,4)]


def test_learning_panels_use_true_components_and_field_roles():
    from training_report_v1.plot import learning_series
    b=bundle()
    b['epochs']=[dict(model='current_s2',role='select',country=country,epoch=1,step=2,train_loss=1.2,train_nll=.9,train_crps=.3,fit_loss=1.4,select_loss=2.1,selection_metric=.8,selected=True) for country in ('China','Germany')]
    fit=learning_series(b,'fit_loss')
    assert len(fit)==1 and fit[0]['role']=='fit_probe' and fit[0]['country']=='China'
    assert fit[0]['rows'][0]['value']==1.4
    select=learning_series(b,'select_loss')
    assert select[0]['role']=='select' and select[0]['rows'][0]['value']==2.1
    assert learning_series(b,'train_nll')[0]['rows'][0]['value']==.9
    assert learning_series(b,'train_crps')[0]['rows'][0]['value']==.3
    for row in b['epochs']:
        del row['train_nll'];del row['train_crps']
    assert learning_series(b,'train_nll')==[]
    assert learning_series(b,'train_crps')==[]
