import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from .report import standard_score, convert_row, risk_target, summarize, input_identity, validate_source, validate_rows, SOURCE_VERSION, SOURCE_RUN


def row(**changes):
    raw=dict(arm='S2',seed='2',role='development',horizon='6',window_id='w',station='s',event='e',source_event_id='e',origin_time='t',legal_mainpeak_minus6='True',qscale='90',truth_m3_s='100',point_m3_s='110',probability='.75',q05_m3_s='80',q95_m3_s='130')
    raw.update(changes)
    return convert_row(raw,'saved.csv',2,'fixture_only')


class Boundaries(unittest.TestCase):
    def test_standard_boundary(self):
        self.assertTrue(standard_score(100,119.999,True)['qualified'])
        self.assertFalse(standard_score(100,120,True)['qualified'])
        self.assertTrue(standard_score(100,80,True)['boundary20'])
        self.assertFalse(standard_score(100,120.001,True)['qualified'])

    def test_truth_failure_and_eligibility(self):
        for q in (0,None,'nan',-1): self.assertEqual(standard_score(q,1,True)['standard_state'],'NOT_EVALUABLE')
        self.assertEqual(standard_score(100,'inf',True)['standard_state'],'MODEL_FAILED')
        self.assertEqual(standard_score(100,100,None)['standard_state'],'NOT_EVALUABLE')
        self.assertEqual(row(horizon='3')['standard_state'],'NOT_EVALUABLE')
        self.assertEqual(row(legal_mainpeak_minus6='False')['standard_state'],'NOT_EVALUABLE')

    def test_s0_and_peak_time(self):
        r=row(arm='S0',probability='0',q05_m3_s='0',q95_m3_s='0')
        self.assertIsNone(r['probability']); self.assertIsNone(r['coverage90'])
        self.assertEqual(r['probability_state'],'NOT_APPLICABLE')
        self.assertEqual(row()['exact_peak_time_state'],'NOT_APPLICABLE')
        self.assertEqual(row(arm='S3')['exact_peak_time_state'],'NOT_APPLICABLE')

    def test_current_high(self):
        self.assertEqual(risk_target('current_high'),'maintenance_or_reexceedance')
        self.assertEqual(risk_target('current_low'),'future_window_exceedance')
        self.assertEqual(row()['current_state'],'UNKNOWN')
        self.assertEqual(row()['escalation_state'],'PENDING_TARGET_DEFINITION')

    def test_weighting_and_failure_denominator(self):
        rows=[row(station='a',source_event_id='e1',probability='0') for _ in range(5)]
        rows += [row(station='a',source_event_id='e2',probability='1'),row(station='b',source_event_id='e1',probability='1',point_m3_s='nan')]
        rows += [row(station='c',source_event_id='e1',truth_m3_s='')]
        s=summarize(rows)
        self.assertEqual(s['metrics']['brier']['value'],.25)
        self.assertEqual(s['standard_denominator'],7)
        self.assertEqual(s['counts']['qualified'],6)
        self.assertEqual(s['metrics']['standard_failure_inclusive']['value'],.5)
        self.assertEqual(s['metrics']['standard_finite_only']['value'],1)
        self.assertEqual(s['counts']['MODEL_FAILED'],1)
        self.assertEqual(s['counts']['NOT_EVALUABLE'],1)

    def test_invalid_probability_and_interval(self):
        r=row(probability='nan',q05_m3_s='140',q95_m3_s='130')
        self.assertEqual(r['probability_state'],'MODEL_FAILED')
        self.assertIsNone(r['brier']); self.assertIsNone(r['width90_m3_s'])

    def test_input_identity(self):
        self.assertIn('ERA5_Land', input_identity('lamah_ice:67'))
        self.assertIn('filled_P', input_identity('camels_de:1'))
        self.assertIn('unknown', input_identity('da_tl_cn:Liqingdian'))
        self.assertIn('unknown', input_identity('unrecognized'))

    def test_source_version_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/SOURCE_RUN/'artifacts'
            source.parent.mkdir()
            record = source.parent/'run_record.json'
            record.write_text(json.dumps(dict(head=SOURCE_VERSION,state='complete',exit_code=0)))
            config = dict(code_version=SOURCE_VERSION,run_id=SOURCE_RUN,change_id='WP-20261011-MULTIWINDOW-PEAK-V1')
            with patch('src.prob_risk_reporting_v1.report.completed_source',return_value=config) as guard:
                self.assertEqual(validate_source(source),SOURCE_VERSION)
                guard.assert_called_once_with(source)
                config['run_id']='different_run'
                with self.assertRaises(ValueError): validate_source(source)
                config['run_id']=SOURCE_RUN
                record.write_text(json.dumps(dict(head='wrong',state='complete',exit_code=0)))
                with self.assertRaises(ValueError): validate_source(source)
            with patch('src.prob_risk_reporting_v1.report.completed_source',side_effect=ValueError('smoke')):
                with self.assertRaises(ValueError): validate_source(source)

    def test_summary_identity(self):
        rows=[dict(arm='S2',seed=2,role=role,horizon=h) for role in ('fit','select','calibration','development') for h in (1,3,6,12)]
        validate_rows(rows,'S2',2)
        with self.assertRaises(ValueError): validate_rows(rows+rows[:1],'S2',2)
        with self.assertRaises(ValueError): validate_rows(rows[:-1],'S2',2)
        with self.assertRaises(ValueError): validate_rows(rows,'S1',2)

if __name__=='__main__': unittest.main()
