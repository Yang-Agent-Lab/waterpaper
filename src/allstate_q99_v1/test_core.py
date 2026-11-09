import unittest
from datetime import datetime,timedelta
import numpy as np
from .core import strategy_mask,routed_lines,metrics,score_events
from .report import saved_scores


def rows():
    return [dict(station_uid='A',source_event_id='E',window_id=str(i),origin_time=(datetime(2020,1,1)+timedelta(hours=i)).isoformat()) for i in range(3)]
def base(reason='no_legal_current_state_origin_covering_first_crossing'):
    return {('A','E'):dict(reason=reason,first_crossing='2020-01-01T03:00:00')}


class Bounds(unittest.TestCase):
    def test_p99_identity(self):
        with self.assertRaises(ValueError):saved_scores([dict(target='first_q95')],'S1')
        self.assertEqual(saved_scores([dict(target='upgrade_q99',probability='.25')],'S2')[0],.25)
    def test_masks_and_routing(self):
        low=np.array([True,False]);high=np.array([False,True]);self.assertTrue(strategy_mask('combined_first_q99',low,high).all())
        np.testing.assert_array_equal(routed_lines('combined_first_q99',low,.2,.7),[.2,.7])
        with self.assertRaises(ValueError):strategy_mask('combined_first_q99',[True],[True])
    def test_combined_count_and_earliest(self):
        r=rows();ev=score_events(r,np.array([.8,.9,.9]),np.array([True,False,False]),np.array([False,True,True]),'combined_first_q99',6,.5,.5,base())
        self.assertEqual(len(ev),1);self.assertTrue(ev[0]['discovered']);self.assertEqual(ev[0]['lead_hours'],3);self.assertEqual(ev[0]['first_issue_state'],'low_start')
    def test_low_crosses_both_same_hour(self):
        ev=score_events(rows(),np.array([0,0,.8]),np.array([True,True,True]),np.zeros(3,bool),'low_start_q99',6,.5,.5,base())[0]
        self.assertTrue(ev['discovered']);self.assertEqual(ev['lead_hours'],1)
    def test_observation_NE_preserved(self):
        ev=score_events(rows(),np.repeat(.9,3),np.ones(3,bool),np.zeros(3,bool),'combined_first_q99',6,.5,.5,base('author_start_already_high'))[0]
        self.assertEqual(ev['state'],'NOT_EVALUABLE');self.assertIsNone(ev['discovered'])
    def test_missing_branch_partial(self):
        ev=score_events(rows(),np.repeat(.9,3),np.array([True,False,False]),np.array([False,True,True]),'combined_first_q99',6,None,.5,base())[0]
        self.assertEqual(ev['state'],'partial');self.assertTrue(ev['known_detected']);self.assertIsNone(ev['discovered']);self.assertIsNone(ev['lead_hours'])
    def test_cross_state_false_segments(self):
        r=rows();m=metrics(np.repeat(.9,3),np.zeros(3,bool),r,np.array([.5,.7,.7]),np.repeat(.2,3),True)
        self.assertEqual(m['false_alarm_origin_count'],3);self.assertEqual(m['false_alarm_segments'],1)
    def test_strict_equal_and_S0(self):
        m=metrics(np.array([.5,1.,0.]),np.array([False,True,False]),rows(),np.array([.5,.5,.5]),np.repeat(.2,3),False)
        self.assertEqual(m['fp'],0);self.assertIsNone(m['brier'])
        s=saved_scores([dict(target='upgrade_q99',probability='',deterministic_decision='True',forecast_state='evaluable')],'S0');self.assertEqual(s[0],1)
        with self.assertRaises(ValueError):saved_scores([dict(target='upgrade_q99',probability='.5')],'S0')

if __name__=='__main__':unittest.main()
