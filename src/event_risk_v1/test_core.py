import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from .core import join_ids,state_masks,choose_line,window_metrics,first_crossing,event_score,false_segments


def row(hour=0):
    t=datetime(2020,1,1)+timedelta(hours=hour)
    return dict(window_id=str(hour),station_uid='A',source_event_id='E',origin_time=t.isoformat(),qscale=10,
      source_times=[(t+timedelta(hours=i-11)).isoformat() for i in range(24)],source_rows=[str(hour+i-11) for i in range(24)])

def points(values):
    t=datetime(2020,1,1)
    return {t+timedelta(hours=i):dict(q=v,source=str(i),conflict=False) for i,v in enumerate(values)}


class Boundaries(unittest.TestCase):
    def test_id_reordering(self):
        np.testing.assert_array_equal(join_ids(['a','b'],['b','a']),[1,0])
        with self.assertRaises(ValueError):join_ids(['a'],['a','a'])
    def test_state_equalities(self):
        r=state_masks(np.array([1,1.1,2,2.01]),np.repeat(2.,4))
        self.assertEqual(r['first_q95'].tolist(),[True,False,False,False]);self.assertEqual(r['upgrade_q99'].tolist(),[False,True,True,False])
    def test_strict_alarm_and_no_negative(self):
        line=choose_line(np.array([.2,.3]),np.array([False,False]),[row(),row(1)],.05)
        self.assertEqual(line['line'],.3);self.assertEqual(line['actual_fpr'],0)
        self.assertIsNone(choose_line(np.array([.5]),np.array([True]),[row()],.05)['line'])
    def test_failure_denominator_reference(self):
        m=window_metrics(np.array([.9,np.nan]),np.array([True,True]),[row(),row(1)],.5,np.array([np.nan,np.nan]))
        self.assertEqual(m['recall'],.5);self.assertEqual(m['failed_windows'],1);self.assertIsNone(m['brier_skill']);self.assertEqual(m['brier_common_windows'],0)
    def test_first_crossing_limits(self):
        p=points([0,1,2]);t,reason=first_crossing(p,p,1);self.assertEqual(t,datetime(2020,1,1,2));self.assertIsNone(reason)
        trusted=dict(p);trusted.pop(datetime(2020,1,1,1));self.assertIsNotNone(first_crossing(p,trusted,1)[1])
        p=points([2,0,2]);self.assertIsNotNone(first_crossing(p,p,1)[1])
        p=points([0,0,2]);p[datetime(2020,1,1)]['conflict']=True;self.assertIsNotNone(first_crossing(p,p,1)[1])
    def test_gap_and_unknown_prefix(self):
        p=points([0,0,0,2]);trusted=dict(p);trusted.pop(min(p));self.assertIsNone(first_crossing(p,trusted,1)[1])
        p.pop(datetime(2020,1,1,1));self.assertIsNotNone(first_crossing(p,trusted,1)[1])
    def test_actual_lead_and_NE(self):
        r=row(11);x=np.zeros((1,12));y=np.array([[0,0,0,0,0,2,0,0,0,0,0,0.]])
        p={datetime.fromisoformat(t):dict(q=float(q*10),source=s,conflict=False) for t,s,q in zip(r['source_times'],r['source_rows'],np.r_[x[0],y[0]])}
        th={'A':{'q95':10,'q99':20}}
        hit=event_score([r],x,y,{('A','E'):p},np.array([.8]),th,'first_q95',6,.5)[0]
        self.assertEqual(hit['lead_hours'],6);self.assertTrue(hit['discovered'])
        miss=event_score([r],x,y,{('A','E'):p},np.array([.2]),th,'first_q95',6,.5)[0];self.assertFalse(miss['discovered']);self.assertEqual(miss['state'],'evaluable')
        ne=event_score([r],x,y,{('A','E'):p},np.array([.8]),th,'first_q95',1,.5)[0];self.assertEqual(ne['state'],'NOT_EVALUABLE')
        r2=row(16);x2=np.zeros((1,12));y2=np.array([[2,0,0,0,0,0,0,0,0,0,0,0.]])
        pp={datetime.fromisoformat(t):dict(q=float(q*10),source=s,conflict=False) for t,s,q in zip(r2['source_times'],r2['source_rows'],np.r_[x2[0],y2[0]])}
        hit=event_score([r2],x2,y2,{('A','E'):pp},np.array([.8]),th,'first_q95',6,.5)[0];self.assertEqual(hit['lead_hours'],1)
    def test_false_segments(self):self.assertEqual(false_segments([row(0),row(1),row(3)],[True]*3),2)
    def test_threshold_side_conflict(self):
        p=points([0,0,1.0000001]);trusted={t:dict(v) for t,v in p.items()}
        trusted[max(trusted)]['q']=1.
        self.assertEqual(first_crossing(p,trusted,1)[1],'author_trusted_threshold_side_conflict')
    def test_invalid_threshold_ne(self):
        r=row();result=event_score([r],np.zeros((1,12)),np.zeros((1,12)),{},np.array([.8]),{'A':dict(state='NOT_EVALUABLE',q95=10,q99=10)},'upgrade_q99',6,.5)[0]
        self.assertEqual(result['reason'],'invalid_fit_threshold_metadata');self.assertIsNone(result['discovered'])
    def test_prepared_scale_identity(self):
        from .report import pool,Inputs
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);r=dict(row(),country='China',role='fit')
            (p/'fit_rows.json').write_text(json.dumps([r]));np.savez(p/'fit.npz',x=np.zeros((1,12,2)),y=np.zeros((1,12)),features=np.zeros((1,8)))
            self.assertEqual(len(pool('fit',Inputs(),p,{'A':dict(qscale=10)})['rows']),1)
            with self.assertRaisesRegex(ValueError,'qscale mismatch'):pool('fit',Inputs(),p,{'A':dict(qscale=11)})

if __name__=='__main__':unittest.main()
