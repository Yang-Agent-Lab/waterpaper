import unittest
import numpy as np
from predictor_benchmark_v1.cases import choose_cases,make_cases
class CaseTests(unittest.TestCase):
    def test_peak_max_tie_earliest_and_ordinary_earliest(self):
        rows=[dict(station_uid='china:Liqingdian',origin_time=t,window_id=str(i),source_event_id='e',qscale=10) for i,t in enumerate(['2020-01-03','2020-01-02','2020-01-01'])]
        rows += [dict(station_uid='china:Luxi',origin_time=t,window_id=str(i+3),source_event_id='e',qscale=10) for i,t in enumerate(['2020-01-03','2020-01-01','2020-01-02'])]
        truth=np.array([[1,2,3,4],[1,2,3,4],[1,2,2,4],[.1,.2,.5,1],[.1,.2,1.1,2],[.1,.2,.8,1]])
        selected=choose_cases(rows,truth);self.assertEqual(selected,[('difficult',1),('ordinary',5)])
        a=make_cases(rows,truth,np.repeat(truth[:,None,:],4,axis=1),'current_s2','development')
        b=make_cases(rows,truth,np.repeat(truth[:,None,:],4,axis=1)*2,'direct_tcn','development')
        self.assertEqual([r['case_id'] for r in a],[r['case_id'] for r in b]);self.assertEqual(a[0]['origin_time'],'2020-01-02');self.assertEqual(a[0]['x'],[1,3,6,12])
        self.assertEqual(make_cases(rows,truth,np.repeat(truth[:,None,:],4,axis=1),'current_s2','select'),[])
