"""Run on server: author model forward/backward and semantic boundary checks."""
import unittest
from unittest.mock import patch,MagicMock
import json,hashlib
import numpy as np
import torch
from neuralforecast.losses.pytorch import MQLoss
from predictor_benchmark_v1.statistics import weights
from native_baselines_v1.model import ALL_MODELS,MODELS,QUANTILES,build_model,coherent,quantile_samples,targets
from predictor_benchmark_v1.data import authorize_role

class NativeTests(unittest.TestCase):
    def test_saved_overall_crps_display(self):
        from native_baselines_v1.runner import saved_point_crps
        rows=[dict(data_policy='native_lstm',horizon='6',crps='.25',scope='',station='',stratum=''),dict(data_policy='native_lstm',horizon='6',crps='9',scope='station',station='a'),dict(data_policy='native_lstm',horizon='6',crps='8',scope='overall',stratum='ordinary'),dict(data_policy='native_tft',horizon='6',crps='',scope='overall')]
        result=saved_point_crps(rows);self.assertEqual(result[('native_lstm',6)],.25);self.assertIsNone(result.get(('native_tft',6)))
        with self.assertRaises(ValueError):saved_point_crps(rows+[dict(data_policy='native_lstm',horizon='6',crps='.5',scope='overall')])
    def test_formal_reuses_verified_legacy_exposures(self):
        from native_baselines_v1.runner import exposure_slots
        rows=[dict(window_id='fit-one')];shared=np.zeros((100,28928),dtype=np.int64);sha=hashlib.sha256(shared.tobytes()).hexdigest()
        fake=MagicMock();fake.__truediv__.side_effect=lambda name:MagicMock(read_text=lambda:json.dumps(rows if name=='fit_rows.json' else dict(slots_sha256=sha)))
        with patch('native_baselines_v1.runner.LEGACY',fake),patch('native_baselines_v1.runner.np.load',return_value=shared):
            actual,digest=exposure_slots(dict(rows=rows),False);self.assertIs(actual,shared);self.assertEqual(digest,sha)
            with self.assertRaises(ValueError):exposure_slots(dict(rows=[dict(window_id='changed')]),False)
    def test_weighted_native_loss_matches_window_then_event_station_mean(self):
        rows=[dict(station_uid=s,source_event_id=e) for s,e in [('a','one'),('a','one'),('a','two'),('b','three')]]
        loss=MQLoss(quantiles=QUANTILES);raw=torch.randn(4,4,17);y=torch.rand(4,4,1);w=torch.from_numpy(weights(rows)).float()
        scalar=torch.tensor([float(loss(y=y[i:i+1],y_hat=loss.domain_map(raw[i:i+1]))) for i in range(4)])
        batch=loss(y=y,y_hat=loss.domain_map(raw),mask=w[:,None,None].expand(-1,4,1))
        self.assertAlmostEqual(float(batch),float(w@scalar),places=6)
    def test_targets_and_coherence(self):
        y=torch.tensor([[1.,3.,2.,9.,0.,4.,2.,1.,1.,1.,12.,0.]])
        self.assertEqual(targets(y).tolist(),[[1.,3.,9.,12.]])
        raw=torch.tensor([[[3.,1.,2.],[0.,-1.,2.],[5.,4.,3.],[2.,1.,0.]]])
        fixed=coherent(raw)
        self.assertTrue((fixed.diff(dim=-1)>=0).all());self.assertTrue((fixed.diff(dim=1)>=0).all())
        q=torch.arange(17.).reshape(1,1,17).expand(2,4,17)
        s=quantile_samples(q,512)
        self.assertEqual(s.shape,(2,512,4));self.assertEqual(float(s[0,0,0]),0.);self.assertEqual(float(s[0,-1,0]),16.)
        for role in ('test','calibration','development'):
            with self.assertRaises(PermissionError):authorize_role(role,False)
    def test_five_author_models_real_backward(self):
        torch.set_num_threads(1)
        for name in MODELS:
            m=build_model(name);x=torch.rand(3,12,2,requires_grad=True);y=torch.rand(3,12)
            out=m.raw(x);self.assertEqual(out.shape,(3,4,17));self.assertEqual(type(m.native.loss).__name__,'MQLoss');self.assertTrue(type(m.native).__module__.startswith('neuralforecast.models.'))
            before={k:p.detach().clone() for k,p in m.named_parameters() if p.requires_grad};opt=torch.optim.Adam(m.parameters(),lr=.001)
            loss,_=m.loss(x,None,y);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(x.grad[:,:,0].abs().sum()),0.,name+' must use historical P')
            opt.step();self.assertTrue(any(not torch.equal(before[k],p) for k,p in m.named_parameters() if p.requires_grad))
            m.eval();a=m.raw(x.detach());torch.manual_seed(7);b=m.raw(x.detach());self.assertTrue(torch.equal(a,b))
            with self.assertRaises(ValueError):m.raw(x[:,:11])
    def test_persistence(self):
        m=build_model('persistence');x=torch.rand(2,12,2);q=m.raw(x)
        self.assertTrue(torch.equal(q[:,0,0],x[:,-1,1]));self.assertEqual(q.shape,(2,4,17));self.assertEqual(sum(p.numel() for p in m.parameters()),0)

if __name__=='__main__':unittest.main()
