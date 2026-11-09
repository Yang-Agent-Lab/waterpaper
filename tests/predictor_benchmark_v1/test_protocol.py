import unittest
import copy
import torch
from torch import nn
from predictor_benchmark_v1.model import build_model, CausalTCN
from predictor_benchmark_v1.diagnostics import intervene
from protected_flood_risk_v1.training import state_hash

class P0(nn.Module):
    def __init__(self):
        super().__init__(); self.layer=nn.Linear(2,64)
    def encode(self,x,f): return self.layer(x[:,-1])

class ProtocolTests(unittest.TestCase):
    def test_common_head_freeze_gradients_and_shapes(self):
        models=[build_model(k,P0(),2) for k in ('current_s2','direct_lstm','direct_tcn')]
        self.assertEqual(len({state_hash(m.decoder) for m in models}),1)
        self.assertEqual(len({state_hash(m.head) for m in models}),1)
        for m in models:
            x=torch.rand(4,12,2); f=torch.rand(4,8); y=torch.rand(4,12)
            loss,_=m.loss(x,f,y,torch.Generator().manual_seed(2)); loss.backward()
            self.assertTrue(torch.isfinite(loss)); self.assertTrue(any(p.grad is not None for p in m.parameters() if p.requires_grad))
            samples=m.sample(m.encode(x,f),16,torch.Generator().manual_seed(2))
            self.assertEqual(samples.shape,(4,16,4)); self.assertTrue((samples.diff(dim=-1)>=0).all())
            if m.protected is not None:self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.protected.parameters()))
    def test_causal_tcn(self):
        m=CausalTCN(); x=torch.rand(3,12,2); z=x.clone(); z[:,8:]+=10
        self.assertTrue(torch.equal(m.sequence(x)[:,:8],m.sequence(z)[:,:8]))
    def test_intervention_restores_state_and_rng(self):
        m=build_model('current_s2',P0(),2); m.train(); before=state_hash(m); rng=torch.get_rng_state().clone()
        x=torch.rand(4,12,2);f=torch.rand(4,8); rng=torch.get_rng_state().clone()
        with intervene(m,'input_rain_zero') as fn:
            h=fn(x,f); self.assertEqual(h.shape,(4,64)); torch.rand(5)
        self.assertEqual(before,state_hash(m));self.assertTrue(m.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_forbidden_roles(self):
        from predictor_benchmark_v1.data import authorize_role
        for r in ('test','calibration','development'):
            with self.assertRaises(PermissionError):authorize_role(r,False)

if __name__=='__main__': unittest.main()

class ForeignBlockTests(unittest.TestCase):
    def test_unequal_block_windows_equal_block_weight(self):
        from predictor_benchmark_v1.statistics import mean,weights
        rows=[dict(country='Germany',station_uid='s',source_event_id='',event='a')]+[dict(country='Germany',station_uid='s',source_event_id='',event='b')]*3
        self.assertAlmostEqual(mean([0,1,1,1],rows),.5)
        self.assertAlmostEqual(weights(rows).sum(),1.)
    def test_probe_country_cap(self):
        from predictor_benchmark_v1.data import fixed_probe
        from collections import Counter
        rows=[dict(country=c,station_uid=str(s),source_event_id='',event=str(e)) for c in ('China','Germany','Iceland') for s in range(300) for e in range(4) for w in range(8)]
        ix=fixed_probe(dict(rows=rows));counts=Counter(rows[i]['country'] for i in ix)
        self.assertEqual(dict(counts),dict(China=256,Germany=256,Iceland=256));self.assertEqual(ix,fixed_probe(dict(rows=rows)))
