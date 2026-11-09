"""Server-only real prepared-role loader and subset invariants."""
import unittest
from predictor_benchmark_v1.data import load,source_config,fixed_probe
from protected_flood_risk_v1.runner import subset
ROOT='._inputs/data_org_v1'
class RealDataTests(unittest.TestCase):
    def test_country_subset_and_probe_keep_manifest(self):
        cfg=source_config(ROOT+'/source_manifest.json',ROOT+'/target_path_map_072b561.json',ROOT+'/Q_only')
        pools,source=load(cfg,['fit','select']);self.assertEqual(len(pools['select']['rows']),10574)
        self.assertEqual({r['country'] for r in pools['select']['rows']},{'China','Germany','Iceland'})
        probe=subset(pools['fit'],fixed_probe(pools['fit']))
        self.assertTrue(probe['input_manifest']);self.assertIn(cfg['data']+'/manifest.json',probe['input_manifest'])
        china,_=load(cfg,['select'],countries=['China']);self.assertEqual(len(china['select']['rows']),2150)
        self.assertTrue(china['select']['input_manifest'])
if __name__=='__main__':unittest.main()
