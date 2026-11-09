import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from predictor_benchmark_v1 import runner
from predictor_benchmark_v1.model import MODELS
from protected_flood_risk_v1.data import digest
class CompletionTests(unittest.TestCase):
    def fixture(self,root):
        out=root/'artifacts';out.mkdir();source=root/'source.json';source.write_text('{}')
        (out/'source_manifest.json').write_text(json.dumps(dict(source_manifest_sha256=digest(source),source={'inputs':{str(source):digest(source)}})))
        (out/'report_bundle.json').write_text('{}');pages=['01_learning.png','02_predictions.png','03_risk.png','04_modules.png']
        (out/'report_manifest.json').write_text(json.dumps(dict(pages=[dict(file=name) for name in pages])))
        (out/'figures').mkdir()
        for name in pages+['training_report.pdf','report_manifest.json']:(out/'figures'/name).write_bytes(b'report')
        for name in MODELS:
            folder=out/'seed_2'/name;folder.mkdir(parents=True);(folder/'best.pt').write_bytes(b'checkpoint')
            (folder/'selection_complete.json').write_text(json.dumps(dict(state='complete',best_file_sha256=digest(folder/'best.pt'))))
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
        (out/'completion.json').write_text(json.dumps(dict(state='complete',code_version='abc',models=list(MODELS),artifact_hashes=files)))
        return out,source
    def test_complete_execution_skips_all_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,source=self.fixture(Path(tmp))
            with patch.object(runner,'source_config',return_value={}),patch.object(runner.subprocess,'check_output',return_value='abc'),patch.object(runner,'load',side_effect=AssertionError('new data load')),patch.object(runner,'final_scores',side_effect=AssertionError('new inference')),patch.object(runner,'predict',side_effect=AssertionError('new predict')):
                result=runner.execute(str(out),True,None,True,str(source),None,None)
            self.assertTrue(result['reused'])
    def test_changed_hash_or_bad_completion_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,source=self.fixture(Path(tmp));(out/'report_bundle.json').write_text('{"changed":1}')
            with self.assertRaises(ValueError):runner.completed_result(out,str(source),'abc')
            (out/'completion.json').write_text('{"state":"partial"}')
            with self.assertRaises(ValueError):runner.completed_result(out,str(source),'abc')
