"""No model training: synthetic failures prove state reporting and exact update identity."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_state_failure_runtime_and_three_status_rows(tmp_path,monkeypatch):
    from pointwise_benchmark_v1 import state_runner as runner
    from pointwise_benchmark_v1.runner import NumericalFailure
    from pointwise_benchmark_v1.status import training_status
    config=json.loads((Path(__file__).parents[1]/'configs/pointwise_state_smoke_v1.json').read_text())
    receipt=tmp_path/'receipt.json';receipt.write_text('{}')
    config['extreme_manifest']=str(receipt)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    out=tmp_path/'output'
    bank=SimpleNamespace(identity='synthetic',exposure=lambda *args:np.zeros((1,2,4),dtype=int))
    monkeypatch.setattr(runner,'SampleBank',lambda *args:bank)
    monkeypatch.setattr(runner,'verify_exposure',lambda *args:None)
    monkeypatch.setattr(runner,'sha256',lambda *args:'synthetic')
    runtime=dict(pid=0,host='synthetic-nonlocal-host',process_start_ticks='synthetic')
    monkeypatch.setattr(runner,'runtime_identity',lambda:runtime)
    def fail(*args,**kwargs):raise NumericalFailure('synthetic loss failure',2,1,'loss')
    monkeypatch.setattr(runner,'train_one',fail)
    monkeypatch.setattr('sys.argv',['state_runner','--config',str(path),'--output',str(out)])
    with pytest.raises(RuntimeError,match='partial_failed'):runner.main()
    assert json.loads((out/'runtime.json').read_text())==runtime
    for item in config['execution_plan']:
        failure=json.loads((out/item['id']/'failure.json').read_text())
        assert (failure['failed_attempted_update'],failure['failed_epoch'],failure['phase'])==(2,1,'loss')
    status=training_status(out,config)
    assert len(status)==3 and all(r['state']=='failed' and r['failed_attempted_update']==2 for r in status)
    assert all(r['model']!='persistence' for r in status)
