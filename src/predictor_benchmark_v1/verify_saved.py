"""Read-only focused verification of saved engineering smoke outputs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from protected_flood_risk_v1.training import state_hash
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.runner import subset
from .data import source_config,load,smoke_subset
from .model import MODELS,crps
from .statistics import mean
from .runner import ROOT

def verify(folder):
    out=Path(folder);runtime=json.loads((out/'effective_config.json').read_text());assert runtime['smoke']
    shared=np.load(out/'slots.npy');assert shared.shape==(2,256);assert hashlib.sha256(shared.tobytes()).hexdigest()==runtime['slots_sha256']
    initial=json.loads((out/'initialization.json').read_text());assert len({r['head'] for r in initial.values()})==1;assert len({r['decoder'] for r in initial.values()})==1
    cfg=source_config(ROOT+'/source_manifest.json',ROOT+'/target_path_map_072b561.json',ROOT+'/Q_only');pools,_=load(cfg,['select']);pool=smoke_subset(pools['select']);rows=pool['rows'];ix=np.array([i for i,r in enumerate(rows) if r['country']=='China']);truth=np.column_stack([pool['y'][:,:h].max(1) for h in (1,3,6,12)]);records=[]
    for name in MODELS:
        folder=out/'seed_2'/name;selection=json.loads((folder/'selection_complete.json').read_text());epochs=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in epochs]==[1,2];assert [r['step'] for r in epochs]==[2,4];assert selection['updates']==4
        chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);assert chosen['model_id']==name;assert state_hash(chosen['model'])==selection['weight_hash'];assert digest(folder/'best.pt')==selection['best_file_sha256']
        with np.load(folder/'best_select.npz',allow_pickle=False) as z:
            draw=z['samples'];assert list(z['window_ids'].astype(str))==[r['window_id'] for r in rows]
        cc=crps(torch.from_numpy(draw),torch.from_numpy(truth)).numpy();value=mean(cc[ix,2],[rows[i] for i in ix]);assert abs(value-selection['best_score'])<1e-12
        assert selection['best_epoch']==min(epochs,key=lambda r:r['selection_score'])['epoch']
        if name=='current_s2':assert selection['protected_hash']==initial[name]['protected']
        records.append(dict(model=name,updates=selection['updates'],best_epoch=selection['best_epoch'],saved_select_crps6=value,encoder_class=chosen['encoder_class'],weight_hash=selection['weight_hash'],sample_sha256=digest(folder/'best_select.npz')))
    assert json.loads((out/'all_selections_stopped.json').read_text())['models']==list(MODELS)
    assert json.loads((out/'alarm_lines_complete.json').read_text())['calibration_only']
    assert json.loads((out/'calibration_source.json').read_text())['state']=='smoke_fit_standin'
    assert json.loads((out/'development_source.json').read_text())['state']=='smoke_select_standin'
    assert len(json.loads((out/'report_bundle.json').read_text())['models'])==3
    return dict(state='PASS',scientific_state='engineering_smoke_not_scientific',training_commit=runtime['code_version'],models=records,common_head_hash=initial['current_s2']['head'],common_decoder_hash=initial['current_s2']['decoder'],slots_sha256=runtime['slots_sha256'],no_duplicate_completed_group=True,formal_calibration_development_read=False,report_state='saved_bundle; rendering verified separately')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);a=p.parse_args();print(json.dumps(verify(a.artifacts),indent=2))
