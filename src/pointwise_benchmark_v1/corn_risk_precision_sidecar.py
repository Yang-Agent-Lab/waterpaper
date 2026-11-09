"""Save canonical select targets beside (never over) a completed CORN-risk run."""
import argparse
import json
from pathlib import Path

import numpy as np

from .corn_risk_check import check_risk_readouts
from .dataset import SampleBank, sha256


FORMAL_HEAD='ed15f290f69f3ea7aa427fe6e06325d77454d0d7'
BANK='./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq'
BANK_ID='2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c'
MODELS=('corn_risk_pq','binary_risk_pq')


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def build_sidecar(formal_run,output):
    formal_run=Path(formal_run).resolve();output=Path(output).resolve()
    if output==formal_run or formal_run in output.parents:
        raise ValueError('precision sidecar output must be separate from the immutable formal run')
    run=json.loads((formal_run/'run_record.json').read_text())
    if (run.get('head'),run.get('state'),run.get('exit_code'))!=(FORMAL_HEAD,'complete',0):
        raise ValueError('expected completed immutable formal run at the approved source commit')
    review_hash_before=sha256(formal_run/'artifacts'/'review.json')
    completion_hash_before=sha256(formal_run/'artifacts'/'completion.json')
    serialization_check=check_risk_readouts('cpu')
    if serialization_check.get('state')!='passed':raise RuntimeError('risk precision serialization self-check failed')
    bank=SampleBank(BANK,'pq',24,24)
    if bank.identity!=BANK_ID:raise ValueError('canonical fit/select bank identity changed')
    rows=bank.rows['select'];sample_ids=np.asarray([r['sample_id'] for r in rows],dtype='U64')
    y=np.asarray(bank.arrays['select']['y']);thresholds=np.asarray(bank.arrays['select']['thresholds_normalized'])
    observed=np.stack((y>thresholds[:,0,None],y>thresholds[:,1,None]),axis=-1).astype(np.uint8)
    if y.dtype!=np.dtype('float64') or thresholds.dtype!=np.dtype('float64'):
        raise ValueError(f'expected canonical float64 arrays, got y={y.dtype}, thresholds={thresholds.dtype}')
    output.mkdir(parents=True,exist_ok=True)
    target_path=output/'canonical_select_targets.npz'
    np.savez_compressed(target_path,sample_ids=sample_ids,truth_normalized=y,
        thresholds_normalized=thresholds,observed=observed)

    group_notes=[];original_hashes={}
    country_rows=np.asarray([r['country'] for r in rows])
    de_metrics=[]
    for model in MODELS:
        folder=formal_run/'artifacts'/model
        with np.load(folder/'select_predictions.npz',allow_pickle=False) as saved:
            stored_ids=np.asarray(saved['sample_ids']).astype('U64')
            stored_observed=np.asarray(saved['observed']).astype(np.uint8)
            old_y=np.asarray(saved['truth_normalized'])
            old_thresholds=np.asarray(saved['thresholds_normalized'])
        group_rows=json.loads((folder/'select_rows.json').read_text())
        if not np.array_equal(stored_ids,sample_ids) or group_rows!=rows:
            raise ValueError(f'{model} prediction/row ordering differs from canonical bank')
        if not np.array_equal(stored_observed,observed):
            raise ValueError(f'{model} stored observed labels differ from canonical labels')
        rounded_observed=np.stack((old_y>old_thresholds[:,0,None],old_y>old_thresholds[:,1,None]),axis=-1).astype(np.uint8)
        mismatch=rounded_observed!=observed
        by_country={}
        for country in sorted(set(country_rows.tolist())):
            mask=country_rows==country
            by_country[country]=dict(mismatched_label_cells=int(mismatch[mask].sum()),
                affected_origins=int(np.any(mismatch[mask],axis=(1,2)).sum()))
        original_hashes[model]=dict(predictions=sha256(folder/'select_predictions.npz'),
            scores=sha256(folder/'scores.json'))
        recorded=json.loads((folder/'scores.json').read_text())
        de=[r for r in recorded if r['country']=='DE']
        if len(de)!=48:raise ValueError(f'{model} expected 48 canonical Germany secondary score rows')
        by_lead={}
        for lead in range(1,25):
            for threshold in ('Q95','Q99'):
                match=[r for r in de if r['horizon']==lead and r['target'].endswith(threshold)]
                if len(match)!=1:raise ValueError(f'{model} missing/duplicate DE {threshold} lead {lead}')
                by_lead.setdefault(str(lead),{})[threshold]=float(match[0]['brier'])
        q95=[by_lead[str(lead)]['Q95'] for lead in range(1,25)]
        q99=[by_lead[str(lead)]['Q99'] for lead in range(1,25)]
        de_metrics.append(dict(model=model,role='secondary_select_descriptive_only',country='DE',
            origins=len(set(r['sample_id'] for r in rows if r['country']=='DE')),
            q95_brier=float(np.mean(q95)),q99_brier=float(np.mean(q99)),
            mean_q95_q99_brier=float(np.mean(q95+q99)),hourly_brier=by_lead,
            canonical_source='existing scores.json; generated from full-precision in-memory bank targets/thresholds'))
        group_notes.append(dict(model=model,prediction_sha256=sha256(folder/'select_predictions.npz'),
            prediction_truth_dtype=str(old_y.dtype),prediction_threshold_dtype=str(old_thresholds.dtype),
            saved_observed_matches_canonical=True,rounded_recomputed_label_mismatch_cells=int(mismatch.sum()),
            rounded_recomputed_label_mismatch_by_country=by_country))
    note=dict(state='complete',purpose='canonical precision sidecar for completed direct-risk outputs; no training or prediction changes',
        formal_run=str(formal_run),formal_source_commit=FORMAL_HEAD,formal_run_record_sha256=sha256(formal_run/'run_record.json'),
        bank_root=BANK,bank_identity=bank.identity,bank_manifest_sha256=sha256(Path(BANK)/'manifest.json'),
        canonical_targets_sha256=sha256(target_path),canonical_truth_dtype=str(y.dtype),canonical_threshold_dtype=str(thresholds.dtype),
        canonical_observed_dtype=str(observed.dtype),select_rows=len(rows),
        historical_review_label_source='review.json recomputed labels from float32 truth_normalized and thresholds_normalized in each prediction NPZ',
        official_model_score_source='scores.json from direct_probability_scores using canonical in-memory fit/select bank arrays',
        future_serialization_check=dict(state=serialization_check['state'],checks=serialization_check['checks']),
        groups=group_notes,original_hashes=original_hashes,
        original_review_sha256=review_hash_before,original_completion_sha256=completion_hash_before)
    de_review_path=output/'corrected_secondary_de_review.json'
    save(de_review_path,dict(state='complete',scope='DE secondary select scores only; not used for model selection',
        source_formal_commit=FORMAL_HEAD,bank_identity=bank.identity,groups=de_metrics,
        interpretation='Use canonical scores.json values; this file does not reconstruct labels from rounded NPZ targets.'))
    note['corrected_secondary_de_review']=dict(path=str(de_review_path),sha256=sha256(de_review_path))
    for model,hashes in original_hashes.items():
        folder=formal_run/'artifacts'/model
        if hashes!={'predictions':sha256(folder/'select_predictions.npz'),
                'scores':sha256(folder/'scores.json')}:
            raise RuntimeError(f'{model} formal predictions/scores changed while creating sidecar')
    if review_hash_before!=sha256(formal_run/'artifacts'/'review.json') or completion_hash_before!=sha256(formal_run/'artifacts'/'completion.json'):
        raise RuntimeError('formal review/completion changed while creating sidecar')
    note['original_npz_scores_review_completion_unchanged']=True
    save(output/'precision_note.json',note)
    print(json.dumps(note,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--formal-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();build_sidecar(args.formal_run,args.output)


if __name__=='__main__':main()
