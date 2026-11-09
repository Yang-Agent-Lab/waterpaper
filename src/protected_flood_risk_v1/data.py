"""Read existing prepared roles only; do not rebuild scales or access future P."""
import hashlib
import json
from pathlib import Path
import numpy as np

ALLOWED={'fit','select','calibration','development'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def validate_metadata(rows, role):
    if role not in ALLOWED: raise ValueError('role forbidden')
    seen=set()
    for row in rows:
        # Actual ab720ba prepared rows carry event/component identities here.
        required={'window_id','station_uid','country','role','event','origin_time','source_rows','source_times','qscale','source_event_id','temporal_overlap_component'}
        if not required.issubset(row) or row['role']!=role: raise ValueError('metadata/role mismatch')
        if len(row['source_rows'])!=24 or len(row['source_times'])!=24: raise ValueError('source identity requires24')
        if not np.isfinite(row['qscale']) or row['qscale']<=0: raise ValueError('invalid fit scale')
        if row['window_id'] in seen: raise ValueError('duplicate identity')
        seen.add(row['window_id'])
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def join_static(rows, attributes, normalization_fit_stations, current_fit_stations):
    if set(normalization_fit_stations)!=set(current_fit_stations): raise ValueError('static normalization fit membership mismatch')
    values=[]
    for row in rows:
        value=attributes[row['station_uid']]['model_features']
        if len(value)!=2 or not np.isfinite(value).all(): raise ValueError('invalid static features')
        values.append(value)
    return values


def real_data_adapter(folder, roles, attribute_rows, normalization, expected_manifest_sha256, *, execution_authorized=False):
    if not execution_authorized: raise PermissionError('Real data execution requires a separately approved run; CLI never grants it in engineering stage')
    if not roles or any(role not in ALLOWED for role in roles): raise ValueError('forbidden role')
    folder=Path(folder); manifest_path=folder/'manifest.json'
    if digest(manifest_path)!=expected_manifest_sha256: raise ValueError('prepared manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text()); pools={}; inputs={str(manifest_path):digest(manifest_path)}
    attrs=json.loads(Path(attribute_rows).read_text()); norm=json.loads(Path(normalization).read_text())
    attrs={a['station_uid']:a for a in attrs}
    fit_rows=json.loads((folder/'fit_rows.json').read_text())
    fit_stations={r['station_uid'] for r in fit_rows}
    if set(norm['fit_station_uids'])!=fit_stations: raise ValueError('static fit set mismatch')
    for path in (Path(attribute_rows),Path(normalization),folder/'fit_rows.json',folder/'scales.json',folder/'rise_cutoffs.json'):
        inputs[str(path)]=digest(path)
    for role in roles:
        path=folder/(role+'.npz'); rows_path=folder/(role+'_rows.json')
        if digest(path)!=manifest['derived_files'][path.name]: raise ValueError('prepared NPZ mismatch')
        rows=json.loads(rows_path.read_text());validate_metadata(rows,role)
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'x','y','features'}: raise ValueError('unexpected arrays including possible future P')
            arrays={k:z[k].copy() for k in z.files}
        n=len(rows)
        if arrays['x'].shape!=(n,12,2) or arrays['y'].shape!=(n,12) or arrays['features'].shape!=(n,8): raise ValueError('prepared shape mismatch')
        if not all(np.isfinite(a).all() for a in arrays.values()): raise ValueError('nonfinite prepared')
        idx=np.array([i for i,r in enumerate(rows) if role=='fit' or r['country']=='China'],dtype=int)
        chosen=[rows[i] for i in idx]
        pools[role]=dict(rows=chosen,static=np.asarray(join_static(chosen,attrs,norm['fit_station_uids'],fit_stations),dtype='float32'),**{k:a[idx] for k,a in arrays.items()})
        inputs[str(path)]=digest(path);inputs[str(rows_path)]=digest(rows_path)
    for pool in pools.values(): pool['input_manifest']=dict(inputs)
    return pools,dict(inputs=inputs,roles=list(roles),prepared_producer=manifest['producer_code_sha'],future_P_read=False,static_context='retrospective area + long-term rainfall; not historical-as-of')
