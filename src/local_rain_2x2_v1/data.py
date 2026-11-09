"""Role-gated future rainfall recovery; no final-role read before selection."""
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from multiwindow_peak_v1.data import load
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.training import atomic_json

ID = re.compile(r'"window_id"\s*:\s*"([^\"]+)"')


def subset(pool, ix):
    ix = np.asarray(ix, dtype=int)
    return {**{k: v for k, v in pool.items() if not isinstance(v, np.ndarray) and k != 'rows'},
            'rows': [pool['rows'][i] for i in ix],
            **{k: v[ix] for k, v in pool.items() if isinstance(v, np.ndarray)}}


def recover(pool, cfg, out, role, selection_gate=False):
    if role not in ('fit', 'select', 'calibration', 'development'):
        raise PermissionError('forbidden role')
    if role in ('calibration', 'development') and not selection_gate:
        raise PermissionError('final rainfall before all selections stopped')
    if any(r['role'] != role for r in pool['rows']):
        raise ValueError('pool role mismatch')
    prepared = Path(cfg['data'])
    manifest = json.loads((prepared/'manifest.json').read_text())
    scales = json.loads((prepared/'scales.json').read_text())
    by_station = defaultdict(list)
    for i, row in enumerate(pool['rows']): by_station[row['station_uid']].append(i)
    sources = defaultdict(list)
    old_role='validation' if role=='development' else 'train'
    for entry in manifest['inputs']:
        if entry['old_role']==old_role:sources[entry['station']].append(entry)
    future = np.full((len(pool['rows']), 6), np.nan, dtype=np.float32)
    receipts = []
    for station, ix in sorted(by_station.items()):
        wanted = {pool['rows'][i]['window_id']: i for i in ix}; seen = set()
        for entry in sources[station]:
            path = Path(entry['path'])
            if digest(path) != entry['sha256']: raise ValueError('upstream hash changed '+str(path))
            found = []
            with gzip.open(path, 'rt') as fp:
                for line in fp:
                    match = ID.search(line)
                    # Mixed old files contain other new roles. NEVER parse those values.
                    if not match or match.group(1) not in wanted: continue
                    record = json.loads(line); wid = record['window_id']; i = wanted[wid]; row = pool['rows'][i]
                    if wid in seen: raise ValueError('duplicate upstream identity '+wid)
                    seen.add(wid); found.append(wid)
                    if (record['station_uid'] != station or record['role'] != entry['old_role']
                        or record['source_rows'] != row['source_rows'] or record['source_times'] != row['source_times']):
                        raise ValueError('source identity mismatch '+wid)
                    comparisons = ((np.asarray(record['x_p'])*entry['original_pscale'], pool['x'][i,:,0]*scales[station]['pscale']),
                                   (np.asarray(record['x_q'])*entry['original_qscale'], pool['x'][i,:,1]*row['qscale']),
                                   (np.asarray(record['y_q'])*entry['original_qscale'], pool['y'][i]*row['qscale']))
                    if not all(np.allclose(a,b,rtol=3e-6,atol=1e-4) for a,b in comparisons):
                        raise ValueError('physical P/Q reconstruction mismatch '+wid)
                    p = np.asarray(record.get('offline_generation_future_p', []), dtype=float)
                    if p.shape != (12,) or not np.isfinite(p[:6]).all() or (p[:6]<0).any():
                        raise ValueError('missing/invalid future rainfall '+wid)
                    future[i] = p[:6]*entry['original_pscale']/scales[station]['pscale']
            receipts.append(dict(role=role, station=station, path=str(path), sha256=entry['sha256'],
                                 old_role=entry['old_role'], selected_ids=found,
                                 values_of_other_roles_parsed=False, filter_before_json_loads=True))
        if len(seen) != len(wanted): raise ValueError('missing paired members '+station)
    if not np.isfinite(future).all(): raise ValueError('unpaired future rainfall')
    atomic_json(out/(role+'_rain_access.json'), dict(state='complete',role=role,n=len(future),hours=6,
        physical_identity_checked=True,normalized_by='unchanged fit-only station pscale',
        information_status='oracle_rain_diagnostic',China_interval_timezone='unknown',sources=receipts))
    return future


def legal_from_saved(pool, path, role):
    saved = json.loads(Path(path).read_text())
    records = saved['records'][role]
    lookup = {r['window_id']: i for i,r in enumerate(pool['rows'])}
    mask = np.zeros(len(lookup), bool)
    for row in records:
        if row['state'] != 'evaluable': continue
        wid = row['window_id']
        if wid not in lookup: raise ValueError('eligible origin outside prepared role')
        actual = pool['rows'][lookup[wid]]
        if actual['station_uid'] != row['station'] or actual['source_event_id'] != row['source_event_id']:
            raise ValueError('eligible origin identity mismatch')
        mask[lookup[wid]] = True
    return mask


def prepare(cfg, runtime, out, roles, selection_gate=False):
    if set(roles)-{'fit','select'} and not selection_gate:
        raise PermissionError('all-selection gate required')
    pools, source = load(cfg, roles)
    expected = {'fit':28814,'select':2150,'calibration':1853,'development':1356}
    for role, pool in pools.items():
        if len(pool['rows']) != expected[role]: raise ValueError('role count differs '+role)
        target_counts={'fit':(162,137),'select':(74,57),'calibration':(56,64),'development':(65,88)}[role]
        for station,n in zip(runtime['targets'],target_counts):
            members=[r for r in pool['rows'] if r['station_uid']==station]
            if len(members)!=n or len({r['source_event_id'] for r in members})!=(3 if role=='fit' else 1):
                raise ValueError('target membership differs '+role+' '+station)
        pool['future'] = recover(pool,cfg,out,role,selection_gate)
        path = runtime['eligibility'] if role in ('fit','select') else runtime['final_eligibility']
        pool['legal'] = legal_from_saved(pool,path,role)
        atomic_json(out/(role+'_eligibility_source.json'),dict(path=path,sha256=digest(path)))
        atomic_json(out/(role+'_rows.json'),pool['rows'])
    atomic_json(out/('fit_select_sources.json' if roles==['fit','select'] else 'final_sources.json'),source)
    return pools
