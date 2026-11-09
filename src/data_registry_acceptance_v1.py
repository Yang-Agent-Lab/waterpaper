"""Independent saved-profile checks; run server-side, never on legacy test Q."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import zipfile
from datetime import datetime

import h5py
import numpy as np

ROOT = Path('.')
EVENT = Path('._event_profile_7a22804/results/experiments/event_profile_span_correction_7a22804_20261013/artifacts/event_profiles.csv')
ICE = Path('./external-home/multicountry_QC_fix_1fa15b9/results/experiments/multicountry_harmonization_full_1fa15b9_20261005/artifacts/registry.jsonl')


def near(a, b):
    assert np.isclose(float(a), float(b), rtol=1e-10, atol=1e-9), (a, b)


def check_series(profile, q, t):
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float)
    valid = np.isfinite(q) & (q >= 0)
    st = profile['statistics']
    assert st['n_rows'] == len(q)
    assert st['n_valid_q'] == int(valid.sum())
    near(st['valid_q_fraction'], valid.mean())
    for key, fraction in [('q05', .05), ('q50', .5), ('q95', .95), ('q99', .99)]:
        near(st[key], np.quantile(q[valid], fraction))
    differences = np.diff(q)[valid[:-1] & valid[1:] & (np.diff(t) == 1)]
    for key, sign in [('max_rise_1h', 1), ('max_recession_1h', -1)]:
        if len(differences):
            near(st[key], max(0., float(np.max(sign * differences))))
        else:
            assert st[key] is None
    return {k: st.get(k) for k in ['n_rows', 'n_valid_q', 'q50', 'q95', 'q99', 'max_rise_1h']}


def main(payload, jp_db, output):
    data = {}
    for kind in ['datasets', 'entities', 'profiles', 'rules', 'classifications']:
        data[kind] = [json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))) for line in (payload / (kind + '.jsonl')).read_text().splitlines()]
        assert all(r['project_slug'] == 'waterpaper' for r in data[kind])
    entities = {r['entity_key']: r for r in data['entities']}
    profiles = {r['profile_key']: r for r in data['profiles']}
    rules = {r['rule_key']: r for r in data['rules']}
    assert all(e['role_label'] for e in entities.values())
    assert all(p['entity_key'] in entities for p in profiles.values())
    assert all(c['entity_key'] in entities and c['rule_key'] in rules and (c['profile_key'] is None or profiles[c['profile_key']]['entity_key'] == c['entity_key']) for c in data['classifications'])
    assert all(c['label'] is None for c in data['classifications'] if c['rule_key'].endswith('rain_soil_process:v1'))
    report = {'counts': {k: len(v) for k, v in data.items()}, 'checks': {}, 'no_test_observation_access': True}
    us = sorted([p for p in profiles.values() if p['dataset_key'] == 'camelsh_us_v6' and p['scope'] == 'full_available_inventory'], key=lambda p: p['entity_key'])
    assert len(us) == 5767
    chosen = us[0]
    with zipfile.ZipFile(ROOT / 'data/Hourly2.zip') as archive:
        with h5py.File(io.BytesIO(archive.read(chosen['provenance']['member'])), 'r') as f:
            d = f['streamflow']; q = d[:].astype(float)
            for field in ['_FillValue', 'missing_value']:
                if field in d.attrs:
                    q[np.isin(q, np.asarray(d.attrs[field]).reshape(-1))] = np.nan
            q = q * float(d.attrs.get('scale_factor', 1)) + float(d.attrs.get('add_offset', 0))
            result = check_series(chosen, q, f['time'][:])
    report['checks']['usa_first_station_raw_nc'] = dict(entity=chosen['entity_key'], **result)

    jp = sorted([p for p in profiles.values() if p['dataset_key'] == 'japan_mlit_tone_existing' and entities[p['entity_key']]['entity_type'] == 'station' and p['n_valid_q'] > 0], key=lambda p: p['entity_key'])
    chosen = jp[0]; native = entities[chosen['entity_key']]['native_id']
    db = sqlite3.connect('file:' + str(jp_db) + '?mode=ro', uri=True)
    rows = db.execute('select t,q,conflict from obs where id=? order by t', (native,)).fetchall()
    result = check_series(chosen, [r[1] if r[1] is not None and not r[2] else np.nan for r in rows], [datetime.fromisoformat(r[0]).timestamp() / 3600 for r in rows])
    report['checks']['japan_first_valid_station_dedup'] = dict(entity=chosen['entity_key'], **result)
    catalogue = json.loads((ROOT / 'data/catalog/japan_mlit_station_list_tone_20261013.json').read_text())['stations']
    name = next(r['tds'][3] for r in catalogue if r['id'] == native)
    assert sum(r['id'].endswith(':0202') and len(r['tds']) >= 4 and r['tds'][3] == name for r in catalogue) == 1
    checked, opened, empty_checked = 0, 0, False
    for path in sorted((ROOT / 'data/raw/japan_mlit_tone_Q_2015_2024/chunks').glob('*.csv')):
        if path.stat().st_size == 10:
            if not empty_checked:
                assert path.read_text().strip() == 'time_jst,'
                empty_checked = True
            continue
        if checked >= 20 and empty_checked:
            break
        if opened >= 2000:
            break
        opened += 1
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if name not in (reader.fieldnames or []):
                continue
            for row in reader:
                try:
                    value = float(row[name])
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(value):
                    continue
                actual = db.execute('select q,conflict from obs where id=? and t=?', (native, row['time_jst'])).fetchone()
                assert actual is not None
                if not actual[1]:
                    near(actual[0], value)
                    checked += 1
                if checked >= 20:
                    break
    assert checked >= 20 and empty_checked, (checked, empty_checked)
    db.close()
    report['checks']['japan_raw_cells_and_empty_file'] = {'matched_nonconflicting_cells': checked, 'raw_files_opened': opened, 'header_only_verified': empty_checked}

    ep = [p for p in profiles.values() if p['profile_kind'] == 'event_morphology']
    assert len(ep) == 67
    for row in csv.DictReader(EVENT.open()):
        if row['station'] == 'da_tl_cn:Liqingdian':
            entity = 'da_tl_cn:event:' + row['station'] + ':' + row['source_event_id']
            p = next(p for p in ep if p['entity_key'] == entity)
            for k, v in row.items():
                try:
                    near(p['statistics'][k], float(v))
                except (ValueError, TypeError):
                    assert p['statistics'][k] == v
            assert p['scope'] in ('fit', 'development')
            report['checks']['china_reused_event'] = {'entity': entity, 'source_sha256': hashlib.sha256(EVENT.read_bytes()).hexdigest(), 'all_fields_match': True}
            break
    assert 'china_reused_event' in report['checks']
    original = next(json.loads(line) for line in ICE.read_text().splitlines() if json.loads(line).get('station_uid') == 'lamah_ice:98')
    p = next(p for p in profiles.values() if p['entity_key'] == 'lamah_ice:station:98' and p['scope'] == 'historical_QC')
    assert p['statistics']['eligible_pq_rows'] == original['coverage']['counts']['eligible_rows'] == 124719
    assert p['statistics']['longest_valid_pq_run_hours'] == original['coverage']['longest_eligible_hourly_run'] == 7798
    report['checks']['ice_repaired_source'] = {'eligible_pq_rows': 124719, 'longest_run': 7798, 'source': str(ICE)}
    report['status'] = 'passed_targeted_saved_profile_verification'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--jp-db', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    main(a.payload, a.jp_db, a.output)
