"""Five-country inventory; no model fitting or test observation access.

Run on teacherbox only. Cache is append-only by fingerprint and method version.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import zipfile
from datetime import datetime, timezone

import numpy as np

VERSION = 'inventory-v1.0'
ROOT = Path('.')
ICE = Path('./external-home/multicountry_QC_fix_1fa15b9/results/experiments/multicountry_harmonization_full_1fa15b9_20261005/artifacts')
FIT = Path('._prob_flood_ab720ba/results/experiments/prob_flood_prepare_ab720ba_20261010/artifacts/data')
EVENT = Path('._event_profile_7a22804/results/experiments/event_profile_span_correction_7a22804_20261013/artifacts/event_profiles.csv')
ATTR = Path('./external-home/protected_risk_v1_inputs_20261010/attribute_rows.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, bytes):
        return x.decode('utf-8', 'replace')
    return x


def dumps(x):
    return json.dumps(clean(x), ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(x):
    return hashlib.sha256(dumps(x).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    p = Path(path)
    s = p.stat()
    return {'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def longest(mask, t):
    if not len(mask):
        return 0
    breaks = np.r_[True, np.diff(t) != 1]
    start = np.flatnonzero(mask & (breaks | ~np.r_[False, mask[:-1]]))
    end = np.flatnonzero(mask & (np.r_[breaks[1:], True] | ~np.r_[mask[1:], False]))
    return int(np.max(end - start + 1)) if len(start) else 0


def stats(q, t, iso):
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    finite = np.isfinite(q)
    valid = finite & (q >= 0)
    values = q[valid]
    out = {'n_rows': len(q), 'n_valid_q': int(valid.sum()),
           'missing_q_rows': int((~finite).sum()), 'negative_q_rows': int((finite & (q < 0)).sum()),
           'zero_q_rows': int((q == 0).sum()),
           'valid_q_fraction': float(valid.mean()) if len(q) else None,
           'longest_valid_run_hours': longest(valid, t),
           'longest_missing_run_hours': max(longest(~valid, t), int(np.max(np.diff(t[valid]) - 1)) if valid.sum() > 1 else 0),
           'nonhourly_intervals': int((np.diff(t) != 1).sum()),
           'valid_q_first': iso(t[np.flatnonzero(valid)[0]]) if len(values) else None,
           'valid_q_last': iso(t[np.flatnonzero(valid)[-1]]) if len(values) else None}
    if len(values):
        out.update(q_min=float(values.min()), q_mean=float(values.mean()), q_std=float(values.std()), q_max=float(values.max()))
        out.update(dict(zip(['q05', 'q25', 'q50', 'q75', 'q95', 'q99'], np.quantile(values, [.05, .25, .5, .75, .95, .99], method='linear'))))
    adjacent = valid[:-1] & valid[1:] & (np.diff(t) == 1)
    indices = np.flatnonzero(adjacent)
    for key, sign in [('max_rise_1h', 1), ('max_recession_1h', -1)]:
        if len(indices):
            delta = np.diff(q)[indices] * sign
            k = int(indices[int(np.argmax(delta))])
            if (q[k + 1] - q[k]) * sign > 0:
                out[key] = float((q[k + 1] - q[k]) * sign)
                out[key + '_context'] = {'before': float(q[k]), 'after': float(q[k + 1]), 'time_before': iso(t[k]), 'time_after': iso(t[k + 1])}
            else:
                out[key] = 0.0
        else:
            out[key] = None
    return clean(out)


class Builder:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.out / 'cache.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS fingerprints (path TEXT, size INTEGER, mtime INTEGER, sha TEXT, PRIMARY KEY(path,size,mtime))')
        self.db.execute('CREATE TABLE IF NOT EXISTS profile_clocks (key TEXT PRIMARY KEY, computed_at TEXT NOT NULL)')
        self.rows = {k: {} for k in ['datasets', 'entities', 'profiles', 'rules', 'classifications']}
        self.keys = dict(datasets='dataset_key', entities='entity_key', profiles='profile_key', rules='rule_key', classifications='assignment_key')
        self.counts = {'processed': 0, 'skipped': 0, 'failed': 0}
        self.failures = []
        self.observed = now()

    def add(self, kind, **row):
        row['project_slug'] = 'waterpaper'
        self.rows[kind][row[self.keys[kind]]] = clean(row)

    def fingerprint(self, p):
        s = p.stat()
        identity = (str(p), s.st_size, s.st_mtime_ns)
        row = self.db.execute('SELECT sha FROM fingerprints WHERE path=? AND size=? AND mtime=?', identity).fetchone()
        if row:
            return row[0]
        h = sha(p)
        self.db.execute('INSERT INTO fingerprints VALUES (?,?,?,?)', (*identity, h))
        self.db.commit()
        return h

    def cached(self, key, make):
        row = self.db.execute('SELECT payload FROM cache WHERE key=?', (key,)).fetchone()
        if row:
            self.counts['skipped'] += 1
            return json.loads(row[0])
        result = make()
        result['computed_at'] = now()
        self.db.execute('INSERT INTO cache VALUES (?,?)', (key, dumps(result)))
        self.db.commit()
        self.counts['processed'] += 1
        return result

    def dataset(self, key, country, name, url, path, variables, interval=60, zone='unknown', status='partial', **summary):
        fp = digest(file_identity(path)) if Path(path).is_file() else digest(summary)
        self.add('datasets', dataset_key=key, name=name, country_code=country, source_url=url,
                 source_version=summary.pop('source_version', 'existing_snapshot'), data_kind=summary.pop('data_kind', 'observed_Q'),
                 nominal_interval_minutes=interval, time_zone=zone, variables=variables, server_uri='teacherbox:' + str(path),
                 source_fingerprint=fp, fingerprint_method='path_size_mtime_or_inventory_index', inventory_status=status,
                 model_readiness=summary.pop('model_readiness', 'Q_only_needs_P' if country == 'US' else ('partial_mapping_Q_only_needs_P' if country == 'JP' else 'legacy_paired_QP')), planned_role=summary.pop('planned_role', 'candidate_external_profiled' if country in ['US', 'JP'] else 'existing_research_source'),
                 summary=summary, observed_at=self.observed)
        self.entity(key + ':dataset', key, 'dataset', key, name=name)

    def entity(self, key, dataset, typ, native, **kw):
        self.add('entities', entity_key=key, dataset_key=dataset, entity_type=typ, native_id=native,
                 station_key=kw.pop('station_key', None), name=kw.pop('name', native), role_label=kw.pop('role_label', None) or ('metadata_only' if typ == 'event' else 'inventory'),
                 time_start=kw.pop('time_start', None), time_end=kw.pop('time_end', None), attributes=kw.pop('attributes', {}),
                 status=kw.pop('status', 'metadata_only'), observed_at=self.observed)

    def profile(self, entity, dataset, scope, fp, result, provenance, status='profiled'):
        key = digest([dataset, entity, result.get('profile_kind', 'coverage_distribution'), scope, VERSION, fp])
        st = result.get('statistics', {})
        computed = result.get('computed_at')
        if computed is None:
            clock = self.db.execute('SELECT computed_at FROM profile_clocks WHERE key=?', (key,)).fetchone()
            computed = clock[0] if clock else self.observed
            if not clock:
                self.db.execute('INSERT INTO profile_clocks VALUES(?,?)', (key, computed))
                self.db.commit()
        provenance = dict(provenance, threshold_fitting_allowed=False, selection_use='inventory_only' if scope == 'full_available_inventory' else 'existing_scope_description_only')
        self.add('profiles', profile_key=key, entity_key=entity, dataset_key=dataset, profile_kind=result.get('profile_kind', 'coverage_distribution'),
                 scope=scope, method_version=VERSION, source_fingerprint=fp, fingerprint_method=provenance.get('fingerprint_method', 'sha256'),
                 n_rows=st.get('n_rows'), n_valid_q=st.get('n_valid_q'), n_valid_p=st.get('n_valid_p'),
                 time_start=result.get('time_start'), time_end=result.get('time_end'), statistics=st,
                 quality=result.get('quality', {}), provenance=provenance, status=status, computed_at=computed)
        return key

    def us(self, limit=None):
        import h5py
        p = ROOT / 'data/Hourly2.zip'
        ident = file_identity(p)
        with zipfile.ZipFile(p) as z:
            members = sorted((i for i in z.infolist() if i.filename.endswith('.nc')), key=lambda i: i.filename)
            self.dataset('camelsh_us_v6', 'US', 'CAMELSH Hourly2 v6', 'https://zenodo.org/records/16729675', p,
                         {'Q': 'm3/s', 'H': 'm', 'P': 'absent_in_observation_archive'}, zone='UTC+0', status='downloaded_profiled',
                         source_version='v6', member_count=len(members), uncompressed_bytes=sum(i.file_size for i in members))
            started = time.monotonic()
            for index, member in enumerate(members[:limit] if limit else members):
                native = Path(member.filename).name.removesuffix('_hourly.nc')
                entity = 'camelsh_us_v6:station:' + native
                fp = digest([ident, member.filename, member.CRC, member.file_size, member.compress_size])
                def read():
                    with h5py.File(io.BytesIO(z.read(member)), 'r') as f:
                        ds = f['streamflow']
                        attrs = dict(ds.attrs)
                        unit = clean(attrs.get('units'))
                        tunits = clean(f['time'].attrs.get('units'))
                        if unit != 'm3 s-1' or tunits != 'hours since 1980-01-01 00:00:00':
                            raise ValueError('unverified units/time schema: ' + str((unit, tunits)))
                        q = ds[:].astype(float)
                        for fill in [attrs.get('_FillValue'), attrs.get('missing_value')]:
                            if fill is not None:
                                q[np.isin(q, np.asarray(fill).reshape(-1))] = np.nan
                        q = q * float(attrs.get('scale_factor', 1)) + float(attrs.get('add_offset', 0))
                        t = f['time'][:].astype(float)
                        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
                            raise ValueError('nonfinite or nonincreasing time')
                        base = np.datetime64('1980-01-01T00:00:00')
                        iso = lambda h: str(base + np.timedelta64(int(h * 3600), 's')) + '+00:00'
                        st = stats(q, t, iso)
                        st['n_valid_p'] = 0
                        return {'statistics': st, 'time_start': iso(t[0]), 'time_end': iso(t[-1]),
                                'quality': {'status': 'basic_numeric_only', 'P': 'absent', 'q_attributes': clean({k: attrs[k] for k in ['units', '_FillValue', 'missing_value', 'scale_factor', 'add_offset'] if k in attrs}), 'time_units': tunits, 'independent_row_qc': 'not_present_in_known_schema'}}
                try:
                    result = self.cached(digest([entity, VERSION, fp]), read)
                    self.entity(entity, 'camelsh_us_v6', 'station', native, time_start=result['time_start'], time_end=result['time_end'], role_label='candidate_external_profiled', status='candidate_external_profiled' if result['statistics']['n_valid_q'] else 'no_valid_Q_in_downloaded_scope')
                    self.profile(entity, 'camelsh_us_v6', 'full_available_inventory', fp, result,
                                 {'source': str(p), 'member': member.filename, 'q_unit': 'm3/s', 'timezone': 'UTC+0', 'P': 'absent', 'quantile_method': 'numpy linear', 'coverage_denominator': 'declared_full_public_time_axis', 'fingerprint_method': 'archive_stat_member_path_crc_sizes_not_full_md5', 'not_for_model_scaling_or_selection': True})
                except Exception as e:
                    self.counts['failed'] += 1
                    self.failures.append({'entity': entity, 'error': str(e)})
                if index == 2 or (index + 1) % 100 == 0:
                    progress = {'stage': 'US', 'members_completed': index + 1, 'total': len(members), 'elapsed_s': time.monotonic() - started, 'cache': self.counts}
                    print(dumps(progress), flush=True)
                    (self.out / 'progress.json').write_text(dumps(progress))

    def japan(self):
        root = ROOT / 'data/raw/japan_mlit_tone_Q_2015_2024'
        paths = sorted((root / 'chunks').glob('*.csv'))
        index = [file_identity(p) for p in paths]
        fp = digest(index)
        station_path = ROOT / 'data/catalog/japan_mlit_station_list_tone_20261013.json'
        profile_fp = digest([fp, self.fingerprint(station_path)])
        mapping_sha = self.fingerprint(station_path)
        directory_cache_key = digest(['jp_directory', VERSION, fp, mapping_sha])
        legacy_db = self.out / ('jp_dedup_' + fp[:16] + '.sqlite')
        sidecar = self.out / ('jp_dedup_' + fp[:16] + '.mapping.json')
        if sidecar.exists():
            original_mapping = json.loads(sidecar.read_text())['catalogue_sha256']
        elif self.db.execute('SELECT 1 FROM cache WHERE key=?', (directory_cache_key,)).fetchone():
            # A completed legacy aggregate cache proves its mapping identity.
            original_mapping = mapping_sha
            sidecar.write_text(dumps({'catalogue_sha256': mapping_sha, 'evidence': 'completed_directory_cache_key'}))
        elif not legacy_db.exists():
            original_mapping = mapping_sha
            sidecar.write_text(dumps({'catalogue_sha256': mapping_sha, 'evidence': 'new_dedup_store'}))
        else:
            original_mapping = None  # Interrupted legacy state has no proven mapping.
        dedup_path = legacy_db if original_mapping == mapping_sha else self.out / ('jp_dedup_' + fp[:16] + '_' + mapping_sha[:16] + '.sqlite')
        catalogue = json.loads(station_path.read_text())['stations']
        names = {}
        for s in catalogue:
            if s['id'].endswith(':0202') and len(s['tds']) >= 4:
                names.setdefault(s['tds'][3], []).append(s)
        self.dataset('japan_mlit_tone_existing', 'JP', 'MLIT Tone existing hourly discharge chunks', 'https://www1.river.go.jp/', root,
                     {'Q': 'm3/s', 'P': 'not_downloaded'}, zone='JST UTC+09:00', directory_fingerprint=fp, files=len(paths), bytes=sum(i['bytes'] for i in index), requested_period='2015-2024_not_verified_complete')
        # Cache the directory result. Individual file parses are content-addressed too.
        def combine():
            db = sqlite3.connect(dedup_path)
            db.execute('CREATE TABLE IF NOT EXISTS obs (id TEXT,t TEXT,q REAL,conflict INTEGER DEFAULT 0,dups INTEGER DEFAULT 0,PRIMARY KEY(id,t))')
            db.execute('CREATE TABLE IF NOT EXISTS done (path TEXT PRIMARY KEY)')
            quality = {'header_only_files': 0, 'unresolved_columns': {}, 'malformed_rows': 0, 'nonhourly_times': 0, 'missing_values': 0, 'parsed_files': 0, 'unknown_flags': 0}
            for p in paths:
                psha = self.fingerprint(p)
                def parse():
                    with p.open(newline='', encoding='utf-8-sig') as f:
                        reader = csv.reader(f)
                        head = next(reader, [])
                        if not head or head[0] != 'time_jst':
                            raise ValueError('unexpected Japan header')
                        mapped = []
                        unresolved = []
                        for n in head[1:]:
                            possible = names.get(n.strip(), [])
                            mapped.append(possible[0]['id'] if len(possible) == 1 else None)
                            if n.strip() and len(possible) != 1:
                                unresolved.append(n)
                        records = []
                        malformed = nonhour = missing = flags = 0
                        for row in reader:
                            if len(row) != len(head):
                                malformed += 1
                                continue
                            try:
                                dt = datetime.fromisoformat(row[0])
                                if dt.utcoffset().total_seconds() != 32400 or dt.minute or dt.second:
                                    nonhour += 1
                                    continue
                            except (ValueError, AttributeError):
                                malformed += 1
                                continue
                            for native, v in zip(mapped, row[1:]):
                                if native is None:
                                    continue
                                value = number(v)
                                if value is None:
                                    missing += 1
                                    if v.strip() not in ['-', '']:
                                        flags += 1
                                records.append([native, row[0], value])
                        return {'records': records, 'header_only': not records and not any(head[1:]), 'unresolved': unresolved,
                                'malformed': malformed, 'nonhour': nonhour, 'missing': missing, 'flags': flags}
                try:
                    item = self.cached(digest(['jp_file', VERSION, str(p), psha, self.fingerprint(station_path)]), parse)
                    quality['header_only_files'] += int(item['header_only'])
                    for n in item['unresolved']:
                        quality['unresolved_columns'][n] = quality['unresolved_columns'].get(n, 0) + 1
                    for k, field in [('malformed_rows', 'malformed'), ('nonhourly_times', 'nonhour'), ('missing_values', 'missing'), ('unknown_flags', 'flags')]:
                        quality[k] += item[field]
                    if not db.execute('SELECT 1 FROM done WHERE path=?', (str(p),)).fetchone():
                        for native, t, q in item['records']:
                            db.execute('INSERT INTO obs(id,t,q) VALUES(?,?,?) ON CONFLICT(id,t) DO UPDATE SET dups=dups+1, conflict=CASE WHEN obs.q IS NOT NULL AND excluded.q IS NOT NULL AND obs.q!=excluded.q THEN 1 ELSE conflict END,q=COALESCE(obs.q,excluded.q)', (native, t, q))
                        db.execute('INSERT INTO done VALUES(?)', (str(p),))
                        db.commit()
                    quality['parsed_files'] += 1
                except Exception as e:
                    self.failures.append({'file': str(p), 'error': str(e)})
                    self.counts['failed'] += 1
            stations = []
            for (native,) in db.execute('SELECT DISTINCT id FROM obs ORDER BY id').fetchall():
                rows = db.execute('SELECT t,q,conflict,dups FROM obs WHERE id=? ORDER BY t', (native,)).fetchall()
                times = [r[0] for r in rows]
                t = np.array([datetime.fromisoformat(v).timestamp() / 3600 for v in times])
                q = np.array([r[1] if r[1] is not None and not r[2] else np.nan for r in rows])
                iso = lambda h: datetime.fromtimestamp(h * 3600, timezone.utc).isoformat()
                st = stats(q, t, iso)
                st.update(n_valid_p=0, duplicate_rows=sum(r[3] for r in rows), conflicting_timestamps=sum(r[2] for r in rows), downloaded_grid_missing_hours=int(t[-1] - t[0] + 1 - len(t)))
                stations.append({'native': native, 'statistics': st, 'time_start': times[0], 'time_end': times[-1], 'quality': {'status': 'basic_numeric_only', 'conflicts_excluded': True}})
            db.close()
            return {'stations': stations, 'quality': quality}
        result = self.cached(directory_cache_key, combine)
        for s in result['stations']:
            native = s['native']
            entity = 'japan_mlit_tone_existing:station:' + native
            name = next((r['tds'][3] for r in catalogue if r['id'] == native), native)
            self.entity(entity, 'japan_mlit_tone_existing', 'station', native, name=name, time_start=s['time_start'], time_end=s['time_end'], role_label='candidate_external_profiled', status='candidate_external_profiled' if s['statistics']['n_valid_q'] else 'no_valid_Q_in_downloaded_scope')
            s['computed_at'] = result['computed_at']
            self.profile(entity, 'japan_mlit_tone_existing', 'full_available_inventory', profile_fp, s,
                         {'source': str(root), 'q_unit': 'm3/s', 'timezone': 'JST UTC+09:00', 'output_valid_extrema_timezone': 'UTC+0', 'mapping': 'unique_Q_station_name_in_native_station_catalogue', 'fingerprint_method': 'sorted_path_size_mtime_index_and_native_catalogue_sha256', 'coverage_denominator': 'deduplicated_downloaded_rows_not_declared_2015_2024', 'quantile_method': 'numpy linear', 'not_for_model_scaling_or_selection': True})
        self.profile('japan_mlit_tone_existing:dataset', 'japan_mlit_tone_existing', 'full_available_inventory', profile_fp,
                     {'quality': result['quality'], 'statistics': {'files': len(paths), 'resolved_stations_with_rows': len(result['stations'])}, 'computed_at': result['computed_at']},
                     {'source': str(root), 'fingerprint_method': 'sorted_path_size_mtime_index_plus_cached_file_sha256', 'unresolved_columns_excluded': True}, status='partial')
        print(dumps({'stage': 'JP', 'stations': len(result['stations']), 'quality': result['quality']}), flush=True)

    def historical(self):
        index = ROOT / 'data/catalog/integrated_station_index_v2.csv'
        self.dataset('da_tl_cn', 'CN', 'DA-TL China nine basins', 'https://zenodo.org/records/15608461', index, {'Q': 'm3/s', 'P': 'mm/h'}, data_kind='observed_event_segments', historical_QC_only=True)
        self.dataset('camels_de_1h', 'DE', 'CAMELS-DE-1h', 'https://doi.org/10.5880/fidgeo.2026.045', index, {'Q': 'm3/s', 'P': 'mm/h'}, historical_QC_only=True)
        reg = ICE / 'registry.jsonl'
        self.dataset('lamah_ice', 'IS', 'LamaH-Ice', 'https://essd.copernicus.org/articles/16/2741/2024/', reg, {'Q': 'm3/s', 'P': 'mm/h reanalysis'}, zone='GMT UTC+0', source_version='QC_fix_1fa15b9', historical_QC_only=True)
        attrs = {r['station_uid']: r for r in json.loads(ATTR.read_text())} if ATTR.exists() else {}
        rows = list(csv.DictReader(index.open()))
        ice_rows = [json.loads(line) for line in reg.open() if '"station_uid"' in line]
        rows += [r for r in ice_rows if r['station_uid'].startswith('lamah_ice:')]
        for r in rows:
            uid = r['station_uid']
            ds = uid.split(':')[0]
            if ds not in ['da_tl_cn', 'camels_de_1h', 'lamah_ice']:
                continue
            entity = ds + ':station:' + uid.split(':', 1)[1]
            self.entity(entity, ds, 'station', uid.split(':', 1)[1], attributes={'original_metadata': r, 'background': attrs.get(uid)}, role_label='existing_research_source', status='historical_QC_imported')
            coverage = r.get('coverage', {})
            counts = r.get('counts', coverage.get('counts', {}))
            st = {'n_rows': number(r.get('rows', counts.get('rows'))), 'n_valid_q': None, 'n_valid_p': None,
                  'valid_pq_fraction': number(r.get('numeric_pq_fraction')), 'longest_valid_pq_run_hours': number(r.get('longest_numeric_pq_run_hours', coverage.get('longest_eligible_hourly_run'))),
                  'eligible_pq_rows': counts.get('eligible_rows'), 'numeric_pq_rows': counts.get('numeric_pq_rows')}
            for key in ['missing_q_rows', 'missing_p_rows', 'negative_q_rows', 'negative_p_rows', 'duplicate_timestamp_rows', 'conflicting_timestamp_rows', 'nonhourly_intervals']:
                st[key] = number(r.get(key, counts.get(key)))
            for key in ['n_rows', 'n_valid_q', 'n_valid_p']:
                if st[key] is not None:
                    st[key] = int(st[key])
            self.profile(entity, ds, 'historical_QC', digest(r), {'statistics': st, 'quality': {'original': coverage if ds == 'lamah_ice' else r, 'full_Q_distribution': 'not_recomputed'}},
                         {'source': str(reg if ds == 'lamah_ice' else index), 'original_method': 'QC_fix_1fa15b9' if ds == 'lamah_ice' else 'cleaning_v2', 'original_computed_at': '2026-10-05' if ds == 'lamah_ice' else '2026-09-30', 'observed_at': self.observed, 'no_test_observations_read': True})
        cat = ROOT / 'data/catalog/china_event_catalog_v1.csv'
        fp = self.fingerprint(cat)
        roles = {r['event_uid']: r for r in csv.DictReader((ICE / 'pooled_china_roles.csv').open())}
        for r in csv.DictReader(cat.open()):
            uid = r['station_uid']
            entity = 'da_tl_cn:event:' + r['event_uid']
            self.entity(entity, 'da_tl_cn', 'event', r['source_event_id'], station_key='da_tl_cn:station:' + uid.split(':', 1)[1], time_start=r['source_start'], time_end=r['source_end'], attributes=dict(r, historical_role=roles.get(r['event_uid'])), role_label=roles.get(r['event_uid'], {}).get('pooled_role'), status='metadata_only')
        for r in csv.DictReader(EVENT.open()):
            entity = 'da_tl_cn:event:' + r['station'] + ':' + r['source_event_id']
            if entity not in self.rows['entities']:
                continue
            numeric = {k: number(v) if number(v) is not None else v for k, v in r.items()}
            self.profile(entity, 'da_tl_cn', 'fit' if r['role'] == 'fit' else 'development', digest([self.fingerprint(EVENT), r]),
                         {'profile_kind': 'event_morphology', 'statistics': numeric, 'quality': {'complete_volume': 'NOT_EVALUABLE', 'complete_duration': 'NOT_EVALUABLE', 'direct_runoff_Ts': 'NOT_EVALUABLE'}, 'time_start': r['trusted_first'], 'time_end': r['trusted_last']},
                         {'source': str(EVENT), 'original_method': '17b26fe_features_7a22804_span_correction', 'original_computed_at': '2026-10-13', 'observed_at': self.observed, 'no_new_event_observations_read': True})

    def fit(self):
        rows = json.loads((FIT / 'fit_rows.json').read_text())
        if any(r.get('role') != 'fit' for r in rows):
            self.failures.append({'stage': 'fit', 'error': 'Non-fit role in approved fit metadata; arrays not opened'})
            self.counts['failed'] += 1
            return
        expected_manifest = 'e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6'
        if self.fingerprint(FIT / 'manifest.json') != expected_manifest:
            self.failures.append({'stage': 'fit', 'error': 'Approved prepared manifest identity differs; arrays not opened'})
            self.counts['failed'] += 1
            return
        fp = digest([self.fingerprint(FIT / 'fit_rows.json'), self.fingerprint(FIT / 'fit.npz'), self.fingerprint(FIT / 'scales.json')])
        def read():
            with np.load(FIT / 'fit.npz') as arrays:
                x, y = arrays['x'], arrays['y']
                print(dumps({'stage': 'fit_schema', 'x': x.shape, 'y': y.shape}), flush=True)
                if x.ndim != 3 or x.shape[1:] != (12, 2) or y.shape[1:] != (12,):
                    raise ValueError('Unexpected prepared fit array schema')
                scales = json.loads((FIT / 'scales.json').read_text())
                buckets = {}
                rain = {}
                for i, row in enumerate(rows):
                    uid = row['station_uid']
                    b = buckets.setdefault(uid, {})
                    q = np.r_[x[i, :, 1], y[i]] * row['qscale']
                    pb = rain.setdefault(uid, {})
                    for k, source_row in enumerate(row['source_rows']):
                        key = (source_row, row['source_times'][k])
                        value = float(q[k])
                        if key in b and not np.isclose(b[key], value, rtol=1e-5, atol=1e-5):
                            raise ValueError('fit source-row Q conflict: ' + uid)
                        b[key] = value
                        if k < 12:
                            p = float(x[i, k, 0] * scales[uid]['pscale'])
                            if key in pb and not np.isclose(pb[key], p, rtol=1e-5, atol=1e-5):
                                raise ValueError('fit history P conflict: ' + uid)
                            pb[key] = p
                out = {}
                for uid, b in buckets.items():
                    # Event snippets have unknown zone; use supplied ISO lexical sorting and integer source-hour offsets.
                    ordered = sorted(b.items(), key=lambda v: v[0][1])
                    t = np.array([datetime.fromisoformat(k[1]).replace(tzinfo=timezone.utc).timestamp() / 3600 if datetime.fromisoformat(k[1]).tzinfo is None else datetime.fromisoformat(k[1]).timestamp() / 3600 for k, v in ordered])
                    iso_map = {h: k[1] for h, (k, v) in zip(t, ordered)}
                    st = stats(np.array([v for k, v in ordered]), t, lambda h: iso_map[h])
                    st['max_unsampled_span_hours'] = st['longest_missing_run_hours']
                    st['longest_missing_run_hours'] = None  # Sampling absence is not an archive missing-data run.
                    pp = np.array(list(rain[uid].values()))
                    st.update(n_valid_p=int(np.sum(np.isfinite(pp) & (pp >= 0))), p_mean=float(np.mean(pp)), p95=float(np.quantile(pp, .95)), p_zero_rows=int((pp == 0).sum()), paired_history_pq_rows=len(rain[uid]))
                    out[uid] = {'statistics': st, 'quality': {'scope': 'sampled_fit_windows_not_full_archive', 'P_scope': 'history_only', 'Q_scope': 'history_and_label', 'timestamp_gap_note': 'Across-event gaps are absence from sampled union, not archive missingness'}, 'time_start': ordered[0][0][1], 'time_end': ordered[-1][0][1]}
                return out
        try:
            result = self.cached(digest(['fit_union', VERSION, fp]), read)
            for uid, r in result.items():
                if uid == 'computed_at':
                    continue
                ds, native = uid.split(':', 1)
                entity = ds + ':station:' + native
                r['computed_at'] = result['computed_at']
                if r['statistics'].get('longest_missing_run_hours') is not None:
                    r['statistics']['max_unsampled_span_hours'] = r['statistics']['longest_missing_run_hours']
                    r['statistics']['longest_missing_run_hours'] = None
                self.profile(entity, ds, 'fit', fp, r, {'source': str(FIT), 'original_method': 'prepared_ab720ba_source_row_time_dedup', 'q_unit': 'm3/s', 'scope_detail': 'sampled_fit_windows_not_full_archive', 'timezone': 'original source metadata; China unknown', 'not_new_model_scales': True})
        except Exception as e:
            self.failures.append({'stage': 'fit', 'error': str(e)})
            self.counts['failed'] += 1

    def rules(self):
        definitions = {
            'availability': ('data availability', ['native_interval', 'variables'], 'Only source-supported hourly_Q/hourly_PQ/daily_auxiliary/sample_only; missing input is not zero'),
            'background': ('basin background', ['area', 'annualP', 'PET/P', 'storage_evidence'], 'Area/annual precipitation descriptors; no climate-storage class without PET/P and storage evidence'),
            'event_morphology': ('event Q morphology', ['continuous_Q', 'event_role'], 'Import existing fit/development continuous-Q descriptors; no new fixed class cutoffs'),
            'rain_soil_process': ('rain and soil process', ['rain_duration', 'antecedent_soil_moisture', 'hourly_applicability'], 'Long/short/excess rain hypotheses need required fields; no daily thresholds imposed on hourly series'),
            'legacy_flow_severity': ('legacy relative flow severity', ['existing_fit_scale', 'existing_event_role'], 'Existing fit scale only, not return period or physical hazard grade')}
        for k, (name, fields, text) in definitions.items():
            self.add('rules', rule_key='waterpaper:' + k + ':v1', name=name, level='event' if k in ['event_morphology', 'rain_soil_process', 'legacy_flow_severity'] else ('dataset_or_station' if k == 'availability' else 'station'), method_version=VERSION,
                     status='descriptor_only' if k != 'availability' else 'active', definition={'description': text}, required_fields=fields,
                     source_refs=['https://doi.org/10.1029/2025EF006060', 'https://doi.org/10.1029/2026EF008397'], notes='SMOTE is augmentation, not missing-value imputation or flood cause. Country is provenance, not flood type.')
        for entity, r in self.rows['entities'].items():
            ds = r['dataset_key']
            if r['entity_type'] == 'dataset' and ds in ['hei_luoma_daily', 'geodata_poyang_samples', 'gxsw_recent_capture']:
                label = {'hei_luoma_daily': 'daily_auxiliary', 'geodata_poyang_samples': 'sample_only', 'gxsw_recent_capture': 'unknown'}[ds]
                self.add('classifications', assignment_key=digest([entity, 'availability', VERSION, self.rows['datasets'][ds]['source_fingerprint']]), entity_key=entity, profile_key=None, rule_key='waterpaper:availability:v1', label=label, status='source_supported_availability', evidence={'metadata_only': True, 'not_hourly_model_input': True}, assigned_at=self.observed)
                continue
            if r['entity_type'] == 'event':
                pp = [p for p in self.rows['profiles'].values() if p['entity_key'] == entity and p['profile_kind'] == 'event_morphology']
                for p in pp:
                    self.add('classifications', assignment_key=digest([entity, 'event_morphology', p['profile_key'], VERSION]), entity_key=entity, profile_key=p['profile_key'], rule_key='waterpaper:event_morphology:v1', label=None,
                             status='descriptor_only', evidence={'scope': p['scope'], 'existing_continuous_Q_features': True, 'missing_requirements': ['physical_event_completeness', 'complete_volume', 'complete_duration', 'direct_runoff_Ts']}, assigned_at=self.observed)
                for k in ['rain_soil_process', 'legacy_flow_severity'] + ([] if pp else ['event_morphology']):
                    self.add('classifications', assignment_key=digest([entity, k, VERSION, r['attributes']]), entity_key=entity, profile_key=None, rule_key='waterpaper:' + k + ':v1', label=None, status='NOT_EVALUABLE',
                             evidence={'scope': 'metadata_only' if not pp else 'existing_fit_development_event_descriptors', 'missing_requirements': definitions[k][1]}, assigned_at=self.observed)
                continue
            if r['entity_type'] != 'station':
                continue
            availability = 'hourly_Q' if ds in ['camelsh_us_v6', 'japan_mlit_tone_existing'] else 'hourly_PQ'
            pp = [p for p in self.rows['profiles'].values() if p['entity_key'] == entity and p['scope'] in ['historical_QC', 'full_available_inventory']]
            p = pp[0] if pp else None
            no_q = r['status'] == 'no_valid_Q_in_downloaded_scope'
            for k in ['availability', 'background']:
                self.add('classifications', assignment_key=digest([entity, k, VERSION, p['profile_key'] if p else None, r['attributes']]), entity_key=entity, profile_key=p['profile_key'] if p else None, rule_key='waterpaper:' + k + ':v1',
                         label=availability if k == 'availability' and not no_q else None,
                         status=('no_valid_Q_in_downloaded_scope' if no_q else 'source_supported_availability') if k == 'availability' else ('features_available' if r['attributes'].get('background') else 'NOT_EVALUABLE'),
                         evidence={'scope': 'inventory_or_historical_metadata', 'native_hourly_Q_field': True, 'valid_Q_observed': not no_q if ds in ['camelsh_us_v6', 'japan_mlit_tone_existing'] else 'historical_QC_only', 'missing_requirements': ['valid_Q_observations'] if no_q and k == 'availability' else ([] if k == 'availability' else (['PET/P', 'storage_evidence'] if r['attributes'].get('background') else [])), 'present_descriptors': ['area', 'annualP'] if k == 'background' and r['attributes'].get('background') else [], 'not_verified_for_classification': ['PET/P', 'storage_evidence'] if k == 'background' else [], 'unverified_descriptors': ['area', 'annualP'] if k == 'background' and not r['attributes'].get('background') else [], 'not_model_training_certification': True}, assigned_at=self.observed)

    def auxiliary(self):
        configs = [
            ('hei_luoma_daily', 'Hei/Luoma published daily flood events', 'hei_luoma_floods_zenodo_20725049', 'https://zenodo.org/records/20725049', 1440, 'auxiliary_daily_units_pending', 'hei_luoma_download_receipt.json', {'Q': 'units_from_receipt', 'P': 'Luoma_units_cumulative_semantics_pending'}, {'workbooks': 2, 'sheets_each': 5, 'events_each': 10, 'interval_evidence': 'daily_verified_by_parent', 'hourly_prediction_eligible': False}),
            ('geodata_poyang_samples', 'Poyang/Yangtze workbook samples', 'geodata_poyang', 'unknown', None, 'sample_only', None, {'variables': 'sample_metadata_only'}, {'workbook_samples': 3, 'rows_with_headers': [4, 4, 5], 'official_provenance': 'unknown', 'hourly_prediction_eligible': False}),
            ('gxsw_recent_capture', 'Guangxi recent dashboard snapshots', 'gxsw_capture', 'http://124.227.12.36:8259/', None, 'recent_snapshots_not_continuous_archive', 'gxsw_capture_system_20261008.json', {'P': 'snapshot', 'H': 'snapshot', 'Q': 'partial_stations_snapshot'}, {'capture_begin': '2026-10-08', 'continuous_all_station_archive': False, 'existing_capture_task': 'not_modified'})]
        for key, name, folder, url, interval, readiness, receipt, variables, description in configs:
            root = ROOT / 'data/raw' / folder
            index = [file_identity(p) for p in sorted(root.iterdir()) if p.is_file()] if root.exists() else []
            description.update(shallow_file_index=index, deeper_directories='not_traversed', current_directory_exists=root.exists(), metadata_only=True)
            if receipt and (ROOT / 'data/catalog' / receipt).exists():
                p = ROOT / 'data/catalog' / receipt
                description.update(receipt_path=str(p), receipt_sha256=self.fingerprint(p), receipt_limits='Use approved metadata description; commands or capture configuration are not exported')
            self.dataset(key, 'CN', name, url, root, variables, interval=interval, model_readiness=readiness, data_kind='auxiliary_metadata_only', status='metadata_only', planned_role='auxiliary_not_hourly_model_input', **description)

    def finish(self, figure=True):
        self.rules()
        for ds, dataset in self.rows['datasets'].items():
            pp = [r for r in self.rows['profiles'].values() if r['dataset_key'] == ds and r['scope'] == 'full_available_inventory' and self.rows['entities'][r['entity_key']]['entity_type'] == 'station']
            if pp:
                descriptors = {}
                for k in ['valid_q_fraction', 'q50', 'q95', 'max_rise_1h', 'longest_valid_run_hours']:
                    v = [r['statistics'][k] for r in pp if number(r['statistics'].get(k)) is not None]
                    if v:
                        descriptors[k] = dict(zip(['station_p05', 'station_p50', 'station_p95'], np.quantile(v, [.05, .5, .95])))
                dataset['summary'].update(profiled_stations=len(pp), station_descriptor_distributions=clean(descriptors), aggregation='equal_station_weighted_descriptors_not_pooled_hourly_Q_or_severity', threshold_fitting_allowed=False)
        for r in self.rows['profiles'].values():
            assert r['entity_key'] in self.rows['entities'], r['entity_key']
            for key in ['n_rows', 'n_valid_q', 'n_valid_p']:
                assert r[key] is None or isinstance(r[key], int), (key, r[key])
        for kind, rows in self.rows.items():
            with (self.out / (kind + '.jsonl')).open('w') as f:
                for key in sorted(rows):
                    f.write(dumps(rows[key]) + '\n')
        batches = self.out / 'batches'
        batches.mkdir(exist_ok=True)
        batch_manifest = []
        for kind, rows in self.rows.items():
            chunks, current, size = [], [], 2
            for key in sorted(rows):
                encoded = dumps(rows[key])
                n = len(encoded.encode('utf-8')) + 1
                if current and size + n > 300000:
                    chunks.append(current)
                    current, size = [], 2
                current.append(rows[key])
                size += n
            if current:
                chunks.append(current)
            for i, chunk in enumerate(chunks):
                path = batches / ('%s_%04d.json' % (kind, i + 1))
                path.write_text(dumps(chunk))
                batch_manifest.append({'kind': kind, 'file': str(path.relative_to(self.out)), 'rows': len(chunk), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        (batches / 'manifest.json').write_text(dumps(batch_manifest))
        manifest = {'method_version': VERSION, 'observed_at': self.observed, 'cache': self.counts, 'failures': self.failures,
                    'payload_counts': {k: len(v) for k, v in self.rows.items()}, 'files': {k + '.jsonl': sha(self.out / (k + '.jsonl')) for k in self.rows}, 'batches_manifest_sha256': sha(batches / 'manifest.json'), 'no_test_observation_access': True}
        (self.out / 'manifest.json').write_text(dumps(manifest))
        if figure:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            ds = ['da_tl_cn', 'camels_de_1h', 'lamah_ice', 'camelsh_us_v6', 'japan_mlit_tone_existing']
            counts = [sum(r['dataset_key'] == d and r['entity_type'] == 'station' for r in self.rows['entities'].values()) for d in ds]
            fig, (ax, coverage_ax) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.5, 1]})
            ax.bar(['China\nevent snippets', 'Germany\nhistorical QC', 'Iceland\nrepaired QC', 'USA\nfull Q inventory', 'Japan\nexisting chunks'], counts, color=['#2171b5'] * 3 + ['#e6550d'] * 2)
            ax.set_ylabel('Stations registered (different data scopes)')
            ax.set_yscale('log')
            for i, n in enumerate(counts):
                ax.text(i, n * 1.08, str(n), ha='center')
            ax.set_title('Registered stations by source')
            fractions = [[r['statistics']['valid_q_fraction'] for r in self.rows['profiles'].values() if r['dataset_key'] == d and r['scope'] == 'full_available_inventory' and r['statistics'].get('valid_q_fraction') is not None] for d in ds[-2:]]
            if all(fractions):
                coverage_ax.boxplot(fractions, labels=['USA\nfull declared axis', 'Japan\ndownloaded rows'], showfliers=False)
                coverage_ax.set_ylim(0, 1.05)
                coverage_ax.set_ylabel('Finite nonnegative Q fraction per station')
            coverage_ax.set_title('Coverage denominators differ')
            fig.text(.5, .01, 'Q fractions: basic numeric only, no measurement QC certification. Japan empty files excluded from station fractions. No forecast performance claims.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, 1))
            figures = self.out / 'figures'
            figures.mkdir(exist_ok=True)
            fig.savefig(figures / 'inventory_scope.png', dpi=150)
            plt.close(fig)
        print(dumps(manifest), flush=True)


def checks():
    iso = lambda x: str(x)
    r = stats([0, 4, np.nan, 100, -2, 3], [0, 1, 2, 4, 5, 6], iso)
    assert r['n_valid_q'] == 4 and r['negative_q_rows'] == 1 and r['zero_q_rows'] == 1
    assert r['max_rise_1h'] == 4 and r['longest_valid_run_hours'] == 2
    assert longest(np.array([True, True, True]), np.array([0, 1, 3])) == 2
    assert clean(float('nan')) is None
    print('focused numeric/time-gap checks passed', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--us-limit', type=int)
    ap.add_argument('--us-only', action='store_true')
    args = ap.parse_args()
    if args.check:
        checks()
        return
    b = Builder(args.out)
    b.us(args.us_limit)
    if not args.us_only:
        b.japan()
        b.historical()
        b.fit()
        b.auxiliary()
    b.finish(figure=not args.us_only)


if __name__ == '__main__':
    main()
