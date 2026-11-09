"""Incremental JP-only day-end timestamp repair; all old artifacts stay read-only."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

import numpy as np

from . import build

METHOD = 'inventory-jp-clock-v1.1'
LEGACY = Path('._registry_final_payload_1bcb5b5')
OLD_ARTIFACTS = Path('._data_registry_a0bb3ec/results/experiments/data_registry_v1/artifacts')
DIRECTORY_FP = '7ead8d1de6dddfa3084afc9e686597b8efbea757bc87c796d89d8815d5b053ba'
OLD_DB = OLD_ARTIFACTS / ('jp_dedup_' + DIRECTORY_FP[:16] + '.sqlite')
CATALOG = build.ROOT / 'data/catalog/japan_mlit_station_list_tone_20261013.json'
SOURCE = build.ROOT / 'data/raw/japan_mlit_tone_Q_2015_2024/chunks'
DATASET = 'japan_mlit_tone_existing'
PATTERN = re.compile(r'^(\d{4}-\d\d-\d\d)T24:00:00\+09:00$')
UPSERT = ('INSERT INTO obs(id,t,q) VALUES(?,?,?) ON CONFLICT(id,t) DO UPDATE SET dups=dups+1, '
          'conflict=CASE WHEN obs.q IS NOT NULL AND excluded.q IS NOT NULL AND obs.q!=excluded.q THEN 1 ELSE conflict END,'
          'q=COALESCE(obs.q,excluded.q)')


def normalize(t):
    m = PATTERN.fullmatch(t)
    if not m:
        return None
    return (datetime.fromisoformat(m[1] + 'T00:00:00+09:00') + timedelta(days=1)).isoformat()


def load(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


class JPFixBuilder(build.Builder):
    def rules(self):
        super().rules()
        # Rule meanings are unchanged. Reuse original identities and definition versions.
        self.rows['rules'] = {r['rule_key']: r for r in load(LEGACY / 'rules.jsonl')}


def fix(out):
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE.glob('*.csv'))
    index = [build.file_identity(p) for p in paths]
    assert build.digest(index) == DIRECTORY_FP, 'Raw directory changed; no repair attempted'
    catsha = build.sha(CATALOG)
    mapping = json.loads((OLD_ARTIFACTS / ('jp_dedup_' + DIRECTORY_FP[:16] + '.mapping.json')).read_text())
    assert catsha == mapping['catalogue_sha256'], 'Native mapping changed; no repair attempted'
    names = {}
    for r in json.loads(CATALOG.read_text())['stations']:
        if r['id'].endswith(':0202') and len(r['tds']) >= 4:
            names.setdefault(r['tds'][3], []).append(r['id'])
    input_fp = build.digest([build.file_identity(OLD_DB), DIRECTORY_FP, catsha, METHOD])
    identity = out / 'source_identity.json'
    if identity.exists():
        assert json.loads(identity.read_text())['fingerprint'] == input_fp, 'Repair source identity changed'
    else:
        identity.write_text(build.dumps({'fingerprint': input_fp, 'old_db': build.file_identity(OLD_DB), 'raw_index_fingerprint': DIRECTORY_FP, 'catalogue_sha256': catsha, 'method': METHOD}))
    repaired = out / 'japan_dedup_midnight_v1_1.sqlite'
    if not repaired.exists():
        print(build.dumps({'stage': 'copy_derived_dedup', 'bytes': OLD_DB.stat().st_size}), flush=True)
        # SQLite backup makes a consistent new database; never writes the source.
        source_db = sqlite3.connect('file:' + str(OLD_DB) + '?mode=ro', uri=True)
        target_db = sqlite3.connect(repaired)
        source_db.backup(target_db)
        target_db.close()
        source_db.close()
    db = sqlite3.connect(repaired)
    db.execute('CREATE TABLE IF NOT EXISTS midnight_done (fingerprint TEXT PRIMARY KEY,path TEXT,details TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS midnight_targets (id TEXT,t TEXT,old_valid INTEGER,old_conflict INTEGER,PRIMARY KEY(id,t))')
    db.execute('CREATE TABLE IF NOT EXISTS repair_state (key TEXT PRIMARY KEY,value TEXT)')
    if not db.execute('SELECT 1 FROM repair_state WHERE key="before"').fetchone():
        before = db.execute('SELECT COUNT(*),SUM(q>=0 AND conflict=0),SUM(conflict) FROM obs').fetchone()
        db.execute('INSERT INTO repair_state VALUES("before",?)', (build.dumps(dict(zip(['rows', 'valid_q', 'conflicts'], before))),))
        db.commit()
    parsed = skipped = rows24 = accepted = 0
    batch = 0
    for p in paths:
        fp = build.digest([build.file_identity(p), catsha, METHOD])
        if db.execute('SELECT 1 FROM midnight_done WHERE fingerprint=?', (fp,)).fetchone():
            skipped += 1
            continue
        info = {'header_only': p.stat().st_size == 10, 'raw_24_rows': 0, 'accepted_timestamp_rows': 0, 'invalid_24_rows': 0,
                'mapped_values': 0, 'nonnegative_values': 0, 'unknown_value_flags': 0, 'unresolved_columns': []}
        if p.stat().st_size > 10:
            parsed += 1
            with p.open(encoding='utf-8-sig', newline='') as f:
                head = next(csv.reader([next(f)]))
                assert head[0] == 'time_jst', 'Unrecognized source header: ' + str(p)
                native = [names.get(n.strip(), [None])[0] if len(names.get(n.strip(), [])) == 1 else None for n in head[1:]]
                info['unresolved_columns'] = [n for n, k in zip(head[1:], native) if n.strip() and k is None]
                for line in f:
                    if 'T24:' not in line:
                        continue  # Do not reparse the already accepted 01..23 Q rows.
                    info['raw_24_rows'] += 1
                    row = next(csv.reader([line]))
                    try:
                        t = normalize(row[0])
                    except (ValueError, IndexError):
                        t = None
                    if t is None or len(row) != len(head):
                        info['invalid_24_rows'] += 1
                        continue
                    info['accepted_timestamp_rows'] += 1
                    for station, value in zip(native, row[1:]):
                        if station is None:
                            continue
                        q = build.number(value)
                        info['mapped_values'] += 1
                        info['nonnegative_values'] += int(q is not None and q >= 0)
                        info['unknown_value_flags'] += int(q is None and value.strip() not in ['', '-'])
                        db.execute('INSERT OR IGNORE INTO midnight_targets VALUES(?,?,COALESCE((SELECT q>=0 AND conflict=0 FROM obs WHERE id=? AND t=?),0),COALESCE((SELECT conflict FROM obs WHERE id=? AND t=?),0))', (station, t, station, t, station, t))
                        db.execute(UPSERT, (station, t, q))
        db.execute('INSERT INTO midnight_done VALUES(?,?,?)', (fp, str(p), build.dumps(info)))
        rows24 += info['raw_24_rows']
        accepted += info['accepted_timestamp_rows']
        batch += 1
        if batch >= 200:
            db.commit()
            batch = 0
        if (parsed + skipped) % 2000 == 0:
            print(build.dumps({'stage': 'midnight_scan', 'new_nonempty_files_parsed': parsed, 'cached_files_skipped': skipped, 'raw_24_rows': rows24, 'elapsed_s': time.monotonic() - started}), flush=True)
    db.commit()
    before = json.loads(db.execute('SELECT value FROM repair_state WHERE key="before"').fetchone()[0])
    after = dict(zip(['rows', 'valid_q', 'conflicts'], db.execute('SELECT COUNT(*),SUM(q>=0 AND conflict=0),SUM(conflict) FROM obs').fetchone()))
    expected_gain, retained_midnight, old_midnight = db.execute('SELECT SUM(CASE WHEN o.q>=0 AND o.conflict=0 THEN 1 ELSE 0 END - m.old_valid),SUM(CASE WHEN o.q>=0 AND o.conflict=0 THEN 1 ELSE 0 END),SUM(m.old_valid) FROM midnight_targets m JOIN obs o ON m.id=o.id AND m.t=o.t').fetchone()
    assert after['valid_q'] - before['valid_q'] == expected_gain, 'Valid Q accounting mismatch'
    details = [json.loads(r[0]) for r in db.execute('SELECT details FROM midnight_done')]
    totals = {k: sum(r[k] for r in details) for k in ['raw_24_rows', 'accepted_timestamp_rows', 'invalid_24_rows', 'mapped_values', 'nonnegative_values', 'unknown_value_flags']}
    receipt = {'method_version': METHOD, 'source_fingerprint': input_fp, 'old_dedup': str(OLD_DB), 'repaired_dedup': str(repaired),
               'new_nonempty_files_parsed_this_run': parsed, 'files_skipped_this_run': skipped, 'before': before, 'after': after,
               'valid_q_gain': after['valid_q'] - before['valid_q'], 'expected_net_gain_from_unique_midnight_targets': expected_gain,
               'unique_retained_valid_midnight_targets': retained_midnight, 'old_valid_midnight_targets': old_midnight, 'incremental_totals': totals,
               'old_malformed_rows_preserved_as_historical': 97647, 'other_malformed_rows': 'not_reparsed_or_claimed_fixed',
               'elapsed_seconds': time.monotonic() - started, 'old_sources_unchanged': True, 'other_countries_reparsed': 0}
    # Cross-day examples are actual values from the repaired store, not fabricated fixtures.
    examples = []
    for name in ['畷橋', '吉川']:
        assert len(names.get(name, [])) == 1
        station = names[name][0]
        rr = db.execute('SELECT t,q,conflict FROM obs WHERE id=? AND t IN (?,?,?) ORDER BY t', (station, '2015-01-01T23:00:00+09:00', '2015-01-02T00:00:00+09:00', '2015-01-02T01:00:00+09:00')).fetchall()
        assert len(rr) == 3 and all(r[1] is not None and r[2] == 0 for r in rr), rr
        assert all((datetime.fromisoformat(rr[i + 1][0]) - datetime.fromisoformat(rr[i][0])).total_seconds() == 3600 for i in [0, 1])
        examples.append({'native_id': station, 'name': name, 'rows': rr, 'strict_adjacent_hour': True})
    receipt['cross_day_examples'] = examples
    build.VERSION = METHOD
    b = JPFixBuilder(out)
    def profiles():
        output = []
        for (native,) in db.execute('SELECT DISTINCT id FROM obs ORDER BY id').fetchall():
            rr = db.execute('SELECT t,q,conflict,dups FROM obs WHERE id=? ORDER BY t', (native,)).fetchall()
            times = [r[0] for r in rr]
            t = np.array([datetime.fromisoformat(v).timestamp() / 3600 for v in times])
            q = np.array([r[1] if r[1] is not None and not r[2] else np.nan for r in rr])
            by_hour = dict(zip(t, times))
            st = build.stats(q, t, lambda h: by_hour[h])
            st.update(n_valid_p=0, duplicate_rows=sum(r[3] for r in rr), conflicting_timestamps=sum(r[2] for r in rr), downloaded_grid_missing_hours=int(t[-1] - t[0] + 1 - len(t)))
            output.append({'native': native, 'statistics': st, 'time_start': times[0], 'time_end': times[-1], 'quality': {'status': 'basic_numeric_only', 'conflicts_excluded': True, 'day_end_24_normalization': 'legal_24_00_only_to_next_day_00_JST', 'other_unknown_flags': 'remain_missing_not_zero'}})
        return {'stations': output}
    result = b.cached(build.digest([METHOD, input_fp, 'repaired_profiles']), profiles)
    old_dataset = next(r for r in load(LEGACY / 'datasets.jsonl') if r['dataset_key'] == DATASET)
    old_dataset.update(source_version='existing_snapshot_clock_repair_v1_1', source_fingerprint=input_fp, fingerprint_method='legacy_derived_stat_raw_index_catalogue_sha256_method', observed_at=b.observed)
    old_dataset['summary'] = dict(old_dataset['summary'], clock_repair_method=METHOD, old_version='inventory-v1.0_parser_limited', repair_source=str(OLD_DB), valid_q_gain=receipt['valid_q_gain'], correction_totals=totals, non_midnight_malformed='not_reparsed_or_claimed_fixed')
    b.rows['datasets'][DATASET] = old_dataset
    for row in load(LEGACY / 'entities.jsonl'):
        if row['dataset_key'] == DATASET:
            row['observed_at'] = b.observed
            b.rows['entities'][row['entity_key']] = row
    old_profiles = {r['entity_key']: r for r in load(LEGACY / 'profiles.jsonl') if r['dataset_key'] == DATASET and r['profile_kind'] == 'coverage_distribution'}
    gain = 0
    run_changes = []
    for r in result['stations']:
        entity = DATASET + ':station:' + r['native']
        old = old_profiles[entity]
        change = r['statistics']['n_valid_q'] - old['n_valid_q']
        gain += change
        run_changes.append({'native_id': r['native'], 'valid_q_gain': change, 'old_longest_valid_run_hours': old['statistics']['longest_valid_run_hours'], 'new_longest_valid_run_hours': r['statistics']['longest_valid_run_hours']})
        r['computed_at'] = result['computed_at']
        b.rows['entities'][entity].update(time_start=r['time_start'], time_end=r['time_end'], status='candidate_external_profiled' if r['statistics']['n_valid_q'] else 'no_valid_Q_in_downloaded_scope')
        b.profile(entity, DATASET, 'full_available_inventory', input_fp, r,
                  {'source': str(SOURCE), 'derived_dedup': str(repaired), 'repair_of_profile': old['profile_key'], 'repair_scope': 'only existing legal raw T24:00 rows; existing 01..23 dedup retained', 'q_unit': 'm3/s', 'timezone': 'JST UTC+09:00', 'fingerprint_method': 'legacy_derived_stat_raw_index_catalogue_sha256_method', 'coverage_denominator': 'deduplicated_downloaded_rows_not_declared_2015_2024', 'quantile_method': 'numpy linear', 'mapping': 'unchanged unique native Q station names', 'threshold_fitting_allowed': False, 'selection_use': 'inventory_only'})
    assert gain == receipt['valid_q_gain'], 'Per-station valid Q gains differ from database accounting'
    receipt['station_changes'] = run_changes
    assert any(r['old_longest_valid_run_hours'] == 23 and r['new_longest_valid_run_hours'] > 24 for r in run_changes), 'No expected daily-boundary improvement'
    b.profile(DATASET + ':dataset', DATASET, 'full_available_inventory', input_fp,
              {'profile_kind': 'coverage_distribution', 'statistics': {'n_rows': after['rows'], 'n_valid_q': after['valid_q'], 'n_valid_p': 0, 'valid_q_gain': receipt['valid_q_gain'], 'files': len(paths), 'resolved_stations_with_rows': len(result['stations'])}, 'quality': {'repair_accounting': True, 'incremental_totals': totals, 'historical_malformed_rows': 97647, 'header_only_files': 19777, 'other_malformed_rows': 'not_reparsed_or_claimed_fixed'}, 'computed_at': result['computed_at']},
              {'source': str(SOURCE), 'old_dedup': str(OLD_DB), 'derived_dedup': str(repaired), 'repair_of_profile': old_profiles[DATASET + ':dataset']['profile_key'], 'timezone': 'JST UTC+09:00', 'q_unit': 'm3/s', 'fingerprint_method': 'legacy_derived_stat_raw_index_catalogue_sha256_method'}, status='partial')
    b.finish(figure=False)
    receipt['profile_cache'] = b.counts
    receipt['payload_counts'] = {k: len(v) for k, v in b.rows.items()}
    receipt['observed_at'] = b.observed
    (out / 'clock_repair_receipt.json').write_text(build.dumps(receipt))
    chart(b, out)
    db.close()
    print(build.dumps({k: v for k, v in receipt.items() if k != 'station_changes'}), flush=True)


def chart(b, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    legacy = load(LEGACY / 'profiles.jsonl')
    old = [r['statistics']['longest_valid_run_hours'] for r in legacy if r['dataset_key'] == DATASET and r['scope'] == 'full_available_inventory' and 'longest_valid_run_hours' in r['statistics']]
    new = [r['statistics']['longest_valid_run_hours'] for r in b.rows['profiles'].values() if r['profile_kind'] == 'coverage_distribution' and 'longest_valid_run_hours' in r['statistics']]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([np.array(old) + 1, np.array(new) + 1], labels=['Old: hour 24 rejected', 'Corrected: next-day 00'], showfliers=False)
    ax.set_yscale('log')
    ax.set_ylabel('Longest valid Q run + 1 hour (log scale)')
    ax.set_title('Japan: repairing day-end timestamps removes artificial daily gaps')
    fig.text(.5, .01, 'Same existing chunks and unique native IDs; basic numeric checks only. Zero-valid stations retained. Other countries unchanged.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures/japan_clock_repair.png', dpi=150)
    plt.close(fig)


def check():
    assert normalize('2015-01-01T24:00:00+09:00') == '2015-01-02T00:00:00+09:00'
    assert normalize('2020-02-29T24:00:00+09:00') == '2020-03-01T00:00:00+09:00'
    assert normalize('2015-12-31T24:00:00+09:00') == '2016-01-01T00:00:00+09:00'
    assert normalize('2015-01-01T24:30:00+09:00') is None
    q = [2.16, 2.11, 2.11]
    t = [datetime.fromisoformat(v).timestamp() / 3600 for v in ['2015-01-01T23:00:00+09:00', normalize('2015-01-01T24:00:00+09:00'), '2015-01-02T01:00:00+09:00']]
    assert build.longest(np.isfinite(q), np.array(t)) == 3
    print('strict day-end, leap-day, year-boundary and cross-day checks passed', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if args.check:
        check()
    else:
        fix(args.out)


if __name__ == '__main__':
    main()
