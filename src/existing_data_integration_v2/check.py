"""Inspect existing source entry metadata without creating model windows or roles."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path('./external-home/multicountry_QC_fix_1fa15b9/results/experiments/multicountry_harmonization_full_1fa15b9_20261005/artifacts/registry.jsonl')


def validate(manifest):
    keys = [s['dataset_key'] for s in manifest['sources']]
    expected = {'da_tl_cn', 'camels_de_1h', 'lamah_ice', 'camelsh_us_v6',
                'japan_mlit_tone_existing', 'jp_jma_maebashi_mlit_kurihashi_2015_2024',
                'waterbench_iowa_v2_7087806', 'hei_luoma_daily',
                'geodata_poyang_samples', 'gxsw_recent_capture'}
    assert len(keys) == len(set(keys)) == 10 and set(keys) == expected
    assert {s['country_code'] for s in manifest['sources']} == {'CN', 'DE', 'IS', 'US', 'JP'}
    assert all(s['planned_role'] == 'unassigned' for s in manifest['sources'])
    assert manifest['forecast_contract']['history_hours'].startswith('unassigned')
    assert manifest['forecast_contract']['future_observations'] == 'supervision_only_never_predictor_input'


def metadata(path):
    p = Path(path)
    result = {'path': str(p), 'exists': p.exists()}
    if p.exists():
        st = p.stat()
        result.update(kind='directory' if p.is_dir() else 'file', bytes=st.st_size,
                      mtime_ns=st.st_mtime_ns)
    return result


def csv_header(path, examples=False):
    result = metadata(path)
    if not result['exists']:
        return result
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        result['header'] = reader.fieldnames
        if examples:
            # Only new unassigned US/JP sources; old split observations are never read.
            result['unassigned_examples'] = [r for _, r in zip(range(3), reader)]
    return result


def check(manifest):
    validate(manifest)
    registry = []
    with REGISTRY.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get('station_uid', '').split(':')[0] in ('da_tl_cn', 'camels_de_1h', 'lamah_ice'):
                registry.append(row)
    findings = []
    for source in manifest['sources']:
        key = source['dataset_key']
        path = Path(source.get('prepared_entry', source['server_uri']).removeprefix('teacherbox:'))
        result = {'dataset_key': key, 'entry': metadata(path), 'checks': [],
                  'declared_units': source['variables'], 'declared_clock': source['native_clock'],
                  'planned_role': 'unassigned'}
        if key in ('da_tl_cn', 'camels_de_1h', 'lamah_ice'):
            selected = [r for r in registry if r['station_uid'].startswith(key + ':')]
            result['registered_station_count'] = len(selected)
            for row in selected[:2]:
                result['checks'].append({'station_uid': row['station_uid'],
                    'clock': {k: row.get(k) for k in ('timezone', 'interval_label', 'available_at')},
                    'provenance': row.get('provenance'),
                    'observation_header_only': csv_header(row['observation_file'])})
        elif key == 'waterbench_iowa_v2_7087806':
            for p in sorted(path.glob('*.csv.gz'))[:2]:
                result['checks'].append(csv_header(p, examples=True))
            result['prepared_station_files'] = sum(1 for _ in path.glob('*.csv.gz'))
            for name in ('existing_usgs_overlap.json', 'nesting.json', 'completion.json'):
                result['checks'].append(metadata(path.parent / name))
        elif key == 'jp_jma_maebashi_mlit_kurihashi_2015_2024':
            result['checks'].append(csv_header(path, examples=True))
        elif key == 'camelsh_us_v6':
            with zipfile.ZipFile(path) as archive:
                members = [m for m in archive.infolist() if m.filename.endswith('.nc')]
                result['archive_nc_members'] = len(members)
                result['member_metadata_examples'] = [{'name': m.filename, 'bytes': m.file_size,
                    'CRC': m.CRC} for m in members[:2]]
                result['raw_members_decompressed'] = 0
        elif key == 'japan_mlit_tone_existing':
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
                result['cache_schema'] = connection.execute(
                    "SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall()
            result['raw_rows_read'] = 0
        else:
            children = sorted(p for p in path.iterdir() if p.is_file())[:5]
            result['checks'] = [csv_header(p) if p.suffix == '.csv' else metadata(p) for p in children]
        result['status'] = 'entry_present_metadata_checked' if result['entry']['exists'] else 'missing_entry'
        findings.append(result)
    return {'status': 'lightweight_access_checked_not_model_ready',
        'host': socket.gethostname(), 'cwd': str(Path.cwd()),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'sources': findings,
        'raw_test_observations_read': False, 'training_started': False,
        'windows_constructed': False, 'thresholds_fitted': False, 'downloads_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(ROOT / 'configs/existing_data_integration_v2.json'))
    parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    validate(manifest)
    if args.self_test:
        broken = json.loads(json.dumps(manifest))
        broken['sources'][0]['planned_role'] = 'fit'
        try:
            validate(broken)
        except AssertionError:
            print('manifest coverage and role rejection checks passed')
            return
        raise AssertionError('all-data fit assignment was not rejected')
    assert args.output, '--output is required'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = check(manifest)
    receipt['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    snapshot = output.parent / 'manifest_snapshot.json'
    with snapshot.open('xb') as f:
        f.write(manifest_bytes)
    receipt['manifest_snapshot'] = str(snapshot)
    with output.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
    assert len(receipt['sources']) == 10
    assert all(s['entry']['exists'] for s in receipt['sources'])
    print(json.dumps({'status': receipt['status'], 'sources': len(receipt['sources']),
                      'host': receipt['host'], 'output': str(output)}))


if __name__ == '__main__':
    main()
