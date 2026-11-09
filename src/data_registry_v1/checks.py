"""Server-only focused fixtures and exported-payload validation."""
import argparse
import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from . import build


def validate(path):
    rows = {k: [json.loads(line) for line in (path / (k + '.jsonl')).open()] for k in ['datasets', 'entities', 'profiles', 'rules', 'classifications']}
    keynames = dict(datasets='dataset_key', entities='entity_key', profiles='profile_key', rules='rule_key', classifications='assignment_key')
    keys = {kind: {r[keynames[kind]] for r in values} for kind, values in rows.items()}
    for kind, values in rows.items():
        assert len(values) == len(keys[kind]), kind
        for r in values:
            assert r['project_slug'] == 'waterpaper'
    entities = {r['entity_key']: r for r in rows['entities']}
    for r in rows['entities']:
        assert r['dataset_key'] in keys['datasets']
        if r['station_key']:
            assert r['station_key'] in keys['entities']
            assert entities[r['station_key']]['entity_type'] == 'station'
            assert entities[r['station_key']]['dataset_key'] == r['dataset_key']
    for r in rows['profiles']:
        assert r['entity_key'] in entities and r['dataset_key'] in keys['datasets']
        assert entities[r['entity_key']]['dataset_key'] == r['dataset_key']
        for field in ['n_rows', 'n_valid_q', 'n_valid_p']:
            assert r[field] is None or isinstance(r[field], int)
        assert isinstance(r['statistics'], dict) and isinstance(r['quality'], dict) and isinstance(r['provenance'], dict)
        if r['scope'] == 'full_available_inventory':
            assert r['provenance']['threshold_fitting_allowed'] is False
        for key in ['max_rise_1h_context', 'max_recession_1h_context']:
            context = r['statistics'].get(key)
            if context:
                before = datetime.fromisoformat(context['time_before'])
                after = datetime.fromisoformat(context['time_after'])
                assert (after - before).total_seconds() == 3600, (r['profile_key'], key)
        st = r['statistics']
        if all(st.get(k) is not None for k in ['q_min', 'q50', 'q95', 'q_max']):
            assert 0 <= st['q_min'] <= st['q50'] <= st['q95'] <= st['q_max']
    for r in rows['classifications']:
        assert r['entity_key'] in keys['entities'] and r['rule_key'] in keys['rules']
        assert r['profile_key'] is None or r['profile_key'] in keys['profiles']
    assert all(r['role_label'] for r in rows['entities'])
    for batch in json.loads((path / 'batches/manifest.json').read_text()):
        assert build.sha(path / batch['file']) == batch['sha256']
        assert len(json.loads((path / batch['file']).read_text())) == batch['rows']
    return {k: len(v) for k, v in rows.items()}


def fixtures(root):
    build.checks()
    build.ROOT = root / 'fixture_source'
    cat = build.ROOT / 'data/catalog'
    cat.mkdir(parents=True, exist_ok=True)
    stations = [{'id': 'unique:0202', 'tds': ['Q', 'Tone', 'Tone', 'Unique', 'x']},
                {'id': 'amb1:0202', 'tds': ['Q', 'Tone', 'Tone', 'Ambiguous', 'x']},
                {'id': 'amb2:0202', 'tds': ['Q', 'Tone', 'Tone', 'Ambiguous', 'x']}]
    (cat / 'japan_mlit_station_list_tone_20261013.json').write_text(json.dumps({'stations': stations}))
    chunks = build.ROOT / 'data/raw/japan_mlit_tone_Q_2015_2024/chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    (chunks / 'empty.csv').write_text('time_jst,\n')
    (chunks / 'a.csv').write_text('time_jst,Unique,Ambiguous\n2020-01-01T01:00:00+09:00,1,100\n2020-01-01T02:00:00+09:00,2,101\n2020-01-01T04:00:00+09:00,9,102\n')
    (chunks / 'b.csv').write_text('time_jst,Unique\n2020-01-01T01:00:00+09:00,1\n2020-01-01T02:00:00+09:00,3\n2020-01-01T05:00:00+09:00,-\n')
    b = build.Builder(root / 'fixture_output')
    b.japan()
    b.finish(figure=False)
    counts = validate(b.out)
    pp = [r for r in b.rows['profiles'].values() if r['entity_key'].endswith('unique:0202')][0]
    assert pp['n_rows'] == 4 and pp['n_valid_q'] == 2
    assert pp['statistics']['duplicate_rows'] == 2 and pp['statistics']['conflicting_timestamps'] == 1
    assert pp['statistics']['max_rise_1h'] is None  # Conflict and missing hour cannot bridge.
    dataset = [r for r in b.rows['profiles'].values() if r['entity_key'].endswith(':dataset')][0]
    assert dataset['quality']['header_only_files'] == 1 and dataset['quality']['unresolved_columns']['Ambiguous'] == 1
    def semantic(value):
        if isinstance(value, dict):
            return {k: semantic(v) for k, v in value.items() if k not in ['observed_at', 'assigned_at']}
        if isinstance(value, list):
            return [semantic(v) for v in value]
        return value
    hashes = {k: build.digest(semantic(b.rows[k])) for k in b.rows}
    b.db.close()
    again = build.Builder(root / 'fixture_output')
    again.japan()
    again.finish(figure=False)
    assert again.counts['processed'] == 0, again.counts
    assert hashes == {k: build.digest(semantic(again.rows[k])) for k in again.rows}
    again.db.close()
    stations[0]['id'] = 'new-native:0202'
    (cat / 'japan_mlit_station_list_tone_20261013.json').write_text(json.dumps({'stations': stations}))
    changed = build.Builder(root / 'fixture_output')
    changed.japan()
    native_ids = {r['native_id'] for r in changed.rows['entities'].values() if r['entity_type'] == 'station'}
    assert native_ids == {'new-native:0202'}, native_ids
    changed.db.close()
    print(build.dumps({'fixture_checks': 'passed', 'contract_counts': counts, 'repeat_raw_parses': again.counts['processed']}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fixtures')
    p.add_argument('--validate')
    p.add_argument('--status')
    args = p.parse_args()
    if args.fixtures:
        fixtures(Path(args.fixtures))
    if args.validate:
        print(build.dumps({'payload_validation': validate(Path(args.validate))}))
    if args.status:
        out = Path(args.status)
        result = {}
        for path in sorted(out.glob('jp_dedup_*.sqlite')):
            db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
            result[path.name] = {'files_committed': db.execute('SELECT COUNT(*) FROM done').fetchone()[0], 'deduplicated_station_times': db.execute('SELECT COUNT(*) FROM obs').fetchone()[0]}
            db.close()
        print(build.dumps(result))


if __name__ == '__main__':
    main()
