"""Package existing registry JSONL as bounded SQL batches; no credentials or raw Q."""
import argparse
import hashlib
import json
from pathlib import Path

TABLES = {
    'datasets': ('research_datasets', 'dataset_key', 'dataset_key project_slug name country_code source_url source_version data_kind nominal_interval_minutes time_zone variables server_uri source_fingerprint fingerprint_method inventory_status model_readiness planned_role summary observed_at'),
    'entities': ('research_data_entities', 'entity_key', 'entity_key project_slug dataset_key entity_type native_id station_key name role_label time_start time_end attributes status observed_at'),
    'rules': ('research_data_classification_rules', 'rule_key', 'rule_key project_slug name level method_version status definition required_fields source_refs notes'),
    'profiles': ('research_data_profiles', 'profile_key', 'profile_key project_slug entity_key dataset_key profile_kind scope method_version source_fingerprint fingerprint_method n_rows n_valid_q n_valid_p time_start time_end statistics quality provenance status computed_at'),
    'classifications': ('research_data_classifications', 'assignment_key', 'assignment_key project_slug entity_key profile_key rule_key label status evidence assigned_at'),
}


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def package(source, out, max_bytes):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'source': str(source), 'raw_time_series_included': False, 'tables': {}, 'batches': []}
    for kind, (table, key, fields) in TABLES.items():
        columns = fields.split()
        path = source / (kind + '.jsonl')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len({r[key] for r in rows}) == len(rows), (kind, 'duplicate keys')
        for row in rows:
            assert row['project_slug'] == 'waterpaper'
            assert set(row) <= set(columns), (kind, sorted(set(row) - set(columns)))
        if kind == 'entities':
            rank = {'dataset': 0, 'station': 1, 'event': 2}
            rows.sort(key=lambda r: (rank[r['entity_type']], r[key]))
        else:
            rows.sort(key=lambda r: r[key])
        manifest['tables'][kind] = {'rows': len(rows), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        chunks, chunk, size = [], [], 2
        for row in rows:
            n = len(dump(row).encode('utf-8')) + 1
            if chunk and size + n > max_bytes:
                chunks.append(chunk)
                chunk, size = [], 2
            chunk.append(row)
            size += n
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            quoted = dump(chunk).replace("'", "''")
            conflict = 'DO NOTHING'
            if kind in ('datasets', 'entities'):
                conflict = 'DO UPDATE SET ' + ','.join(f'{c}=excluded.{c}' for c in columns if c != key)
            cols = ','.join(columns)
            query = ("set standard_conforming_strings=on;\n"
                     f'with written as (insert into public.{table} ({cols}) '
                     f"select {cols} from jsonb_populate_recordset(null::public.{table},'{quoted}'::jsonb) "
                     f'on conflict({key}) {conflict} returning {key}) '
                     f"select '{kind}' as registry_table,count(*) as affected from written;\n")
            name = f'{len(manifest["batches"]):04d}_{kind}.sql'
            query += '-- registry_batch_end:' + name + '\n'
            data = query.encode('utf-8')
            (out / name).write_bytes(data)
            manifest['batches'].append({'file': name, 'kind': kind, 'rows': len(chunk), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(dump({'tables': manifest['tables'], 'batch_count': len(manifest['batches']), 'out': str(out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=300000)
    args = parser.parse_args()
    package(args.source, args.out, args.max_bytes)
