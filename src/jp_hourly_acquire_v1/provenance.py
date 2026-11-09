"""Source-pointer-only derivative; never reacquire or reinterpret observations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--original-cwd', required=True)
    ap.add_argument('--october-csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    source = Path(args.source_dir).resolve(strict=True)
    cwd = Path(args.original_cwd).resolve(strict=True)
    october = Path(args.october_csv).resolve(strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    name = 'point_P_Q_2015_2024_uniform_CSV_quality.csv'
    rows = list(csv.DictReader((source / name).open()))
    changes = {'P_source': 0, 'Q_source': 0}
    paths = set()
    for row in rows:
        before = dict(row)
        for key in changes:
            old = row[key]
            path = october if key == 'P_source' and 'official_2019_october_uniform_CSV.raw' in old else Path(old)
            if not path.is_absolute():
                path = cwd / path
            row[key] = str(path.resolve(strict=True))
            paths.add(row[key])
            changes[key] += row[key] != old
        assert all(row[k] == v for k, v in before.items() if k not in changes)
    dataset = out / name
    with dataset.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = json.loads((source / 'receipt.json').read_text())
    receipt.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   provenance_only_fix=True, prior_artifacts=str(source),
                   original_relative_path_cwd=str(cwd), source_pointer_changes=changes,
                   all_source_paths_exist=True, n_unique_source_paths=len(paths),
                   all_non_source_fields_identical=True, requests=[])
    receipt['artifact_sha256'][name] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    receipt['reused_qualification_artifacts'] = {
        n: str(source / n) for n in receipt['artifact_sha256'] if n != name}
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
