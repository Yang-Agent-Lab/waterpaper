"""Independent Japan clock-correction checks on existing derived stores."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3

import numpy as np
from data_registry_acceptance_v1 import check_series, near


def main(a):
    profiles = [json.loads(line) for line in (a.payload / 'profiles.jsonl').read_text().splitlines()]
    stations = [p for p in profiles if ':station:' in p['entity_key']]
    assert len(stations) == 112
    assert all(p['method_version'] == 'inventory-jp-clock-v1.1' for p in profiles)
    assert all(p['dataset_key'] == 'japan_mlit_tone_existing' for p in profiles)
    conns = [sqlite3.connect('file:' + str(p) + '?mode=ro', uri=True) for p in [a.old_db, a.new_db]]
    counts = [c.execute('select count(*),sum(case when q>=0 and conflict=0 then 1 else 0 end),sum(conflict),sum(dups) from obs').fetchone() for c in conns]
    assert sum(p['n_rows'] for p in stations) == counts[1][0]
    assert sum(p['n_valid_q'] for p in stations) == counts[1][1]
    assert counts[1][0] > counts[0][0] and counts[1][1] > counts[0][1]
    chosen = next(p for p in sorted(stations, key=lambda p: p['entity_key']) if p['n_valid_q'] > 0)
    native = chosen['entity_key'].split(':station:', 1)[1]
    rows = conns[1].execute('select t,q,conflict from obs where id=? order by t', (native,)).fetchall()
    checked = check_series(chosen, [r[1] if r[1] is not None and not r[2] else np.nan for r in rows], [datetime.fromisoformat(r[0]).timestamp()/3600 for r in rows])
    # This station and date are chosen by native identity, not prediction performance.
    midnight = next(t for t, q, conflict in rows if 'T00:00:00' in t and q is not None and q >= 0 and not conflict)
    t0 = datetime.fromisoformat(midnight).timestamp()
    neighbours = sorted([r for r in rows if abs(datetime.fromisoformat(r[0]).timestamp() - t0) <= 3600], key=lambda r: r[0])
    assert len(neighbours) == 3
    assert np.array_equal(np.diff([datetime.fromisoformat(r[0]).timestamp() for r in neighbours]), [3600, 3600])
    assert conns[0].execute('select count(*) from obs where id=? and t=?', (native, midnight)).fetchone()[0] == 0
    for t, q, conflict in [neighbours[0], neighbours[2]]:
        previous = conns[0].execute('select q,conflict from obs where id=? and t=?', (native, t)).fetchone()
        assert previous is not None and previous[1] == conflict
        if q is None:
            assert previous[0] is None
        else:
            near(previous[0], q)
    report = {'status': 'passed_independent_japan_clock_verification', 'old_counts_rows_valid_conflict_duplicates': counts[0], 'new_counts_rows_valid_conflict_duplicates': counts[1], 'added_unique_rows': counts[1][0]-counts[0][0], 'added_valid_q_rows': counts[1][1]-counts[0][1], 'station_profile_count': len(stations), 'sample_entity': chosen['entity_key'], 'sample_statistics': checked, 'sample_midnight_and_neighbours': neighbours, 'old_neighbours_unchanged': True, 'raw_original_not_modified': 'checked_by_worker_receipt_not_independently_rehashed'}
    for c in conns:
        c.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['payload', 'old-db', 'new-db', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
