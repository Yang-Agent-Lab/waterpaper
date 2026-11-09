"""Read-only complete CN identity/array comparison; no new data or role allocation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True)
    p.add_argument('--portable', required=True)
    p.add_argument('--accepted-reader-root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    # Existing accepted streaming reader; loading it does not run rank experiments.
    sys.path.insert(0, a.accepted_reader_root)
    from src.fit_rank_template_v1.core import json_items, numeric_rows
    bank, portable, out = map(Path, (a.bank, a.portable, a.out))
    if out.exists():
        raise FileExistsError(out)
    bm = json.loads((bank / 'manifest.json').read_text())
    pm = json.loads((portable / 'data_manifest.json').read_text())
    assert sha(bank / 'manifest.json') == pm['bank_manifest_sha256'] == '2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c'
    report = dict(state='READ_ONLY_AUDIT', roles={}, no_model_forward=True,
                  no_data_reallocation=True, test_or_cal_read=False,
                  bank_manifest_sha256=sha(bank / 'manifest.json'),
                  portable_manifest_sha256=sha(portable / 'data_manifest.json'))
    identities = {}
    for role in ('fit', 'select'):
        for name, expected in bm['roles'][role].items():
            assert sha(bank / name) == expected
        for name in (role + '.npz', role + '_rows.json'):
            assert sha(portable / name) == pm['files'][name]
        source = [(i, r) for i, r in enumerate(json_items(bank / (role + '_rows.json'))) if r['country'] == 'CN']
        rows = json.loads((portable / (role + '_rows.json')).read_text())
        pi = np.asarray([i for i, r in enumerate(rows) if r['country'] == 'CN'])
        compact = [rows[i] for i in pi]
        assert len(source) == len(compact) == (3388 if role == 'fit' else 1706)
        indices = np.asarray([i for i, r in source])
        fields = ('sample_id', 'canonical_station', 'country', 'role', 'event',
                  'source_event_identity', 'origin_time', 'input_start', 'target_end',
                  'q_scale', 'q95', 'q99', 'current_state')
        for (index, original), small in zip(source, compact):
            assert small['global_bank_index'] == index
            assert all(original[k] == small[k] for k in fields)
            assert original['route'] == 'pq' and original['current_information_included']
            assert original['target_leads'] == list(range(1, 25))
            assert original['threshold_sources']['Q95']['role'] == original['threshold_sources']['Q99']['role'] == 'fit'
        identities[role] = {(r['canonical_station'], r['source_event_identity']) for r in compact}
        with np.load(portable / (role + '.npz'), allow_pickle=False) as z:
            old_x, _ = numeric_rows(bank / (role + '.npz'), 'x', indices, (24, 2))
            old_y, _ = numeric_rows(bank / (role + '.npz'), 'y', indices, (24,))
            old_physical, _ = numeric_rows(bank / (role + '.npz'), 'y_q_m3_s', indices, (24,))
            old_threshold, _ = numeric_rows(bank / (role + '.npz'), 'thresholds_normalized', indices, (2,))
            assert np.array_equal(old_x, z['history'][pi])
            assert np.array_equal(old_physical, z['future_q'][pi])
            assert z['history_mask'][pi].all() and z['future_mask'][pi].all()
            scale = np.asarray([r['q_scale'] for r in compact])
            native_y = z['future_q'][pi] / scale[:, None]
            assert np.array_equal(native_y, old_y)
            th = np.stack([z['q95_fit'][pi], z['q99_fit'][pi]], -1) / scale[:, None]
            assert np.allclose(th, old_threshold, rtol=1e-12, atol=1e-10)
            assert np.array_equal(np.asarray([r['global_bank_index'] for r in compact]), z['global_bank_index'][pi])
        report['roles'][role] = dict(rows=len(compact), stations=len({r['canonical_station'] for r in compact}),
            source_units=len(identities[role]), native_x_exact=True, native_y_exact=True,
            physical_truth_exact=True, row_fields_exact=list(fields),
            sample_ids_sha256=hashlib.sha256('\n'.join(r['sample_id'] for r in compact).encode()).hexdigest(),
            per_station=dict(Counter(r['canonical_station'] for r in compact)))
        print(role, report['roles'][role], flush=True)
    assert not identities['fit'] & identities['select']
    report.update(state='PASS', fit_select_source_unit_disjoint=True,
                  interpretation='3388 is full accepted CN fit scope; 3187 was unique origins actually exposed in earlier D500 training')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
