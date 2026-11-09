"""Independent pre-training checks of the committed full pointwise bank.

Reads only produced fit/select artifacts; never accesses reserved observations.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from src.pointwise_benchmark_v1.dataset import SampleBank, sha256


def check(config_path, output):
    config = json.loads(Path(config_path).read_text())
    expected = dict(CN=320000, DE=128000, IS=64000, US=96000, JP=32000)
    root = Path(config['banks']['pq']).parent
    completion = json.loads((root / 'completion.json').read_text())
    assert completion['state'] == 'complete' and completion['mode'] == 'full'
    inputs = json.loads((root / 'input_manifest.json').read_text())
    assert inputs['role_values_allowed'] == ['fit', 'select']
    assert inputs['permission_sha256'] == 'bd0074a0e9ddf6df9a9dc077e835330e7741f29ffa7f87d57edee3bd1bca2e53'
    assert sha256(inputs['metadata_access_receipt_path']) == inputs['metadata_access_receipt_sha256']
    results = []
    for route in ('pq', 'q_only'):
        bank = SampleBank(config['banks'][route], route, 24, 24)
        manifest = bank.manifest
        assert manifest['raw_supervision_preserved']
        assert not manifest['roles_reallocated'] and not manifest['fit_scales_recomputed']
        assert not manifest['calibration_test_values_read']
        assert manifest['fit_scale_identity']['prepared_manifest_sha256'] == 'adb945fa9e7e054ab25ad74ed85c4c6eb3ede21098bfc9ed82a19986a3fbf5c7'
        exposure = bank.exposure(config['exposures'][route], 50, 200, 64)
        actual = Counter(bank.rows['fit'][int(i)]['country'] for i in exposure.flat)
        assert dict(actual) == expected
        country_stations = {}
        intervals = defaultdict(list)
        sample_ids = set()
        for role in ('fit', 'select'):
            rows = bank.rows[role]
            cn = [r for r in rows if r['country'] == 'CN']
            assert len({r['canonical_station'] for r in cn}) == (8 if role == 'fit' else 9)
            if role == 'select':
                assert len(cn) == 1706
                count = Counter((r['country'], r['canonical_station']) for r in rows if r['country'] != 'CN')
                assert count and max(count.values()) <= 2
            arrays = bank.arrays[role]
            for index, row in enumerate(rows):
                assert row['sample_id'] not in sample_ids
                sample_ids.add(row['sample_id'])
                # Producer uses each source's native hourly clock; never infer a
                # timezone or DST shift from the server hosting this check.
                start, origin, end = [(datetime.fromisoformat(row[k]).replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()
                                      for k in ('input_start', 'origin_time', 'target_end')]
                assert origin - start == 23 * 3600 and end - origin == 24 * 3600
                assert len(row['source_rows']) == 48
                assert 'current_state' in row and set(row['threshold_sources']) == {'Q95', 'Q99'}
                for j, name in enumerate(('Q95', 'Q99')):
                    threshold = row['threshold_sources'][name]
                    assert threshold['value'] == row[name.lower()]
                    assert np.isclose(threshold['value'] / row['q_scale'], arrays['thresholds_normalized'][index, j], rtol=1e-12)
                intervals[(row['country'], row['canonical_station'])].append((start, end, role))
            country_stations[role] = {c: len({r['canonical_station'] for r in rows if r['country'] == c}) for c in expected}
        # Overlapping windows within one role are expected. Physical observation
        # hours must not overlap between fit and select at the same station.
        for spans in intervals.values():
            latest = {'fit': float('-inf'), 'select': float('-inf')}
            for start, end, role in sorted(spans):
                other = 'select' if role == 'fit' else 'fit'
                assert start > latest[other], 'cross-role physical time overlap'
                latest[role] = max(latest[role], end)
        results.append(dict(route=route, bank_sha256=bank.identity,
            exposure_sha256=sha256(config['exposures'][route]), actual_exposures=dict(actual),
            actual_origins={role: len(rows) for role, rows in bank.rows.items()},
            country_stations=country_stations, raw_supervision_scale_consistent=True,
            source_hour_overlap_between_fit_select=False))
        del bank, exposure, intervals
    receipt = dict(state='accepted', condition_sha256=sha256(config_path),
        scope='produced fit/select identities, raw supervision, role isolation and common exposures; not model performance',
        input_manifest_sha256=sha256(root / 'input_manifest.json'), routes=results)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check(args.config, args.output)
