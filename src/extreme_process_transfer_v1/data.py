"""Read and validate the approved full prepared bank, without changing its roles."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from src.pointwise_benchmark_v1.dataset import sha256

EXPECTED = {'fit.npz':'328a949a6581b71c679f66b358d6cacad51f2c36b77c5b207c85535739b45224',
            'fit_rows.json.gz':'5f16e8d9f7c3defc5fb07c28873361afac621a20b9e21a77993cbfa41b59df1e',
            'indices.npz':'8984c77f2efd249045e9a5a1b10423b0839f98a365e8fb5cb785de713ea695e1',
            'select.npz':'d5438117352686cff45e2b67d574072c016298dbae727a01b6338ab59202d9ba',
            'select_rows.json':'1b27d000c279d78743f994ae5266d0cf1343ee644f06e0fe61fb051f33691a3e',
            'exposure_B.npz':'57a0ef8892e9663ab2e6114308ad2f70c0513dfa69969eb17dc4e881439e1a7a'}


class Bank:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('state') != 'accepted' or self.manifest.get('execution_cache', False):
            raise ValueError('requires accepted complete prepared bank')
        if self.manifest.get('fit_select_overlap_windows') != 0:
            raise ValueError('fit/select overlap')
        if self.manifest.get('source_bank_sha256') != '2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c':
            raise ValueError('unexpected source bank')
        for name in ('fit.npz', 'fit_rows.json.gz', 'indices.npz', 'select.npz',
                     'select_rows.json', 'exposure_B.npz'):
            if self.manifest['files'][name] != EXPECTED[name] or sha256(self.root/name) != EXPECTED[name]:
                raise ValueError('input hash mismatch: '+name)
        with gzip.open(self.root/'fit_rows.json.gz', 'rt') as file:
            self.fit_rows = json.load(file)
        self.select_rows = json.loads((self.root/'select_rows.json').read_text())
        with np.load(self.root/'indices.npz', allow_pickle=False) as z:
            self.probability = z['p_B'].copy()
            self.severity = z['severity'].copy()
            self.early = z['early_extreme'].copy()
            np.testing.assert_array_equal(z['B'], np.arange(268009))
        self.arrays = {}
        for role, expected in (('fit', 268009), ('select', 262)):
            with np.load(self.root/(role+'.npz'), allow_pickle=False) as z:
                a = {name: z[name].copy() for name in z.files}
            if a['history'].shape != (expected, 24, 2) or a['future_q'].shape != (expected, 24):
                raise ValueError('unexpected data shape '+role)
            if not all(np.isfinite(value).all() for value in a.values()):
                raise ValueError('nonfinite raw data '+role)
            if np.any(a['flow_scale'] <= 0):
                raise ValueError('nonpositive scale')
            np.testing.assert_allclose(a['flow_scale'], a['q95_fit'], rtol=1e-12)
            self.arrays[role] = a
        if len(self.fit_rows) != 268009 or len(self.select_rows) != 262:
            raise ValueError('row identity count mismatch')
        if any(r['role'] != 'fit' for r in self.fit_rows) or any(r['role'] != 'select' for r in self.select_rows):
            raise ValueError('unexpected role')
        expected = ((self.arrays['fit']['future_q'].max(1) > self.arrays['fit']['q95_fit']).astype('int8') +
                    (self.arrays['fit']['future_q'].max(1) > self.arrays['fit']['q99_fit']).astype('int8'))
        np.testing.assert_array_equal(expected, self.severity)

    def batch(self, role, indices, device):
        a = self.arrays[role]; indices = np.asarray(indices)
        x = torch.as_tensor(a['history'][indices], dtype=torch.float32, device=device)
        y = torch.as_tensor(a['future_q'][indices]/a['flow_scale'][indices,None], dtype=torch.float32, device=device)
        return x, y
