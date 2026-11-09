"""Strict read interface for a separately accepted, immutable pointwise sample bank.

This module does not expand legacy roles or create a five-country data pack.
Window qualification and source access remain the responsibility of a versioned
bank producer. The runner consumes the same persisted exposure indices for all
models in a route.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):
            h.update(part)
    return h.hexdigest()


class SampleBank:
    def __init__(self, root, route, history_hours, forecast_hours):
        self.root=Path(root);self.route=route
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        if m['state']!='accepted' or m['target']!='pointwise_Q' or m['route']!=route:
            raise PermissionError('accepted pointwise bank for this route required')
        if (m['history_hours'],m['forecast_hours'])!=(history_hours,forecast_hours):
            raise ValueError('bank window differs from committed model condition')
        if set(m['roles'])!={'fit','select'}:
            raise PermissionError('bank must contain only fit/select')
        if not m['source_qualification_receipt'] or not m['fit_scale_identity']:
            raise ValueError('source qualification and inherited fit scale evidence required')
        self.identity=sha256(self.root/'manifest.json');self.arrays={};self.rows={}
        for role in ('fit','select'):
            files=m['roles'][role]
            if set(files)!={role+'.npz',role+'_rows.json'}:
                raise PermissionError('bank role file allowlist differs from fit/select arrays and identities')
            for name,expected in files.items():
                if sha256(self.root/name)!=expected:
                    raise ValueError('sample bank file hash mismatch: '+name)
            self.rows[role]=json.loads((self.root/(role+'_rows.json')).read_text())
            with np.load(self.root/(role+'.npz'),allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files}
            n=len(self.rows[role]);x,y=arrays['x'],arrays['y']
            if x.shape!=(n,history_hours,2) or y.shape!=(n,forecast_hours):
                raise ValueError('pointwise bank shape mismatch')
            if arrays['mask'].shape!=x.shape or arrays['thresholds_normalized'].shape!=(n,2):
                raise ValueError('bank mask/threshold shape mismatch')
            if not np.isfinite(x[:,:,1]).all() or not arrays['mask'][:,:,1].all() or not np.isfinite(y).all():
                raise ValueError('qualified historical/future Q required')
            if route=='pq' and (not arrays['mask'][:,:,0].all() or not np.isfinite(x[:,:,0]).all()):
                raise ValueError('pq requires actual finite historical P')
            if route=='q_only' and (arrays['mask'][:,:,0].any() or not np.isnan(x[:,:,0]).all()):
                raise ValueError('q_only must explicitly mask P')
            if not np.isfinite(arrays['thresholds_normalized']).all() or (arrays['thresholds_normalized']<=0).any():
                raise ValueError('two positive fit-only research thresholds required')
            if m.get('raw_supervision_preserved'):
                raw=arrays['y_q_m3_s'];scale=np.asarray([r['q_scale'] for r in self.rows[role]],dtype=float)
                if raw.shape!=y.shape or not np.isfinite(raw).all() or not np.allclose(y*scale[:,None],raw,rtol=1e-12,atol=1e-10):
                    raise ValueError('saved raw supervision differs from inherited-scale normalization')
            for row in self.rows[role]:
                if row['role']!=role or row['route']!=route or not row['sample_id']:
                    raise PermissionError('sample identity/role mismatch')
                if not row['current_information_included'] or row['target_leads']!=list(range(1,forecast_hours+1)):
                    raise ValueError('current point and every future hour required')
            arrays['x_model']=x if route=='pq' else x[:,:,1:2]
            self.arrays[role]=arrays
        fit_countries={r['country'] for r in self.rows['fit']}
        if fit_countries!=set(m['training_countries']):
            raise ValueError('declared training countries differ from actual fit samples')

    def exposure(self,path,epochs,updates,batch):
        with np.load(path,allow_pickle=False) as z:
            indices=z['indices']
            bank_identity=str(z['bank_identity'].item())
        if bank_identity!=self.identity or indices.shape!=(epochs,updates,batch):
            raise ValueError('persisted shared exposure shape/identity mismatch')
        if indices.dtype.kind not in ('i','u') or indices.min()<0 or indices.max()>=len(self.rows['fit']):
            raise ValueError('exposure contains invalid fit indices')
        return indices


def hierarchy_weights(rows):
    """Within one country: station, then inherited event/block, then origin equal."""
    from collections import Counter
    count=Counter((r['canonical_station'],r['event']) for r in rows)
    events=Counter(station for station,event in count)
    return np.asarray([1/(len(events)*events[r['canonical_station']]*count[r['canonical_station'],r['event']]) for r in rows])
