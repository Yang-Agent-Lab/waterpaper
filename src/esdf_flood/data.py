"""Read prepared Chinese event windows without refitting or changing roles."""
import gzip
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def predictor_input(row):
    """Only past P/Q enter the forecaster; offline future P and labels do not."""
    return torch.tensor(list(zip(row['x_p'], row['x_q'])), dtype=torch.float32)


def joint_input(row):
    return torch.tensor([row['x_p'] + row['offline_generation_future_p'],
                         row['x_q'] + row['y_q']], dtype=torch.float32)


def load_transforms(folder):
    meta = json.loads((Path(folder) / 'training_transforms.json').read_text())
    for name in ('p_mm_h', 'q_m3_s'):
        values = meta['scales'][name]
        if not all(torch.isfinite(torch.tensor(values[k])).item() for k in ('mean', 'scale')) or values['scale'] <= 0:
            raise ValueError(f'Invalid training transform: {name}')
    if not (0 <= meta['q95_m3_s'] <= meta['q99_m3_s'] < float('inf')):
        raise ValueError('Invalid training Q95/Q99')
    return meta


def raw_joint(joint, transforms):
    scales = transforms['scales']
    means = joint.new_tensor([scales[k]['mean'] for k in ('p_mm_h', 'q_m3_s')])[None, :, None]
    stds = joint.new_tensor([scales[k]['scale'] for k in ('p_mm_h', 'q_m3_s')])[None, :, None]
    return joint * stds + means


def valid_synthetic(joint, transforms):
    if joint.ndim != 3 or joint.shape[1:] != (2, 24):
        raise ValueError('Synthetic joint must have shape [B,2,24]')
    raw = raw_joint(joint, transforms)
    finite = torch.isfinite(raw).all(dim=(1, 2))
    nonnegative = (raw >= 0).all(dim=(1, 2))
    high95 = (raw[:, 1, 12:] > transforms['q95_m3_s']).any(dim=1)
    mask = finite & nonnegative & high95
    return mask, {'attempted': len(joint), 'accepted': int(mask.sum()),
                  'nonfinite': int((~finite).sum()),
                  'nonfinite_count': int((~finite).sum()),
                  'raw_negative_P_count': int((raw[:, 0] < 0).any(1).sum()),
                  'raw_negative_Q_count': int((raw[:, 1] < 0).any(1).sum()),
                  'q95_pass_count': int(high95.sum()),
                  'negative': int((finite & ~nonnegative).sum()),
                  'not_high95': int((finite & nonnegative & ~high95).sum())}


class WindowDataset(Dataset):
    def __init__(self, folder, role, *, final_evaluation=False):
        if role not in ('train', 'validation', 'test'):
            raise ValueError(f'Unknown data role: {role}')
        if role == 'test' and not final_evaluation:
            raise ValueError('Test is available only for explicit final evaluation')
        self.role = role
        self.folder = Path(folder)
        self.transforms = load_transforms(folder)
        with gzip.open(self.folder / f'{role}.jsonl.gz', 'rt') as handle:
            self.rows = [json.loads(line) for line in handle]
        if not self.rows:
            raise ValueError(f'No windows: {self.folder}/{role}')
        for row in self.rows:
            for field in ('x_p', 'x_q', 'y_q', 'offline_generation_future_p', 'high95_hours', 'high99_hours'):
                if len(row[field]) != 12:
                    raise ValueError(f'Invalid window length: {field}')
            if not torch.isfinite(joint_input(row)).all():
                raise ValueError('Nonfinite prepared window')
            if len(row['source_rows']) != 24 or len(set(row['source_rows'])) != 24:
                raise ValueError('Invalid source-row provenance')
            event = row['event_uid']
            if not event.startswith(self.transforms['station_uid'] + ':'):
                raise ValueError('Window belongs to another target')
            in_fit = event in self.transforms['fit_event_ids']
            if (role == 'train') != in_fit:
                raise ValueError('Training-transform event/role leakage')
            raw = raw_joint(joint_input(row)[None], self.transforms)[0]
            for threshold, field in (('q95_m3_s', 'high95_hours'), ('q99_m3_s', 'high99_hours')):
                # float32 roundoff at a threshold does not change the saved labels.
                expected = torch.tensor(row[field], dtype=torch.bool)
                distance = raw[1, 12:] - self.transforms[threshold]
                reliable = distance.abs() > max(1e-4, abs(self.transforms[threshold]) * 1e-6)
                if not torch.equal((distance > 0)[reliable], expected[reliable]):
                    raise ValueError('Saved high-flow label disagrees with training threshold')
            if bool(row['generation_high95']) != any(row['high95_hours']):
                raise ValueError('Generation label mismatch')
        self.x = torch.stack([predictor_input(row) for row in self.rows])
        self.y = torch.tensor([row['y_q'] for row in self.rows], dtype=torch.float32)
        self.joint = torch.stack([joint_input(row) for row in self.rows])
        self.labels = torch.tensor([int(row['generation_high95']) for row in self.rows], dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def check_role_isolation(train, validation):
    if train.transforms != validation.transforms:
        raise ValueError('Roles use different transforms')
    events = [{row['event_uid'] for row in dataset.rows} for dataset in (train, validation)]
    if events[0] & events[1]:
        raise ValueError('Event crosses train/validation')
    keys = [{(row['event_uid'], source) for row in dataset.rows for source in row['source_rows']}
            for dataset in (train, validation)]
    if keys[0] & keys[1]:
        raise ValueError('Source row crosses train/validation')


def discover_targets(data_root):
    return sorted(path.name for path in Path(data_root).iterdir()
                  if path.is_dir() and (path / 'training_transforms.json').is_file())
