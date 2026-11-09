"""One focused executable check; fixture artifacts remain in a new output folder."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from .data import WindowDataset, check_role_isolation, discover_targets, predictor_input, raw_joint, valid_synthetic
from .diffusion import check_diffusion
from .forecasting import Baseline, Separation, metrics, mmd, orthogonal_loss, separation_loss
from .train import write_json


def fixture(folder):
    folder.mkdir()
    transforms = dict(station_uid='China:fixture', fit_rows=48, fit_event_ids=['China:fixture:train'],
                      q95_m3_s=20., q99_m3_s=50., scales={
                          'p_mm_h': dict(mean=1., scale=1.), 'q_m3_s': dict(mean=10., scale=2.)})
    write_json(folder / 'training_transforms.json', transforms)
    for role in ('train', 'validation'):
        with gzip.open(folder / f'{role}.jsonl.gz', 'wt') as handle:
            for i in range(4):
                y = [float(v) for v in range(1, 13)]
                row = dict(event_uid=f'China:fixture:{role}', origin_source_time=f'2000-01-01T{i:02}:00:00',
                    source_rows=[str(v + 24 * i) for v in range(24)], x_p=[0.] * 12,
                    x_q=[float(i)] * 12, y_q=y, offline_generation_future_p=[2.] * 12,
                    high95_hours=[10 + 2 * v > 20 for v in y], high99_hours=[False] * 12,
                    generation_high95=True)
                handle.write(json.dumps(row) + '\n')
    return transforms


def run_checks(out_dir, data_root=None, device='cpu'):
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2)
    transforms = fixture(out_dir / 'fixture')
    train = WindowDataset(out_dir / 'fixture', 'train')
    validation = WindowDataset(out_dir / 'fixture', 'validation')
    check_role_isolation(train, validation)
    assert train.x.shape == (4, 12, 2) and train.y.shape == (4, 12)
    assert train.joint.shape == (4, 2, 24) and train.labels.dtype == torch.long
    row = dict(train.rows[0])
    changed = dict(row, offline_generation_future_p=[999.] * 12, y_q=[888.] * 12,
                   generation_high95=False, high95_hours=[False] * 12)
    assert torch.equal(predictor_input(row), predictor_input(changed))
    try:
        WindowDataset(out_dir / 'fixture', 'test')
    except ValueError as exc:
        assert 'explicit final evaluation' in str(exc)
    else:
        raise AssertionError('Training could load real test')
    leaking = validation.rows[0]['event_uid']
    validation.rows[0]['event_uid'] = train.rows[0]['event_uid']
    try:
        check_role_isolation(train, validation)
    except ValueError:
        pass
    else:
        raise AssertionError('Event leakage not rejected')
    validation.rows[0]['event_uid'] = leaking

    baseline = Baseline(16).to(device)
    separation = Separation(16).to(device)
    separation.initialize_shared(baseline)
    assert all(torch.equal(value, separation.shared_encoder.state_dict()[key])
               for key, value in baseline.encoder.state_dict().items())
    x, y = train.x.to(device), train.y.to(device)
    assert baseline(x).shape == (4, 12) and separation(x).shape == (4, 12)
    assert separation.branch(x, 'real')[1].shape == x.shape
    for model in (baseline, separation):
        model.zero_grad(set_to_none=True)
        terms = separation_loss(model, x, y, x.flip(0), y.flip(0), {}) if isinstance(model, Separation) else {
            'loss': torch.nn.functional.l1_loss(model(x), y)}
        if isinstance(model, Separation):
            expected = terms['prediction_loss'] + .1 * terms['reconstruction_loss'] + terms['mmd_loss'] + .5 * terms['orthogonal_loss']
            assert torch.allclose(expected, terms['loss'])
        assert all(torch.isfinite(value) for value in terms.values())
        terms['loss'].backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    a, b = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])
    expected = 2 * (5 - sum(math.exp(-2 / float(i + 1) ** 2) for i in range(5)))
    assert math.isclose(float(mmd(a, b)), expected, rel_tol=1e-5)
    assert float(mmd(a, a)) == 0
    assert math.isclose(float(orthogonal_loss(torch.eye(2), torch.eye(2))), 1., rel_tol=1e-6)
    assert torch.isfinite(orthogonal_loss(torch.zeros(2, 2), torch.zeros(2, 2)))

    samples = torch.zeros(5, 2, 24)
    samples[:, 1, 12:] = 100.
    samples[1, 0, 0] = -2.
    samples[2, 1, 0] = -6.
    samples[3, 1, 12:] = 0.
    samples[4, 0, 0] = float('nan')
    before = samples.clone()
    keep, counts = valid_synthetic(samples, transforms)
    assert keep.tolist() == [True, False, False, False, False]
    assert counts['raw_negative_P_count'] == counts['raw_negative_Q_count'] == counts['nonfinite_count'] == 1
    assert counts['accepted'] == 1
    assert torch.allclose(before, samples, equal_nan=True)
    assert float(raw_joint(samples[:1], transforms)[0, 1, 12]) == 210.
    observed = torch.arange(1., 13.)[None].repeat(2, 1)
    score = metrics(observed + 2, observed, dict(q95_m3_s=5., q99_m3_s=20.))
    assert score['h1']['all']['mae'] == score['h12']['all']['rmse'] == 2.
    assert score['h1']['q95']['n'] == 0 and score['h1']['q95']['mae'] is None
    assert score['h12']['q95']['n'] == 2 and score['h12']['q99']['status'] == 'NOT_EVALUABLE'
    assert score['window_peak']['magnitude_mae_m3_s'] == 2.
    assert score['window_peak']['timing_mae_hours'] == 0.

    diffusion = check_diffusion(device=device)
    # Fresh interpreter: imports must not launch jobs, write artifacts, or emit logs.
    import_folder = out_dir / 'import_check'
    import_folder.mkdir()
    src = str(Path(__file__).resolve().parents[1])
    script = (f'import sys; sys.path.insert(0, {src!r}); '
              'import esdf_flood.data, esdf_flood.forecasting, esdf_flood.diffusion, esdf_flood.train')
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=import_folder,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert not result.stdout and not result.stderr and not list(import_folder.iterdir())
    real_data = {}
    if data_root is not None:
        for target in discover_targets(data_root):
            real_train = WindowDataset(data_root / target, 'train')
            real_validation = WindowDataset(data_root / target, 'validation')
            check_role_isolation(real_train, real_validation)
            real_data[target] = dict(train_windows=len(real_train), validation_windows=len(real_validation),
                train_events=len({row['event_uid'] for row in real_train.rows}),
                validation_events=len({row['event_uid'] for row in real_validation.rows}),
                high95_train_windows=int(real_train.labels.sum()), test_read=False)
        assert real_data, 'No real prepared targets found'
    report = dict(state='PASS', scope='implementation_correctness_not_scientific_benefit',
                  checks=['roles_and_test_block', 'forecast_input_excludes_future', 'shapes_and_gradients',
                          'shared_initialization', 'author_MMD_and_orthogonal_formula',
                          'raw_sample_filter_without_clipping', 'metrics_and_missing_highflow',
                          'diffusion_guidance_direction', 'import_without_execution'],
                  synthetic_filter_counts=counts, diffusion=diffusion, real_data=real_data,
                  fixture_directory=str(out_dir / 'fixture'))
    write_json(out_dir / 'check_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(run_checks(args.out_dir, args.data_root, args.device), ensure_ascii=False))


if __name__ == '__main__':
    main()
