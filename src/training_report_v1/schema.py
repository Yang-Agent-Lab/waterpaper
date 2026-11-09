"""Validate identities and forecasting semantics before drawing."""
import math
from datetime import datetime, timedelta

ROLES = {'fit', 'fit_probe', 'select', 'calibration', 'development', 'smoke_fit_standin', 'smoke_select_standin', 'fit_select'}
X_KINDS = {'origin_time_h6_peak', 'horizon_window_peak', 'valid_time_pointwise', 'lead_hour_pointwise'}


def validate_bundle(bundle):
    if bundle.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    for key in ('run_id', 'models', 'epochs', 'cases', 'risk', 'health', 'interventions', 'metadata'):
        if key not in bundle:
            raise ValueError(f'missing {key}')
    ids = [r['id'] for r in bundle['models']]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('model identities must be unique and nonempty')
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite number: use null with state/reason')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        if isinstance(value, list):
            for v in value: finite(v)
    finite(bundle)
    pointwise = bundle.get('target') == 'pointwise_exceedance'
    if pointwise:
        forecast_hours = bundle['metadata'].get('forecast_hours')
        if type(forecast_hours) is not int or forecast_hours < 1:
            raise ValueError('pointwise forecast_hours must be a positive integer')
        if len(ids) != 1:
            raise ValueError('pointwise reports require one model per bundle')
        if bundle['metadata'].get('input_group') not in ('pq', 'q_only'):
            raise ValueError('pointwise reports require input_group pq or q_only')
        if bundle['epochs'] and not bundle['metadata'].get('loss_definition'):
            raise ValueError('pointwise learning requires a shared loss_definition')
        if len({r['country'] for r in bundle['risk']}) > 5:
            raise ValueError('pointwise page supports at most five countries')
        if bundle['metadata'].get('risk_only_report'):
            curves=bundle['metadata'].get('risk_case_curves')
            if not isinstance(curves,list) or not curves or len(curves)>4:
                raise ValueError('risk-only report requires 1-4 fixed risk_case_curves')
            case_ids=[row.get('case_id') for row in curves]
            if any(not value for value in case_ids) or len(set(case_ids))!=len(case_ids):
                raise ValueError('risk-only case identities must be unique and nonempty')
            for row in curves:
                if row.get('country')!='CN' or row.get('role')!='select' or not row.get('station'):
                    raise ValueError('risk-only cases must be fixed China select samples')
                hours=row.get('lead_hours')
                if hours!=list(range(1,forecast_hours+1)):
                    raise ValueError('risk-only case leads must be 1..forecast_hours')
                for field in ('p95','p99','observed_q95','observed_q99'):
                    values=row.get(field)
                    if not isinstance(values,list) or len(values)!=forecast_hours:
                        raise ValueError('risk-only case probability/observation length mismatch')
                for field in ('p95','p99'):
                    values=row[field]
                    if any(value is None or not 0<=value<=1 for value in values):
                        raise ValueError('risk-only probabilities must be finite values in [0,1]')
                for field in ('observed_q95','observed_q99'):
                    values=row[field]
                    if any(value not in (0,1,False,True) for value in values):
                        raise ValueError('risk-only observed exceedance must be binary')
        for row in bundle['epochs']:
            for key in ('train_loss_definition', 'fit_loss_definition', 'select_loss_definition'):
                if key in row and row[key] != bundle['metadata'].get('loss_definition'):
                    raise ValueError('learning loss definitions must match')
    for section in ('epochs', 'cases', 'risk', 'health', 'interventions'):
        for row in bundle[section]:
            if row.get('model') not in ids:
                raise ValueError(f'{section}: unknown model')
            if row.get('role') not in ROLES:
                raise ValueError(f'{section}: unsupported/missing role')
            if not row.get('country'):
                raise ValueError(f'{section}: missing country')
            if any(v is None for v in row.values()) and not (row.get('state') and row.get('reason')):
                raise ValueError(f'{section}: null requires state and reason')
            if section == 'cases':
                if row.get('x_kind') not in X_KINDS:
                    raise ValueError('four window peaks are not hourly flow')
                n = len(row['x'])
                for name in ('truth', 'median', 'lower', 'upper'):
                    if len(row[name]) != n: raise ValueError('case length mismatch')
                if row['x_kind'] == 'horizon_window_peak' and row['x'] != [1, 3, 6, 12]:
                    raise ValueError('window horizons must be 1/3/6/12h')
                if row['x_kind'].endswith('_pointwise'):
                    if not pointwise or row.get('unit') != 'm3/s' or not n:
                        raise ValueError('pointwise cases require target, nonempty x, and unit m3/s')
                    if n != forecast_hours:
                        raise ValueError('pointwise case length must equal forecast_hours')
                    origin = datetime.fromisoformat(row['origin_time'])
                    expected = list(range(1, n + 1))
                    if row['x_kind'] == 'lead_hour_pointwise' and row['x'] != expected:
                        raise ValueError('pointwise leads must be 1..H')
                    if row['x_kind'] == 'valid_time_pointwise' and [datetime.fromisoformat(v) for v in row['x']] != [origin + timedelta(hours=h) for h in expected]:
                        raise ValueError('valid times must be origin + 1..H hours')
                elif pointwise:
                    raise ValueError('pointwise bundle cannot contain window peaks')
            if section == 'risk' and pointwise:
                if row.get('target') not in ('pointwise_exceedance_Q95', 'pointwise_exceedance_Q99'):
                    raise ValueError('pointwise risk target required')
                if type(row.get('horizon')) is not int or row['horizon'] < 1:
                    raise ValueError('pointwise horizon must be a positive integer')
                for key in ('brier', 'crps', 'reliability_bins'):
                    if key not in row:
                        raise ValueError(f'pointwise risk missing {key}')
                if row['brier'] is not None and not 0 <= row['brier'] <= 1:
                    raise ValueError('Brier must be in [0,1]')
                if row['crps'] is not None and row['crps'] < 0:
                    raise ValueError('CRPS must be nonnegative')
                for bin_row in row['reliability_bins']:
                    for key in ('mean_probability', 'observed_fraction'):
                        if bin_row.get(key) is not None and not 0 <= bin_row[key] <= 1:
                            raise ValueError('reliability values must be in [0,1]')
    if pointwise:
        groups = {(r['country'], r['role'], r['target']) for r in bundle['risk']}
        for group in groups:
            horizons = sorted(r['horizon'] for r in bundle['risk'] if (r['country'], r['role'], r['target']) == group)
            if horizons != list(range(1, forecast_hours + 1)):
                raise ValueError('risk horizons must be unique and contiguous 1..H; use null for missing scores')
    return bundle
