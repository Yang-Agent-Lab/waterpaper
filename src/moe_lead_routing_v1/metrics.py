"""Pilot summaries using the accepted empirical V-score and source-unit weighting."""
import numpy as np

from src.d500_process_adapter_v1.scoring import score as inherited_score
from src.m_distribution_metrics_v1.metrics import crps, macro, support, weights


def _pointwise_lower_median(draws):
    k = draws.shape[1]
    return np.sort(draws, axis=1, kind="stable")[:, int(np.ceil(k * 0.5)) - 1, :]


def _lower_median_2d(values):
    k = values.shape[1]
    return np.sort(values, axis=1, kind="stable")[:, int(np.ceil(k * 0.5)) - 1]


def _origin_mean(values, rows, mask):
    ix = np.flatnonzero(mask)
    if not len(ix):
        return None
    keys = [(rows[int(i)]["canonical_station"], rows[int(i)]["source_event_identity"]) for i in ix]
    return float(weights(keys) @ np.asarray(values, dtype=np.float64)[ix])


def pilot_metrics(draws, truth, thresholds, rows, truth_physical, q95_physical):
    """Score [B,K,H] draws; primary top and point-curve peak metrics are separate."""
    draws = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    q95_physical = np.asarray(q95_physical, dtype=np.float64)
    n, k, horizon = draws.shape
    if truth.shape != (n, horizon) or thresholds.shape != (n, 2) or len(rows) != n:
        raise ValueError("prediction/truth/threshold/row identity mismatch")
    scenario_weights = np.full((n, k), 1.0 / k, dtype=np.float64)
    crps_cells = crps(draws, scenario_weights, truth)
    overall_crps, overall_crps_by_lead = macro(crps_cells, rows)
    censored = crps(np.maximum(draws - 1.0, 0.0), scenario_weights, np.maximum(truth - 1.0, 0.0))
    twcrps, twcrps_by_lead = macro(censored, rows)

    top = {}
    for label, threshold_index in (("q95", 0), ("q99", 1)):
        mask = truth > thresholds[:, threshold_index, None]
        value, by_lead = macro(crps_cells, rows, mask)
        top[label] = dict(
            conditional_crps_q95=value,
            by_lead=by_lead,
            support=support(rows, mask),
            condition=f"observed flow > training-fit {label.upper()} at the same lead",
        )

    brier = {}
    for label, threshold_index in (("q95", 0), ("q99", 1)):
        observed = truth > thresholds[:, threshold_index, None]
        probability = (draws > thresholds[:, None, threshold_index, None]).mean(axis=1)
        cells = (probability - observed.astype(np.float64)) ** 2
        value, by_lead = macro(cells, rows)
        brier[label] = dict(value=value, by_lead=by_lead, support=support(rows, np.ones_like(observed, dtype=bool)))

    point_curve = _pointwise_lower_median(draws)
    peak_amplitude = {}
    peak_distribution = {}
    for window in (6, 24):
        actual = truth[:, :window].max(axis=1)
        point_peak = point_curve[:, :window].max(axis=1)
        scenario_peaks = draws[:, :, :window].max(axis=2)
        scenario_peak_median = _lower_median_2d(scenario_peaks)
        signed = point_peak - actual
        peak_amplitude[str(window)] = dict(
            lower_median_point_curve_peak_q95=_origin_mean(point_peak, rows, np.ones(n, dtype=bool)),
            observed_peak_q95=_origin_mean(actual, rows, np.ones(n, dtype=bool)),
            signed_error_q95=_origin_mean(signed, rows, np.ones(n, dtype=bool)),
            absolute_error_q95=_origin_mean(np.abs(signed), rows, np.ones(n, dtype=bool)),
            signed_error_m3_s=_origin_mean(signed * q95_physical, rows, np.ones(n, dtype=bool)),
            absolute_error_m3_s=_origin_mean(np.abs(signed) * q95_physical, rows, np.ones(n, dtype=bool)),
            conditional={},
            support=dict(origins=n, stations=len({r["canonical_station"] for r in rows}),
                         source_units=len({(r["canonical_station"], r["source_event_identity"]) for r in rows})),
            definition="maximum of the leadwise lower-median point curve; normalized by each origin's fit Q95",
        )
        for label, threshold_index in (("q95", 0), ("q99", 1)):
            mask = actual > thresholds[:, threshold_index]
            peak_amplitude[str(window)]["conditional"][label] = dict(
                signed_error_q95=_origin_mean(signed, rows, mask),
                absolute_error_q95=_origin_mean(np.abs(signed), rows, mask),
                signed_error_m3_s=_origin_mean(signed * q95_physical, rows, mask),
                absolute_error_m3_s=_origin_mean(np.abs(signed) * q95_physical, rows, mask),
                support=support(rows, mask[:, None]),
                condition=f"observed {window}h window peak > training-fit {label.upper()}",
            )
        scenario_signed = scenario_peak_median - actual
        peak_distribution[str(window)] = dict(
            lower_median_of_scenario_peaks_q95=_origin_mean(scenario_peak_median, rows, np.ones(n, dtype=bool)),
            signed_error_q95=_origin_mean(scenario_signed, rows, np.ones(n, dtype=bool)),
            absolute_error_q95=_origin_mean(np.abs(scenario_signed), rows, np.ones(n, dtype=bool)),
            definition="lower median of per-scenario window maxima; reported separately from point-curve peak",
        )

    source_detail = []
    unit_keys = sorted({(row["canonical_station"], row["source_event_identity"]) for row in rows})
    for station, source_event in unit_keys:
        ix = np.asarray([
            i for i, row in enumerate(rows)
            if row["canonical_station"] == station and row["source_event_identity"] == source_event
        ], dtype=np.int64)
        local_rows = [rows[int(i)] for i in ix]
        local_top = {}
        for label, threshold_index in (("q95", 0), ("q99", 1)):
            local_mask = truth[ix] > thresholds[ix, threshold_index, None]
            local_top[label] = dict(
                conditional_crps_q95=macro(crps_cells[ix], local_rows, local_mask)[0],
                supported_origin_lead_cells=int(local_mask.sum()),
            )
        local_peak = {}
        for window in (6, 24):
            actual = truth[ix, :window].max(axis=1)
            predicted = point_curve[ix, :window].max(axis=1)
            signed = predicted - actual
            conditional = {}
            for label, threshold_index in (("q95", 0), ("q99", 1)):
                mask = actual > thresholds[ix, threshold_index]
                conditional[label] = dict(
                    signed_error_q95=_origin_mean(signed, local_rows, mask),
                    absolute_error_q95=_origin_mean(np.abs(signed), local_rows, mask),
                    supported_origins=int(mask.sum()),
                )
            local_peak[str(window)] = dict(
                absolute_error_q95=_origin_mean(np.abs(signed), local_rows, np.ones(len(ix), dtype=bool)),
                signed_error_q95=_origin_mean(signed, local_rows, np.ones(len(ix), dtype=bool)),
                conditional=conditional,
            )
        source_detail.append(dict(
            station=station,
            source_event_identity=source_event,
            origins=len(ix),
            independent_event_claim=False,
            top=local_top,
            peak_amplitude=local_peak,
        ))

    inherited = inherited_score(
        draws, truth, thresholds, rows, truth_physical=truth_physical, q95_physical=q95_physical
    )
    return dict(
        state="EVALUABLE" if n else "NOT_EVALUABLE",
        origins=n,
        scenarios=k,
        horizon_hours=horizon,
        overall=dict(
            crps_q95=overall_crps,
            crps_q95_by_lead=overall_crps_by_lead,
            q95_twcrps_q95=twcrps,
            q95_twcrps_by_lead=twcrps_by_lead,
            brier=brier,
        ),
        top=top,
        point_curve_peak_amplitude=peak_amplitude,
        scenario_peak_distribution=peak_distribution,
        source_detail=source_detail,
        inherited_summary=inherited["summary"],
        inherited_constraints=inherited["constraints"],
        inherited_phase_state="EVALUABLE" if inherited["phase"] is not None else "NOT_EVALUABLE_subset_not_full1706",
        score_definition="accepted empirical V-statistic; station/source/origin weighting; top masks use fit thresholds",
    )
