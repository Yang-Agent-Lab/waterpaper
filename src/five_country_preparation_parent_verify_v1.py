"""Independent selected-source quantile and episode checks; no producer import."""
import argparse
import csv
import gzip
import io
import json
import math
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np


def h(time):
    return int(np.datetime64(datetime.fromisoformat(str(time)).replace(tzinfo=None), 'h').astype('int64'))


def valid(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else math.nan
    except (ValueError, TypeError):
        return math.nan


def quantile(values, p):
    ordered = sorted(v for v in values if math.isfinite(v) and v >= 0)
    if not ordered:
        return None
    rank = (len(ordered)-1)*p
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(rank-lo)


def reference_episodes(times, q, threshold):
    if threshold is None or threshold <= 0:
        return []
    result, active, quiet, previous = [], None, 0, None
    for time, value in zip(times, q):
        if previous is not None and time != previous+1:
            if active:
                result.append((*active, False))
            active, quiet = None, 0
        if not math.isfinite(value):
            if active:
                result.append((*active, False))
            active, quiet = None, 0
        elif value > threshold:
            if active is None:
                active = [int(time), int(time), float(value), quiet >= 24]
            else:
                active[1], active[2] = int(time), max(active[2], float(value))
            quiet = 0
        else:
            quiet += 1
            if active and quiet == 24:
                result.append((*active, True))
                active = None
        previous = time
    if active:
        result.append((*active, False))
    return result


def source_arrays(table):
    source = table['source']
    path = Path(table['input_identity']['path'])
    is_jp = source.startswith('jp_') or source == 'japan_mlit_tone_existing'
    split = h('2021-01-01T00:00:00' if is_jp else '2015-10-01T00:00:00')
    stop = h('2022-01-01T00:00:00' if is_jp else '2016-10-01T00:00:00')
    times, q, rain = [], [], []
    if source == 'camelsh_us_v6':
        import h5py
        with zipfile.ZipFile(path) as archive, h5py.File(io.BytesIO(archive.read(table['input_identity']['member']))) as f:
            base = h(f['time'].attrs['units'].decode().split('since ')[1])
            all_times = base + np.asarray(f['time'][:], dtype=np.int64)
            n = int(np.searchsorted(all_times, stop))
            times = all_times[:n]
            q = np.asarray(f['streamflow'][:n], dtype=float)
            missing = list(np.asarray(f['streamflow'].attrs.get('_FillValue', [])).ravel())
            missing += list(np.asarray(f['streamflow'].attrs.get('missing_value', [])).ravel())
            if f['streamflow'].id.get_create_plist().fill_value_defined() == 2:
                missing.append(f['streamflow'].fillvalue)
            for value in missing:
                q[q == value] = np.nan
            q[(q < 0) | ~np.isfinite(q)] = np.nan
        rain = None
    elif source == 'japan_mlit_tone_existing':
        with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as c:
            for time, value, conflict in c.execute('select t,q,conflict from obs where id=? and t<? order by t',
                                                   (table['station'], '2022-01-01T00:00:00+09:00')):
                times.append(h(time)); q.append(valid(value) if not conflict else math.nan)
        rain = None
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                time = h(row['time_jst'] if is_jp else row['datetime_native'])
                if time >= stop:
                    break
                times.append(time)
                if is_jp:
                    q.append(valid(row['Q_m3_s']) if row['Q_conflict'].lower() not in ['true', '1'] else math.nan)
                    rain.append(valid(row['P_mm']) if row['P_quality']=='8' and row['P_usable'].lower() in ['true','1'] else math.nan)
                else:
                    q.append(valid(row['discharge_m3_per_s']) if row['q_observed_available']=='1' else math.nan)
                    rain.append(valid(row['precipitation_mm_per_hour']) if row['p_observed_available']=='1' else math.nan)
                if time == stop-1:
                    break
    times, q = np.asarray(times, dtype=np.int64), np.asarray(q, dtype=float)
    rain = None if rain is None else np.asarray(rain, dtype=float)
    return times, q, rain, split


def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifacts', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); manifest = json.loads((Path(a.artifacts)/'manifest.json').read_text())
    assert manifest['state']=='complete' and not manifest['smoke']
    rows = manifest['stations']; assert len(rows) == 6005
    assert Counter(row['state'] for row in rows) == manifest['counts']
    assert not manifest['reserved_values_read'] and not manifest['training_started']
    selected = [('camelsh_us_v6','05464500'), ('waterbench_iowa_v2_7087806','1609'),
                ('japan_mlit_tone_existing','303031283301030:0202'),
                ('jp_jma_maebashi_mlit_kurihashi_2015_2024','303031283301030')]
    checked, process_count = [], 0
    for source, station in selected:
        row = next(r for r in rows if r['source']==source and r['station']==station)
        prepared_path = Path(row['prepared_file'])
        if not prepared_path.is_absolute():
            prepared_path = Path(a.artifacts)/'stations'/prepared_path.name
        with gzip.open(prepared_path, 'rt') as f: table=json.load(f)
        t, q, rain, split = source_arrays(table)
        fit = t < split
        expected95, expected99 = quantile(q[fit], .95), quantile(q[fit], .99)
        assert math.isclose(expected95, table['scale']['q95'], rel_tol=1e-12, abs_tol=1e-8)
        assert math.isclose(expected99, table['scale']['q99'], rel_tol=1e-12, abs_tol=1e-8)
        assert int(np.isfinite(q[fit]).sum()) == table['scale']['q_count']
        if rain is not None:
            positive = rain[fit & np.isfinite(rain) & (rain > 0)]
            assert math.isclose(quantile(positive,.95) if len(positive) else 1, table['scale']['p_scale'], rel_tol=1e-12)
        for role, keep in [('fit',fit), ('select',~fit)]:
            expected = reference_episodes(t[keep], q[keep], expected95)
            actual = table['roles'][role]['processes']
            assert len(expected)==len(actual), (source,role,len(expected),len(actual))
            for reference, process in zip(expected, actual):
                assert reference[:2] == (process['first_exceed_hour'],process['last_exceed_hour'])
                assert math.isclose(reference[2],process['peak_q'],rel_tol=1e-12)
                assert reference[3] == (not process['left_censored'])
                assert reference[4] == (not process['right_censored'])
                assert process['complete'] == (reference[3] and reference[4])
            process_count += len(actual)
        checked.append(dict(source=source,station=station,fit_q_count=table['scale']['q_count'],q95=expected95,q99=expected99))
    eligible = [r for r in rows if r['q_only_pool_eligible']]
    assert len({r['canonical_station'] for r in eligible}) == len(eligible)
    kurihashi = [r for r in eligible if r['canonical_station']=='MLIT:303031283301030']
    assert len(kurihashi)==1 and kurihashi[0]['source']=='jp_jma_maebashi_mlit_kurihashi_2015_2024'
    result=dict(state='passed',producer_commit=manifest['code'],source_stations=len(rows),
                status_counts=manifest['counts'],independent_source_checks=checked,
                independent_threshold_process_checks=process_count,
                duplicate_route_check='passed_latest_Kurihashi_Q_preferred',
                evaluation_values_read=False,training=False,
                limitation='Four selected source stations independently recomputed; remaining stations covered by producer run records, not independent full reprocessing')
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
