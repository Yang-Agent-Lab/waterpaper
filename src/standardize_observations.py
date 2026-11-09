"""Conservative first-batch harmonization; never edits source archives."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
import math
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

FIELDS = ['station_uid', 'source_dataset', 'source_row', 'event_id', 'time_source',
          'time_utc', 'q_value', 'p_value', 'p_gapfilled_value', 't_value',
          'water_level_m', 'precipitation_uncovered_pct', 'qc_flags']


def number(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'na', 'null'):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def excel_rows(blob):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:sheetData/s:row', ns):
            values = {}
            for cell in row.findall('s:c', ns):
                col = re.sub('[0-9]', '', cell.attrib['r'])
                value = cell.find('s:v', ns)
                value = value.text if value is not None else None
                if cell.attrib.get('t') == 's':
                    value = strings[int(value)]
                elif cell.attrib.get('t') == 'inlineStr':
                    value = ''.join(cell.find('s:is', ns).itertext())
                elif value is not None:
                    value = float(value)
                values[col] = value
            yield values


def write_station(path, station, rows):
    counts = Counter()
    previous = {}
    seen = {}
    first = last = None
    with path.open('xb') as binary, gzip.GzipFile(fileobj=binary, mode='wb', compresslevel=1) as packed:
        with io.TextIOWrapper(packed, encoding='utf-8', newline='') as text:
            writer = csv.DictWriter(text, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                flags = []
                timestamp = datetime.fromisoformat(row['time_source'])
                key = row['event_id'] or 'continuous'
                if key in previous and abs((timestamp - previous[key]).total_seconds() - 3600) > 1:
                    flags.append('nonhourly_interval')
                previous[key] = timestamp
                signature = tuple(row.get(k) for k in ['q_value', 'p_value', 'p_gapfilled_value', 't_value', 'water_level_m'])
                if row['time_source'] in seen:
                    flags.append('duplicate_same_values' if seen[row['time_source']] == signature else 'duplicate_conflicting_values')
                seen[row['time_source']] = signature
                for k in ['q_value', 'p_value', 't_value']:
                    if row.get(k) is None:
                        flags.append('missing_or_invalid_' + k)
                for k in ['q_value', 'p_value', 'p_gapfilled_value']:
                    if row.get(k) is not None and row[k] < 0:
                        flags.append('negative_' + k)
                if not row['time_utc']:
                    flags.append('timezone_unverified')
                if station['units_status'] != 'documented':
                    flags.append('units_unverified')
                row['qc_flags'] = '|'.join(flags)
                writer.writerow(row)
                counts['rows'] += 1
                counts.update(flags)
                first = min(first, timestamp) if first else timestamp
                last = max(last, timestamp) if last else timestamp
    return {'station_uid': station['station_uid'], 'output': str(path),
            'first_time_source': first.isoformat(), 'last_time_source': last.isoformat(),
            'counts': dict(counts), 'status': 'staging_only_not_training_ready'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        assert number('NaN') is None and number('-1') == -1 and number('inf') is None
        assert number('0') == 0 and number('bad') is None
        assert (datetime(1899, 12, 30) + timedelta(days=43831)).isoformat() == '2020-01-01T00:00:00'
        assert number('77') / 100 == 0.77
        print('Self-check passed: missing/nonfinite values, negative retention, Excel date and cm-to-m conversion.')
        return
    root = Path(__file__).resolve().parents[1]
    assert str(root) == '.', 'Run on server only'
    out = root / 'data/interim/standardized_v1_pilot'
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Preserve an existing pilot; inspect it before rerunning.'
    stations = []
    reports = []
    dezip = root / 'data/raw/camels_de_1h/CAMELS-DE-1h.zip'
    with zipfile.ZipFile(dezip) as z:
        def table(suffix):
            name = next(n for n in z.namelist() if n.endswith(suffix))
            with z.open(name) as f:
                return {r['gauge_id']: r for r in csv.DictReader(io.TextIOWrapper(f))}
        top = table('_topographic_attributes.csv')
        hydro = table('_hydrologic_attributes.csv')
        human = table('_humaninfluence_attributes.csv')
        names = sorted(n for n in z.namelist() if '/timeseries/CAMELS_DE_1h_hydromet_timeseries_' in n and n.endswith('.csv'))
        for name in names:
            sid = Path(name).stem.rsplit('_', 1)[1]
            assert re.fullmatch(r'DE[A-Z0-9]+', sid)
            assert sid in top
            station = {'station_uid': 'camels_de_1h:' + sid, 'source_dataset': 'camels_de_1h',
                       'station_id': sid, 'country': 'Germany', 'source_member': name,
                       'layout': 'continuous_hourly', 'timezone': 'UTC', 'q_unit': 'm3/s',
                       'p_unit': 'mm/h', 't_unit': 'degC', 'water_level_output_unit': 'm',
                       'units_status': 'documented', 'latitude': top[sid].get('gauge_lat'),
                       'longitude': top[sid].get('gauge_lon'), 'area_km2': top[sid].get('area'),
                       'source_flow_start': hydro.get(sid, {}).get('flow_period_start'),
                       'source_flow_end': hydro.get(sid, {}).get('flow_period_end'),
                       'source_flow_completeness_pct': hydro.get(sid, {}).get('flow_perc_complete'),
                       'dams_num': human.get(sid, {}).get('dams_num')}
            stations.append(station)
        for station in stations[:3]:
            with z.open(station['source_member']) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw))
                assert {'date', 'discharge_vol_obs', 'precipitation_mean', 'air_temperature_mean', 'water_level_obs'} <= set(reader.fieldnames)
                def mapped():
                    for i, row in enumerate(reader, 2):
                        time_source = datetime.fromisoformat(row['date']).isoformat()
                        level = number(row['water_level_obs'])
                        yield {'station_uid': station['station_uid'], 'source_dataset': 'camels_de_1h',
                               'source_row': i, 'event_id': '', 'time_source': time_source,
                               'time_utc': time_source + '+00:00', 'q_value': number(row['discharge_vol_obs']),
                               'p_value': number(row['precipitation_mean']),
                               'p_gapfilled_value': number(row['precipitation_mean_gapfilled']),
                               't_value': number(row['air_temperature_mean']),
                               'water_level_m': level / 100 if level is not None else None,
                               'precipitation_uncovered_pct': number(row['perc_nan_precipitation_original'])}
                reports.append(write_station(out / (station['station_id'] + '.csv.gz'), station, mapped()))
    with zipfile.ZipFile(root / 'data/raw/da_tl_zenodo_15608461/Hydro_Extremes_DataAugmentation_Data.zip') as z:
        for name in sorted(n for n in z.namelist() if n.startswith('basin_data/') and n.endswith('.xlsx')):
            sid = Path(name).stem
            station = {'station_uid': 'da_tl_cn:' + sid, 'source_dataset': 'da_tl_cn', 'station_id': sid,
                       'country': 'China', 'source_member': name, 'layout': 'event_segments',
                       'timezone': 'unknown', 'q_unit': 'unknown', 'p_unit': 'unknown', 't_unit': 'unknown',
                       'water_level_output_unit': 'not_available', 'units_status': 'unverified'}
            stations.append(station)
            reader = iter(excel_rows(z.read(name)))
            assert next(reader) == {'A': 'time', 'B': 'Q', 'C': 'P', 'D': 'T', 'E': 'events'}
            def mapped_cn():
                for i, row in enumerate(reader, 2):
                    yield {'station_uid': station['station_uid'], 'source_dataset': 'da_tl_cn',
                           'source_row': i, 'event_id': station['station_uid'] + ':' + str(row.get('E')),
                           'time_source': (datetime(1899, 12, 30) + timedelta(days=row['A'])).isoformat(),
                           'time_utc': '', 'q_value': number(row.get('B')), 'p_value': number(row.get('C')),
                           'p_gapfilled_value': None, 't_value': number(row.get('D')), 'water_level_m': None,
                           'precipitation_uncovered_pct': None}
            reports.append(write_station(out / ('CN_' + sid + '.csv.gz'), station, mapped_cn()))
    assert len({s['station_uid'] for s in stations}) == len(stations)
    catalog = root / 'data/catalog'
    with (catalog / 'stations_unified_v1.csv').open('x', newline='') as f:
        fields = list(dict.fromkeys(k for s in stations for k in s))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stations)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'catalog_stations': len(stations),
              'pilot_stations': len(reports), 'reports': reports,
              'rules': ['Raw archives unchanged; each inspected ZIP member read through EOF with CRC checking.',
                        'No outlier deletion, imputation, normalization, flood threshold or model training.',
                        'Missing/nonfinite/unparseable selected numeric values become empty with flags; negatives retained and flagged.',
                        'Original and provider-gapfilled precipitation remain separate.',
                        'China time zone and Q/P/T units remain unknown; no assumed UTC or Kelvin conversion.',
                        'Rows and duplicates retained; flags support later decisions.',
                        'German first three lexical station IDs are a processing pilot, not a representative scientific sample.',
                        'Source full-period hydrologic summaries in station catalog must not be reused as model predictors without leakage review.']}
    with (catalog / 'standardization_pilot_v1.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
