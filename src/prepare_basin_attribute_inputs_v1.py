"""Prepare retrospective basin metadata only; never open observation/window files."""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AUDIT_COMMIT = 'feeec8819c6160b408c8e3a66665ecf445316344'
MANIFEST = './external-home/pooled_esdf_d781a3f_v2/results/experiments/esdf_projection_full_d781a3f_20261005/artifacts/model_manifest.json'
DE_ZIP = './data/raw/camels_de_1h/CAMELS-DE-1h.zip'
ICE_MEMBER = 'lamah_ice_hourly/A_basins_total_upstrm/1_attributes/Catchment_attributes.csv'


def number(value):
    if value is None or str(value).strip().lower() in ('', 'na', 'nan', 'null'):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite metadata value')
    return result


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def read_remote():
    program = f'''import csv,io,json,zipfile
z=zipfile.ZipFile({DE_ZIP!r})
tables={{}}
for kind in ['topographic','climatic']:
 member='CAMELS-DE-1h/CAMELS_DE_1h_'+kind+'_attributes.csv'
 tables[kind]={{'member':member,'rows':list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))}}
with open({MANIFEST!r}) as f: manifest=json.load(f)
if manifest.get('scope') != 'projected_train_validation_only' or manifest.get('test_access') is not False or manifest.get('cache_access') is not False:
 raise ValueError('Unexpected model manifest access scope')
stations=[{{k:s[k] for k in ['station_uid','country','counts','coverage_state','scales'] if k in s}} for s in manifest['stations']]
print(json.dumps({{'tables':tables,'manifest_stations':stations}},allow_nan=False))
'''
    result = subprocess.run(
        ['ssh', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o',
         'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o',
         'ServerAliveCountMax=2', 'teacherbox', 'python3', '-'],
        input=program, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Single bounded metadata SSH failed: ' + result.stderr.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory; existing outputs are never overwritten')
    cn = json.loads((ROOT / 'data/catalog/china_basin_attributes_paper_20261006.json').read_text())
    audit = json.loads(subprocess.check_output(
        ['git', 'show', AUDIT_COMMIT + ':data/catalog/multicountry_sources_audit_20261005.json'],
        cwd=ROOT, text=True))['iceland']
    with (ROOT / 'data/catalog/stations_unified_v2.csv').open() as handle:
        de_index = [r for r in csv.DictReader(handle) if r['country'] == 'Germany']
    remote = read_remote()
    tables = remote['tables']
    by_id = {}
    for kind, table in tables.items():
        rows = table['rows']
        by_id[kind] = {r['gauge_id']: r for r in rows}
        if len(by_id[kind]) != len(rows):
            raise ValueError('Duplicate German gauge_id in ' + kind)
        if set(by_id[kind]) != {r['station_id'] for r in de_index}:
            raise ValueError('German source/local station ID mismatch: ' + kind)
    stations = remote['manifest_stations']
    if isinstance(stations, dict):
        manifest_by_uid = stations
    elif isinstance(stations, list):
        manifest_by_uid = {r['station_uid']: r for r in stations}
        if len(manifest_by_uid) != len(stations):
            raise ValueError('Duplicate model manifest station_uid')
    else:
        raise ValueError('Unknown manifest stations structure')
    rows = []

    def add(uid, country, area, precip, unit, period, source, original_p, model_uid=None):
        area, precip = number(area), number(precip)
        if (area is not None and area <= 0) or (precip is not None and precip < 0):
            raise ValueError('Invalid background metadata: ' + uid)
        model_uid = model_uid or uid
        matched = model_uid in manifest_by_uid
        metadata = manifest_by_uid.get(model_uid)
        counts = metadata['counts'] if matched else None
        scales = metadata.get('scales') if matched else None
        scale_values = [number(scales.get(k)) for k in ['qscale', 'pscale']] if isinstance(scales, dict) else [None, None]
        has_scale = all(v is not None and v > 0 for v in scale_values)
        if counts is not None and counts['train'] > 0 and not has_scale:
            raise ValueError('Training station lacks positive finite qscale/pscale: ' + uid)
        country_matches = matched and metadata.get('country') == country
        rows.append(dict(station_uid=uid, station_id=uid.split(':', 1)[1], country=country,
                         area_km2=area, mean_annual_precip_mm_equiv=precip,
                         original_precip_value=number(original_p), original_precip_unit=unit,
                         precipitation_reference_period=period, source=source,
                         model_manifest_status=('MATCHED' if country_matches else
                                                'COUNTRY_MISMATCH' if matched else 'NOT_IN_MANIFEST'),
                         model_scale_status='AVAILABLE' if has_scale else 'NO_MODEL_SCALE',
                         model_station_uid=model_uid if matched else None, counts=counts,
                         model_station_metadata=metadata))

    for r in cn['basins']:
        add(r['station_uid'], 'China', r['area_km2'], r['annual_mean_precipitation_mm'],
            'mm/year', 'unknown', cn['source_url'] + ' Table 1', r['annual_mean_precipitation_mm'])
    for r in de_index:
        sid = r['station_id']; t = by_id['topographic'][sid]; c = by_id['climatic'][sid]
        p = number(c['p_mean'])
        add(r['station_uid'], 'Germany', t['area'], None if p is None else p * 8766,
            'mm/hr', '2001-01-01/2024-12-31', DE_ZIP, p)
    ice_rows = audit['attributes'][ICE_MEMBER]['rows']
    ice_by_id = {r['id']: r for r in ice_rows}
    if len(ice_by_id) != len(ice_rows):
        raise ValueError('Duplicate Iceland attributes id')
    ice_ids = audit['station_ids']['gauges']
    if len(set(ice_ids)) != len(ice_ids):
        raise ValueError('Duplicate Iceland hourly Q id')
    ice_model_uids = {}
    for uid, metadata in manifest_by_uid.items():
        if metadata.get('country') != 'Iceland':
            continue
        match = re.fullmatch(r'[^:]+:(?:ID_)?([0-9]+)', uid)
        if not match or match[1] in ice_model_uids:
            raise ValueError('Unknown/duplicate Iceland manifest station ID: ' + uid)
        ice_model_uids[match[1]] = uid
    for sid in sorted(ice_ids, key=int):
        r = ice_by_id[sid]; p = number(r['p_mean_ERA5L'])
        add('lamah_ice:' + sid, 'Iceland', r['area_calc'], None if p is None else p * 365.25,
            'mm/day', '1989-10-01/2009-09-30', AUDIT_COMMIT + ':' + ICE_MEMBER, p,
            model_uid=ice_model_uids.get(sid, 'NOT_IN_MANIFEST:' + sid))
    if len(rows) != 1696 or len({r['station_uid'] for r in rows}) != len(rows):
        raise ValueError('Expected 1696 unique candidate source stations')
    matched_uids = [r['model_station_uid'] for r in rows if r['model_station_uid'] is not None]
    extra_manifest = sorted(set(manifest_by_uid) - set(matched_uids))
    country_mismatches = [r['station_uid'] for r in rows if r['model_manifest_status'] == 'COUNTRY_MISMATCH']
    ready = (all(r['area_km2'] is not None and r['mean_annual_precip_mm_equiv'] is not None
                 and r['model_manifest_status'] == 'MATCHED' for r in rows)
             and len(manifest_by_uid) == 1696 and len(set(matched_uids)) == 1696 and not extra_manifest)
    fit_rows = [r for r in rows if r['counts'] is not None and r['counts']['train'] > 0]
    if not fit_rows:
        raise ValueError('No training-qualified basins for attribute transforms')
    normalization = {'fit_scope': 'counts.train>0 unique basins, equal basin weight',
                     'fit_station_uids': [r['station_uid'] for r in fit_rows], 'fields': {}}
    for field in (['area_km2', 'mean_annual_precip_mm_equiv'] if ready else []):
        values = [math.log1p(r[field]) for r in fit_rows]
        mean, std = statistics.fmean(values), statistics.pstdev(values)
        if std == 0:
            raise ValueError('Constant transformed attribute: ' + field)
        normalization['fields'][field] = {'transform': 'log1p', 'mean': mean, 'std': std,
                                           'std_definition': 'population', 'fit_basins': len(fit_rows)}
    for r in rows:
        r['model_features'] = ([(math.log1p(r[field]) - spec['mean']) / spec['std']
                               for field, spec in normalization['fields'].items()] if ready else None)
    fields = ['area', 'longterm_precip', 'elevation', 'slope', 'river_length', 'soil', 'landcover', 'coordinates']
    states = {'China': [0, 0, 2, 2, 0, 2, 2, 2],
              'Germany': [0, 0, 0, 2, 2, 1, 1, 0],
              'Iceland': [0, 0, 0, 0, 2, 1, 1, 0]}
    coverage = {country: {'candidate_stations': len(rr := [r for r in rows if r['country'] == country]),
                         'area_finite': sum(r['area_km2'] is not None for r in rr),
                         'precip_finite': sum(r['mean_annual_precip_mm_equiv'] is not None for r in rr),
                         'manifest_matched': sum(r['model_manifest_status'] == 'MATCHED' for r in rr)}
                for country in states}
    summary = dict(status='READY_RETROSPECTIVE_BACKGROUND' if ready else 'PARTIAL_METADATA_OR_MANIFEST',
                   total=len(rows), coverage=coverage,
                   de_area_metadata_different=sum(number(r['area']) != number(r['area_metadata'])
                                                  for r in by_id['topographic'].values()),
                   de_area_metadata_finite=sum(number(r['area_metadata']) is not None
                                               for r in by_id['topographic'].values()),
                   no_model_scale=[r['station_uid'] for r in rows if r['model_scale_status'] != 'AVAILABLE'],
                   missing_manifest=[r['station_uid'] for r in rows if r['model_manifest_status'] != 'MATCHED'],
                   extra_manifest=extra_manifest, country_mismatches=country_mismatches,
                   transform_fit_country_counts={c: sum(r['country'] == c for r in fit_rows) for c in states},
                   climatology_policy='Retrospective long-term background; periods/products differ, not forecast-time available climatology',
                   fitted_transforms=ready, transform_fit_basins=len(fit_rows) if ready else 0, training_run=False,
                   figure_rendering='PENDING_SERVER_RENDER')
    plot_data = dict(rows=rows, coverage=coverage, fields=fields, states=states,
                     state_labels=['已核实资料', '口径或覆盖待核实', '未取得'],
                     coverage_evidence='basin_attributes_mapping_20261006.md; no uncounted coverage percentages',
                     periods={'China': '基准期未知', 'Germany': '2001–2024', 'Iceland': '1989–2009（ERA5-Land）'})
    source = dict(china=cn, germany=dict(zip_path=DE_ZIP, **tables),
                  iceland=dict(audit_commit=AUDIT_COMMIT, member=ICE_MEMBER, rows=ice_rows, hourly_q_ids=ice_ids),
                  model_manifest_path=MANIFEST, model_manifest_stations=stations,
                  annualization={'China': 1, 'Germany': 8766, 'Iceland': 365.25})
    # Validate JSON before creating the unique output directory.
    for obj in [source, rows, summary, plot_data, normalization]:
        json.dumps(obj, allow_nan=False)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'source_metadata.json', source)
    write_json(args.output_dir / 'attribute_rows.json', rows)
    write_json(args.output_dir / 'summary.json', summary)
    write_json(args.output_dir / 'normalization.json', normalization)
    figures = args.output_dir / 'figures'; figures.mkdir()
    write_json(figures / 'basin_attribute_plot_data.json', plot_data)
    plot_source = ROOT / 'src/plot_basin_attribute_inputs_v1.py'
    script = figures / plot_source.name
    with script.open('xb') as handle:
        handle.write(plot_source.read_bytes())
    if not ready:
        raise SystemExit('PARTIAL: attributes or manifest association incomplete; no transforms or figures executed')
    # Figures are rendered separately on the server under the project policy.
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
