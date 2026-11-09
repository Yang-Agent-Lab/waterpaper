"""Reusable station/role locators from metadata, never all-source observation scans."""
import json
import sqlite3
import zipfile
from collections import defaultdict
from .reader import REGISTRY, OLD


def station_index(reader):
    index = []
    permissions = defaultdict(dict)
    event_roles=defaultdict(set)
    for role in ('fit','select','calibration','development'):
        path = reader.legacy_dir/(role+'_rows.json')
        reader.access.append(str(path))
        rows = json.loads(path.read_text())
        for r in rows:
            if r['role'] != role:
                raise ValueError('role metadata mismatch')
            uid = r['station_uid']
            if r['source_event_id']:
                event_roles[(uid,r['source_event_id'])].add(role)
            entry = permissions[uid].setdefault(role,dict(windows=0,events=set(),
                time_start=r['source_times'][0],time_end=r['source_times'][-1],metadata=str(path)))
            entry['windows'] += 1
            entry['events'].add(r['event'])
            entry['time_start'] = min(entry['time_start'],r['source_times'][0])
            entry['time_end'] = max(entry['time_end'],r['source_times'][-1])
    conflicts={str(k):sorted(v) for k,v in event_roles.items() if len(v)>1}
    if conflicts:
        raise ValueError('known source event crosses legacy roles: '+str(conflicts))
    reader.access.append(str(REGISTRY))
    with REGISTRY.open() as f:
        for line in f:
            r = json.loads(line)
            key = r.get('station_uid','').split(':')[0]
            if key not in OLD:
                continue
            roles = permissions.get(r['station_uid'],{})
            for v in roles.values():
                v['events']=sorted(v['events'])
            index.append(dict(source=key,station=r['station_id'],canonical_station=r['station_uid'],
                locator={'adapter':'legacy_prepared','directory':str(reader.legacy_dir)},
                status='legacy_role_available' if roles else 'no_saved_non_test_windows',roles=roles,
                raw_observations_read=False,registered_raw_locator=r['observation_file'],
                history_support='saved_non_test_contiguous_coverage_only; no prefix padding',
                quality='accepted_prepared_numeric_only; original quality flags unavailable'))
    for key in reader.sources:
        if key in OLD:
            continue
        p = reader.entry(key)
        if key=='camelsh_us_v6':
            reader.access.append(str(p)+'!central_directory')
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith('_hourly.nc'):
                        station=n.split('/')[-1].split('_')[0]
                        index.append(dict(source=key,station=station,canonical_station=reader.canonical(key,station),
                            locator={'archive':str(p),'member':n},status='Q_only_unassigned',roles={},
                            timezone='unknown',P='missing_NaN_mask',H='unverified_not_model_available'))
        elif key=='waterbench_iowa_v2_7087806':
            for file in sorted(p.glob('*.csv.gz')):
                station=file.name.split('.')[0]
                index.append(dict(source=key,station=station,canonical_station=reader.canonical(key,station),
                    locator={'csv':str(file)},status='PQ_unassigned',roles={},timezone='unknown'))
        elif key=='japan_mlit_tone_existing':
            reader.access.append(str(p)+'!station_time_metadata')
            with sqlite3.connect(f'file:{p}?mode=ro',uri=True) as c:
                for station,start,end in c.execute('SELECT id,min(t),max(t) FROM obs GROUP BY id'):
                    index.append(dict(source=key,station=station,canonical_station=reader.canonical(key,station),
                        locator={'sqlite':str(p),'table':'obs','id':station},status='Q_only_unassigned',
                        time_start=start,time_end=end,roles={}))
        elif key=='jp_jma_maebashi_mlit_kurihashi_2015_2024':
            index.append(dict(source=key,station='303031283301030',canonical_station='MLIT:303031283301030',
                locator={'csv':str(p)},status='point_PQ_unassigned',roles={}))
        else:
            files=sorted(p.rglob('*.xlsx')) if key!='gxsw_recent_capture' else sorted(p.glob('*.csv'))
            for file in files:
                index.append(dict(source=key,station=file.stem,canonical_station='unresolved',
                    locator={'file':str(file),'selector':'workbook substring or *'},status='auxiliary_only',roles={}))
    for r in index:
        country=reader.sources[r['source']]['country_code']
        if r['source'] in reader.role_bounds:
            r['roles']={role:dict(source_clock_start=b[0],source_clock_end_exclusive=b[1],
                access='allowed_fit_select_only' if role in ('fit','select') else 'metadata_only',
                availability='actual_release_unknown') for role,b in reader.role_bounds[r['source']].items()}
            r['status']=r['status'].replace('_unassigned','_allocated')
            r['boundary_event_integrity']='pending; time partition does not prove separate storms'
    camelsh={r['canonical_station'] for r in index if r['source']=='camelsh_us_v6'}
    for r in index:
        r['q_only_pool_eligible']=not (r['source']=='waterbench_iowa_v2_7087806' and r['canonical_station'] in camelsh)
        if r['canonical_station']=='MLIT:303031283301030':
            r['q_only_pool_eligible']=r['source']==reader.policy['identity_policy']['MLIT:303031283301030']['primary_Q_source_both_routes']
        r['pq_pool_eligible']=r['source'] in OLD or r['source'] in ('waterbench_iowa_v2_7087806','jp_jma_maebashi_mlit_kurihashi_2015_2024')
        if r['status']=='auxiliary_only':
            r['q_only_pool_eligible']=r['pq_pool_eligible']=False
        r['pool_composition']='mutually_exclusive_single_route_dataset; do_not_concatenate_PQ_and_Q_only_pools'
    return index
