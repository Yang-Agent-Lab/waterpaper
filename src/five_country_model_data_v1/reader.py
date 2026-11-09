"""Bounded source adapters and explicit hindcast/operational sample construction."""
import csv
import gzip
import hashlib
import io
import json
import math
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REGISTRY = Path('./external-home/multicountry_QC_fix_1fa15b9/results/experiments/multicountry_harmonization_full_1fa15b9_20261005/artifacts/registry.jsonl')
LEGACY = Path('./external-home/data_org_fixed_export_173d0b6/prepared')
OLD = {'da_tl_cn', 'camels_de_1h', 'lamah_ice'}
ROLES = {'fit', 'select'}


def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) and x >= 0 else math.nan
    except (TypeError, ValueError):
        return math.nan


def stamp(v):
    if isinstance(v, datetime):
        return v
    value = str(v).replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(value.replace('/','-').replace('.','-'),fmt)
            except ValueError:
                pass
        raise


def truth(v):
    return str(v).lower() in ('true', '1')


def csv_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


class Reader:
    def __init__(self, manifest, legacy_dir=LEGACY, role_policy=None, prepared_dir=None):
        self.sources = {s['dataset_key']: s for s in json.loads(Path(manifest).read_text())['sources']}
        policy_path=Path(role_policy) if role_policy else Path(manifest).with_name('five_country_model_data_roles_v1.json')
        self.policy=json.loads(policy_path.read_text())
        self.legacy_dir = Path(legacy_dir)
        self.role_bounds={}
        for group in self.policy['new_source_roles']:
            for key in group['datasets']:
                self.role_bounds[key]={}
                for period in group['periods']:
                    bounds=[]
                    for field in ('start','end_exclusive'):
                        value=period[field]
                        if value and group['clock']=='JST':
                            value=stamp(value).replace(tzinfo=timezone(timedelta(hours=9))).isoformat()
                        bounds.append(value)
                    self.role_bounds[key][period['role']]=bounds
        self.access = []
        self.access.append(str(policy_path))
        self._us_crosswalk=None
        self._legacy_inputs={}
        self.prepared_dir=Path(prepared_dir) if prepared_dir else None
        self._prepared={}
        self._prepared_manifest=None
        self.sample_rejections={}

    def entry(self, key):
        s = self.sources[key]
        return Path(s.get('prepared_entry', s['server_uri']).removeprefix('teacherbox:'))

    def canonical(self, key, station):
        if key == 'camelsh_us_v6':
            return 'USGS:' + station.zfill(8)
        if key == 'waterbench_iowa_v2_7087806':
            path = self.entry(key).parent / 'existing_usgs_overlap.json'
            if self._us_crosswalk is None:
                self.access.append(str(path))
                self._us_crosswalk=json.loads(path.read_text())['crosswalk']
            return 'USGS:' + self._us_crosswalk[station]['usgs_id']
        if key == 'jp_jma_maebashi_mlit_kurihashi_2015_2024':
            return 'MLIT:303031283301030'
        if key == 'japan_mlit_tone_existing':
            return 'MLIT:' + station.split(':')[0]
        return key + ':' + station

    def record(self, key, station, time, p, q, *, role='unassigned', event='continuous',
               quality=None, p_ok=True, q_ok=True, interval=60, label='unknown',
               source_row=None, support='unknown', available_at=None):
        s = self.sources[key]
        t = stamp(time)
        end = t + timedelta(minutes=interval) if label == 'start' else t
        p, q = number(p), number(q)
        return dict(source=key, canonical_station=self.canonical(key, station), station=station,
                    country=s['country_code'], role=role, event=event,
                    time=t.isoformat(), interval_end=end.isoformat(), interval_minutes=interval,
                    interval_label=label, timezone=s['time_zone'], available_at=available_at,
                    availability_state='actual_release_unknown' if available_at is None else 'verified',
                    q=q, p=p, q_mask=bool(q_ok and math.isfinite(q)),
                    p_mask=bool(p_ok and math.isfinite(p)), quality=quality or {},
                    p_spatial_support=support, p_origin=s['p_origin'], q_origin=s['q_origin'],
                    source_row=source_row)

    def read(self, key, station, *, start=None, end=None, role='unassigned', event=None, limit=256):
        """Yield native records. Legacy is physically separated; no raw mixed split opens."""
        if limit <= 0:
            raise ValueError('positive bounded limit required')
        if key in OLD:
            if role not in ROLES:
                raise PermissionError('explicit legacy non-test role required')
            records = self._legacy(key, station, role, event)
        else:
            if key in self.role_bounds and role=='unassigned':
                raise PermissionError('explicit allocated fit/select role required')
            if role != 'unassigned':
                bounds = self.role_bounds.get(key,{})
                if role not in self.policy['role_read_policy']['preparation_real_samples'] or role not in bounds:
                    raise PermissionError('new source role observation access not authorized')
                lower,upper = bounds[role]
                if lower and (start is None or stamp(start)<stamp(lower)):
                    start=lower
                upper_inclusive=stamp(upper)-timedelta(hours=1)
                if end is None or stamp(end)>upper_inclusive:
                    end=upper_inclusive
                if start and stamp(start)>stamp(end):
                    raise PermissionError('requested slice outside allowed role')
                if start and lower and stamp(start)<stamp(lower):
                    raise PermissionError('requested start outside allowed role')
            records = self._new(key, station, start, end, limit, event)
        count = 0
        for r in records:
            r['role']=role
            r['role_bounds']=self.role_bounds.get(key,{}).get(role)
            if event is not None and r['event'] != event:
                continue
            t = stamp(r['interval_end'])
            if start is not None and t < stamp(start):
                continue
            if end is not None and t > stamp(end):
                continue
            yield r
            count += 1
            if count >= limit:
                break

    def iter_samples(self, key, station, *, role, route, history_hours,
                     horizons_hours=(1,3,6,12), start=None, end=None, event=None,
                     max_samples=32, record_limit=4096, technical_only=False):
        """Bounded lazy station batch. Caller chooses history and information route."""
        if route not in ('pq','q_only') or max_samples<1:
            raise ValueError('explicit pq/q_only route and positive max_samples required')
        horizons=tuple(horizons_hours)
        if not horizons or any(h<1 or int(h)!=h for h in horizons):
            raise ValueError('positive integer horizons required')
        prepared=None
        if not technical_only and key not in OLD:
            prepared=self.prepared_station(key,station)
        records=list(self.read(key,station,role=role,start=start,end=end,event=event,limit=record_limit))
        groups={}
        for r in records:
            groups.setdefault(r['event'],[]).append(r)
        count=0
        for group in groups.values():
            for r in group:
                try:
                    sample=build_sample(group,r['interval_end'],history_hours=history_hours,
                                        forecast_hours=max(horizons))
                except ValueError:
                    continue
                if route=='pq' and any(not mask[0] for mask in sample['mask']):
                    continue
                if route=='q_only':
                    for x,mask in zip(sample['x'],sample['mask']):
                        x[0]=math.nan;mask[0]=False
                sample['route']=route
                sample['window_max_targets']={str(h):max(sample['y_q'][:h]) for h in horizons}
                sample['sample_id']=hashlib.sha256(json.dumps([sample['identity'],sample['origin'],
                    history_hours,horizons,route],ensure_ascii=False).encode()).hexdigest()
                sample['boundary_event_integrity']='inherited_legacy_roles' if key in OLD else 'pending_continuous_event_boundary_identification'
                if prepared:
                    from .prepare import qualify_sample
                    reason=qualify_sample(sample,prepared,route)
                    if reason:
                        self.sample_rejections[reason]=self.sample_rejections.get(reason,0)+1
                        continue
                elif key in OLD:
                    scale=self._load_legacy(role)[1][key+':'+station]
                    sample['x_normalized']=[[x[0]/scale['pscale'],x[1]/scale['qscale']] for x in sample['x']]
                    sample['window_max_targets_normalized']={h:v/scale['qscale'] for h,v in sample['window_max_targets'].items()}
                    sample['scale_source']='existing_legacy_fit_scale_no_new_fit'
                yield sample
                count+=1
                if count>=max_samples:
                    return

    def prepared_station(self,key,station):
        from .prepare import station_name,input_identity
        cache_key=(key,station)
        if cache_key not in self._prepared:
            if self.prepared_dir is None:
                raise PermissionError('prepared_dir required for model samples; technical_only is not training data')
            path=self.prepared_dir/'stations'/station_name(key,station)
            self.access.append(str(path))
            if self._prepared_manifest is None:
                manifest_path=self.prepared_dir/'manifest.json'
                self.access.append(str(manifest_path))
                self._prepared_manifest=json.loads(manifest_path.read_text())
            entry=next((r for r in self._prepared_manifest['stations'] if r['source']==key and r['station']==station),None)
            if entry is None or entry['prepared_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('prepared station artifact hash differs from manifest')
            with gzip.open(path,'rt') as f:table=json.load(f)
            policy_sha=hashlib.sha256(Path(self.access[0]).read_bytes()).hexdigest()
            row={'source':key,'station':station,'locator':{'member':'Hourly2/'+station.zfill(8)+'_hourly.nc'}}
            if table['state']!='complete' or table['role_policy_sha256']!=policy_sha or table['input_identity']!=input_identity(self,row):
                raise ValueError('prepared station missing qualification or identity changed')
            self._prepared[cache_key]=table
        return self._prepared[cache_key]

    def _load_legacy(self, role):
        import numpy as np
        if role in self._legacy_inputs:
            return self._legacy_inputs[role]
        folder = self.legacy_dir
        metadata = folder / (role + '_rows.json')
        array_path = folder / (role + '.npz')
        manifest_path = folder / 'manifest.json'
        self.access.extend(map(str, (metadata, array_path, manifest_path, folder / 'scales.json')))
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!='e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6':
            raise ValueError('legacy manifest identity mismatch')
        export=folder.parent/'manifest.json'
        if hashlib.sha256(export.read_bytes()).hexdigest()!='e8355e8240975589b0da5081dc471f40f387da2251ad0a05ee6e242e0f42ca35':
            raise ValueError('legacy export identity mismatch')
        expected={r['relative']:r['sha256'] for r in json.loads(export.read_text())['files']}
        for file in (metadata,folder/'scales.json'):
            if hashlib.sha256(file.read_bytes()).hexdigest()!=expected['prepared/'+file.name]:
                raise ValueError('legacy role metadata/scale identity mismatch')
        self.access.append(str(export))
        digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
        if digest != manifest['derived_files'][array_path.name]:
            raise ValueError('legacy array identity mismatch')
        rows = json.loads(metadata.read_text())
        if any(r['role'] != role for r in rows):
            raise ValueError('legacy role metadata mismatch')
        scales = json.loads((folder / 'scales.json').read_text())
        with np.load(array_path,allow_pickle=False) as z:
            result=rows,scales,z['x'],z['y']
        self._legacy_inputs[role]=result
        return result

    def _legacy(self, key, station, role, event):
        import numpy as np
        rows,scales,xx,yy=self._load_legacy(role)
        uid = key + ':' + station
        sc = scales[uid]
        merged = {}
        for i, r in enumerate(rows):
                if r['station_uid'] != uid or (event is not None and r['event'] != event):
                    continue
                x = xx[i] * np.array([sc['pscale'], sc['qscale']])
                y = yy[i] * sc['qscale']
                for j, (t, source_row) in enumerate(zip(r['source_times'], r['source_rows'])):
                    identity = (r['event'], source_row)
                    p = float(x[j, 0]) if j < len(x) else math.nan
                    q = float(x[j, 1]) if j < len(x) else float(y[j-len(x)])
                    record = self.record(key, station, t, p, q, role=role, event=r['event'],
                        label='start' if key == 'lamah_ice' else 'unknown', source_row=source_row,
                        support='catchment_mean' if key != 'da_tl_cn' else 'source_support_unknown',
                        quality={'state': 'accepted_prepared_numeric_only', 'raw_flags': 'not_retained',
                                 'derived_from_prepared': True,
                                 'overlap_group': r['temporal_overlap_component'],
                                 'physical_values': 'float32_saved_scale_reconstruction'})
                    old = merged.get(identity)
                    if old and not math.isclose(old['q'], q, rel_tol=1e-5, abs_tol=1e-4):
                        raise ValueError('conflicting legacy Q identity')
                    if old and old['p_mask']:
                        record['p'], record['p_mask'] = old['p'], True
                    merged[identity] = record
        yield from sorted(merged.values(), key=lambda r: (r['event'], r['interval_end']))

    def _new(self, key, station, start, end, limit, event):
        path = self.entry(key)
        if key == 'waterbench_iowa_v2_7087806':
            path = path / (station + '.csv.gz')
            self.access.append(str(path))
            for i, r in enumerate(csv_rows(path)):
                if start and stamp(r['datetime_native']) < stamp(start):
                    continue
                if end and stamp(r['datetime_native']) > stamp(end):
                    break
                yield self.record(key, station, r['datetime_native'], r['precipitation_mm_per_hour'],
                    r['discharge_m3_per_s'], p_ok=truth(r['p_observed_available']),
                    q_ok=truth(r['q_observed_available']), support='catchment_mean', source_row=i,
                    quality={k: r[k] for k in ('p_observed_available', 'q_observed_available')})
                if end and stamp(r['datetime_native']) >= stamp(end):
                    break
        elif key == 'jp_jma_maebashi_mlit_kurihashi_2015_2024':
            if station != '303031283301030':
                raise ValueError('paired JP route only maps Kurihashi')
            self.access.append(str(path))
            for i, r in enumerate(csv_rows(path)):
                if start and stamp(r['time_jst']) < stamp(start):
                    continue
                if end and stamp(r['time_jst']) > stamp(end):
                    break
                yield self.record(key, station, r['time_jst'], r['P_mm'], r['Q_m3_s'],
                    label='end', p_ok=truth(r['P_usable']) and r['P_quality']=='8',
                    q_ok=not truth(r['Q_conflict']), support='point:Maebashi:s47624', source_row=i,
                    quality={k: r[k] for k in ('P_quality','P_usable','Q_conflict','Q_acquisition_state')})
                if end and stamp(r['time_jst']) >= stamp(end):
                    break
        elif key == 'japan_mlit_tone_existing':
            self.access.append(str(path))
            native_id = station if ':' in station else station + ':0202'
            clauses, args = ['id=?'], [native_id]
            for bound, op in ((start, '>='), (end, '<=')):
                if bound:
                    clauses.append('t'+op+'?'); args.append(stamp(bound).isoformat())
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as c:
                for r in c.execute('SELECT t,q,conflict,dups FROM obs WHERE '+ ' AND '.join(clauses)+' ORDER BY t LIMIT ?', args+[limit]):
                    yield self.record(key, station, r[0], math.nan, r[1], label='end',
                        q_ok=not r[2], quality={'conflict':r[2], 'dups':r[3]}, source_row=r[0])
        elif key == 'camelsh_us_v6':
            import h5py
            member = 'Hourly2/' + station.zfill(8) + '_hourly.nc'
            self.access.append(str(path)+'!'+member)
            with zipfile.ZipFile(path) as z, h5py.File(io.BytesIO(z.read(member))) as f:
                units = f['time'].attrs['units'].decode()
                if not units.startswith('hours since ') or f['streamflow'].attrs['units'].decode() != 'm3 s-1':
                    raise ValueError('unexpected CAMELSH time/Q units')
                base = stamp(units.removeprefix('hours since '))
                first = max(0, math.ceil((stamp(start)-base).total_seconds()/3600)) if start else 0
                last = min(len(f['time']), first+limit)
                if end:
                    last = min(last, math.floor((stamp(end)-base).total_seconds()/3600)+1)
                from .prepare import h5_missing
                missing=h5_missing(f['streamflow'])
                for i, (hour, q) in enumerate(zip(f['time'][first:last], f['streamflow'][first:last]), first):
                    if any(math.isfinite(v) and float(q)==v for v in missing):q=math.nan
                    yield self.record(key, station, base+timedelta(hours=float(hour)), math.nan, q,
                        source_row=i, quality={'state':'source_numeric_only', 'H':'unverified_not_read'})
        else:
            yield from self._auxiliary(key, station, limit, event)

    def _auxiliary(self, key, station, limit, event):
        path = self.entry(key)
        if key == 'gxsw_recent_capture':
            for name, pfield, qfield in (('river_hourly.csv',None,'流量(m3/s)'), ('rain_hourly.csv','累计雨量(mm)',None), ('reservoir_hourly.csv',None,None)):
                file = path/name; self.access.append(str(file))
                for i, r in enumerate(csv_rows(file)):
                    if station != '*' and r.get('站名') != station:
                        continue
                    yield self.record(key,r.get('站名','unknown'),r['时间'].replace('/','-'),
                        r.get(pfield) if pfield else None, r.get(qfield) if qfield else None,
                        interval=None, source_row=f'{name}:{i}', p_ok=False,
                        quality={'state':'auxiliary_snapshot', 'fetched_at':r.get('fetched_at'),
                                 'H':r.get('水位(m)'), 'raw':r})
        elif key in ('hei_luoma_daily', 'geodata_poyang_samples'):
            import openpyxl
            for file in sorted(path.rglob('*.xlsx')):
                if station != '*' and station not in file.name:
                    continue
                self.access.append(str(file))
                w = openpyxl.load_workbook(file, read_only=True, data_only=True)
                try:
                    for sheet in w:
                        rows = sheet.iter_rows(values_only=True)
                        headers = next(rows)
                        if key == 'hei_luoma_daily':
                            next(rows)
                            for i, r in enumerate(rows):
                                for col in range(0, len(headers), 3):
                                    eid = f'{file.name}:{sheet.title}:{headers[col]}'
                                    if r[col] is None or (event and event != eid):
                                        continue
                                    t = r[col] if isinstance(r[col],datetime) else stamp(str(r[col]).replace('.','-'))
                                    yield self.record(key,file.stem,t,r[col+2],r[col+1],event=eid,
                                        interval=1440, source_row=f'{sheet.title}:{i+3}:{col}', p_ok=False,
                                        quality={'state':'auxiliary_daily','P_semantics':'pending','headers':list(headers)})
                        else:
                            for i, r in enumerate(rows):
                                four = len(headers)==4
                                yield self.record(key,str(r[0]),r[2] if four else r[1],None,r[3] if four else r[2],
                                    interval=60,source_row=f'{file.name}:{sheet.title}:{i+2}',
                                    quality={'state':'auxiliary_sample_provenance_pending'})
                finally:
                    w.close()
        else:
            raise KeyError(key)


def build_sample(records, origin, *, history_hours, forecast_hours, mode='hindcast'):
    """Hourly Q supervision, historical P/Q with masks; no future P in returned sample."""
    if mode not in ('hindcast','operational') or history_hours < 1 or forecast_hours < 1:
        raise ValueError('invalid sample contract')
    rows = list(records)
    if not rows:
        raise ValueError('empty records')
    identity = {(r['source'],r['canonical_station'],r['event'],r['role']) for r in rows}
    if len(identity)!=1 or any(r['interval_minutes']!=60 for r in rows):
        raise ValueError('cannot cross source/station/event/role or use auxiliary daily/snapshots')
    if rows[0]['role'] not in ROLES:
        raise PermissionError('model sample role not permitted for preparation')
    if rows[0]['source'] in ('geodata_poyang_samples','gxsw_recent_capture'):
        raise ValueError('auxiliary source cannot create model sample')
    t = stamp(origin)
    by_time = {}
    for r in rows:
        time = stamp(r['interval_end'])
        if time in by_time:
            raise ValueError('duplicate interval identity')
        by_time[time] = r
    times = [t+timedelta(hours=i) for i in range(1-history_hours,forecast_hours+1)]
    if any(time not in by_time for time in times):
        raise ValueError('incomplete contiguous window; never pad gaps')
    history = [by_time[time] for time in times[:history_hours]]
    future = [by_time[time] for time in times[history_hours:]]
    for r in history+future:
        bounds=r.get('role_bounds')
        if bounds:
            lower,upper=bounds
            current=stamp(r['interval_end'])
            if (lower and current<stamp(lower)) or (upper and current>=stamp(upper)):
                raise ValueError('sample crosses allocated role boundary')
    if any(not r['q_mask'] for r in history+future):
        raise ValueError('unqualified/missing Q')
    if mode == 'operational':
        for r in history:
            if r['source']=='lamah_ice':
                raise ValueError('reanalysis route is hindcast only')
            if r['available_at'] is None or stamp(r['available_at']) > t:
                raise ValueError('actual issue-time availability unverified or future')
            if r['interval_label']=='unknown' or r['timezone']=='unknown':
                raise ValueError('operational clock semantics unverified')
    return dict(identity=list(identity)[0],origin=t.isoformat(),history_hours=history_hours,
                forecast_hours=forecast_hours,mode=mode,
                x=[[r['p'] if r['p_mask'] else math.nan,r['q']] for r in history],
                mask=[[r['p_mask'],r['q_mask']] for r in history],
                y_q=[r['q'] for r in future],input_times=[r['interval_end'] for r in history],
                window_max_targets={str(h):max(r['q'] for r in future[:h]) for h in (1,3,6,12) if h<=forecast_hours},
                target_times=[r['interval_end'] for r in future],
                quality=[r['quality'] for r in history],availability=[r['available_at'] for r in history],
                p_spatial_support=history[0]['p_spatial_support'])


ROUTE_PRIORITY = {'Q_only': {'USGS': ['camelsh_us_v6','waterbench_iowa_v2_7087806'],
                  'MLIT:303031283301030': ['jp_jma_maebashi_mlit_kurihashi_2015_2024']},
                  'PQ': {'USGS':['waterbench_iowa_v2_7087806'],
                  'MLIT:303031283301030':['jp_jma_maebashi_mlit_kurihashi_2015_2024']}}
# Native clocks are never merged; priority chooses a route, not extra independent Q.
