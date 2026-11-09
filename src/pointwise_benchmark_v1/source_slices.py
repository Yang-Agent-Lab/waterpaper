"""Read only planned fit/select slices, reusing each station's physical source."""
import io
import math
import zipfile
import numpy as np
from datetime import timedelta
from src.five_country_model_data_v1.prepare import hour,iso,h5_missing
from src.five_country_model_data_v1.reader import csv_rows,stamp,truth
from .coverage import merge


def batch_records(reader,source,station,plans):
    wanted={}
    for role in ('fit','select'):
        wanted[role]=merge([[p['origin_hour']-p['history_hours']+1,p['origin_hour']+p['forecast_hours']]
                           for p in plans if p['role']==role])
        lower,upper=reader.role_bounds[source][role]
        a=hour(lower) if lower else None;b=hour(upper)
        if any((a is not None and first<a) or last>=b for first,last in wanted[role]):
            raise PermissionError('planned observation slice crosses allocated role boundary')
    result={role:[] for role in wanted};cursors={role:0 for role in wanted}
    def wanted_role(t):
        for role,ranges in wanted.items():
            while cursors[role]<len(ranges) and ranges[cursors[role]][1]<t:cursors[role]+=1
            j=cursors[role]
            if j<len(ranges) and ranges[j][0]<=t<=ranges[j][1]:return role
        return None
    if source=='camelsh_us_v6':
        import h5py
        path=reader.entry(source);member='Hourly2/'+station.zfill(8)+'_hourly.nc'
        reader.access.append(str(path)+'!'+member)
        with zipfile.ZipFile(path) as z,h5py.File(io.BytesIO(z.read(member))) as f:
            units=f['time'].attrs['units'].decode()
            if not units.startswith('hours since ') or f['streamflow'].attrs['units'].decode()!='m3 s-1':
                raise ValueError('CAMELSH source units differ from accepted adapter')
            base=stamp(units.removeprefix('hours since '));base_hour=hour(base)
            # Read clock metadata first. Never infer Q indices from a hypothetical
            # gap-free axis: a clock gap could otherwise expose another role's Q.
            offsets=np.asarray(f['time'][:],dtype=float)
            if not np.isfinite(offsets).all() or not np.equal(offsets,np.floor(offsets)).all() or np.any(np.diff(offsets)<=0):
                raise ValueError('CAMELSH time metadata must be unique increasing integer hours')
            native_hours=offsets.astype(np.int64)+base_hour
            selections=[]
            for role,ranges in wanted.items():
                for a,b in ranges:
                    first=int(np.searchsorted(native_hours,a));last=int(np.searchsorted(native_hours,b,side='right'))
                    if not np.array_equal(native_hours[first:last],np.arange(a,b+1,dtype=np.int64)):
                        raise ValueError('planned CAMELSH slice is not fully supported by actual permitted hourly clock metadata')
                    lower,upper=reader.role_bounds[source][role]
                    if (lower and np.any(native_hours[first:last]<hour(lower))) or np.any(native_hours[first:last]>=hour(upper)):
                        raise PermissionError('actual CAMELSH clock indices cross a role boundary')
                    selections.append((role,first,last))
            missing=h5_missing(f['streamflow'])
            # Every numeric slice index above is already proven to belong to the
            # requested role and source-native hours before streamflow is read.
            for role,first,last in selections:
                for index,q in enumerate(f['streamflow'][first:last],first):
                    time=base+timedelta(hours=float(offsets[index]))
                    if any(math.isfinite(v) and float(q)==v for v in missing):q=math.nan
                    result[role].append(reader.record(source,station,time,math.nan,q,role=role,
                        source_row=index,quality={'state':'source_numeric_only','H':'unverified_not_read'}))
    elif source in ('waterbench_iowa_v2_7087806','jp_jma_maebashi_mlit_kurihashi_2015_2024'):
        pair=source=='jp_jma_maebashi_mlit_kurihashi_2015_2024'
        path=reader.entry(source) if pair else reader.entry(source)/(station+'.csv.gz')
        reader.access.append(str(path));last=max(b for ranges in wanted.values() for a,b in ranges);previous=None
        for index,raw in enumerate(csv_rows(path)):
            time=raw['time_jst'] if pair else raw['datetime_native'];t=hour(time)
            if previous is not None and t<=previous:raise ValueError('source clock duplicate or out of order')
            previous=t
            if t>last:break
            role=wanted_role(t)
            if role is not None:
                if pair:
                    record=reader.record(source,station,time,raw['P_mm'],raw['Q_m3_s'],role=role,label='end',
                        p_ok=truth(raw['P_usable']) and raw['P_quality']=='8',q_ok=not truth(raw['Q_conflict']),
                        support='point:Maebashi:s47624',source_row=index,
                        quality={k:raw[k] for k in ('P_quality','P_usable','Q_conflict','Q_acquisition_state')})
                else:
                    record=reader.record(source,station,time,raw['precipitation_mm_per_hour'],raw['discharge_m3_per_s'],role=role,
                        p_ok=truth(raw['p_observed_available']),q_ok=truth(raw['q_observed_available']),
                        support='catchment_mean',source_row=index,
                        quality={k:raw[k] for k in ('p_observed_available','q_observed_available')})
                result[role].append(record)
            if t==last:break
    elif source=='japan_mlit_tone_existing':
        for role,ranges in wanted.items():
            # Read-only SQL predicates exclude all other roles before numeric Q
            # is returned; each needed interval is queried once.
            for a,b in ranges:
                result[role].extend(reader.read(source,station,role=role,start=iso(a,'+09:00'),end=iso(b,'+09:00'),limit=b-a+1))
    else:raise ValueError('unrecognized new source')
    return result
