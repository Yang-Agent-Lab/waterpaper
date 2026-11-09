"""Metadata-only common sample/exposure plans, prior to any model comparison."""
from collections import defaultdict,Counter
import hashlib
import json
import numpy as np


def units(coverage,permission,route,history=24,future=24):
    result=[]
    for row in coverage['rows']:
        if row['source'] in ('camels_de_1h','lamah_ice'):continue
        if row['route']!=route or row['history_hours']!=history or not row['threshold_eligible']:continue
        for a,b in row['origin_ranges']:
            result.append(dict(source=row['source'],station=row['station'],canonical_station=row['canonical_station'],
                country=row['country'],role=row['role'],event=row['event'],start_origin=int(a),end_origin=int(b),
                grouping_unit='inherited_event' if row['source']=='da_tl_cn' else 'continuous_qualification_segment',
                source_group_id=row['event'] if row['source']=='da_tl_cn' else 'continuous:%s:%s'%(a,b)))
    for block in permission['blocks']:
        shift=block['interval_end_shift_hours'];a=block['start_native_hour']+shift+history-1;b=block['end_native_hour']+shift-future
        if a>b:continue
        result.append(dict(source=block['source'],station=block['station'],canonical_station=block['canonical_station'],
            country=block['country'],role=block['role'],event=block['event'],start_origin=a,end_origin=b,
            grouping_unit='existing_allocated_original_block',source_group_id=block['event']))
    # Priority has already been enforced by coverage's actual eligible pools.
    # It remains a station-wide alternative route, never an hourly source splice.
    identities={}
    for unit in result:
        key=(unit['country'],unit['canonical_station'])
        previous=identities.setdefault(key,unit['source'])
        if previous!=unit['source']:raise ValueError('duplicate physical station sources in the same route pool')
    return result


def pool(units,role):
    result=defaultdict(lambda:defaultdict(list))
    for index,unit in enumerate(units):
        if unit['role']==role:result[unit['country']][unit['canonical_station']].append(index)
    return {c:dict(sorted(stations.items())) for c,stations in sorted(result.items())}


def sample_unit(rng,unit_ids,units):
    groups=defaultdict(list)
    for index in unit_ids:groups[units[index]['source_group_id']].append(index)
    keys=sorted(groups);indices=groups[keys[int(rng.integers(len(keys)))]]
    capacity=np.asarray([units[i]['end_origin']-units[i]['start_origin']+1 for i in indices],dtype=float)
    index=int(rng.choice(indices,p=capacity/capacity.sum()));u=units[index]
    return index,int(rng.integers(u['start_origin'],u['end_origin']+1))


def make_plan(all_units,route,config,smoke=False):
    rng=np.random.default_rng(config['seed']);fit=pool(all_units,'fit');select=pool(all_units,'select')
    countries=config['training_countries']
    if set(fit)!=set(countries):raise ValueError('positive country weight without qualified fit data')
    unique=[];lookup={};fit_schedule=[];selection=[];retained_fit=[]
    def add(unit_index,origin):
        key=(unit_index,origin)
        if key not in lookup:
            u=all_units[unit_index];row=dict(u,origin_hour=origin,route=route,
                history_hours=config['history_hours'],forecast_hours=config['forecast_hours'])
            row.pop('start_origin');row.pop('end_origin')
            row['plan_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            lookup[key]=len(unique);unique.append(row)
        return lookup[key]
    total=8 if smoke else config['epochs']*config['updates_per_epoch']*config['batch_size']
    if smoke:labels=['CN']*4+['DE','IS','US','JP']
    else:
        exact=np.asarray([config['country_weights'][c]*total for c in countries]);counts=np.floor(exact).astype(int)
        for index in np.argsort(-(exact-counts))[:total-int(counts.sum())]:counts[index]+=1
        labels=[c for c,n in zip(countries,counts) for _ in range(int(n))]
    rng.shuffle(labels)
    smoke_fit={}
    if smoke:
        for country in countries:
            chosen=set();stations=list(fit[country])
            while len(chosen)<2:
                station=stations[int(rng.integers(len(stations)))];chosen.add(sample_unit(rng,fit[country][station],all_units))
            smoke_fit[country]=[add(i,t) for i,t in sorted(chosen)];retained_fit.extend(smoke_fit[country])
    for country in labels:
        if smoke:fit_schedule.append(int(rng.choice(smoke_fit[country])))
        else:
            stations=list(fit[country]);station=stations[int(rng.integers(len(stations)))]
            index,origin=sample_unit(rng,fit[country][station],all_units);fit_schedule.append(add(index,origin))
    for country,stations in select.items():
        if smoke and country!='CN':stations=dict(list(stations.items())[:1])
        for station,ids in stations.items():
            if country=='CN':
                for index in ids:
                    unit=all_units[index]
                    for origin in range(unit['start_origin'],unit['end_origin']+1):selection.append(add(index,origin))
            else:
                capacity=sum(all_units[index]['end_origin']-all_units[index]['start_origin']+1 for index in ids)
                chosen=set()
                if capacity<=config['foreign_select_origins_per_station']:
                    chosen={(i,t) for i in ids for t in range(all_units[i]['start_origin'],all_units[i]['end_origin']+1)}
                else:
                    while len(chosen)<config['foreign_select_origins_per_station']:chosen.add(sample_unit(rng,ids,all_units))
                selection.extend(add(i,t) for i,t in sorted(chosen))
    if smoke:
        # Fixed observation/source choices, shared by all models; no prediction
        # values participate in these case identities or in the real tiny bank.
        preferred=['Liqingdian','Luanchuan'];cn=[]
        for station in preferred:
            candidates=[i for i in selection if unique[i]['country']=='CN' and unique[i]['station']==station]
            found=min(candidates,key=lambda i:unique[i]['origin_hour']) if candidates else None
            if found is not None:cn.append(found)
        selection=cn+[i for i in selection if unique[i]['country']!='CN']
    selection=list(dict.fromkeys(selection))
    used=list(dict.fromkeys(retained_fit+fit_schedule+selection));remap={old:new for new,old in enumerate(used)}
    selected_rows=[unique[i] for i in used]
    return dict(schema='pointwise_shared_exposure_plan_v1',route=route,seed=config['seed'],
        mode='engineering_smoke' if smoke else 'exploratory_single_seed',unique_samples=selected_rows,
        fit_schedule=[remap[i] for i in fit_schedule],select_indices=[remap[i] for i in selection],
        requested_country_weights=config['country_weights'],actual_fit_exposures=dict(Counter(labels)),
        select_actual_counts=dict(Counter(unique[i]['country'] for i in selection)),
        country_station_event_origin_sampling=True,source_group_not_independent_storm=True,
        foreign_select_rule='2 per eligible canonical station; uniform source group then origin, without replacement; all if fewer than 2',
        china_select_rule='all qualified H24/F24 origins' if not smoke else '2 fixed source identities only for engineering smoke',
        observation_values_read=False)
