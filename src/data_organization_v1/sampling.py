"""Fit-only weights. A/B/C share random slots; process units are not storms."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from train_gap_audit_v1 import fragments
from protected_flood_risk_v1.training import atomic_json


def categorical(probability, u):
    p=np.asarray(probability,dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
        raise ValueError('invalid sampling probability')
    return np.minimum(np.searchsorted(np.cumsum(p),u,side='right'),len(p)-1)


def cells(q,p,rise,threshold):
    # Nonpositive is a separate class; strong boundaries include equality.
    return ((q>=threshold['q_mm_h']).astype(int)*2+(p>=threshold['p6_mm']).astype(int))*3+np.where(rise<=0,0,np.where(rise>=threshold['rise_mm_h'],2,1))


def prepare(pool,attrs,scales):
    rows=pool['rows']
    if any(r['role']!='fit' for r in rows):raise ValueError('fit-only weights')
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['station_uid']].append(i)
    groups={s:np.asarray(v) for s,v in sorted(groups.items())}
    targets=sorted(s for s,ix in groups.items() if rows[ix[0]]['country']=='China')
    countries=sorted({r['country'] for r in rows})
    if countries!=['China','Germany','Iceland'] or len(targets)!=9:raise ValueError('fixed three countries and nine targets required')
    stations={c:sorted(s for s,ix in groups.items() if rows[ix[0]]['country']==c) for c in countries}
    mapping,records=fragments(rows)
    components=defaultdict(set)
    for i,r in enumerate(rows):components[mapping[i]].add(str(r['temporal_overlap_component']))
    for record in records:record['temporal_overlap_components']=sorted(components[record['fragment_id']])
    unit_ids=sorted({mapping[i] for i in range(len(rows))});unit_lookup={v:i for i,v in enumerate(unit_ids)}
    unit=np.array([unit_lookup[mapping[i]] for i in range(len(rows))]);units={s:[np.flatnonzero(unit==k) for k in sorted(set(unit[ix]))] for s,ix in groups.items()}
    area=np.array([attrs[r['station_uid']]['area_km2'] for r in rows],float)
    annual={s:float(attrs[s]['mean_annual_precip_mm_equiv']) for s in groups}
    if not np.isfinite(area).all() or (area<=0).any() or any(not np.isfinite(v) or v<=0 for v in annual.values()):raise ValueError('invalid background attributes')
    qscale=np.array([r['qscale'] for r in rows]);now=pool['x'][:,-1,1].astype(float);q=3.6*now*qscale/area
    p6=pool['x'][:,-6:,0].astype(float).sum(1)*np.array([scales[r['station_uid']]['pscale'] for r in rows])
    rise=3.6*(pool['y'][:,:6].astype(float).max(1)-now)*qscale/area
    thresholds={};support={};weights={};background={};probability_rows=[]
    for t in targets:
        ix=groups[t];positive_p=p6[ix][p6[ix]>0];positive_r=rise[ix][rise[ix]>0]
        valid=bool(len(positive_p) and len(positive_r))
        th=dict(state='evaluable' if valid else 'NOT_EVALUABLE',q_mm_h=float(3.6*qscale[ix[0]]/area[ix[0]]),p6_mm=float(np.quantile(positive_p,.75,method='linear')) if len(positive_p) else None,rise_mm_h=float(np.quantile(positive_r,.75,method='linear')) if len(positive_r) else None)
        thresholds[t]=th;boost=np.ones(len(rows));counts=np.zeros(12,int)
        if valid:
            cell=cells(q,p6,rise,th)
            for indices in units[t]:counts[np.unique(cell[indices])]+=1
            history=set((cell[ix]//3).tolist());mask=np.isin(cell//3,list(history))
            boost[mask]=1+3/(1+counts[cell[mask]])
        support[t]=dict(distinct_process_units_by_cell=counts.tolist(),fit_windows=len(ix),fit_units=len(units[t]),unit='same-station adjacent/overlapping full24h coverage, not independent storm')
        weights[t]={}
        for s,items in units.items():
            count=np.array([len(v) for v in items]);mean_boost=np.array([boost[v].mean() for v in items]);a=count/count.sum();c=count*mean_boost;c=c/c.sum()
            weights[t][s]=(a,c)
            for j,indices in enumerate(items):probability_rows.append(dict(target=t,station=s,unit=unit_ids[unit[indices[0]]],windows=len(indices),mean_boost=float(mean_boost[j]),A_B_unit_probability=float(a[j]),C_unit_probability=float(c[j]),within_unit_probability=1/len(indices)))
        background[t]={}
        for country,ss in stations.items():
            d2=np.array([(np.log(attrs[s]['area_km2']/attrs[t]['area_km2'])/np.log(2))**2+(np.log(annual[s]/annual[t])/np.log(1.5))**2 for s in ss]);logit=-.5*d2;prob=np.exp(logit-logit.max());prob/=prob.sum();background[t][country]=prob
    if not any(np.max(abs(a-c))>1e-12 for w in weights.values() for a,c in w.values()):raise ValueError('NOT_EVALUABLE: C and B policies indistinguishable')
    return dict(rows=rows,groups=groups,targets=targets,countries=countries,stations=stations,unit=unit,unit_ids=unit_ids,units=units,thresholds=thresholds,support=support,weights=weights,background=background,process_records=records,probability_rows=probability_rows)


def slots(prepared,epochs,size,seed):
    p=prepared;rng=np.random.default_rng(seed);shape=(epochs,size);n=epochs*size
    target=rng.integers(len(p['targets']),size=n);country=rng.integers(len(p['countries']),size=n)
    uniforms=rng.random((3,n));result={a:np.empty(n,np.int64) for a in 'ABC'}
    for ti,t in enumerate(p['targets']):
        for ci,country_name in enumerate(p['countries']):
            pos=np.flatnonzero((target==ti)&(country==ci));ss=p['stations'][country_name]
            a_station=categorical(np.ones(len(ss))/len(ss),uniforms[0,pos]);bc_station=categorical(p['background'][t][country_name],uniforms[0,pos])
            for arm,selected in [('A',a_station),('B',bc_station),('C',bc_station)]:
                for sj in np.unique(selected):
                    at=pos[selected==sj];s=ss[sj];items=p['units'][s];weight=p['weights'][t][s][arm=='C'];chosen=categorical(weight,uniforms[1,at])
                    for uj in np.unique(chosen):
                        where=at[chosen==uj];indices=items[uj];result[arm][where]=indices[np.minimum((uniforms[2,where]*len(indices)).astype(int),len(indices)-1)]
    rows=p['rows'];bc_equal=all(rows[b]['station_uid']==rows[c]['station_uid'] and rows[b]['country']==rows[c]['country'] for b,c in zip(result['B'],result['C']))
    countries_equal=all(rows[a]['country']==rows[b]['country'] for a,b in zip(result['A'],result['B']))
    if not bc_equal or not countries_equal:raise AssertionError('shared slot identity failed')
    if np.array_equal(result['B'],result['C']):raise ValueError('actual slots B/C indistinguishable')
    audit=dict(shape=list(shape),B_C_station_country_position_equal=bc_equal,A_B_C_country_position_equal=countries_equal,actual_window_difference_count=int((result['B']!=result['C']).sum()),exposure={})
    for a,ix in result.items():
        audit['exposure'][a]=dict(total=len(ix),country=dict(Counter(rows[i]['country'] for i in ix)),station=dict(Counter(rows[i]['station_uid'] for i in ix)),unit=dict(Counter(p['unit_ids'][p['unit'][i]] for i in ix)),sha256=hashlib.sha256(ix.tobytes()).hexdigest())
    return {a:v.reshape(shape) for a,v in result.items()},audit,dict(target_index=target.reshape(shape),country_index=country.reshape(shape),uniforms=uniforms.reshape(3,*shape))


def save(out,p,slot_arrays,audit,latents):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'thresholds.json',p['thresholds']);atomic_json(out/'condition_support.json',p['support']);atomic_json(out/'process_units.json',p['process_records'])
    atomic_json(out/'unit_probabilities.json',p['probability_rows']);atomic_json(out/'station_probabilities.json',{t:{c:dict(zip(p['stations'][c],w.tolist())) for c,w in by.items()} for t,by in p['background'].items()})
    atomic_json(out/'slot_audit.json',audit)
    np.savez_compressed(out/'shared_random_slots.npz',**latents)
    for a,ix in slot_arrays.items():np.save(out/('slots_'+a+'.npy'),ix)
