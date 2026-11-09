"""Allowlisted windows, indivisible source components and fit-only transforms."""
import csv
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def table(path, rows):
    if not rows:
        return
    with Path(path).open('x',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def hierarchy_weights(rows, country_equal=False):
    counts = Counter((r['country'], r['station_uid'], r['event']) for r in rows)
    events = Counter((c, s) for c, s, _ in counts)
    stations = Counter(c for c, _ in events)
    if country_equal:
        w = [1/(len(stations)*stations[r['country']]*events[r['country'],r['station_uid']]*counts[r['country'],r['station_uid'],r['event']]) for r in rows]
    else:
        w = [1/(len(events)*events[r['country'],r['station_uid']]*counts[r['country'],r['station_uid'],r['event']]) for r in rows]
    return np.array(w, dtype=np.float64)


def features(x):
    """Eight values; only supplied historical P/Q participate."""
    p, q = x[:, :, 0], x[:, :, 1]
    return np.column_stack([p[:, -h:].sum(1) for h in (1,3,6,12)]+[
        p[:, -3:].max(1),q[:, -1],q[:, -1]-q[:, -2],q[:, -1]-q[:, -4]])


def components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a,b=find(i),find(j)
        if a != b: parent[max(a,b)]=min(a,b)
    event, temporal, source = {}, {}, {}
    for i, r in enumerate(rows):
        key=(r['station_uid'],r['event'])
        if key in event:union(i,event[key])
        else:event[key]=i
        # International basins sharing the same clock are not connected.
        if r['country']=='China':
            assert r['temporal_component'], 'China temporal component absent'
            key=r['temporal_component']
            if key in temporal:union(i,temporal[key])
            else:temporal[key]=i
        for line in r['source_rows']:
            key=(r['station_uid'],line)
            if key in source:union(i,source[key])
            else:source[key]=i
    # A second interval check connects source-clock overlaps within a station,
    # even when two source rows carry different IDs for the same timestamp.
    by=defaultdict(list)
    for i,r in enumerate(rows):by[r['station_uid']].append(i)
    for ids in by.values():
        ids.sort(key=lambda i:rows[i]['source_times'][0]); anchor=None; right=None
        for i in ids:
            lo,hi=rows[i]['source_times'][0],rows[i]['source_times'][-1]
            if right is not None and lo<=right:union(i,anchor);right=max(right,hi)
            else:anchor=i;right=hi
    return np.array([find(i) for i in range(len(rows))])


def assign_roles(rows, group, seed):
    """Random whole components, with China development-station fit support."""
    rng=np.random.default_rng(seed);mapping={};reserved=[]
    china=defaultdict(set)
    for i,r in enumerate(rows):
        if r['country']=='China':china[r['station_uid']].add(int(group[i]))
    unique=np.array(sorted(set().union(*china.values())))
    # Metadata-only support check: never inspect any target to choose a split.
    support_ok=False
    for attempt in range(512):
        rng.shuffle(unique);n=len(unique);a=int(.6*n);b=int(.8*n)
        mapping.update({int(g):('fit' if i<a else 'select' if i<b else 'calibration') for i,g in enumerate(unique)})
        for st,gs in sorted(china.items()):
            if len(gs)<3:continue
            for missing in ['fit','select','calibration']:
                if any(mapping[g]==missing for g in gs):continue
                candidates=[]
                for g in sorted(gs):
                    old=mapping[g]
                    if all(sum(mapping[z]==old for z in other)>1 for other in china.values() if g in other):candidates.append(g)
                if candidates:mapping[candidates[int(rng.integers(len(candidates)))]]=missing
        if all(len({mapping[g] for g in gs})==3 for gs in china.values() if len(gs)>=3):support_ok=True;break
    assert support_ok,'NOT_EVALUABLE: no whole-component China split supports all required roles after metadata-only attempts'
    reserved.append(dict(china_metadata_draws=attempt+1,rule='First seeded component partition supporting all three roles for China stations with at least three components'))
    foreign=defaultdict(set)
    for i,r in enumerate(rows):
        if r['country']!='China':foreign[r['station_uid']].add(int(group[i]))
    for st,gs in sorted(foreign.items()):
        ids=np.array(sorted(gs));rng.shuffle(ids);n=len(ids)
        if n==1:assigned=['fit']
        elif n==2:assigned=['fit','select' if rng.random()<.5 else 'calibration']
        else:
            nf=max(1,min(n-2,round(.6*n)));ns=max(1,min(n-nf-1,round(.2*n)))
            assigned=['fit']*nf+['select']*ns+['calibration']*(n-nf-ns)
        mapping.update({int(g):s for g,s in zip(ids,assigned)})
        if n<3:reserved.append(dict(station=st,components=n,roles=assigned,rule='No component split; one component fit, two components fit plus one seeded role'))
    for station in sorted({r['station_uid'] for r in rows if r['country']=='China'}):
        gs=sorted(set(int(group[i]) for i,r in enumerate(rows) if r['station_uid']==station))
        if not any(mapping[g]=='fit' for g in gs):
            # Deterministic support reservation, recorded rather than borrowing scales.
            selected=gs[0];reserved.append(dict(station=station,component=selected,previous_role=mapping[selected]))
            mapping[selected]='fit'
    return np.array([mapping[int(g)] for g in group]),reserved


def validate_roles(rows, group, role):
    events, lines, comp = {}, {}, {}
    for r,g,s in zip(rows,group,role):
        for store,key in ((events,(r['station_uid'],r['event'])),(comp,int(g))):
            assert key not in store or store[key]==s,'source component crosses roles'
            store[key]=s
        for line in r['source_rows']:
            key=(r['station_uid'],line)
            assert key not in lines or lines[key]==s,'source row crosses roles'
            lines[key]=s


def load_rows(config):
    mp=Path(config['source_manifest']);m=json.loads(mp.read_text(encoding='utf-8'))
    assert m['source_data_commit']==config['source_commit']
    rows=[];dev=[];sources=[]
    for st in m['stations']:
        if st['country'] not in config['countries']:continue
        for oldrole,dest in [('train',rows)]+([('validation',dev)] if st['country']=='China' else []):
            if not st['counts'].get(oldrole,0):continue
            path=Path(st['files'][oldrole]['path'])
            assert path.name==oldrole+'_windows.jsonl.gz'
            assert path.parent.name==st['station_uid'].replace(':','__')
            assert path.is_relative_to(Path(m['stageB_root'])/'stations')
            sources.append(dict(station=st['station_uid'],country=st['country'],old_role=oldrole,path=str(path),sha256=digest(path),
                original_qscale=st['scales']['qscale'],original_pscale=st['scales']['pscale']))
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for line in f:
                    raw=json.loads(line)
                    assert raw['role']==oldrole and raw['station_uid']==st['station_uid']
                    r={k:raw[k] for k in ['window_id','station_uid','country','origin_time','source_rows','source_times','x_p','x_q','y_q']}
                    r['event']=raw.get('overlap_group_id') or raw['group_id']
                    r['source_event_id']=raw.get('source_event_id')
                    r['temporal_component']=raw.get('temporal_overlap_component')
                    r['old_role']=oldrole
                    oldq,oldp=st['scales']['qscale'],st['scales']['pscale']
                    r['x_physical']=np.column_stack((np.array(r.pop('x_p'))*oldp,np.array(r.pop('x_q'))*oldq))
                    r['y_physical']=np.array(r.pop('y_q'))*oldq
                    assert r['x_physical'].shape==(12,2) and r['y_physical'].shape==(12,)
                    assert len(r['source_rows'])==len(r['source_times'])==24
                    assert np.isfinite(r['x_physical']).all() and np.isfinite(r['y_physical']).all()
                    assert (r['x_physical']>=0).all() and (r['y_physical']>=0).all()
                    dest.append(r)
    assert Counter(r['country'] for r in rows)=={'China':8557,'Germany':20000,'Iceland':20000}
    assert len(dev)==1356 and len({r['station_uid'] for r in dev})==9
    assert len({r['window_id'] for r in rows+dev})==len(rows)+len(dev)
    train_lines={(r['station_uid'],line) for r in rows if r['country']=='China' for line in r['source_rows']}
    assert not any((r['station_uid'],line) in train_lines for r in dev for line in r['source_rows']),'Original train/development share source rows'
    train_components={r['temporal_component'] for r in rows if r['country']=='China'}
    assert not any(r['temporal_component'] in train_components for r in dev),'Original train/development share China temporal component'
    return rows,dev,sources


def prepare(config, folder):
    rows,dev,sources=load_rows(config)
    group=components(rows);roles,reserved=assign_roles(rows,group,config['split_seed'])
    validate_roles(rows,group,roles)
    scales={}; inventories=[]
    for station in sorted({r['station_uid'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['station_uid']==station and roles[i]=='fit']
        q,p={},{}
        for i in ix:
            r=rows[i]
            for line,val in zip(r['source_rows'],np.r_[r['x_physical'][:,1],r['y_physical']]):
                if line in q:assert np.isclose(q[line],val,rtol=2e-6,atol=1e-5)
                q[line]=float(val)
            for line,val in zip(r['source_rows'][:12],r['x_physical'][:,0]):
                if line in p:assert np.isclose(p[line],val,rtol=2e-6,atol=1e-5)
                p[line]=float(val)
        positive_p=[v for v in p.values() if v>0]
        q95=float(np.quantile(list(q.values()),.95)) if q else None
        q99=float(np.quantile(list(q.values()),.99)) if q else None
        good=q95 is not None and q95>0
        scales[station]=dict(state='evaluable' if good else 'NOT_EVALUABLE',qscale=q95,q99=q99,
            pscale=float(np.quantile(positive_p,.95)) if positive_p else 1.,
            pscale_rule='fit positive historical-P Q95, all-zero P uses 1',all_zero_fit_P=not bool(positive_p),fit_positive_P_rows=len(positive_p),fit_Q_rows=len(q),fit_P_rows=len(p),
            fit_Q_source_rows_sha256=hashlib.sha256(json.dumps(sorted(q)).encode()).hexdigest(),
            fit_P_source_rows_sha256=hashlib.sha256(json.dumps(sorted(p)).encode()).hexdigest())
    for r in dev:assert scales[r['station_uid']]['state']=='evaluable','China dev station has no fit scale'
    allrows=rows+dev;allroles=list(roles)+['development']*len(dev)
    standard={};cutoffs={}
    for st,sc in scales.items():
        fit=[r for r,s in zip(allrows,allroles) if r['station_uid']==st and s=='fit']
        if sc['state']!='evaluable':continue
        x=np.stack([r['x_physical']/[sc['pscale'],sc['qscale']] for r in fit])
        f=features(x);w=hierarchy_weights(fit);mu=w@f;sd=np.sqrt(w@((f-mu)**2));sd[sd==0]=1
        standard[st]=dict(mean=mu.tolist(),scale=sd.tolist())
        rise=np.maximum(0,np.array([r['y_physical'][:6].max()-r['x_physical'][-1,1] for r in fit])/sc['qscale'])
        positive=rise[rise>0];cutoffs[st]=float(np.quantile(positive,.9)) if len(positive) else None
    pools={}; ids=[];unknown=[]
    for role in ['fit','select','calibration','development']:
        selected=[];x=[];y=[];f=[]
        for i,(r,s) in enumerate(zip(allrows,allroles)):
            if s!=role:continue
            sc=scales[r['station_uid']]
            ids.append(dict(window_id=r['window_id'],station=r['station_uid'],country=r['country'],event=r['event'],role=role,
                            component=int(group[i]) if i<len(rows) else None,scale_state=sc['state']))
            if sc['state']!='evaluable':
                unknown.append(dict(window_id=r['window_id'],role=role,station=r['station_uid'],reason='No fit-supported positive Q95'))
                continue
            a=r['x_physical']/[sc['pscale'],sc['qscale']];b=r['y_physical']/sc['qscale']
            norm=standard[r['station_uid']];z=(features(a[None])[0]-norm['mean'])/norm['scale']
            selected.append(dict(window_id=r['window_id'],station_uid=r['station_uid'],country=r['country'],event=r['event'],
                origin_time=r['origin_time'],source_rows=r['source_rows'],source_times=r['source_times'],qscale=sc['qscale'],
                source_event_id=r['source_event_id'],temporal_overlap_component=r['temporal_component'],
                strong_rise_cutoff=cutoffs[r['station_uid']],role=role))
            x.append(a);y.append(b);f.append(z)
        arrays=dict(x=np.array(x,dtype='float32'),y=np.array(y,dtype='float32'),features=np.array(f,dtype='float32'))
        assert len(selected)>0 and all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(folder/(role+'.npz'),**arrays)
        save(folder/(role+'_rows.json'),selected)
        pools[role]=dict(rows=selected,**arrays)
        allstations={r['station_uid'] for r in selected}|{r['station_uid'] for r in rows if r['country']=='China'}
        for st in sorted(allstations):
            indices=[i for i,r in enumerate(selected) if r['station_uid']==st];yy=arrays['y'][indices];xx=arrays['x'][indices]
            inventories.append(dict(role=role,station=st,country=selected[indices[0]]['country'] if indices else 'China',windows=len(indices),
                state='evaluable' if indices else 'NOT_EVALUABLE',
                events=len({selected[i]['event'] for i in indices}),current_not_high=int((xx[:,-1,1]<=1).sum()),
                high6=int((yy[:,:6].max(1)>1).sum()),nonhigh6=int((yy[:,:6].max(1)<=1).sum())))
    # Eight candidates have different declared meanings/time scales; correlation
    # alone is insufficient to remove one. Preserve all and publish the fit check.
    from scipy.stats import spearmanr
    corr=spearmanr(pools['fit']['features']).statistic
    corr=np.nan_to_num(corr,nan=0).tolist()
    save(folder/'feature_screening.json',dict(names=['P1','P3','P6','P12','P3max','qnow','dq1','dq3'],
        retained=list(range(8)),removed=[],fit_spearman=corr,
        reason='Distinct time scales, rainfall sum/max, flow level/change; no same-meaning pair eligible for automatic removal.',
        status='All eight retained after fit-only eligibility check; redundancy not established across events'))
    save(folder/'scales.json',scales);save(folder/'standardization.json',standard);save(folder/'rise_cutoffs.json',cutoffs)
    table(folder/'role_ids.csv',ids);table(folder/'role_inventory.csv',inventories);table(folder/'unknown_scale_windows.csv',unknown)
    manifest=dict(source_commit=config['source_commit'],source_manifest=config['source_manifest'],source_manifest_sha256=digest(config['source_manifest']),
        producer_code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=sources,
        roles=dict(Counter(allroles)),components=len(np.unique(group)),china_fit_reservations=reserved,
        unknown_scale_windows=len(unknown),derived_files={},fit_float32_reconstruction=True,
        limits=['Physical values reconstructed from original float32 normalized records, not raw source observations.',
                'Future P never retrieved; fit P scale uses only historical P source rows.',
                'Development repeatedly used historically; current results exploratory.'])
    for path in sorted(folder.glob('*.npz')):manifest['derived_files'][path.name]=digest(path)
    save(folder/'manifest.json',manifest)
    return pools


def load(folder):
    pools={}
    for role in ['fit','select','calibration','development']:
        a=np.load(folder/(role+'.npz'));pools[role]=dict(rows=json.loads((folder/(role+'_rows.json')).read_text(encoding='utf-8')),**{k:a[k] for k in a.files})
    return pools
