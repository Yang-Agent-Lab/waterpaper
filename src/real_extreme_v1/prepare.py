"""Fit-only expansion, exact sampling probabilities and bank-supported process audit."""
import argparse, csv, gzip, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from src.pointwise_benchmark_v1.dataset import sha256, hierarchy_weights

BANK_SHA="2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c"
COUNTRIES={"CN":.5,"DE":.2,"IS":.1,"US":.15,"JP":.05}
FIELDS=("sample_id","canonical_station","station","country","role","event",
        "source_event_identity","origin_time","input_start","target_end",
        "q_scale","q95","q99","timezone","information_mode","availability")

def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def table(path,rows):
    if not rows: return
    with Path(path).open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def probability(rows,indices):
    p=np.zeros(len(rows),dtype=float)
    for country,mass in COUNTRIES.items():
        ix=np.asarray([i for i in indices if rows[int(i)]["country"]==country],dtype=int)
        if not len(ix): raise ValueError("country missing: "+country)
        p[ix]=mass*hierarchy_weights([rows[i] for i in ix])
    np.testing.assert_allclose(p.sum(),1,atol=1e-12)
    return p

def mixed_probability(rows,labels,base):
    balanced=np.zeros(len(rows))
    for country,mass in COUNTRIES.items():
        ids=np.asarray([i for i,r in enumerate(rows) if r["country"]==country])
        classes=np.unique(labels[ids])
        for cls in classes:
            ix=ids[labels[ids]==cls]
            balanced[ix]=mass/len(classes)*hierarchy_weights([rows[i] for i in ix])
    p=.5*base+.5*balanced
    np.testing.assert_allclose(p.sum(),1,atol=1e-12)
    if not (p>0).all(): raise ValueError("sampling support lost")
    return p

def compact(row,index):
    r={k:row[k] for k in FIELDS if k in row}
    r.update(global_bank_index=int(index),thresholds_fit_only=True,route="pq")
    r.setdefault("station",r["canonical_station"].split(":")[-1])
    return r

def audit_processes(rows,arrays,out):
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[(r["country"],r["canonical_station"],r["event"])].append(i)
    processes=[]; runs=[]; stats=[]; country=defaultdict(Counter)
    for (cn,station,event),ix in groups.items():
        ids=np.asarray(ix); origins=np.asarray([rows[i]["origin_time"] for i in ids],dtype="datetime64[h]").astype("i8")
        times=(origins[:,None]+np.arange(1,25)).ravel()
        values=arrays["future_q"][ids].ravel()
        order=np.argsort(times,kind="stable"); st=times[order]; sv=values[order]
        unique,first,count=np.unique(st,return_index=True,return_counts=True)
        vmin=np.minimum.reduceat(sv,first);vmax=np.maximum.reduceat(sv,first)
        if np.any(vmax-vmin>1e-8*np.maximum(1,np.abs(vmax))): raise ValueError("conflicting repeated future truth")
        q95=arrays["q95_fit"][ids[0]];q99=arrays["q99_fit"][ids[0]]
        if not (arrays["q95_fit"][ids]==q95).all() or not (arrays["q99_fit"][ids]==q99).all(): raise ValueError("threshold changed within source group")
        # Strict contiguous superlevel runs; no claim that these are independent storms.
        hit=unique[vmax>q95]
        parts=np.split(hit,np.flatnonzero(np.diff(hit)!=1)+1) if len(hit) else []
        for j,ts in enumerate(parts):
            take=np.searchsorted(unique,ts);vs=vmax[take];peak=int(np.argmax(vs))
            left=int(ts[0]-1);right=int(ts[-1]+1)
            processes.append(dict(country=cn,station=station,source_event=event,process_id=f"{station}|{event}|{j}",
                first_above_q95=str(np.datetime64(int(ts[0]),"h")),last_above_q95=str(np.datetime64(int(ts[-1]),"h")),
                above_q95_hours=len(ts),above_q99_hours=int((vs>q99).sum()),
                peak_m3_s=float(vs[peak]),peak_time=str(np.datetime64(int(ts[peak]),"h")),
                left_censored=bool(left not in unique),right_censored=bool(right not in unique),
                support="qualified_future_union_only",independent_storm=False))
        sorted_origins=np.sort(np.unique(origins))
        blocks=np.split(sorted_origins,np.flatnonzero(np.diff(sorted_origins)!=1)+1)
        for ts in blocks:
            runs.append(dict(role=rows[ix[0]]["role"],country=cn,station=station,source_event=event,
                first_origin=str(np.datetime64(int(ts[0]),"h")),last_origin=str(np.datetime64(int(ts[-1]),"h")),
                consecutive_origins=len(ts),annual_false_alarm_supported=False,whole_chain_unseen=False,
                exposure="fit_seen" if rows[ix[0]]["role"]=="fit" else "select_used_for_selection",
                support="bank-qualified consecutive origins; source event completeness inherited"))
        c=country[cn];c["windows"]+=len(ids);c["unique_future_hours"]+=len(unique)
        c["q95_unique_hours"]+=int((vmax>q95).sum());c["q99_unique_hours"]+=int((vmax>q99).sum())
        c["q95_runs"]+=len(parts);c["source_units"]+=1
        stats.append(dict(country=cn,station=station,source_event=event,windows=len(ix),
            unique_future_hours=len(unique),q99_hours=int((vmax>q99).sum())))
    table(out/"process_catalog.csv",processes);table(out/"continuous_replay_inventory.csv",runs)
    table(out/"source_unit_support.csv",stats)
    return [dict(country=cn,stations=len({r["canonical_station"] for r in rows if r["country"]==cn}),**dict(c)) for cn,c in sorted(country.items())]

def prepare(bank,small,out):
    bank=Path(bank);small=Path(small);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if sha256(bank/"manifest.json")!=BANK_SHA: raise ValueError("unexpected accepted bank")
    m=json.loads((bank/"manifest.json").read_text())
    for role in ("fit","select"):
        for name,digest in m["roles"][role].items():
            if sha256(bank/name)!=digest: raise ValueError("source hash "+name)
    sm=json.loads((small/"data_manifest.json").read_text())
    if sm["bank_manifest_sha256"]!=BANK_SHA: raise ValueError("small pool source differs")
    for name in ("fit_rows.json","fit.npz"):
        if sha256(small/name)!=sm["files"][name]: raise ValueError("small pool changed")
    print("source hashes accepted",flush=True)
    raw=json.loads((bank/"fit_rows.json").read_text())
    rows=[compact(r,i) for i,r in enumerate(raw)]
    for r in raw:
        if r["role"]!="fit" or r["threshold_sources"]["Q95"]["role"]!="fit" or r["threshold_sources"]["Q99"]["role"]!="fit":
            raise ValueError("fit-only source/threshold violation")
        if not r["current_information_included"] or r["target_leads"]!=list(range(1,25)): raise ValueError("task changed")
        if r["source_rows"]!=list(range(r["source_rows"][0],r["source_rows"][0]+48)): raise ValueError("nonconsecutive source rows")
        start=np.datetime64(r["input_start"],"h");origin=np.datetime64(r["origin_time"],"h");end=np.datetime64(r["target_end"],"h")
        if (origin-start)!=np.timedelta64(23,"h") or (end-origin)!=np.timedelta64(24,"h"): raise ValueError("time geometry")
    del raw
    with np.load(bank/"fit.npz",allow_pickle=False) as z:
        if not z["mask"].all(): raise ValueError("masked fit input")
        a=dict(history=z["x"],future_q=z["y_q_m3_s"],q95_fit=np.asarray([r["q95"] for r in rows]),
               q99_fit=np.asarray([r["q99"] for r in rows]),flow_scale=np.asarray([r["q_scale"] for r in rows]))
        np.testing.assert_allclose(z["y"]*a["flow_scale"][:,None],a["future_q"],rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(z["thresholds_normalized"]*a["flow_scale"][:,None],np.stack([a["q95_fit"],a["q99_fit"]],-1),rtol=1e-12)
    if not all(np.isfinite(v).all() for v in a.values()): raise ValueError("nonfinite accepted values")
    if len({r["sample_id"] for r in rows})!=len(rows): raise ValueError("duplicate sample identity")
    little=json.loads((small/"fit_rows.json").read_text())
    ia=np.asarray([r["global_bank_index"] for r in little],dtype=np.int64)
    if any(rows[i]["sample_id"]!=r["sample_id"] for i,r in zip(ia,little)): raise ValueError("small identity alignment")
    with np.load(small/"fit.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["history"],a["history"][ia])
        np.testing.assert_array_equal(z["future_q"],a["future_q"][ia])
    peak=a["future_q"].max(1)
    labels=(peak>a["q95_fit"]).astype("i1")+(peak>a["q99_fit"]).astype("i1")
    early=(a["history"][:,-1,1]*a["flow_scale"]<=a["q95_fit"])&(labels==2)
    pa=probability(rows,ia);pb=probability(rows,np.arange(len(rows)));pc=mixed_probability(rows,labels,pb)
    np.savez_compressed(out/"fit.npz",**a)
    with gzip.open(out/"fit_rows.json.gz","wt",encoding="utf-8") as f: json.dump(rows,f,ensure_ascii=False,separators=(",",":"))
    np.savez_compressed(out/"indices.npz",A=ia,B=np.arange(len(rows)),C=np.arange(len(rows)),
        p_A=pa,p_B=pb,p_C=pc,importance_C_to_B=pb/pc,severity=labels,early_extreme=early)
    # Persist a common schedule consumed by both model families, with country weights unchanged.
    for group,p in (("A",pa),("B",pb),("C",pc)):
        rng=np.random.default_rng(280002)
        np.savez_compressed(out/f"exposure_{group}.npz",indices=rng.choice(len(rows),size=(3,1000,64),p=p))
    full=json.loads((bank/"select_rows.json").read_text())
    cn=np.asarray([i for i,r in enumerate(full) if r["country"]=="CN"])
    if len(cn)!=1706: raise ValueError("China select identity changed")
    byunit=defaultdict(list)
    for i in cn: byunit[(full[i]["canonical_station"],full[i]["event"])].append(int(i))
    picked=set()
    for ix in byunit.values():
        ix=sorted(ix,key=lambda i:full[i]["origin_time"])
        picked.update(ix[j] for j in np.unique(np.linspace(0,len(ix)-1,min(16,len(ix)),dtype=int)))
    picked.update(i for i in cn if full[i]["sample_id"] in m["report_case_ids"])
    select_ix=np.asarray(sorted(picked))
    sr=[compact(full[i],i) for i in select_ix]
    for r in sr:
        if r["role"]!="select": raise ValueError("select changed")
    # Disjoint full 48-hour ranges, including histories; only role metadata are used here.
    fit_ranges=defaultdict(list)
    for r in rows: fit_ranges[r["canonical_station"]].append((np.datetime64(r["input_start"],"h").astype("i8"),np.datetime64(r["target_end"],"h").astype("i8")))
    overlap=0
    for station,rr in fit_ranges.items():
        rr.sort();starts=np.asarray([s for s,e in rr]);ends=np.maximum.accumulate([e for s,e in rr])
        for r in full:
            if r["canonical_station"]!=station: continue
            s=np.datetime64(r["input_start"],"h").astype("i8");e=np.datetime64(r["target_end"],"h").astype("i8")
            k=np.searchsorted(starts,e,side="right")-1
            overlap+=int(k>=0 and ends[k]>=s)
    if overlap: raise ValueError(f"fit/select full-window temporal overlaps: {overlap}")
    with np.load(bank/"select.npz",allow_pickle=False) as z:
        if not z["mask"][select_ix].all(): raise ValueError("masked select")
        np.savez_compressed(out/"select.npz",history=z["x"][select_ix],future_q=z["y_q_m3_s"][select_ix],
            q95_fit=np.asarray([r["q95"] for r in sr]),q99_fit=np.asarray([r["q99"] for r in sr]),
            flow_scale=np.asarray([r["q_scale"] for r in sr]))
    save(out/"select_rows.json",sr);save(out/"select_full_metadata.json",[compact(full[i],i) for i in cn])
    print("portable arrays and probabilities written",flush=True)
    support=audit_processes(rows,a,out)
    for s in support:
        ids=np.asarray([i for i,r in enumerate(rows) if r["country"]==s["country"]])
        s.update(ordinary_windows=int((labels[ids]==0).sum()),high_windows=int((labels[ids]==1).sum()),
            extreme_windows=int((labels[ids]==2).sum()),early_extreme_windows=int(early[ids].sum()))
    table(out/"country_support.csv",support)
    manifest=dict(state="accepted",experiment="WP-20261028-REAL-EXTREME-ABC-V1",source_bank_sha256=BANK_SHA,
        source_paths=dict(bank=str(bank),small=str(small)),source_role_file_hashes=m["roles"],fit_scale_identity=m["fit_scale_identity"],
        rows=dict(fit=len(rows),A=len(ia),B=len(rows),C=len(rows),select_pilot=len(sr),select_full_metadata=len(cn)),
        country_weights=COUNTRIES,fit_select_overlap_windows=overlap,
        history_hours=24,forecast_hours=24,thresholds_recomputed=False,roles_reallocated=False,
        selection_rule="metadata-only up to16 evenly spaced origins per station/source unit plus inherited report cases",
        C_rule="0.5 base +0.5 country->available severity equal->station->source unit->origin",
        C_training="prepared_only; decision after A/B",importance_rule="p_B/p_C; no self-normalization",
        process_rule="strict consecutive Q>Q95 in qualified future union; zero-gap merge; right/left censor flags; not independent storms",
        continuous_scope="bank-supported only; no annual frequency or OOF policy-training claim",
        input_qc="inherit accepted raw QC; recheck source hashes, masks, role, consecutive source rows, time geometry, finite values, scale roundtrip, duplicate future observations",
        limitations=["timezones and release availability inherited, sometimes unknown","source-group QC is inherited; no new raw sensor flag audit","no new independent China floods","pilot select reused for model selection; not test","fit coverage not equivalent to actual model exposure"],
        report_case_ids=m["report_case_ids"],report_case_kinds=m["report_case_kinds"],support=support)
    manifest["files"]={p.name:sha256(p) for p in out.iterdir() if p.is_file()}
    save(out/"manifest.json",manifest)
    print(json.dumps(dict(state="complete",rows=manifest["rows"],support=support)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--bank",required=True);p.add_argument("--small",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();prepare(a.bank,a.small,a.out)
if __name__=="__main__":main()
