"""Lossless execution cache for the exact prespecified A/B pilot exposures."""
import argparse,gzip,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .prepare import save
from src.pointwise_benchmark_v1.dataset import sha256

def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    m=json.loads((root/"manifest.json").read_text())
    with gzip.open(root/"fit_rows.json.gz","rt") as f:rows=json.load(f)
    with np.load(root/"indices.npz") as z:prob={g:z["p_"+g] for g in ("A","B")}
    schedules={};all_ids=[np.flatnonzero(np.asarray([r["country"]=="CN" for r in rows]))]
    for g,p in prob.items():
        with np.load(root/f"exposure_{g}.npz") as z:schedules["base_"+g]=z["indices"]
        schedules["state_"+g]=np.random.default_rng(280002).choice(len(p),size=(2,500,64),p=p)
        cn=p*np.asarray([r["country"]=="CN" for r in rows]);cn/=cn.sum()
        schedules["cn_"+g]=np.random.default_rng(290002).choice(len(p),size=(2,500,64),p=cn)
        schedules["m_"+g]=np.random.default_rng(450002).choice(len(p),size=(3,400,32),p=cn)
        schedules["z_"+g]=np.random.default_rng(440002).choice(len(p),size=4096,p=p)
    all_ids.extend(v.ravel() for v in schedules.values())
    ids=np.unique(np.concatenate(all_ids));inverse=np.full(len(rows),-1,dtype=np.int64);inverse[ids]=np.arange(len(ids))
    local={k:inverse[v] for k,v in schedules.items()}
    for k,v in schedules.items():np.testing.assert_array_equal(ids[local[k]],v)
    with np.load(root/"fit.npz") as z:
        arrays={k:z[k][ids] for k in z.files}
        np.savez_compressed(out/"fit.npz",**arrays)
    with gzip.open(out/"fit_rows.json.gz","wt") as f:json.dump([rows[i] for i in ids],f,separators=(",",":"))
    np.savez_compressed(out/"indices.npz",A=np.flatnonzero(prob["A"][ids]>0),B=np.arange(len(ids)),
        p_A=prob["A"][ids],p_B=prob["B"][ids],parent_fit_indices=ids)
    np.savez_compressed(out/"planned_stages.npz",**local)
    for g in ("A","B"):np.savez_compressed(out/f"exposure_{g}.npz",indices=local["base_"+g])
    for name in ("select.npz","select_rows.json"):shutil.copy2(root/name,out/name)
    # Independent persisted-cache roundtrip and slot-alignment acceptance.
    with np.load(out/"fit.npz") as z:
        for k in arrays:np.testing.assert_array_equal(z[k],arrays[k])
    with gzip.open(out/"fit_rows.json.gz","rt") as f:saved=json.load(f)
    for i,j in enumerate(ids):
        if saved[i]["sample_id"]!=rows[j]["sample_id"]:raise ValueError("cache identity")
    result=dict(state="PASS",parent_manifest_sha256=sha256(root/"manifest.json"),cached_fit_windows=len(ids),
        logical_A_windows=len(np.flatnonzero(prob["A"])),logical_B_windows=len(rows),
        exact_planned_slots_verified={k:int(v.size) for k,v in schedules.items()},
        no_new_draws_or_sampling_conditioning=True,all_CN_fit_retained=True,
        claim="lossless cache of exact planned exposures; omitted candidates never requested by this pilot")
    save(out/"cache_acceptance.json",result)
    m.update(execution_cache=True,execution_parent_manifest_sha256=result["parent_manifest_sha256"],
        cached_fit_windows=len(ids),execution_cache_rule=result["claim"])
    m["files"]={p.name:sha256(p) for p in out.iterdir()}
    save(out/"manifest.json",m)
    with tarfile.open(str(out)+".tar.gz","w:gz") as tar:tar.add(out,arcname=out.name)
    print(json.dumps(dict(**result,archive=str(out)+".tar.gz")),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.data,a.out)
if __name__=="__main__":main()
