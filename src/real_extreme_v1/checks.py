"""Independent acceptance of persisted data, exact correction and exposure support."""
import argparse,gzip,json
from collections import Counter
from pathlib import Path
import numpy as np
from src.pointwise_benchmark_v1.dataset import sha256
from .prepare import save

def run(root,out):
    root=Path(root);m=json.loads((root/"manifest.json").read_text())
    for name,digest in m["files"].items():
        if sha256(root/name)!=digest:raise ValueError("changed portable file "+name)
    with gzip.open(root/"fit_rows.json.gz","rt") as f:rows=json.load(f)
    with np.load(root/"indices.npz") as z:ix={k:z[k] for k in z.files}
    with np.load(root/"fit.npz") as z:
        y=z["future_q"];q95=z["q95_fit"];q99=z["q99_fit"];x=z["history"];sc=z["flow_scale"]
    labels=(y.max(1)>q95).astype(int)+(y.max(1)>q99).astype(int)
    np.testing.assert_array_equal(labels,ix["severity"])
    np.testing.assert_array_equal((x[:,-1,1]*sc<=q95)&(y.max(1)>q99),ix["early_extreme"])
    for group in ("A","B","C"):
        p=ix["p_"+group];np.testing.assert_allclose(p.sum(),1)
        support=np.flatnonzero(p>0)
        np.testing.assert_array_equal(np.sort(ix[group]),support)
        count=Counter((rows[i]["country"],rows[i]["canonical_station"],rows[i]["event"]) for i in support)
        events=Counter((c,s) for c,s,e in count)
        stations=Counter(c for c,s in events)
        if group in ("A","B"):
            ref=np.zeros(len(rows))
            for i in support:
                r=rows[i];c=r["country"];s=r["canonical_station"];e=r["event"]
                ref[i]=m["country_weights"][c]/stations[c]/events[c,s]/count[c,s,e]
            np.testing.assert_allclose(p,ref,rtol=1e-12)
        with np.load(root/f"exposure_{group}.npz") as z:ex=z["indices"]
        if ex.shape!=(3,1000,64) or not (p[ex]>0).all():raise ValueError("exposure outside support")
    # Exact finite-population importance identity for nonconstant observables.
    correction=ix["importance_C_to_B"]
    for values in (labels.astype(float),(labels==2).astype(float),np.arange(len(rows),dtype=float)):
        np.testing.assert_allclose(np.dot(ix["p_C"]*correction,values),np.dot(ix["p_B"],values),rtol=1e-12,atol=1e-12)
    # Independent station-hour set count across ALL source units, not a window count.
    extreme=set()
    for i,h in np.argwhere(y>q99[:,None]):
        hour=int(np.datetime64(rows[i]["origin_time"],"h").astype("i8"))+int(h)+1
        extreme.add((rows[i]["canonical_station"],hour))
    if len(extreme)!=64715:raise ValueError("full-pool station-hour count changed")
    aext=set()
    for i in ix["A"]:
        for h in np.flatnonzero(y[i]>q99[i]):
            aext.add((rows[i]["canonical_station"],int(np.datetime64(rows[i]["origin_time"],"h").astype("i8"))+int(h)+1))
    if len(aext)!=3908:raise ValueError("small-pool station-hour count changed")
    cn={k for k in extreme if k[0].startswith("da_tl_cn:")}
    if len(cn)!=54:raise ValueError("China extreme support changed")
    save(out,dict(state="PASS",manifest_sha256=sha256(root/"manifest.json"),
        files_verified=len(m["files"]),A_windows=len(ix["A"]),B_windows=len(rows),
        A_unique_q99_station_hours=len(aext),B_unique_q99_station_hours=len(extreme),
        China_unique_q99_station_hours=len(cn),new_foreign_q99_station_hours=len(extreme-aext),
        correction_verified="exact p_C*(p_B/p_C) expectation equals p_B",
        labels_for_sampling_only=True,test_values_read=False))
    print(Path(out).read_text(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.data,a.out)
if __name__=="__main__":main()
