"""Server-only coverage figure from accepted fit metadata; no new model evaluation."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .prepare import save
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    m=json.loads((Path(a.data)/"manifest.json").read_text());support=m["support"]
    fig,ax=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    labels=["Small pool A","Expanded B"]
    ax[0].bar(labels,[3908,64715],color=["#64748b","#2563eb"])
    ax[0].set_ylabel("Unique Q99 station-hours (fit only)")
    ax[0].set_title("Real extreme coverage")
    for i,v in enumerate([3908,64715]):ax[0].text(i,v+1000,f"{v:,}",ha="center")
    ax[0].set_ylim(0,74000)
    countries=[s["country"] for s in support];counts=[s["early_extreme_windows"] for s in support]
    ax[1].bar(countries,counts,color="#2563eb");ax[1].set_yscale("log")
    ax[1].set_title("Currently Q <= Q95; future24h > Q99")
    ax[1].set_ylabel("Qualified fit windows (log scale)")
    for i,v in enumerate(counts):ax[1].text(i,v*1.12,f"{v:,}",ha="center",fontsize=9)
    ax[1].set_ylim(10,max(counts)*2)
    fig.suptitle("Fit-only data expansion; China remains 54 unique Q99 hours",fontsize=12)
    fig.savefig(out/"coverage.png",dpi=150);fig.savefig(out/"coverage.pdf")
    save(out/"sources.json",dict(data_manifest=str(Path(a.data)/"manifest.json"),source="accepted fit only",
        limitations="station-hours and overlapping windows are different units; no independent flood-count or performance claim"))
if __name__=="__main__":main()
