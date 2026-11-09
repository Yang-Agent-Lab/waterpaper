"""Read only the approved D500 select source, never retained test."""
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source = Path(a.source)
    rows = json.loads((source/'select_rows.json').read_text())
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(str(r.get('canonical_station')), str(r.get('source_event_identity')))].append(r)
    segments=[]
    for key, rr in sorted(groups.items()):
        times=sorted(datetime.fromisoformat(r['origin_time'].replace('Z','+00:00')) for r in rr)
        blocks=[1]
        for t,u in zip(times,times[1:]):
            if (u-t).total_seconds()==3600: blocks[-1]+=1
            else: blocks.append(1)
        segments.append(dict(group=key, rows=len(rr), blocks=blocks, first=str(times[0]),last=str(times[-1])))
    with np.load(source/'select_predictions.npz', allow_pickle=False) as z:
        inventory={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
    versions={}
    for lib in ('numpy','torch','gymnasium','stable-baselines3'):
        try: versions[lib]=importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError: versions[lib]=None
    result=dict(rows=len(rows), row_keys=list(rows[0]), example_row=rows[0], groups=segments, inventory=inventory, versions=versions)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
