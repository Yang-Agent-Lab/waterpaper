# Read-only parent inspection of saved results; no training or inference.
# Executed via SSH stdin on 2026-10-18, then archived here unchanged below.
import json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime,timedelta
import numpy as np
root=Path("./results/experiments/pointwise_data_mix_formal_20261018_83db4f4")
read=lambda p:json.loads(Path(p).read_text())
dirs={(g,s):root/"artifacts"/g/f"seed_{s}"/"model" for s in (2,13,42) for g in ("A","B","C")}
missing=[f"{g}/{s}" for (g,s),d in dirs.items() if not (d/"completion.json").is_file()]
if missing:
 print(json.dumps({"state":"pending","missing":missing}));raise SystemExit
rows=read(dirs["A",2]/"select_rows.json")
assert len(rows)==1706 and len({r["canonical_station"] for r in rows})==9
keys=[(r["canonical_station"],r["event"]) for r in rows]
ids=[r["sample_id"] for r in rows]
ne=Counter(keys); ns=Counter(k[0] for k in ne)
weights=np.array([1/(len(ns)*ns[k[0]]*ne[k]) for k in keys])
assert len(ne)==17 and abs(weights.sum()-1)<1e-12
times=[[datetime.fromisoformat(r["origin_time"])+timedelta(hours=h) for h in range(1,25)] for r in rows]
truthfile=np.load(dirs["A",2]/"select_predictions.npz",allow_pickle=False)
truth=truthfile["truth_q_m3_s"]
thresholds=np.array([[r["q95"],r["q99"]] for r in rows],dtype=float)
observed=truth[:,:,None]>thresholds[:,None,:]
timeline=defaultdict(dict); event_indices=defaultdict(list)
for i,k in enumerate(keys):
 event_indices[k].append(i)
 for h,t in enumerate(times[i]):
  entry=(float(truth[i,h]),tuple(observed[i,h].tolist()))
  if t in timeline[k]: assert timeline[k][t]==entry
  timeline[k][t]=entry
phases={p:defaultdict(set) for p in ("entry_exceedance_first_3h","middle_exceedance")}
episodes=[]
for k,values in timeline.items():
 ordered=sorted(values); segments=[]; seg=[]
 for t in ordered:
  if seg and t-seg[-1]!=timedelta(hours=1):segments.append(seg);seg=[]
  seg.append(t)
 if seg:segments.append(seg)
 for tt in segments:
  flags=[values[t][1][0] for t in tt];i=0
  while i<len(tt):
   if not flags[i]:i+=1;continue
   start=i
   while i+1<len(tt) and flags[i+1]:i+=1
   end=i;left=start==0;right=end==len(tt)-1;duration=end-start+1
   episodes.append((k,start,end,left,right))
   if not left:phases["entry_exceedance_first_3h"][k].update(tt[start:min(start+3,end+1)])
   if not left and not right and duration>6:phases["middle_exceedance"][k].update(tt[start+3:end-2])
   i+=1
def phase_brier(prob,phase):
 leads=[]
 for h in range(24):
  station_scores=defaultdict(list)
  for k,points in phases[phase].items():
   ix=[i for i in event_indices[k] if times[i][h] in points]
   if ix:station_scores[k[0]].append(float(np.mean((prob[ix,h,0]-observed[ix,h,0])**2)))
  if station_scores:leads.append(float(np.mean([np.mean(v) for v in station_scores.values()])))
 return float(np.mean(leads)) if leads else None
records=[]; maxerr=0.; config_hashes=defaultdict(set); china_sequences={}
for (g,s),d in dirs.items():
 done=read(d/"completion.json")
 assert done["updates"]==3000 and all(done[k] for k in ("training_ended","selection_ended","final_scoring_ended"))
 assert [v["updates"] for v in done["candidate_checkpoints"]]==[1000,2000,3000]
 assert [r["sample_id"] for r in read(d/"select_rows.json")]==ids
 z=np.load(d/"select_predictions.npz",allow_pickle=False)
 assert z["draws_normalized"].shape==(1706,256,24)
 assert np.array_equal(z["truth_q_m3_s"],truth) and np.array_equal(z["thresholds_q95_q99_m3_s"],thresholds)
 assert np.array_equal(z["observed"],observed)
 base_bank=Path("./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq")
 reference_rows=read(base_bank/"select_rows.json")
 cn_index=[i for i,r in enumerate(reference_rows) if r["country"]=="CN"]
 assert [reference_rows[i]["sample_id"] for i in cn_index]==ids
 normthresholds=np.load(base_bank/"select.npz",allow_pickle=False)["thresholds_normalized"][cn_index]
 prob=(z["draws_normalized"][:,:,:,None]>normthresholds[:,None,None,:]).mean(axis=1)
 assert np.array_equal(prob,z["probability"])
 perlead=(weights[:,None,None]*(prob-observed)**2).sum(axis=0)
 saved=read(d/"scores.json")
 scores=np.array([[next(v["brier"] for v in saved if v["horizon"]==h and v["target"]=="pointwise_exceedance_"+q) for q in ("Q95","Q99")] for h in range(1,25)])
 err=float(np.max(np.abs(perlead-scores)));maxerr=max(maxerr,err);assert err<1e-12
 ds=np.asarray(z["draws_normalized"],dtype=float);y=z["truth_normalized"];count=ds.shape[1]
 crps=np.mean(np.abs(ds-y[:,None,:]),axis=1)-np.sum((2*np.arange(1,count+1)-count-1)[None,:,None]*np.sort(ds,axis=1),axis=1)/(count*count)
 crps_value=float(np.mean(np.sum(weights[:,None]*crps,axis=0)))
 crps_saved=float(np.mean([v["crps"] for v in saved]));assert abs(crps_value-crps_saved)<1e-12
 config=read(d/"model_config.json");config_hashes[s].add(config["initial_state_sha256"])
 expo=np.load(d.parent/"data/exposure.npz",allow_pickle=False)
 china_sequences[g,s]=hashlib.sha256(expo["china_sequence"].tobytes()).hexdigest()
 records.append({"group":g,"seed":s,"brier":float(perlead.mean()),"q95_brier":float(perlead[:,0].mean()),"q99_brier":float(perlead[:,1].mean()),"normalized_crps":crps_value,"entry":phase_brier(prob,"entry_exceedance_first_3h"),"middle":phase_brier(prob,"middle_exceedance"),"selected_updates":done["selected"]["updates"],"selected_score_128draw":done["selected"]["score"]})
assert all(len(v)==1 for v in config_hashes.values())
assert all(china_sequences["B",s]==china_sequences["C",s] for s in (2,13,42))
metrics=("brier","q95_brier","q99_brier","normalized_crps","entry","middle")
stats={}
for g in ("A","B","C"):
 rr=[r for r in records if r["group"]==g]
 stats[g]={m:{"mean":float(np.mean([r[m] for r in rr])),"sample_sd":float(np.std([r[m] for r in rr],ddof=1)),"min":min(r[m] for r in rr),"max":max(r[m] for r in rr)} for m in metrics}
contrasts={}
lookup={(r["group"],r["seed"]):r for r in records}
for ref,cur in (("A","B"),("A","C"),("B","C")):
 contrasts[f"{cur}_vs_{ref}"]={m:{"pct_from_group_means":(stats[ref][m]["mean"]-stats[cur][m]["mean"])/stats[ref][m]["mean"]*100,"paired_seed_pct":[(lookup[ref,s][m]-lookup[cur,s][m])/lookup[ref,s][m]*100 for s in (2,13,42)]} for m in metrics}
support={p:{"stations":len({k[0] for k,v in pp.items() if v}),"source_events":sum(bool(v) for v in pp.values()),"unique_valid_hours":sum(len(v) for v in pp.values())} for p,pp in phases.items()}
print(json.dumps({"state":"independently_recomputed","max_432_brier_error":maxerr,"n_cn":len(rows),"n_cn_source_events":len(ne),"unique_truth_valid_hours":sum(map(len,timeline.values())),"q95_episodes":len(episodes),"phase_support":support,"per_seed":records,"stats":stats,"contrasts":contrasts},allow_nan=False))
