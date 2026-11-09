"""Locked GPU launcher; exact A/B pilot jobs after actual family smoke acceptance."""
import argparse,fcntl,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from .prepare import save
from src.pointwise_benchmark_v1.dataset import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument("--family",choices=("diffpts","nbeatsx"),required=True)
    p.add_argument("--data",required=True);p.add_argument("--release",required=True);p.add_argument("--out",required=True)
    p.add_argument("--gpus",default="0,1")
    a=p.parse_args();gpus=[int(v) for v in a.gpus.split(",")]
    if not 1<=len(gpus)<=2 or len(set(gpus))!=len(gpus) or min(gpus)<0:raise ValueError("one or two distinct GPU indices required")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    release=json.loads(Path(a.release).read_text())
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    data_sha=sha256(Path(a.data)/"manifest.json")
    if release["state"]!="PASS" or release["code_sha"]!=head or release["data_sha"]!=data_sha:raise ValueError("unaccepted exact source/data")
    if a.family not in release["smokes"]:raise ValueError("this family's smoke acceptance is missing")
    for family,smoke in release["smokes"].items():
        if not smoke["complete"] or smoke["smoke"] is not True:raise ValueError("failed smoke: "+family)
    locks=[]
    for gpu in gpus:
        lock=open(Path.cwd()/f"results/experiments/real_extreme_gpu{gpu}.lock","a+")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    def free(required):
        memory=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
        inventory={int(row.split(",")[0]):int(row.split(",")[1]) for row in memory.splitlines()}
        if any(gpu not in inventory or inventory[gpu]>256 for gpu in required):raise RuntimeError("requested GPU occupied before launch")
    free(gpus);jobs=[];logs=[]
    def sync(state):
        save(out/"queue_status.json",dict(state=state,code_sha=head,data_sha=data_sha,at_utc=datetime.now(timezone.utc).isoformat(),
            requested_gpus=gpus,planned_groups=["A","B"],jobs=[{k:v for k,v in j.items() if k!="process"} for j in jobs]))
    for i,group in enumerate(("A","B")):
        if len(gpus)==1 and jobs:
            jobs[-1]["exit_code"]=jobs[-1]["process"].wait();sync("running")
            if jobs[-1]["exit_code"]!=0:sync("failed");sys.exit(1)
            free(gpus)
        gpu=gpus[i%len(gpus)];run=out/f"{group}_{a.family}"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        command=[sys.executable,"src/run_versioned.py","--run-dir",str(run),"--",sys.executable,"-m","src.real_extreme_v1.train",
            "--data",a.data,"--group",group,"--family",a.family,"--out",str(run/"artifacts")]
        log=open(out/f"{group}.launch.log","xb");logs.append(log)
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(dict(group=group,family=a.family,gpu=gpu,pid=child.pid,run=str(run),process=child));sync("running")
    for j in jobs:j["exit_code"]=j["process"].wait();sync("running")
    sync("complete" if all(j["exit_code"]==0 for j in jobs) else "failed")
    if any(j["exit_code"]!=0 for j in jobs):sys.exit(1)
if __name__=="__main__":main()
