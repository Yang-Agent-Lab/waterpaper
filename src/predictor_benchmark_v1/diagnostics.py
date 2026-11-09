"""Isolated temporary interventions and module health; no optimizer updates."""
from contextlib import contextmanager
import torch
from protected_flood_risk_v1.diagnostics import isolated_diagnostic,cached_diagnostic
from protected_flood_risk_v1.training import state_hash,content_digest
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.runner import predict
from multiwindow_peak_v1.model import crps
from .statistics import mean
import numpy as np

class _NoOptimizer:
    def state_dict(self):return {}
    def load_state_dict(self,state):pass

@contextmanager
def intervene(model,kind):
    with isolated_diagnostic(model,_NoOptimizer()):
        model.eval()
        with torch.no_grad():yield lambda x,f:model.encode(x,f,kind)

def interventions(model,pools,ids,out,code,checkpoint):
    identity=dict(code_version=code,checkpoint_sha256=state_hash(model),data={r:content_digest(subset(p,ids[r])) for r,p in pools.items()},sample_ids={r:[p['rows'][i]['window_id'] for i in ids[r]] for r,p in pools.items()},data_role_id='prepared-ab720ba',protocol='S2-window-peak-Q95',schema_version=1,content_digest={r:content_digest(subset(p,ids[r])) for r,p in pools.items()},algorithm_version='predictor-intervention-v1',model=model.name,scenarios=128,seed=93481)
    def compute():
        records=[]
        for role,pool in pools.items():
            sub=subset(pool,ids[role]);countries=sorted({r['country'] for r in sub['rows']})
            with intervene(model,None):base=predict(model,sub,128,93481)
            truth=np.column_stack([sub['y'][:,:h].max(1) for h in (1,3,6,12)])
            def metrics(draw,ix):
                rows=[sub['rows'][i] for i in ix];c=crps(torch.from_numpy(draw[ix]),torch.from_numpy(truth[ix])).numpy()[:,2]
                return dict(crps=mean(c,rows),brier=mean(((draw[ix,:,2]>1).mean(1)-(truth[ix,2]>1))**2,rows),peak_mae=mean(abs(np.median(draw[ix,:,2],axis=1)-truth[ix,2]),rows))
            for kind in ('input_rain_zero','input_flow_zero',('adapter_zero' if model.name=='current_s2' else 'encoder_zero')):
                with intervene(model,kind):draw=predict(model,sub,128,93481,shadow=kind)
                for country in countries:
                    ix=np.array([i for i,r in enumerate(sub['rows']) if r['country']==country]);a=metrics(base,ix);b=metrics(draw,ix)
                    for metric in a:records.append(dict(model=model.name,module=kind,kind='temporary_intervention',role=role,country=country,metric=metric,target='window_exceedance_Q95' if metric=='brier' else 'cumulative_window_peak',horizon=6,delta=b[metric]-a[metric],n=len(ix),checkpoint=checkpoint,state='evaluable'))
            for module in ('decoder','head'):records.append(dict(model=model.name,module=module,kind='temporary_intervention',role=role,country='all',metric='crps',horizon=6,delta=None,n=len(sub['rows']),checkpoint=checkpoint,state='NOT_EVALUABLE',reason='no meaningful substitute specified'))
        return records
    return cached_diagnostic(out/'diagnostic_cache',identity,compute)
