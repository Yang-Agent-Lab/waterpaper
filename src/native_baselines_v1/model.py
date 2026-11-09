"""Input/output adaptation only; author architectures and MQLoss are unchanged."""
import inspect
import hashlib
from pathlib import Path
import neuralforecast
import numpy as np
import torch
from torch import nn
from neuralforecast.models import LSTM, TCN, NHITS, NBEATSx, TFT
from neuralforecast.losses.pytorch import MQLoss

QUANTILES=[.005,.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995]
CLASSES=dict(native_lstm=LSTM,native_tcn=TCN,native_nhits=NHITS,native_nbeatsx=NBEATSx,native_tft=TFT)
MODELS=tuple(CLASSES)
ALL_MODELS=('persistence',)+MODELS
H=(1,3,6,12)

def targets(y):return torch.stack([y[:,:h].amax(1) for h in H],1)

def coherent(raw):
    return raw.sort(dim=-1).values.cummax(dim=1).values

def quantile_samples(raw,n):
    q=coherent(raw)
    levels=torch.tensor(QUANTILES,device=q.device,dtype=q.dtype)
    u=(torch.arange(n,device=q.device,dtype=q.dtype)+.5)/n
    hi=torch.searchsorted(levels,u).clamp(1,len(QUANTILES)-1);lo=hi-1
    weight=((u-levels[lo])/(levels[hi]-levels[lo])).clamp(0,1)
    return (q[:,:,lo]+(q[:,:,hi]-q[:,:,lo])*weight).transpose(1,2)

class Native(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name;self.protected=None
        if neuralforecast.__version__!='3.1.0':raise RuntimeError('approved author package must be NeuralForecast 3.1.0')
        if name=='persistence':
            self.register_buffer('_device',torch.zeros(()));self.config={'class':'persistence','trained':False};return
        cls=CLASSES[name]
        loss=MQLoss(quantiles=QUANTILES)
        kw=dict(h=4,input_size=12,hist_exog_list=['P'],futr_exog_list=None,stat_exog_list=None,loss=loss,valid_loss=MQLoss(quantiles=QUANTILES),scaler_type='identity',random_seed=2,learning_rate=.001,max_steps=22600,batch_size=128,windows_batch_size=128)
        self.native=cls(**kw)
        self.config={'class':cls.__module__+'.'+cls.__name__,'overrides':{k:v for k,v in kw.items() if k not in ('loss','valid_loss')},'quantiles':QUANTILES,'constructor_defaults':{k:repr(v.default) for k,v in inspect.signature(cls.__init__).parameters.items() if k not in ('self',) and v.default is not inspect.Parameter.empty},'parameter_count':sum(p.numel() for p in self.native.parameters() if p.requires_grad),'normalization':'prepared fixed station fit Q95; no native scaling or target recentering','output_positions':'nested window peaks 1/3/6/12h; not hourly trajectory','tft_no_future_inputs':'author forward repeats last observed Q internally when futr_exog=None; no future observation'}
        self.config['author_source_sha256']={inspect.getfile(c):hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in (cls,MQLoss)}
    def raw(self,x):
        if x.ndim!=3 or x.shape[1:]!=(12,2):raise ValueError('history must be [B,12,P/Q]')
        if self.name=='persistence':return x[:,-1,1,None,None].expand(-1,4,len(QUANTILES))
        batch=dict(insample_y=x[:,:,1:2],insample_mask=torch.ones_like(x[:,:,1:2]),hist_exog=x[:,:,0:1],futr_exog=None,stat_exog=None)
        result=self.native(batch)
        if result.shape!=(len(x),4,len(QUANTILES)):raise ValueError('native quantile shape '+str(result.shape))
        return result
    def encode(self,x,f=None,shadow=None):
        if shadow:
            if shadow not in ('input_rain_zero','input_flow_zero'):raise ValueError(shadow)
            x=x.clone();x[:,:,0 if shadow=='input_rain_zero' else 1]=0
        return self.raw(x)
    def sample(self,raw,n,generator=None):return quantile_samples(raw,n)
    def loss(self,x,f,y,generator=None):
        raw=self.raw(x);value=self.native.loss(y=targets(y).unsqueeze(-1),y_hat=self.native.loss.domain_map(raw))
        return value,dict(mqloss=float(value.detach()))
    def health_modules(self):
        return dict(self.native.named_children()) if self.name!='persistence' else {}

def build_model(name):
    if name not in ALL_MODELS:raise ValueError(name)
    torch.manual_seed(2)
    return Native(name)

def raw_predictions(model,pool,batch=128):
    device=next(model.parameters(),model._buffers.get('_device')).device
    model.eval();parts=[]
    with torch.no_grad():
        for lo in range(0,len(pool['rows']),batch):parts.append(model.raw(torch.as_tensor(pool['x'][lo:lo+batch],device=device)).cpu().numpy())
    raw=np.concatenate(parts)
    if not np.isfinite(raw).all():raise FloatingPointError('nonfinite native quantiles')
    return raw

def predict(model,pool,n=512,seed=None,batch=128,shadow=None):
    if shadow is None:raw=torch.from_numpy(raw_predictions(model,pool,batch))
    else:
        altered=dict(pool,x=pool['x'].copy());altered['x'][:,:,0 if shadow=='input_rain_zero' else 1]=0
        raw=torch.from_numpy(raw_predictions(model,altered,batch))
    return quantile_samples(raw,n).numpy()

def quantile_diagnostics(raw):
    adapted=coherent(torch.from_numpy(raw)).numpy()
    return dict(negative_fraction=float((raw<0).mean()),quantile_crossing_fraction=float((np.diff(raw,axis=-1)<0).mean()),nested_window_violation_fraction=float((np.diff(raw,axis=1)<0).mean()),adapted_value_fraction=float((raw!=adapted).mean()),negative_values_clipped=False,algorithm='sort_quantiles_then_cummax_windows; inverse_linear_quantiles_midpoints512; fixed_extreme_tails_v1')
