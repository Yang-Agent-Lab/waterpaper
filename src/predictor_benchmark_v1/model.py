"""Three history encoders; the original S2 probability mathematics is reused unchanged."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from multiwindow_peak_v1.model import PeakModel, blocks, maxima, crps, nll
from protected_flood_risk_v1.model import TemporalAdapter

MODELS=('current_s2','direct_lstm','direct_tcn')
HEAD_SEED=57002

class CausalTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv1d(2 if i==0 else 64,64,3,dilation=d) for i,d in enumerate((1,2,4))])
        self.residual=nn.Conv1d(2,64,1)
    def sequence(self,x):
        z=x.transpose(1,2)
        for i,conv in enumerate(self.layers):
            residual=self.residual(z) if i==0 else z
            z=F.relu(conv(F.pad(z,(2*conv.dilation[0],0)))+residual)
        return z.transpose(1,2)
    def forward(self,x):return self.sequence(x)[:,-1]

class Predictor(PeakModel):
    def __init__(self,name,p0,decoder,head):
        nn.Module.__init__(self);self.arm='S2';self.name=name;self.point=None
        self.protected=copy.deepcopy(p0).requires_grad_(False).eval() if name=='current_s2' else None
        self.adapter=TemporalAdapter() if name=='current_s2' else None
        self.encoder=nn.LSTM(2,64,batch_first=True) if name=='direct_lstm' else CausalTCN() if name=='direct_tcn' else None
        self.decoder=copy.deepcopy(decoder);self.head=copy.deepcopy(head)
    def train(self,mode=True):
        nn.Module.train(self,mode)
        if self.protected is not None:self.protected.eval()
        return self
    def encode(self,x,f,shadow=None):
        if shadow in ('input_rain_zero','input_flow_zero'):
            x=x.clone();x[:,:,0 if shadow=='input_rain_zero' else 1]=0
        if self.name=='current_s2':
            with torch.no_grad():h=self.protected.encode(x,f)
            value=self.adapter(x,'adapter_off' if shadow=='adapter_zero' else None)[0]
            h=h+value
        elif self.name=='direct_lstm':h=self.encoder(x)[1][0][-1]
        else:h=self.encoder(x)
        return torch.zeros_like(h) if shadow=='encoder_zero' else h
    def health_modules(self):
        modules={'decoder':self.decoder,'head':self.head}
        if self.protected is not None:modules.update(protected=self.protected,adapter=self.adapter,rain_encoder=self.adapter.rain,flow_encoder=self.adapter.flow,lag_query=self.adapter.query,lag_key=self.adapter.key,lag_value=self.adapter.value)
        else:modules['encoder']=self.encoder
        return modules

def build_model(name,p0,seed=2):
    if name not in MODELS:raise ValueError('unknown model '+name)
    # The common probability head has a separate RNG stream; encoder construction
    # cannot consume or perturb its initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HEAD_SEED);decoder=nn.GRUCell(68,64);head=nn.Linear(64,2)
        torch.manual_seed(seed);return Predictor(name,p0,decoder,head)
