"""Bounded historical state residuals on the repaired original DiffPTS pq model."""
import torch
from torch import nn
from torch.nn import functional as F
from .models import DiffPTS, build_model, random_state

WINDOWS=(3,6,12,24)
SINGLE_INTERFACE_VARIANTS={
    'state_mean_only':('mean',),
    'state_variance_only':('variance',),
    'state_denoise_only':('denoiser',),
}
STATE_VARIANTS=('state_q','state_pq','state_decay_tau6','state_constant_mean',*SINGLE_INTERFACE_VARIANTS)
STATE_DECAY_TAU_HOURS=6.


def state_variance(g,delta):
    inverse=g+torch.log(-torch.expm1(-g))
    return F.softplus(inverse+delta).clamp_min(1e-8)


def history_state(x,include_rain):
    if x.ndim!=3 or tuple(x.shape[1:])!=(24,2) or not torch.isfinite(x).all():
        raise ValueError('state features require finite existing 24-hour historical P/Q only')
    q=x[:,:,1];p=x[:,:,0]
    values=[q[:,-1]]
    for h in WINDOWS:
        recent=q[:,-h:]
        values.extend([recent.mean(1),recent.std(1,unbiased=False),q[:,-1]-recent[:,0]])
    rain=[p[:,-1]]+[p[:,-h:].sum(1) for h in WINDOWS]+[p[:,-1]-p[:,-3]]
    if not include_rain:rain=[torch.zeros_like(v) for v in rain]
    features=torch.stack(values+rain,dim=1)
    return features.sign()*torch.log1p(features.abs())


class StateDiffPTS(DiffPTS):
    def __init__(self,variant,history_hours,forecast_hours,device):
        if variant not in STATE_VARIANTS or (history_hours,forecast_hours)!=(24,24):
            raise ValueError('approved state variants require H24/F24 pq')
        super().__init__('pq',history_hours,forecast_hours,device,variance_floor=1e-8)
        self.name=variant;self.include_rain=variant in ('state_pq','state_decay_tau6','state_constant_mean',*SINGLE_INTERFACE_VARIANTS)
        if variant in ('state_decay_tau6','state_constant_mean',*SINGLE_INTERFACE_VARIANTS):
            decay=torch.exp(-torch.arange(forecast_hours,dtype=torch.float32,device=device)/STATE_DECAY_TAU_HOURS)
            weights=decay if variant!='state_constant_mean' else decay.mean().expand_as(decay).clone()
            self.register_buffer('state_lead_gate',weights)
        self.state_encoder=nn.Sequential(nn.Linear(19,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.state_mean=nn.Linear(32,forecast_hours)
        self.state_variance_residual=nn.Linear(32,forecast_hours)
        self.state_hidden=nn.Linear(32,128)
        for head in (self.state_mean,self.state_variance_residual,self.state_hidden):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self._current_state=None
        self.diffusion.diffussion_model.lin1.register_forward_hook(self._inject_hidden)
        self.config.update(name=variant,state_condition=variant,state_feature_dimension=19,state_hidden_dimension=32,
            state_encoder=[19,32,32],state_added_parameters=7504,
            state_features='Q current + mean/population std/current-first at 3/6/12/24h; P current + sums3/6/12/24 + current-first3h',
            state_transform='signed log1p per feature; existing bank units; no fitted scaling',
            state_rain_slots='actual historical P' if self.include_rain else 'six zeros; original mean still sees P/Q',
            state_injection='add hourly mean residual; stable inverse-softplus variance + residual then softplus/floor1e-8; add shared128 residual after denoiser lin1 before softplus',
            state_zero_initialized_heads=True,variance_floor=1e-8,
            stage1_loss_unchanged=True,state_future_information=False)
        if variant in ('state_decay_tau6','state_constant_mean'):
            self.config.update(state_injection='lead-gated hourly mean residual; interpolate base/state variance in variance space with exact 0/1 endpoints; lead-gate shared128 residual after denoiser lin1 before softplus',
                state_lead_gate_kind='exp(-(h-1)/6)' if variant=='state_decay_tau6' else 'constant arithmetic mean of exp(-(h-1)/6), h=1..24',
                state_lead_gate=weights.tolist(),state_gate_trainable=False,state_added_parameters=7504)
        if variant in SINGLE_INTERFACE_VARIANTS:
            self.state_active_interfaces=SINGLE_INTERFACE_VARIANTS[variant]
            self.state_inactive_heads=[head for interface,head in (('mean','state_mean'),('variance','state_variance_residual'),('denoiser','state_hidden')) if interface not in self.state_active_interfaces]
            active_modules=[self.state_encoder]+[getattr(self,head) for head in ('state_mean','state_variance_residual','state_hidden') if head not in self.state_inactive_heads]
            active_parameters=sum(p.numel() for module in active_modules for p in module.parameters())
            self.config.update(state_active_interfaces=list(self.state_active_interfaces),state_inactive_heads=self.state_inactive_heads,
                state_inactive_heads_gradient='not observed by design',state_active_state_parameters=active_parameters,
                state_inactive_state_parameters=7504-active_parameters,state_lead_gate_kind='exp(-(h-1)/6)',
                state_lead_gate=weights.tolist(),state_gate_trainable=False,state_added_parameters=7504,
                state_injection='only the declared active interface receives the lead-gated state residual; inactive interfaces follow the original floor DiffPTS path')

    def conditions(self,x):
        fx,gx=super().conditions(x)
        self._current_state=self.state_encoder(history_state(x,self.include_rain))
        if hasattr(self,'state_active_interfaces'):
            if 'mean' in self.state_active_interfaces:
                fx=fx+(self.state_mean(self._current_state)*self.state_lead_gate).unsqueeze(-1)
            if 'variance' in self.state_active_interfaces:
                variance_state=state_variance(gx,self.state_variance_residual(self._current_state).unsqueeze(-1))
                weight=self.state_lead_gate.view(1,-1,1)
                blended=gx+(variance_state-gx)*weight
                gx=torch.where(weight==0,gx,torch.where(weight==1,variance_state,blended))
            return fx,gx
        mean_residual=self.state_mean(self._current_state)
        variance_state=state_variance(gx,self.state_variance_residual(self._current_state).unsqueeze(-1))
        if hasattr(self,'state_lead_gate'):
            weights=self.state_lead_gate
            fx=fx+(mean_residual*weights).unsqueeze(-1)
            weight=weights.view(1,-1,1)
            blended=gx+(variance_state-gx)*weight
            gx=torch.where(weight==0,gx,torch.where(weight==1,variance_state,blended))
        else:
            fx=fx+mean_residual.unsqueeze(-1)
            gx=variance_state
        return fx,gx

    def _inject_hidden(self,module,inputs,output):
        if self._current_state is None:raise RuntimeError('state conditions must precede denoising')
        if hasattr(self,'state_active_interfaces') and 'denoiser' not in self.state_active_interfaces:return output
        state=self._current_state
        if len(output)%len(state):raise ValueError('denoiser/state batch identity mismatch')
        state=state.repeat_interleave(len(output)//len(state),dim=0)
        residual=self.state_hidden(state).unsqueeze(1)
        if hasattr(self,'state_lead_gate'):
            if output.ndim!=3 or tuple(output.shape[1:])!=(self.forecast_hours,128):
                raise ValueError('gated denoiser output must be [batch,forecast_hours,128]')
            residual=residual*self.state_lead_gate.view(1,-1,1)
        return output+residual


def build_state_model(name,route,history_hours,forecast_hours,seed=2,device='cpu'):
    if route!='pq':raise ValueError('all stage2 groups use the same original pq bank')
    device=torch.device(device)
    if name=='extreme_resampling':
        model=build_model('diffpts',route,history_hours,forecast_hours,seed,device,variance_floor=1e-8)
        model.config.update(name=name,state_condition='none; original repaired architecture',sampling_intervention='station_preserving_extreme_fit_v1')
        return model
    with random_state(device,seed=seed):model=StateDiffPTS(name,history_hours,forecast_hours,device).to(device)
    model.config.update(seed=seed,parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
