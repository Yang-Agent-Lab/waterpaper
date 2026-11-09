"""Author methods adapted to history-only inputs and future hourly Q supervision."""
import inspect
import hashlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from .vendor import DIFFPTS_SHA, diffpts_definitions, ppm_definitions

QUANTILES=[.005,.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995]
NATIVE_NAMES=('nbeatsx','lstm','tcn','tft')
PPM_SHA='6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9'


@contextmanager
def random_state(device, generator=None, seed=None):
    if generator is not None:
        seed=int(torch.randint(0,2**31-1,(),generator=generator,device=generator.device).item())
    if seed is None:
        yield;return
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:torch.cuda.manual_seed(seed)
        yield


def quantile_samples(raw,n):
    if type(n) is not int or n<1:raise ValueError('sample count must be positive')
    q=raw.sort(dim=-1).values
    levels=torch.tensor(QUANTILES,device=q.device,dtype=q.dtype)
    u=(torch.arange(n,device=q.device,dtype=q.dtype)+.5)/n
    hi=torch.searchsorted(levels,u).clamp(1,len(QUANTILES)-1);lo=hi-1
    weight=((u-levels[lo])/(levels[hi]-levels[lo])).clamp(0,1)
    return (q[:,:,lo]+(q[:,:,hi]-q[:,:,lo])*weight).transpose(1,2)


class ForecastModel(nn.Module):
    def __init__(self,name,route,history_hours,forecast_hours):
        super().__init__()
        if route not in ('pq','q_only'):raise ValueError('route must be pq or q_only')
        if any(type(h) is not int or h<1 for h in (history_hours,forecast_hours)):raise ValueError('positive integer history/forecast required')
        self.name=name;self.route=route;self.history_hours=history_hours;self.forecast_hours=forecast_hours
        self.config=dict(name=name,route=route,history_hours=history_hours,forecast_hours=forecast_hours,
                         input_channels=['P','Q'] if route=='pq' else ['Q'],target='hourly_Q',
                         normalization='prepared station fit Q95; no future P',physical_projection=False)

    def check(self,x,y=None):
        if x.ndim!=3 or tuple(x.shape[1:])!=(self.history_hours,2 if self.route=='pq' else 1):
            raise ValueError('history-only input shape mismatch')
        if not torch.isfinite(x).all():raise ValueError('nonfinite history')
        if y is not None and (tuple(y.shape)!=(len(x),self.forecast_hours) or not torch.isfinite(y).all()):
            raise ValueError('supervision must contain future Q only [B,F]')

    def health_modules(self):
        return dict(self.named_children())

    def predict_samples(self,x,n,seed=2):
        self.check(x)
        if type(n) is not int or n<1:raise ValueError('sample count must be positive')
        training=self.training
        try:
            self.eval()
            with torch.no_grad(),random_state(x.device,seed=seed):
                result=self._samples(x,n)
            if result.shape!=(len(x),n,self.forecast_hours) or not torch.isfinite(result).all():
                raise FloatingPointError('invalid forecast sample shape or values')
            return result
        finally:self.train(training)


class Native(ForecastModel):
    def __init__(self,name,route,history_hours,forecast_hours,seed):
        super().__init__(name,route,history_hours,forecast_hours)
        import neuralforecast
        from neuralforecast.models import NBEATSx,LSTM,TCN,TFT
        from neuralforecast.losses.pytorch import MQLoss
        if neuralforecast.__version__!='3.1.0':raise RuntimeError('NeuralForecast 3.1.0 required')
        cls=dict(nbeatsx=NBEATSx,lstm=LSTM,tcn=TCN,tft=TFT)[name]
        kw=dict(h=forecast_hours,input_size=history_hours,hist_exog_list=['P'] if route=='pq' else None,
                futr_exog_list=None,stat_exog_list=None,loss=MQLoss(quantiles=QUANTILES),
                valid_loss=MQLoss(quantiles=QUANTILES),scaler_type='identity',random_seed=seed,
                learning_rate=.001,max_steps=10000,batch_size=64,windows_batch_size=64)
        self.native=cls(**kw)
        self.config.update(author='NeuralForecast 3.1.0',author_class=cls.__module__+'.'+cls.__name__,
                           loss_definition='native MQLoss17 on future hourly Q',quantiles=QUANTILES,
                           sample_method='per-hour sorted quantile linear inverse; fixed .005/.995 tails; midpoint grid',
                           temporal_joint_distribution='not established; shared quantile grid is a marginal sampling approximation',
                           overrides={k:v for k,v in kw.items() if k not in ('loss','valid_loss')},
                           constructor_defaults={k:repr(v.default) for k,v in inspect.signature(cls.__init__).parameters.items() if k!='self' and v.default is not inspect.Parameter.empty},
                           author_source_sha256={c.__name__:hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in (cls,MQLoss)})

    def raw_quantiles(self,x):
        self.check(x)
        q=x[:,:,-1:]
        batch=dict(insample_y=q,insample_mask=torch.ones_like(q),hist_exog=x[:,:,:1] if self.route=='pq' else None,
                   futr_exog=None,stat_exog=None)
        raw=self.native(batch)
        if raw.shape!=(len(x),self.forecast_hours,len(QUANTILES)):raise ValueError('native quantile shape mismatch')
        return raw

    def loss(self,x,y,generator=None):
        self.check(x,y)
        with random_state(x.device,generator=generator):
            raw=self.raw_quantiles(x)
            value=self.native.loss(y=y.unsqueeze(-1),y_hat=self.native.loss.domain_map(raw))
        return value,dict(mqloss=float(value.detach()),loss_definition=self.config['loss_definition'])

    def _samples(self,x,n):return quantile_samples(self.raw_quantiles(x),n)


class PPM(ForecastModel):
    def __init__(self,route,history_hours,forecast_hours):
        super().__init__('ppm',route,history_hours,forecast_hours)
        if route!='q_only':raise ValueError('original PPM MLP is q_only')
        ns,hashes=ppm_definitions(PPM_SHA)
        self.args=SimpleNamespace(seq_len=history_hours,pred_len=forecast_hours,d_model=256,d_ff=512,K=4,tau=.1,var=.3,nu=5)
        self.native=ns['Model'](self.args)
        self._kde_loss=ns['kde_loss'];self._mean_loss=ns['mean_loss']
        self.config.update(author_sha=PPM_SHA,author_source_sha256=hashes,author_class='model_ppm.PPMF.MLP_PPM.Model',
                           overrides=vars(self.args),loss_definition='tau * author Gaussian KDE NLL + sample-mean MSE; K=4,tau=.1,bandwidth=.3',
                           sample_method='author conditional Gaussian latent + MLP mapping',extra_pretraining=False)

    def _samples(self,x,n):return self.native(x,n).squeeze(-1)

    def loss(self,x,y,generator=None):
        self.check(x,y)
        with random_state(x.device,generator=generator):
            pred=self.native(x,self.args.K)
            target=y.unsqueeze(-1)
            kde=self._kde_loss(self,pred,target,self.args.tau,kernel='gaussian',nu=self.args.nu)
            mse=self._mean_loss(self,pred,target)
            value=self.args.tau*kde+mse
        return value,dict(kde_nll=float(kde.detach()),mean_mse=float(mse.detach()),loss_definition=self.config['loss_definition'])


class DiffPTS(ForecastModel):
    def __init__(self,route,history_hours,forecast_hours,device,variance_floor=0.):
        super().__init__('diffpts',route,history_hours,forecast_hours)
        if variance_floor not in (0.,1e-8):raise ValueError('only original zero floor or approved paper floor 1e-8')
        self.variance_floor=float(variance_floor)
        ns,hashes=diffpts_definitions();self.author_functions=ns
        channels=2 if route=='pq' else 1
        args=SimpleNamespace(timesteps=20,beta_schedule='linear',beta_start=.0001,beta_end=.01,
                             enc_in=channels,CART_input_x_embed_dim=32,embed='timeF',freq='h',dropout=.05)
        self.diffusion=ns['DiffPTS'](args,device)
        # Author history embedding and denoiser used the same enc_in; only Q is supervised here.
        self.diffusion.diffussion_model=ns['ConditionalGuidedModel'](args.timesteps,1)
        self.mean=ns['iTransformerEnc'](seq_len=history_hours,pred_len=forecast_hours,enc_in=channels,
                                      factor=3,n_heads=8,d_ff=1024,activation='gelu',e_layers=2,d_model=512,dropout=.05)
        self.variance=ns['SigmaEstimation'](history_hours,forecast_hours,1,512)
        self.config.update(author_sha=DIFFPTS_SHA,author_source_sha256=hashes,
                           cond_backbone='itransformer',author_default_cond_backbone='patchtst',
                           loss_definition='Q-only joint GaussianNLL (.5*(log gx + (fx-y)^2/gx)) + denoising noise MSE',
                           diffusion_steps=20,beta_schedule='linear',beta_start=.0001,beta_end=.01,
                           mean_config=dict(d_model=512,n_heads=8,d_ff=1024,e_layers=2,dropout=.05,factor=3),
                           task_adaptation='historical P/Q tokens in author iTransformer; Q mean token only; Q history sigma; Q-only denoiser; history embedding retains input channel dimension',
                           unused_author_modules=['diffusion.enc_embedding','diffusion.diffussion_model.sigma_lin (variance return unused by original loss/sampler)'],
                           sample_method='author p_sample_loop, 20 steps',temporal_joint_distribution='marginals only; no validated temporal dependence')
        if self.variance_floor:
            self.config.update(variance_floor=self.variance_floor,
                numerical_alignment='paper variance clamp_min; same gx used in NLL, forward noise, denoising and reverse sampling; no new loss')

    def clamp_variance(self,value):
        return value.clamp_min(self.variance_floor) if self.variance_floor else value

    def conditions(self,x):
        self.check(x)
        channels=x.shape[-1]
        fx=self.mean(x).reshape(len(x),channels,self.forecast_hours).transpose(1,2)[:,:,-1:]
        gx=self.clamp_variance(self.variance(x[:,:,-1:]))
        return fx,gx

    def loss(self,x,y,generator=None,observer=None):
        self.check(x,y)
        with random_state(x.device,generator=generator):
            fx,gx=self.conditions(x);target=y.unsqueeze(-1);n=len(x);m=self.diffusion
            t=torch.randint(0,m.num_timesteps,(n//2+1,),device=x.device)
            t=torch.cat([t,m.num_timesteps-1-t])[:n]
            nll=(.5*(torch.log(gx)+(fx-target).square()/gx)).mean()
            e=torch.randn_like(target)
            forward_variance=self.author_functions['cal_forward_noise'](m.betas_bar,gx,t)
            noise=e*torch.sqrt(forward_variance)
            yt=self.author_functions['q_sample'](target,fx,m.alphas_bar_sqrt,m.one_minus_alphas_bar_sqrt,t,noise=noise)
            pred,_=m(x,None,yt,fx,gx,t)
            denoise=(e-pred).square().mean();value=nll+denoise
            if observer is not None:
                observer(dict(mu=fx,gx=gx,target=target,forward_variance=forward_variance,
                    sampled_noise=noise,noisy_target=yt,noise_prediction=pred,
                    gaussian_nll=nll,denoise_mse=denoise,total_loss=value))
        return value,dict(gaussian_nll=float(nll.detach()),denoise_mse=float(denoise.detach()),loss_definition=self.config['loss_definition'])

    def _samples(self,x,n):
        fx,gx=self.conditions(x);m=self.diffusion
        tiled_x=x.repeat_interleave(n,dim=0)
        fx=fx.repeat_interleave(n,dim=0);gx=gx.repeat_interleave(n,dim=0)
        seq=self.author_functions['p_sample_loop'](m,tiled_x,None,fx,gx,fx,m.num_timesteps,
             m.alphas,m.one_minus_alphas_bar_sqrt,m.alphas_cumprod,m.alphas_cumprod_sum,
             m.alphas_cumprod_prev,m.alphas_cumprod_sum_prev,m.betas_tilde,m.betas_bar,m.betas_tilde_m_1,m.betas_bar_m_1)
        return seq[-1].reshape(len(x),n,self.forecast_hours)


def build_model(name,route,history_hours,forecast_hours,seed=2,device='cpu',variance_floor=0.):
    device=torch.device(device)
    name=name.lower().removeprefix('native_')
    if name!='diffpts' and variance_floor:raise ValueError('variance floor applies only to DiffPTS')
    with random_state(device,seed=seed):
        if name in NATIVE_NAMES:model=Native(name,route,history_hours,forecast_hours,seed)
        elif name=='ppm':model=PPM(route,history_hours,forecast_hours)
        elif name=='diffpts':model=DiffPTS(route,history_hours,forecast_hours,device,variance_floor=variance_floor)
        else:raise ValueError('unsupported model '+name)
    model=model.to(device)
    model.config.update(seed=seed,parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
