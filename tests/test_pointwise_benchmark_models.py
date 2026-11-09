"""Synthetic model checks; execute only in the isolated server environment."""
import io
import pytest
import torch
from pointwise_benchmark_v1.models import build_model,quantile_samples,QUANTILES

CONFIGS=[(name,route) for route in ('pq','q_only') for name in ('diffpts','nbeatsx','lstm','tcn','tft')]+[('ppm','q_only')]


def test_quantile_adapter_does_not_accumulate_time():
    raw=torch.tensor([[[10.]*17,[2.]*17,[8.]*17]])
    result=quantile_samples(raw,11)
    assert result.shape==(1,11,3)
    assert torch.equal(result[0,0],torch.tensor([10.,2.,8.]))
    with pytest.raises(ValueError):quantile_samples(raw,0)


@pytest.mark.parametrize('name,route',CONFIGS)
def test_real_update_save_restore_and_future_input_boundary(name,route):
    torch.set_num_threads(1)
    model=build_model(name,route,24,4,seed=2)
    x=torch.linspace(.1,1.,2*24*(2 if route=='pq' else 1)).reshape(2,24,-1)
    y=torch.tensor([[.5,.8,.4,.3],[.2,.3,.7,.4]])
    # The interface accepts neither future covariates nor multi-channel labels.
    with pytest.raises(ValueError):model.loss(torch.cat([x,x[:,:4]],dim=1),y)
    with pytest.raises(ValueError):model.loss(x,torch.stack([y,y],dim=-1))
    before={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    generator=torch.Generator().manual_seed(9)
    loss,details=model.loss(x,y,generator=generator)
    assert loss.ndim==0 and torch.isfinite(loss)
    assert details['loss_definition']==model.config['loss_definition']
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    pred=model.predict_samples(x,7,seed=31)
    assert pred.shape==(2,7,4) and torch.isfinite(pred).all()
    assert torch.equal(pred,model.predict_samples(x,7,seed=31))
    saved=io.BytesIO();torch.save(model.state_dict(),saved);saved.seek(0)
    restored=build_model(name,route,24,4,seed=42)
    restored.load_state_dict(torch.load(saved,weights_only=True))
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert torch.equal(pred,restored.predict_samples(x,7,seed=31))
    assert model.config['physical_projection'] is False
    if route=='pq':
        model.eval();changed=x.clone();changed[:,:,0]+=1
        # Same random numbers isolate actual P dependence, not sampling variance.
        assert not torch.allclose(model.predict_samples(x,7,seed=31),model.predict_samples(changed,7,seed=31),atol=1e-7,rtol=1e-7)
        if name=='diffpts':
            original_mean,original_sigma=model.conditions(x)
            changed_mean,changed_sigma=model.conditions(changed)
            assert not torch.allclose(original_mean,changed_mean)
            assert torch.equal(original_sigma,changed_sigma)


def test_original_ppm_rejects_rain_route():
    with pytest.raises(ValueError,match='q_only'):build_model('ppm','pq',24,4)


def test_isolated_original_nn_modules_without_dataset_imports():
    import subprocess
    import sys
    code='''
import importlib.abc
import inspect
import sys
from pathlib import Path
from importlib.metadata import distribution
class BlockDatasetStack(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch_timeseries','sktime'):
            raise AssertionError('top-level dataset import attempted: '+fullname)
sys.meta_path.insert(0,BlockDatasetStack())
from pointwise_benchmark_v1.vendor import torch_timeseries_nn
symbols,provenance=torch_timeseries_nn()
assert provenance['torch_timeseries.version']=='0.2.5'
root=Path(distribution('torch-timeseries').locate_file('torch_timeseries/nn')).resolve()
for cls in symbols.values():
    assert Path(inspect.getfile(cls)).resolve().parent==root
assert not any(name=='sktime' or name.startswith('torch_timeseries') for name in sys.modules)
assert len([value for key,value in provenance.items() if key.endswith('.py') and len(value)==64])==3
import torch
embedding=symbols['DataEmbedding_inverted'](4,8,dropout=0)
attention=symbols['AttentionLayer'](symbols['FullAttention'](False,attention_dropout=0),8,2)
encoder=symbols['Encoder']([symbols['EncoderLayer'](attention,8,16,dropout=0)])
x=torch.randn(2,4,2,requires_grad=True)
y,_=encoder(embedding(x,None))
assert y.shape==(2,2,8)
y.square().mean().backward()
assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
'''
    subprocess.run([sys.executable,'-c',code],check=True)
