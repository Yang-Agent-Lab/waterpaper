"""Load selected author definitions without importing either project's training stack."""
import ast
import hashlib
import base64
import importlib
import importlib.metadata
import importlib.machinery
import os
import subprocess
import sys
from types import ModuleType
from pathlib import Path

DIFFPTS_SHA = 'e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424'


def torch_timeseries_nn():
    """Import installed original NN files without package/dataset initializers."""
    dist=importlib.metadata.distribution('torch-timeseries')
    if dist.version!='0.2.5':raise RuntimeError('torch-timeseries 0.2.5 required')
    root=Path(dist.locate_file('torch_timeseries/nn')).resolve()
    provenance={'torch_timeseries.version':dist.version}
    records={str(item):item for item in dist.files or []}
    for name in ('embedding','encoder','attention'):
        relative=f'torch_timeseries/nn/{name}.py';path=root/(name+'.py')
        source=path.read_bytes();digest=hashlib.sha256(source).digest()
        record=records.get(relative)
        if record is None or record.hash is None or record.hash.mode!='sha256' or base64.urlsafe_b64encode(digest).decode().rstrip('=')!=record.hash.value:
            raise RuntimeError(f'installed NN source differs from wheel RECORD: {relative}')
        provenance[relative]=digest.hex()
        # Checked against v0.2.5 source; fail if the installed files gain data-stack imports.
        for node in ast.walk(ast.parse(source,filename=str(path))):
            if isinstance(node,ast.Import):imports=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:raise RuntimeError('unexpected relative NN dependency')
                imports=[node.module or '']
            else:continue
            if any(item.split('.')[0] not in ('torch','numpy','math') for item in imports):
                raise RuntimeError(f'unexpected NN dependency: {imports}')
    namespace='_waterpaper_torch_timeseries_nn_v025'
    if namespace not in sys.modules:
        package=ModuleType(namespace)
        package.__path__=[str(root)]
        package.__spec__=importlib.machinery.ModuleSpec(namespace,loader=None,is_package=True)
        package.__spec__.submodule_search_locations=package.__path__
        sys.modules[namespace]=package
    elif sys.modules[namespace].__path__!=[str(root)]:
        raise RuntimeError('isolated NN namespace points to another installation')
    embedding=importlib.import_module(namespace+'.embedding')
    encoder=importlib.import_module(namespace+'.encoder')
    attention=importlib.import_module(namespace+'.attention')
    symbols=dict(DataEmbedding=embedding.DataEmbedding,DataEmbedding_inverted=embedding.DataEmbedding_inverted,
                 Encoder=encoder.Encoder,EncoderLayer=encoder.EncoderLayer,
                 FullAttention=attention.FullAttention,AttentionLayer=attention.AttentionLayer)
    return symbols,provenance


def source_root(name, expected_sha):
    key='WATERPAPER_'+name.upper()+'_SOURCE'
    if key not in os.environ:
        raise RuntimeError(f'{key} must point to the fixed author Git checkout')
    root=Path(os.environ[key]).resolve()
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=expected_sha:
        raise RuntimeError(f'{name} source SHA mismatch: {actual}')
    return root


def definitions(root, relative, names, namespace, hashes):
    """Keep exact definition ASTs; inject dependencies instead of author src imports."""
    path=root/relative
    source=path.read_bytes()
    tracked=subprocess.check_output(['git','-C',str(root),'show',f'HEAD:{relative}'])
    if source!=tracked:
        raise RuntimeError(f'author source modified: {relative}')
    hashes[relative]=hashlib.sha256(source).hexdigest()
    tree=ast.parse(source,filename=str(path))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names)]
    if names is not None and {node.name for node in selected}!=set(names):
        raise RuntimeError(f'missing author definitions: {relative}')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def diffpts_definitions():
    import math
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    root=source_root('diffpts',DIFFPTS_SHA);hashes={}
    symbols,package_hashes=torch_timeseries_nn();hashes.update(package_hashes)
    ns=dict(torch=torch,nn=nn,F=F,math=math,np=np,EPS=1e-7,**symbols)
    definitions(root,'src/layer/diffpts_utils.py',None,ns,hashes)
    definitions(root,'src/nn/tmdm_diffusion_utils.py',{'make_beta_schedule'},ns,hashes)
    definitions(root,'src/layer/denoise.py',{'ConditionalLinear','ConditionalGuidedModel'},ns,hashes)
    definitions(root,'src/layer/g_backbone.py',{'SigmaEstimation'},ns,hashes)
    definitions(root,'src/nn/iTransformerBackbone.py',{'iTransformerEnc'},ns,hashes)
    definitions(root,'src/models/DiffPTS.py',{'compute_gx_term','compute_tilde_alpha','compute_hat_alpha','DiffPTS'},ns,hashes)
    return ns,hashes


def ppm_definitions(sha):
    import torch
    from torch import nn
    from torch.nn import functional as F
    root=source_root('ppm',sha);hashes={}
    ns=dict(torch=torch,nn=nn,F=F)
    definitions(root,'model_ppm/PPMF/MLP_PPM.py',{'Model'},ns,hashes)
    # Loss is extracted without importing data_provider, test loaders, or Exp_Basic.
    path='model_ppm/exp/exp_main.py'
    source=(root/path).read_bytes()
    if source!=subprocess.check_output(['git','-C',str(root),'show',f'HEAD:{path}']):
        raise RuntimeError('PPM author loss source modified')
    hashes[path]=hashlib.sha256(source).hexdigest()
    import math
    ns['math']=math
    cls=next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='Exp_Main')
    methods=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ('kde_loss','mean_loss')]
    if len(methods)!=2:raise RuntimeError('PPM original loss definitions missing')
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(root/path),'exec'),ns)
    return ns,hashes
