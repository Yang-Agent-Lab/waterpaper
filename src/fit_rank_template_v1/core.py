"""Selection sees two historical features only; rank mapping uses fit outcomes."""
import hashlib
import json
import zipfile
from collections import Counter
import numpy as np

def json_items(path):
    """Stream array; adapted from accepted 64101ff data-quality items()."""
    decoder=json.JSONDecoder();buf='';pos=0;end=False
    with open(path) as f:
        while True:
            if pos>=len(buf)-1 and not end:buf=buf[pos:]+f.read(1048576);pos=0
            while pos<len(buf) and buf[pos] in ' \r\n\t[,':pos+=1
            if pos<len(buf) and buf[pos]==']':return
            try:val,j=decoder.raw_decode(buf,pos)
            except json.JSONDecodeError:
                more=f.read(1048576)
                if not more:raise
                buf=buf[pos:]+more;pos=0;continue
            pos=j;yield val

def numeric_rows(path,member,indices,shape_tail):
    """Only numeric-decode selected CN fit rows, streaming skipped bytes."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(member+'.npy') as f:
            version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
            assert not fortran and not dtype.hasobject and shape[1:]==shape_tail
            start=f.tell();row_bytes=int(np.prod(shape_tail))*dtype.itemsize
            assert np.all((indices>=0)&(indices<shape[0]))
            result=np.empty((len(indices),*shape_tail),dtype=dtype)
            for destination in np.argsort(indices):
                f.seek(start+int(indices[destination])*row_bytes);raw=f.read(row_bytes);assert len(raw)==row_bytes
                result[destination]=np.frombuffer(raw,dtype=dtype).reshape(shape_tail)
    return result,dict(shape=list(shape),dtype=str(dtype),numeric_rows_decoded=len(indices))

def historical_features(q0,qminus6,q95):
    assert np.all(q0>=0) and np.all(q95>0)
    return np.column_stack([np.log1p(q0/q95),np.arcsinh((q0-qminus6)/q95)])

def select_templates(fit_features,target_features,fit_event_ids,fit_sample_ids):
    """No select future/truth, station preference or calendar filtering argument."""
    center=np.median(fit_features,axis=0);q=np.quantile(fit_features,[.25,.75],axis=0,method='linear');iqr=q[1]-q[0];iqr[iqr==0]=1
    x=(fit_features-center)/iqr;y=(target_features-center)/iqr
    assert sum(min(16,v) for v in Counter(fit_event_ids).values())>=256
    selected=np.empty((len(y),256),np.int32);distances=np.empty((len(y),256),float)
    ids=np.asarray(fit_sample_ids)
    assert len(set(ids))==len(ids)
    for i,target in enumerate(y):
        d=np.sqrt(np.square(x-target).sum(axis=1));order=np.lexsort((ids,d));counter=Counter();keep=[]
        for j in order:
            key=fit_event_ids[j]
            if counter[key]<16:
                keep.append(j);counter[key]+=1
                if len(keep)==256:break
        assert len(keep)==256 and len(set(keep))==256 and max(counter.values())<=16
        selected[i]=keep;distances[i]=d[keep]
    return selected,distances,dict(median=center.tolist(),IQR=iqr.tolist(),fit_only=True)

def rank_templates(relative_changes,fit_ids):
    """Complete ordinal permutations; SHA is secondary to actual fit value."""
    tiekeys=np.asarray([[hashlib.sha256((sid+':'+str(h)+':FRT256-v1').encode()).hexdigest() for h in range(1,7)] for sid in fit_ids])
    return tiekeys

def reorder_marginals(sorted_draws,relative_changes,selected,all_tiekeys):
    ranks=np.empty((len(selected),256,6),np.int16);ties=np.empty((len(selected),6),float)
    for i,ids in enumerate(selected):
        vals=relative_changes[ids]
        for h in range(6):
            order=np.lexsort((all_tiekeys[ids,h],vals[:,h]));rank=np.empty(256,np.int16);rank[order]=np.arange(256)
            assert np.array_equal(np.sort(rank),np.arange(256));ranks[i,:,h]=rank
            _,n=np.unique(vals[:,h],return_counts=True);ties[i,h]=n[n>1].sum()/256
    derived=np.take_along_axis(sorted_draws,ranks,axis=1)
    assert np.array_equal(np.sort(derived,axis=1),sorted_draws)
    return derived,ranks,ties
