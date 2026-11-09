"""Synthetic proof: a CAMELSH clock gap rejects before any Q slice is read."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
from .source_slices import batch_records
from .coverage import save
from src.five_country_model_data_v1.prepare import hour


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/'synthetic_clock.zip'
    with zipfile.ZipFile(path,'x') as z:z.writestr('Hourly2/00000001_hourly.nc',b'synthetic_no_observations')
    reads=[]
    class Dataset:
        def __init__(self,name,values,attrs):self.name=name;self.values=values;self.attrs=attrs
        def __getitem__(self,index):
            if self.name=='streamflow':
                reads.append(index);raise AssertionError('Q observation accessed before rejecting clock gap')
            return self.values[index]
    class File:
        def __init__(self,*args,**kwargs):
            self.datasets={'time':Dataset('time',np.asarray([0.,2.,3.]),{'units':b'hours since 2015-09-30 22:00:00'}),
                           'streamflow':Dataset('streamflow',None,{'units':b'm3 s-1'})}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,name):return self.datasets[name]
    reader=SimpleNamespace(entry=lambda key:path,access=[],role_bounds={'camelsh_us_v6':{
        'fit':[None,'2015-10-01T00:00:00'],'select':['2015-10-01T00:00:00','2016-10-01T00:00:00']}})
    original=sys.modules.get('h5py');sys.modules['h5py']=SimpleNamespace(File=File)
    try:
        try:batch_records(reader,'camelsh_us_v6','00000001',[dict(role='fit',origin_hour=hour('2015-09-30T22:00:00'),history_hours=1,forecast_hours=1)])
        except ValueError as error:
            if 'actual permitted hourly clock metadata' not in str(error):raise
        else:raise AssertionError('missing actual permitted hour was accepted')
        assert not reads
    finally:
        if original is None:sys.modules.pop('h5py',None)
        else:sys.modules['h5py']=original
    save(out/'clock_check.json',dict(state='complete',synthetic_only=True,Q_numeric_slices_accessed=0,
        gap_at_last_fit_hour_rejected_before_reserved_Q=True))


if __name__=='__main__':main()
