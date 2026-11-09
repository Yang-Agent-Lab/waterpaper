"""Resolve physical I/O while keeping immutable historical data identities."""
import gzip
import hashlib
import json
from pathlib import Path
from src.five_country_model_data_v1.reader import Reader
from src.five_country_model_data_v1.prepare import station_name,input_identity
from src.pointwise_flood_risk_v1.adapter import iter_samples as original_pointwise_samples


class PathMap:
    def __init__(self,result):
        receipt=json.loads(Path(result).read_text())
        if receipt['state']!='complete':raise ValueError('completed relocation receipt required')
        self.forward=receipt['forward_mapping'];self.inverse={v:k for k,v in self.forward.items()}
        if len(self.inverse)!=len(self.forward):raise ValueError('nonunique physical mapping')

    @staticmethod
    def apply(path,mapping):
        value=Path(path)
        for old,new in sorted(mapping.items(),key=lambda pair:len(pair[0]),reverse=True):
            try:return Path(new)/value.relative_to(old)
            except ValueError:continue
        return value

    def physical(self,path):return self.apply(path,self.forward)
    def logical(self,path):return self.apply(path,self.inverse)


class RelocatedReader(Reader):
    def __init__(self,manifest,*,mapping_result,legacy_dir,role_policy,prepared_dir):
        self.path_map=PathMap(mapping_result)
        self.logical_manifest=manifest
        super().__init__(self.path_map.physical(manifest),legacy_dir=self.path_map.physical(legacy_dir),
            role_policy=self.path_map.physical(role_policy),prepared_dir=self.path_map.physical(prepared_dir))

    def entry(self,key):return self.path_map.physical(super().entry(key))

    def prepared_station(self,key,station):
        cache_key=(key,station)
        if cache_key not in self._prepared:
            path=self.prepared_dir/'stations'/station_name(key,station)
            if self._prepared_manifest is None:
                manifest_path=self.prepared_dir/'manifest.json'
                self.access.append(str(manifest_path));self._prepared_manifest=json.loads(manifest_path.read_text())
            entry=next((r for r in self._prepared_manifest['stations'] if r['source']==key and r['station']==station),None)
            if entry is None or entry['prepared_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('prepared artifact hash mismatch')
            with gzip.open(path,'rt') as f:table=json.load(f)
            row={'source':key,'station':station,'locator':{'member':'Hourly2/'+station.zfill(8)+'_hourly.nc'}}
            current=input_identity(self,row)
            current['path']=str(self.path_map.logical(current['path']))
            policy_sha=hashlib.sha256(Path(self.access[0]).read_bytes()).hexdigest()
            if table['state']!='complete' or table['role_policy_sha256']!=policy_sha or current!=table['input_identity']:
                raise ValueError('historical input identity or role policy differs')
            self.access.append(str(path));self._prepared[cache_key]=table
        return self._prepared[cache_key]


def iter_pointwise_samples(reader,*args,**kwargs):
    for sample in original_pointwise_samples(reader,*args,**kwargs):
        physical_evidence=sample['threshold_source']['evidence']
        sample['threshold_source']['evidence']=str(reader.path_map.logical(physical_evidence))
        # The original contract hashes logical evidence paths; runtime locations stay outside it.
        sample['sample_id']=hashlib.sha256(json.dumps([sample['identity'],sample['origin_time'],sample['history_hours'],
            sample['forecast_hours'],sample['route'],sample['data_version'],sample['threshold_source']],sort_keys=True).encode()).hexdigest()
        sample['physical_io']={'threshold_evidence':physical_evidence,'manifest':str(reader.path_map.physical(reader.logical_manifest))}
        yield sample
