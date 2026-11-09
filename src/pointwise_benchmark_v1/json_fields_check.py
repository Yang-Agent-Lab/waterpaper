"""Synthetic proof that opaque peak JSON is never passed to a value decoder."""
import argparse
from pathlib import Path
from . import json_fields
from .coverage import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/'synthetic_station.json'
    path.write_text('{"peak_eligibility":[{"peak_q":12345,"text":"} \\\" ["}],"entry":{"station_uid":"S"},"blocks":[{"id":1}],"source_fingerprint":{"bytes":2}}')
    original=json_fields.json.loads;decoded=[]
    def guarded(value,*args,**kwargs):
        if '"peak_q"' in value:raise AssertionError('opaque peak metadata value decoded')
        decoded.append(value);return original(value,*args,**kwargs)
    json_fields.json.loads=guarded
    try:
        actual=json_fields.top_fields(path,{'entry','blocks','source_fingerprint'})
        assert actual==dict(entry={'station_uid':'S'},blocks=[{'id':1}],source_fingerprint={'bytes':2})
        assert not any('12345' in value for value in decoded)
    finally:json_fields.json.loads=original
    save(out/'json_fields_check.json',dict(state='complete',synthetic_only=True,unrequested_peak_values_decoded=False))


if __name__=='__main__':main()
