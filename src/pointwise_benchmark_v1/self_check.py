"""Bounded server-only checks for scores and permission-before-value decoding."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from . import legacy_blocks
from .coverage import save,eligible,intersect
from .metrics import marginal_scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    assert eligible([[0,47]],[[0,23]],24,24,'pq')==[[23,23]]
    assert eligible([[0,46]],[[0,23]],24,24,'q_only')==[]
    assert intersect([[0,4],[8,10]],[[3,9]])==[[3,4],[8,9]]
    draws=np.asarray([[[1,0,0],[0,1,0],[0,1,1]]],dtype=float)
    rows=[dict(country='CN',role='select',canonical_station='S',event='E')]
    scored=marginal_scores(draws,np.asarray([[1,0,1]]),np.asarray([[.5,.75]]),rows)
    assert np.allclose(scored['probability'][0,:,0],[1/3,2/3,1/3])
    assert np.allclose(scored['crps'],[[4/9,4/9,4/9]])
    assert np.isclose(scored['selection_score'],4/9)
    path=output/'synthetic_roles.csv.gz';base=legacy_blocks.hour('2010-01-01T00:00:00')
    fields=['source_row','time_axis','q_m3_s','p_mm_h','pq_numeric_valid','qc_flags']
    with gzip.open(path,'xt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i in range(150):
            permitted=i<48 or 72<=i<120
            writer.writerow(dict(source_row=i+2,time_axis=str(np.datetime64(base+i,'h')),
                q_m3_s='1' if permitted else 'FORBIDDEN_VALUE',p_mm_h='2' if permitted else 'FORBIDDEN_VALUE',
                pq_numeric_valid='1',qc_flags=''))
    stat=path.stat();blocks=[]
    for role,start,end in [('fit',0,47),('select',72,119)]:
        blocks.append(dict(station_uid='camels_de_1h:S',source='camels_de_1h',event=role,role=role,original_role='train',
            start_index=start,end_index=end,start_native_hour=base+start,end_native_hour=base+end,
            interval_end_shift_hours=0,physical_observation_file=str(path),input_fingerprint=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)))
    original=legacy_blocks.number
    def guarded(value):
        if value=='FORBIDDEN_VALUE':raise AssertionError('unauthorized role observation decoded')
        return original(value)
    legacy_blocks.number=guarded
    try:
        stats={};actual=legacy_blocks.read_blocks(blocks,stats)
        assert [len(actual[k]) for k in ('fit','select')]==[48,48]
        assert stats['numeric_QP_rows_decoded']==96 and stats['source_clock_rows_scanned']==120
        try:legacy_blocks.read_blocks([dict(blocks[0],role='calibration')])
        except PermissionError:pass
        else:raise AssertionError('direct forbidden-role block accepted')
    finally:legacy_blocks.number=original
    save(output/'self_check.json',dict(state='complete',synthetic_only=True,
        pointwise_probability=scored['probability'].tolist(),window_ever_probability=1.,
        forbidden_values_never_decoded=True,clock_rows_scanned=stats['source_clock_rows_scanned'],numeric_rows_decoded=96))
    print(json.dumps(dict(state='complete',synthetic_only=True)),flush=True)


if __name__=='__main__':main()
