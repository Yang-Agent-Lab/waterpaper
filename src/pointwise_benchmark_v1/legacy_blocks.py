"""Read complete, already allocated foreign blocks; do not change old roles/scales.

First emit a metadata permission list. Numeric access requires its explicit hash.
No mixed preparation cache is opened. Sequential CSV metadata is inspected before
any Q/P decoding, and scanning stops at the final permitted original-train row.
"""
import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from src.five_country_model_data_v1.prepare import hour
from src.five_country_model_data_v1.reader import number,truth
from src.workspace_consolidation_v1.relocation import RelocatedReader
from .coverage import save,runs,coverage_row,summarize
from .dataset import sha256
from .json_fields import top_fields

FOREIGN={'camels_de_1h':'DE','lamah_ice':'IS'}
OLD_TRAIN_END=hour('2017-01-01T00:00:00')


def metadata(reader):
    export=json.loads((reader.legacy_dir.parent/'manifest.json').read_text())
    original=next(r['source'] for r in export['files'] if r['relative']=='prepared/fit_rows.json')
    role_path=reader.path_map.physical(original).parent/'role_ids.csv'
    preparation=json.loads((reader.legacy_dir/'manifest.json').read_text())
    model_path=reader.path_map.physical(preparation['source_manifest'])
    if sha256(model_path)!=preparation['source_manifest_sha256']:raise ValueError('original model manifest hash mismatch')
    model=top_fields(model_path,{'stageB_root'});stage_b=reader.path_map.physical(model['stageB_root'])
    known={};anchors=defaultdict(list)
    for role in ('fit','select','calibration'):
        path=reader.legacy_dir/(role+'_rows.json')
        expected=next(r['sha256'] for r in export['files'] if r['relative']=='prepared/'+path.name)
        if sha256(path)!=expected:raise ValueError('legacy role metadata hash mismatch')
        for r in json.loads(path.read_text()):
            if r['role']!=role:raise ValueError('role metadata differs from its physical file')
            if r['station_uid'].split(':',1)[0] in FOREIGN:
                known[r['window_id']]=(r['station_uid'],r['event'],role)
                anchors[(r['station_uid'],r['event'])].append(r)
    allocated=defaultdict(set);counts=defaultdict(int);found=set()
    with role_path.open() as f:
        for r in csv.DictReader(f):
            if r['station'].split(':',1)[0] not in FOREIGN:continue
            key=(r['station'],r['event']);allocated[key].add(r['role']);counts[key]+=1
            if r['window_id'] in known and known[r['window_id']]!=(r['station'],r['event'],r['role']):
                raise ValueError('role_ids disagree with accepted exported metadata')
            if r['window_id'] in known:found.add(r['window_id'])
    if found!=set(known):raise ValueError('accepted window missing from original role_ids')
    scales=reader._load_legacy('fit')[1];output=[];excluded=[]
    dependencies={str(role_path):sha256(role_path),str(model_path):sha256(model_path)}
    by_station=defaultdict(list)
    for key,roles in allocated.items():by_station[key[0]].append((key,roles))
    for uid,items in sorted(by_station.items()):
        source,station=uid.split(':',1)
        meta_path=stage_b/'stations'/uid.replace(':','__')/'station.json'
        meta=top_fields(meta_path,{'entry','blocks','source_fingerprint'});dependencies[str(meta_path)]=sha256(meta_path)
        if meta['entry']['station_uid']!=uid:raise ValueError('station metadata identity mismatch')
        blocks={int(b['id']):b for b in meta['blocks']}
        intervals=sorted((b['start'],b['end']) for b in blocks.values())
        if any(b[0]<=a[1] for a,b in zip(intervals,intervals[1:])):raise ValueError('original source-index blocks overlap')
        candidates=[];ranges=[]
        for key,roles in sorted(items):
            event=key[1]
            if len(roles)!=1:
                excluded.append(dict(station_uid=uid,event=event,reason='multiple_roles_in_one_block'));continue
            role=next(iter(roles));block_id=int(event.rsplit(':block:',1)[1]);block=blocks[block_id]
            records=anchors.get(key,[])
            if not records:
                if role in ('fit','select'):excluded.append(dict(station_uid=uid,event=event,reason='no_accepted_anchor_or_fit_scale'))
                continue
            inferred=[]
            for r in records:
                for source_row,t in zip(r['source_rows'],r['source_times']):
                    index=int(source_row)-2
                    if not block['start']<=index<=block['end']:raise ValueError('accepted source row outside original block')
                    inferred.append(hour(t)-(index-block['start']))
            if len(set(inferred))!=1:raise ValueError('inconsistent accepted block source-row/time anchors')
            start=inferred[0];end=start+block['end']-block['start'];ranges.append((start,end,role,event))
            if role not in ('fit','select'):continue
            reason=None
            if int(block['role'])!=1:reason='original_block_not_old_train'
            elif end>=OLD_TRAIN_END:reason='old_validation_or_test_clock'
            elif uid not in scales or not scales[uid]['qscale'] or scales[uid]['qscale']<=0:reason='no_existing_positive_fit_scale'
            if reason:
                excluded.append(dict(station_uid=uid,event=event,reason=reason));continue
            fingerprint=meta['source_fingerprint'];physical=reader.path_map.physical(meta['entry']['observation_file']);stat=physical.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(fingerprint['bytes'],fingerprint['mtime_ns']):raise ValueError('observation source stat identity changed')
            candidates.append(dict(source=source,station=station,station_uid=uid,country=FOREIGN[source],
                canonical_station=reader.canonical(source,station),event=event,role=role,block_id=block_id,
                start_index=int(block['start']),end_index=int(block['end']),start_source_row=int(block['start'])+2,
                end_source_row=int(block['end'])+2,start_native_hour=start,end_native_hour=end,
                interval_end_shift_hours=1 if source=='lamah_ice' else 0,block_kind=block['kind'],original_role='train',
                selected_window_count=counts[key],original_observation_file=meta['entry']['observation_file'],
                physical_observation_file=str(physical),input_fingerprint=fingerprint,
                station_metadata_sha256=dependencies[str(meta_path)],fit_scale=scales[uid]))
        for b in candidates:
            conflict=next((event for a,z,role,event in ranges if event!=b['event'] and role!=b['role']
                and max(a,b['start_native_hour'])<=min(z,b['end_native_hour'])),None)
            if conflict:excluded.append(dict(station_uid=uid,event=b['event'],reason='time_overlap_other_role',conflict_event=conflict))
            else:output.append(b)
    return dict(schema='pointwise_legacy_block_permission_v1',state='metadata_checked',blocks=output,excluded=excluded,
        dependencies=dependencies,legacy_fit_scales_sha256=sha256(reader.legacy_dir/'scales.json'),
        checks=['one_block_one_existing_role','disjoint_source_indices','all_accepted_clock_anchors_agree',
            'no_clock_overlap_other_allocated_role','original_train_before_2017','existing_fit_scales_unchanged'],
        index_rule='original Stage B index = canonical source_row - 2',observation_values_read=False,
        full_original_blocks_are_newly_restored_observations=True,physical_role_separation=False)


def read_blocks(blocks,stats=None):
    started=time.monotonic()
    blocks=sorted(blocks,key=lambda b:b['start_index'])
    if not blocks or len({b['station_uid'] for b in blocks})!=1:raise ValueError('one nonempty station permission list required')
    if any(b['role'] not in ('fit','select') or b['original_role']!='train' or b['end_native_hour']>=OLD_TRAIN_END for b in blocks):
        raise PermissionError('only already allocated original-train fit/select blocks can be decoded')
    path=Path(blocks[0]['physical_observation_file']);before=path.stat();fp=blocks[0]['input_fingerprint']
    if (before.st_size,before.st_mtime_ns)!=(fp['bytes'],fp['mtime_ns']):raise ValueError('source identity changed')
    result={b['event']:[] for b in blocks};j=0
    with gzip.open(path,'rt',newline='') as f:
        for index,raw in enumerate(csv.DictReader(f)):
            if index>blocks[-1]['end_index']:break
            if int(raw['source_row'])!=index+2:raise ValueError('source row/index relation differs from original producer')
            t=hour(raw['time_axis']) if raw['time_axis'] else None
            if t is not None and t>=OLD_TRAIN_END:break
            while j<len(blocks) and index>blocks[j]['end_index']:j+=1
            if j==len(blocks):break
            b=blocks[j]
            if index<b['start_index']:continue
            if t!=b['start_native_hour']+index-b['start_index']:raise ValueError('permitted block source clock mismatch')
            # Q/P are accessed only after the source row and time are authorized.
            q=number(raw['q_m3_s']);p=number(raw['p_mm_h'])
            ok=truth(raw.get('observation_eligible',raw.get('pq_numeric_valid','0')))
            flags=raw.get('qc_flags','');ok=ok and not any(s in flags for s in ('duplicate_timestamp','conflicting_timestamp','off_hour'))
            if b['source']=='lamah_ice':ok=ok and float(raw['q_quality_code']) in (40,80)
            result[b['event']].append(dict(source_row=index+2,native_hour=t,interval_end_hour=t+b['interval_end_shift_hours'],
                q=q,p=p,q_mask=ok and q==q,p_mask=ok and p==p,quality=flags))
            if index==blocks[-1]['end_index']:break
    for b in blocks:
        if len(result[b['event']])!=b['end_index']-b['start_index']+1:raise ValueError('block incomplete before old-train boundary')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('source changed during read')
    if stats is not None:stats.update(elapsed_seconds=time.monotonic()-started,
        source_clock_rows_scanned=blocks[-1]['end_index']+1,
        numeric_QP_rows_decoded=sum(len(v) for v in result.values()),source_compressed_bytes=before.st_size)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--current-entry',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['metadata','smoke'],default='metadata');p.add_argument('--permission');p.add_argument('--permission-sha256')
    a=p.parse_args();out=Path(a.output);reader=RelocatedReader(**json.loads(Path(a.current_entry).read_text())['reader_kwargs'])
    reader._load_legacy('fit');reader._load_legacy('select')
    if a.mode=='metadata':
        permission=metadata(reader);save(out/'permission.json',permission)
        save(out/'metadata_access.json',dict(state='complete',permission_sha256=sha256(out/'permission.json'),
            access_method='lexical_top_level_projection_without_decoding_skipped_values',
            projected_top_fields={'station.json':['entry','blocks','source_fingerprint'],'model_manifest.json':['stageB_root']},
            unrequested_peak_values_decoded=False,raw_hourly_values_read=False,
            historical_e97_note='whole station/model JSON was decoded in e97; peak fields were not accessed by permission logic or used by a model; training was zero'))
        print(json.dumps(dict(state='metadata_checked',blocks=len(permission['blocks']),excluded=len(permission['excluded']),permission_sha256=sha256(out/'permission.json'))),flush=True)
    else:
        if not a.permission or sha256(a.permission)!=a.permission_sha256:raise PermissionError('explicit permission manifest hash required')
        permission=json.loads(Path(a.permission).read_text());receipts=[];coverage=[]
        for source in FOREIGN:
            candidates=[b for b in permission['blocks'] if b['source']==source and b['end_index']-b['start_index']+1>=48]
            station=next(b['station'] for b in candidates if len({x['role'] for x in candidates if x['station']==b['station']})==2)
            blocks=[next(b for b in candidates if b['station']==station and b['role']==role) for role in ('fit','select')]
            stats={};data=read_blocks(blocks,stats)
            for b in blocks:
                points=data[b['event']];qr=runs(r['interval_end_hour'] for r in points if r['q_mask']);pr=runs(r['interval_end_hour'] for r in points if r['p_mask'])
                coverage.extend(coverage_row(source,station,b['canonical_station'],b['country'],b['role'],b['event'],qr,pr,[24],24,
                    high_times=[r['interval_end_hour'] for r in points if r['q_mask'] and r['q']>b['fit_scale']['qscale']]))
                receipts.append(dict(permission=b,points=points,station_scan_stats=stats))
        save(out/'smoke.json',dict(state='complete',rows=receipts,coverage=coverage,summary=summarize(coverage),
            permission_sha256=a.permission_sha256,no_training=True,old_files_modified=False))
        print(json.dumps(dict(state='complete',summary=summarize(coverage))),flush=True)


if __name__=='__main__':main()
