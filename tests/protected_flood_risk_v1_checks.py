"""CPU synthetic engineering checks, no research role arrays/checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,json,math,random,time,hashlib
from pathlib import Path
START=time.monotonic()
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from protected_flood_risk_v1.model import build_candidates, CausalTower
from protected_flood_risk_v1.labels import make_labels,hazard_nll,public_risk,cumulative_risk
from protected_flood_risk_v1.diagnostics import isolated_diagnostic,cache_key,health,module_registry
from protected_flood_risk_v1.recording import Recorder,MLflowSink,METRIC_CONTRACT
from protected_flood_risk_v1.data import validate_metadata,join_static,real_data_adapter


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); parser.add_argument('--ledger',required=True)
    args=parser.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Exclusive cumulative wall-time account, including failures/imports.
    import fcntl
    ledger=Path(args.ledger)
    with Path(str(ledger)+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        account=json.loads(ledger.read_text()) if ledger.exists() else dict(cpu_seconds=0,attempts=[])
        if account['cpu_seconds']>=300: raise RuntimeError('CPU engineering budget exhausted')
        import signal
        signal.signal(signal.SIGALRM,lambda *unused: (_ for _ in ()).throw(TimeoutError('CPU cumulative budget')))
        signal.setitimer(signal.ITIMER_REAL,max(.001,300-account['cpu_seconds']-(time.monotonic()-START)))
        state='failed'
        try:
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            assert not torch.cuda.is_available()
            torch.manual_seed(2);random.seed(2);np.random.seed(2)
            config=json.loads(Path('configs/prob_flood_v1.json').read_text())
            original=Forecaster('P0',config).eval(); models=build_candidates(original)
            x=torch.rand(8,12,2);features=torch.randn(8,8);static=torch.randn(8,2)
            x[:,-1,1]=torch.tensor([1.,.9,.9,.9,.9,1.01,.9,.9])
            y=torch.zeros(8,12); y[0,0]=1.01;y[1,1]=1.01;y[2,3]=1.01;y[3,5]=1.01;y[4,0]=1.;y[5,0]=2.
            labels=make_labels(x,y)
            assert labels['tau'].tolist()==[1,2,4,6,7,1,7,7]
            assert labels['mask'][0].tolist()==[True,False,False]
            assert labels['mask'][5].tolist()==[False]*3
            loss,denom=hazard_nll(torch.zeros(8,3),labels)
            assert abs(float(loss)-18/7*math.log(2))<1e-6 and denom['n']==7
            empty=dict(labels,at_risk=torch.zeros(8,dtype=torch.bool));assert hazard_nll(torch.zeros(8,3),empty)[1]['state']=='NOT_EVALUABLE'
            extremes=cumulative_risk(torch.tensor([[-1000.,0.,1000.],[1000.,-1000.,0.]]))
            assert torch.isfinite(extremes).all() and (extremes[:,1:]>=extremes[:,:-1]).all()
            with torch.no_grad(): baseline=original.sample(x,features,1,None)[:,0]
            before=models['Risk-C'](x,features,static)['hazard_logits']
            assert torch.equal(before,models['Risk-D'](x,features,static)['hazard_logits'])
            assert equal(models['Risk-B'].head.state_dict(),models['Risk-C'].head.state_dict())
            assert equal(models['Risk-C'].adapter.state_dict(),models['Risk-D'].adapter.state_dict())
            tower=CausalTower(); xx=torch.randn(2,12,1); changed=xx.clone();changed[:,7:]+=100
            assert torch.equal(tower(xx)[:,:7],tower(changed)[:,:7])
            summary={}
            for name,model in models.items():
                model.train(); opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
                protected=copy.deepcopy(model.protected.state_dict())
                result=model(x,features,static,diagnostics=True)
                if model.adapter is not None:
                    att=result['diagnostics']['attention'];assert torch.equal(att.triu(1),torch.zeros_like(att))
                    xa=x.clone();xa[:,7:]+=100
                    _,da=model.adapter(x);_,db=model.adapter(xa)
                    assert torch.equal(da['rain_encoder'][:,:7],db['rain_encoder'][:,:7]) and torch.equal(da['flow_encoder'][:,:7],db['flow_encoder'][:,:7])
                riskloss,_=hazard_nll(result['hazard_logits'],labels);riskloss.backward()
                assert all(p.grad is None for p in model.protected.parameters())
                grads={k:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) for k,module in model.named_children() if k!='protected'}
                assert all(v>0 for v in grads.values())
                opt.step(); assert equal(protected,model.protected.state_dict())
                assert torch.equal(baseline,model(x,features,static)['protected_flow'])
                states=copy.deepcopy(model.state_dict());ostates=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone();pyrng=random.getstate();nprng=np.random.get_state();modes={n:m.training for n,m in model.named_modules()};gradient=copy.deepcopy({n:p.grad for n,p in model.named_parameters()})
                with isolated_diagnostic(model,opt):
                    torch.rand(3);random.random();np.random.rand()
                    for shadow in ('rain_off','flow_off','adapter_off','film_identity'):
                        model(x,features,static,diagnostics=True,shadow=shadow)
                assert equal(states,model.state_dict()) and equal(ostates,opt.state_dict()) and torch.equal(rng,torch.get_rng_state())
                assert pyrng==random.getstate() and np.array_equal(nprng[1],np.random.get_state()[1])
                assert modes=={n:m.training for n,m in model.named_modules()} and equal(gradient,{n:p.grad for n,p in model.named_parameters()})
                assert public_risk(result)[5]['risk']==[None]*3
                summary[name]=dict(loss=float(riskloss.detach()),gradient_sums=grads,health=health(model,result,labels),protected_bitwise_unchanged=True)
            identity=dict(code_version='fixture',data_role_id='synthetic-fit',protocol='first-origin-crossing',checkpoint_sha256='synthetic',algorithm_version='1',sample_ids=['a'],schema_version=1,content_digest='synthetic-values')
            assert cache_key(identity)!=cache_key(dict(identity,checkpoint_sha256='changed'))
            rec=Recorder(out/'records.jsonl','synthetic-run','attempt1');assert rec.write('one',metrics={'brier_6h':.2}) and not rec.write('one')
            with (out/'records.jsonl').open('a') as f:f.write('{torn')
            rec=Recorder(out/'records.jsonl','synthetic-run','attempt2');assert rec.write('two')
            assert not Recorder(out/'records.jsonl','synthetic-run','attempt3').write('one')
            class Mock:
                def log_metrics(self,*a,**kw):
                    self.args=(a,kw)
                    if not hasattr(self,'history'):self.history=[]
                    self.history.append((a,kw))
                def log_artifact(self,*a,**kw):self.artifact=(a,kw)
            mock=Mock();sink=MLflowSink(mock,'mock-only');assert sink.log_metrics({'brier_6h':.2},step=50)
            assert mock.args[1]['step']==50 and mock.args[1]['synchronous'] is True
            assert sink.log_artifact(out/'records.jsonl')
            assert not MLflowSink().log_metrics({},0)
            class Broken:
                def log_metrics(self,*a,**kw):raise ConnectionError('fixture')
            broken=MLflowSink(Broken(),'mock');assert not broken.log_metrics({},1) and broken.state=='tracking_unavailable'
            rows=[dict(window_id='w',station_uid='s',country='China',role='fit',source_rows=list(range(24)),source_times=list(range(24)),event='e',origin_time='synthetic',qscale=1.,source_event_id='e',temporal_overlap_component='c')]
            validate_metadata(rows,'fit');assert join_static(rows,{'s':{'model_features':[.1,.2]}},['s'],['s'])==[[.1,.2]]
            for forbidden in ('test','futureP'):
                try:validate_metadata(rows,forbidden)
                except ValueError:pass
                else:raise AssertionError('forbidden role')
            try:real_data_adapter(None,None,None,None,None)
            except PermissionError:pass
            else:raise AssertionError('real adapter enabled')
            # Batch dimensions fail explicitly instead of broadcasting labels.
            for callback in (lambda:make_labels(x,y[:7]),lambda:models['Risk-A'](x,features[:7],static)):
                try: callback()
                except ValueError: pass
                else: raise AssertionError('batch mismatch accepted')
            from unittest.mock import patch
            with patch('torch.cuda.is_initialized',return_value=True), patch('torch.cuda.get_rng_state_all',return_value=['synthetic-state']) as get_cuda, patch('torch.cuda.set_rng_state_all') as set_cuda:
                with isolated_diagnostic(models['Risk-C'],opt): pass
                get_cuda.assert_called_once();set_cuda.assert_called_once_with(['synthetic-state'])
            # Newline-terminated torn final record is also preserved and recoverable.
            torn=out/'newline_torn.jsonl';torn.write_text('{"record_id":"ok"}\n{torn\n');old=torn.read_bytes()
            rr=Recorder(torn,'run','attempt');rr.write('after');assert torn.read_bytes().startswith(old)
            assert not Recorder(torn,'run','attempt2').write('after')
            corrupt=out/'interior_corrupt.jsonl';corrupt.write_text('{torn\n{"record_id":"later"}\n')
            try:Recorder(corrupt,'r','a')
            except ValueError:pass
            else:raise AssertionError('interior corruption accepted')
            from protected_flood_risk_v1.diagnostics import cached_diagnostic
            count=[0]
            def calculate():count[0]+=1;return {'count':count[0]}
            assert not cached_diagnostic(out/'count_cache',identity,calculate)['cache_hit']
            assert cached_diagnostic(out/'count_cache',identity,calculate)['cache_hit'] and count[0]==1
            cached_diagnostic(out/'count_cache',dict(identity,checkpoint_sha256='new'),calculate)
            cached_diagnostic(out/'count_cache',dict(identity,sample_ids=['different']),calculate)
            cached_diagnostic(out/'count_cache',dict(identity,content_digest='changed-values-same-IDs'),calculate)
            assert count[0]==4
            # Full integration on synthetic roles only, with explicitly shortened periods.
            from protected_flood_risk_v1.training import run_training,content_digest,hierarchical,event_summary,fixed_diagnostic,capture_protection_reference
            from protected_flood_risk_v1.data import digest
            pools={}
            for role in ('fit','select','calibration','development'):
                synthetic_rows=[]
                for i in range(8):
                    country='Germany' if role=='fit' and i%2 else 'China'
                    synthetic_rows.append(dict(rows[0],window_id=role+str(i),station_uid='de' if country=='Germany' else 'cn',country=country,role=role,event='event'+str(i//2),source_event_id='event'+str(i//2),synthetic=True,strong_rise_cutoff=.2))
                pools[role]=dict(rows=synthetic_rows,x=x.numpy().copy(),y=y.numpy().copy(),features=features.numpy().copy(),static=static.numpy().copy())
            fake=out/'synthetic_prepared';fake.mkdir()
            derived={}
            for role,pool in pools.items():
                np.savez_compressed(fake/(role+'.npz'),**{k:pool[k] for k in ('x','y','features')});derived[role+'.npz']=digest(fake/(role+'.npz'))
                # Simulate a foreign reserved select/cal row; adapter must filter it, never recycle.
                metadata=copy.deepcopy(pool['rows'])
                if role in ('select','calibration'): metadata[-1]['country']='Germany';metadata[-1]['station_uid']='de'
                (fake/(role+'_rows.json')).write_text(json.dumps(metadata))
            (fake/'scales.json').write_text('{}');(fake/'rise_cutoffs.json').write_text('{}')
            (fake/'manifest.json').write_text(json.dumps(dict(derived_files=derived,producer_code_sha='synthetic')))
            attribute=fake/'attribute_rows.json';attribute.write_text(json.dumps([dict(station_uid=st,model_features=[.1,.2]) for st in ('cn','de')]))
            normalization=fake/'normalization.json';normalization.write_text(json.dumps(dict(fit_station_uids=['cn','de'])))
            adapted,source=real_data_adapter(fake,list(pools),attribute,normalization,digest(fake/'manifest.json'),execution_authorized=True)
            assert len(adapted['fit']['rows'])==8 and len(adapted['select']['rows'])==7 and len(adapted['calibration']['rows'])==7
            original_digest=content_digest(adapted['fit'])
            changed_normalization=fake/'changed_normalization.json';changed_normalization.write_text(json.dumps(dict(fit_station_uids=['cn','de'],synthetic_transform_revision=2)))
            changed_adapter,_=real_data_adapter(fake,['fit'],attribute,changed_normalization,digest(fake/'manifest.json'),execution_authorized=True)
            assert content_digest(changed_adapter['fit'])!=original_digest
            assert all(row['country']=='China' for row in adapted['select']['rows']) and source['future_P_read'] is False
            pools=adapted  # Train and score exactly the adapter-filtered, ordered members.
            changed=copy.deepcopy(pools['fit']);changed['y'][0,0]+=.1
            assert content_digest(changed)!=content_digest(pools['fit']) and changed['rows']==pools['fit']['rows']
            fresh=build_candidates(original)
            training_config=dict(run_id='synthetic-integration',attempt_id='fixture',epochs=2,epoch_size=8,batch_size=4,seed=2,learning_rate=.001,health_every_steps=1,diagnostic_every_epochs=1,code_version='fixture',data_role_id='synthetic-only',fixed_diagnostic_ids={role:[pool['rows'][i]['window_id'] for i in (0,3,5)] for role,pool in pools.items() if role in ('fit','select')})
            tracking_mock=Mock();tracking_sink=MLflowSink(tracking_mock,'mock-shared-run')
            result=run_training(fresh,pools,training_config,out/'integration',synthetic=True,tracking=tracking_sink)
            role_receipt=json.loads((out/'integration/roles_manifest.json').read_text())
            assert {role:entry['n'] for role,entry in role_receipt.items()}==dict(fit=8,select=7,calibration=7,development=8)
            for role,pool in pools.items():
                assert json.loads((out/'integration'/role_receipt[role]['rows_file']).read_text())==pool['rows']
                assert json.loads((out/'integration'/role_receipt[role]['ids_file']).read_text())==[row['window_id'] for row in pool['rows']]
                assert role_receipt[role]['content_digest']==content_digest(pool)
                assert all(digest(path)==expected for path,expected in role_receipt[role]['input_hashes'].items())
            assert json.loads((out/'integration/effective_config.json').read_text())['max_updates']==3600
            assert result['state']=='complete' and all(v['updates']==4 for v in result['arms'].values())
            events=[json.loads(line) for line in (out/'integration/metrics.jsonl').read_text().splitlines()]
            assert sum(e['kind']=='shadow_only' for e in events)==8 and sum(e['kind']=='health' for e in events)==16
            assert all(all(name in e['metrics']['module_health'] for name in ['rain_encoder','flow_encoder','lag_interaction','risk_adapter','film','risk_module','protected']) for e in events if e['kind']=='health')
            assert len(list((out/'integration').glob('Risk-*/epoch_*/checkpoint.pt')))==8
            exposure=json.loads((out/'integration/exposure.json').read_text());assert len({json.dumps(v,sort_keys=True) for v in exposure.values()})==1
            support=json.loads((out/'integration/support.json').read_text());assert support['before']==8 and support['after']==7
            try:run_training(fresh,pools,training_config,out/'forbidden')
            except PermissionError:pass
            else:raise AssertionError('real training enabled')
            # Prepared event identity controls denominator even when source_event_id differs.
            event_rows=[dict(pools['fit']['rows'][0],event='a' if i<3 else 'b',source_event_id='same-source') for i in range(4)]
            assert hierarchical(np.array([0.,0.,0.,1.]),event_rows,np.ones(4,bool))==.5
            event_saved=dict(at_risk=np.ones(4,bool),risk=np.tile(np.array([[0.],[0.],[0.],[1.]]),(1,3)),truth=np.zeros((4,3),bool))
            assert len(event_summary(event_saved,event_rows))==2
            # Shadow Brier and absent module states; deliberate fixed-branch corruption is detected.
            diagnostic_records=[e for e in events if e['kind']=='shadow_only']
            for e in diagnostic_records:
                for role,values in e['diagnostic']['result']['roles'].items():
                    assert values['full_scores']['n_at_risk']==2
                    for shadow,v in values['shadows'].items():
                        if v['state']=='shadow_only':
                            np.testing.assert_allclose(v['brier_shadow_minus_full'],np.array(v['scores']['brier'])-values['full_scores']['brier'])
                        else:assert v['state']=='NOT_APPLICABLE' and 'risk_mean_absolute_change' not in v
            damaged=build_candidates(original)['Risk-C'];damage_opt=torch.optim.Adam([p for p in damaged.parameters() if p.requires_grad])
            reference=capture_protection_reference(damaged,pools,training_config['fixed_diagnostic_ids'])
            with torch.no_grad():damaged.protected.point[-1].bias.add_(1)
            detected=fixed_diagnostic(damaged,damage_opt,pools,training_config['fixed_diagnostic_ids'],out/'damage_cache',dict(identity,checkpoint_sha256='damaged'),reference)
            assert detected['result']['protected_parameter_hash_unchanged'] is False
            assert all(v['protected_flow_max_abs_change']>.9 for v in detected['result']['roles'].values())
            namespaces={name.split('/')[0] for args,kw in tracking_mock.history for name in args[0]}
            assert namespaces==set(fresh) and all('/' in name for args,kw in tracking_mock.history for name in args[0])
            # max_updates stops halfway through epoch2 and records actual work.
            limited=run_training(build_candidates(original),pools,dict(training_config,max_updates=3),out/'limited',synthetic=True)
            assert all(v['updates']==3 and v['complete_epochs']==1 and v['epochs_started']==2 and v['final_epoch_fraction']==.5 and v['stop_reason']=='max_updates' for v in limited['arms'].values())
            final=json.loads((out/'integration/final_summary.json').read_text())
            assert all('ap' in v['development']['metrics'] and 'reliability' in v['development'] and 'alarms' in v['development'] for v in final.values())
            assert (out/'integration/alarm_lines.json').exists()
            # Loss/gradient/probability failures must stop before any optimizer update and preserve evidence.
            for failure_kind in ('probability','gradient','loss'):
                bad=build_candidates(original)
                if failure_kind=='probability':
                    with torch.no_grad():bad['Risk-A'].head.bias.fill_(float('nan'))
                if failure_kind=='gradient':bad['Risk-A'].head.weight.register_hook(lambda grad:torch.full_like(grad,float('nan')))
                from contextlib import nullcontext
                patcher=patch('protected_flood_risk_v1.training.hazard_nll',return_value=(torch.tensor(float('nan'),requires_grad=True),dict(n=4))) if failure_kind=='loss' else nullcontext()
                try:
                    with patcher:run_training(bad,pools,training_config,out/('failed_'+failure_kind),synthetic=True)
                except FloatingPointError:pass
                else:raise AssertionError('nonfinite was not stopped')
                failure=json.loads((out/('failed_'+failure_kind)/'completion.json').read_text())
                assert failure['state']=='failed' and failure['completed_updates']==0 and (out/('failed_'+failure_kind)/'failure_state.pt').exists()
            (out/'checks.json').write_text(json.dumps(dict(state='PASS',synthetic_only=True,real_rows_loaded=0,gpu_seconds=0,threads=torch.get_num_threads(),original_forecaster=True,training_integration=result,max_updates_check=limited,nonfinite_failure_types=['loss','gradient','probability'],prepared_event_denominator_checked=True,protected_corruption_detected=True,tracking_namespaces=sorted(namespaces),cache_compute_calls=count[0],diagnostic_events=8,health_events=16,candidates=summary,modules=module_registry(),metric_contract=METRIC_CONTRACT,torch=torch.__version__,numpy=np.__version__),indent=2))
            state='complete';print('PASS: synthetic CPU checks; zero real samples and zero GPU')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            elapsed=time.monotonic()-START;account['cpu_seconds']+=elapsed;account['attempts'].append(dict(state=state,seconds=elapsed))
            ledger.write_text(json.dumps(account,indent=2));(out/'budget.json').write_text(json.dumps(account,indent=2))

if __name__=='__main__':main()
