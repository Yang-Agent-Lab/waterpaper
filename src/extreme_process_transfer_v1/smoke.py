"""Real-data engineering checks, deliberately not a release receipt."""
import argparse
from pathlib import Path
import numpy as np
import torch
from src.d500_process_adapter_v1.runtime import fingerprint
from src.d500_process_adapter_v1.sampling import NoisePlan
from src.pointwise_benchmark_v1.models import random_state
from .data import Bank
from .objective import objective
from .worker import GROUPS, construct, run, save_json, make_optimizer, checkpoint, restore, train_step
from .paths import checked_output


def assert_nested_equal(left,right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise AssertionError('state keys differ')
        for key in left:assert_nested_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise AssertionError('state lengths differ')
        for a,b in zip(left,right):assert_nested_equal(a,b)
    elif left!=right:
        raise AssertionError('state scalar differs')


def center_checks(bank, artifacts, device, out):
    model, teacher = construct('M_N_ME', artifacts, device)
    indices = np.arange(8); x,y = bank.batch('fit',indices,device)
    student = model.process
    frozen_before = fingerprint(student.base); teacher_before = fingerprint(teacher)
    student.eval(); teacher.eval()
    with torch.no_grad():
        mu,g = student.conditions(x); old,old_g=teacher.conditions(x)
        torch.testing.assert_close(mu,old,atol=0,rtol=0)
        torch.testing.assert_close(g,old_g,atol=0,rtol=0)
        plan=NoisePlan.create(mu,[bank.fit_rows[i]['sample_id'] for i in indices],2,412)
        v,_=student.sample(x,plan); u,_=teacher.sample(x,plan)
        torch.testing.assert_close(v,u,atol=0,rtol=0)
        student.center_head[-1].bias.fill_(.125)
        changed,changed_g=student.conditions(x)
        captured=[]
        hook=student.base.diffusion.register_forward_pre_hook(lambda module,args: captured.append(args[3].detach().clone()))
        try:
            _,(states,_) = student.sample(x,plan,trace=True)
        finally:
            hook.remove()
        torch.testing.assert_close(states[0],changed.repeat_interleave(2,0)+changed_g.repeat_interleave(2,0).sqrt()*plan.initial.reshape(-1,24,1),atol=0,rtol=0)
        if len(captured)!=20:raise AssertionError('reverse step count')
        for seen in captured:
            torch.testing.assert_close(seen,changed.repeat_interleave(2,0),atol=0,rtol=0)
        student.center_head[-1].bias.zero_()
    ordinary=torch.as_tensor(bank.severity[indices]==0,device=device,dtype=torch.float32)
    weight=torch.ones(len(indices),device=device)
    _,parts=objective(student,teacher,x,y,weight,ordinary,914)
    heads={'center':student.center_head[-1].weight,'epsilon':student.adapter.output.weight}
    grads={}
    for key in ('noise','crps','energy'):
        for name,head in heads.items():
            grad=torch.autograd.grad(parts[key],head,retain_graph=True,allow_unused=False)[0]
            if not torch.isfinite(grad).all() or float(grad.norm())==0:
                raise FloatingPointError('missing '+name+' gradient: '+key)
            grads[key+'_'+name]=float(grad.norm())
    if fingerprint(student.base)!=frozen_before or fingerprint(teacher)!=teacher_before:
        raise AssertionError('frozen state changed during forward')
    save_json(out/'center_checks.json',dict(zero_head_conditions='exact',zero_head_full20_prediction='exact',
        changed_mu_initial_state='exact',changed_mu_all_20_denoiser_calls='exact',
        final_head_gradients=grads,base_state_unchanged=True,teacher_state_unchanged=True,
        scope='forward/gradient and 20 denoiser calls; optimizer update checked separately'))


def restoration_check(group, bank, artifacts, device, schedules, ratio, out):
    mode, _, train_adapter, nb = GROUPS[group]
    model,teacher=construct(group,artifacts,device); optimizer=make_optimizer(model,nb)
    model_before=fingerprint(model)
    base_before=fingerprint(model.process.base) if not nb else None
    teacher_before=fingerprint(teacher) if teacher is not None else None
    adapter_before=fingerprint(model.process.adapter) if not nb else None
    center_before=fingerprint(model.process.center_head) if not nb and model.process.center else None
    for i in range(3):
        train_step(model,teacher,optimizer,bank,schedules[mode][i],ratio,mode,nb,device,i)
    identity=dict(scope='engineering_resume_check',group=group)
    path=out/'resume_step3.pt';torch.save(checkpoint(model,optimizer,3,group,device,identity),path)
    loss_a,_=train_step(model,teacher,optimizer,bank,schedules[mode][3],ratio,mode,nb,device,3)
    expected={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    del model, optimizer, teacher
    model,teacher=construct(group,artifacts,device);optimizer=make_optimizer(model,nb)
    state=torch.load(path,map_location=device,weights_only=False)
    assert restore(state,model,optimizer,group,device,identity)==3
    assert_nested_equal(optimizer.state_dict(),state['optimizer'])
    loss_b,_=train_step(model,teacher,optimizer,bank,schedules[mode][3],ratio,mode,nb,device,3)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(),expected[key],rtol=0,atol=0)
    if loss_a != loss_b:raise AssertionError('replayed loss differs')
    if fingerprint(model)==model_before:raise AssertionError('trainable model did not change')
    if not nb:
        if fingerprint(model.process.base)!=base_before or fingerprint(teacher)!=teacher_before:
            raise AssertionError('frozen model changed during optimization')
        if not train_adapter and fingerprint(model.process.adapter)!=adapter_before:
            raise AssertionError('simple-center adapter changed')
        if train_adapter and fingerprint(model.process.adapter)==adapter_before:
            raise AssertionError('trainable epsilon adapter did not change')
        if center_before is not None and fingerprint(model.process.center_head)==center_before:
            raise AssertionError('trainable center head did not change')
    save_json(out/'restore_checks.json',dict(state='exact',steps='3+1 versus 4',
         same_loss=loss_b,model_state='exact',optimizer_state='exact',
         frozen_base_unchanged=True if not nb else 'NOT_APPLICABLE',
         teacher_unchanged=True if not nb else 'NOT_APPLICABLE',
         simple_adapter_unchanged=True if not nb and not train_adapter else 'NOT_APPLICABLE',
         expected_trainable_modules_changed=True,engineering_updates=5))


def mask_checks(bank, artifacts, device, ratio, out):
    model,teacher=construct('M_X_ME',artifacts,device)
    labels=bank.severity;countries=np.asarray([r['country'] for r in bank.fit_rows])
    selected=[]
    for country in ('CN','DE','IS','US','JP'):
        matches=np.flatnonzero((countries==country)&(labels==2))
        if not len(matches):raise ValueError('no actual extreme in country '+country)
        selected.append(int(matches[0]))
    selected.extend(np.flatnonzero(labels==2)[:3].tolist());ids=np.asarray(selected)
    x,y=bank.batch('fit',ids,device);w=torch.as_tensor(ratio[ids],dtype=x.dtype,device=device)
    ordinary=torch.zeros(len(ids),dtype=x.dtype,device=device)
    total,parts=objective(model.process,teacher,x,y,w,ordinary,502)
    if float(parts['keep'])!=0. or not torch.isfinite(total):raise AssertionError('zero ordinary keep semantics')
    ordinary_ids=np.flatnonzero((countries!='CN') & (labels==0))[:8]
    x,y=bank.batch('fit',ordinary_ids,device)
    w=torch.as_tensor(ratio[ordinary_ids],dtype=x.dtype,device=device)
    mask=torch.ones(len(ordinary_ids),dtype=x.dtype,device=device)
    with torch.no_grad():model.process.center_head[-1].bias.fill_(.125)
    _,parts=objective(model.process,teacher,x,y,w,mask,503)
    with torch.no_grad():
        smu,sg=model.process.conditions(x);tmu,tg=teacher.conditions(x)
        m=model.process.base.diffusion;ns=model.process.base.author_functions
        with random_state(device,seed=503):
            t=torch.randint(0,20,(len(x)//2+1,),device=device);t=torch.cat((t,19-t))[:len(x)]
            noise=torch.randn_like(y.unsqueeze(-1))
        var=ns['cal_forward_noise'](m.betas_bar,tg,t)
        shared=ns['q_sample'](y.unsqueeze(-1),tmu,m.alphas_bar_sqrt,m.one_minus_alphas_bar_sqrt,t,noise=noise*var.sqrt())
        ix=torch.arange(len(x),device=device)
        se,_=model.process.view(x,smu,sg,ix)(x,None,shared,smu,sg,t)
        te,_=teacher.view(x,tmu,tg,ix)(x,None,shared,tmu,tg,t)
        expected=(w*((se-te).square().mean((1,2))+(smu-tmu).square().mean((1,2)))).mean()
        torch.testing.assert_close(parts['keep'],expected,rtol=1e-6,atol=1e-8)
        if float(expected)<=0:raise AssertionError('expected nonzero weighted ordinary proxy')
    save_json(out/'mask_checks.json',dict(zero_ordinary_keep=0.,fit_countries=['CN','DE','IS','US','JP'],
        indices=ids.tolist(),sample_ids=[bank.fit_rows[i]['sample_id'] for i in ids],
        weighted_ordinary_proxy=float(expected), weighted_ordinary_samples=len(ordinary_ids),
        source='real fit, all extreme and foreign ordinary; shared teacher-noise calculation'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True)
    p.add_argument('--data',required=True);p.add_argument('--exposure',required=True)
    p.add_argument('--artifacts',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out,_,host=checked_output(a.out)
    if host!='configured-compute':raise ValueError('model smoke requires the public checkout compute role')
    out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('smoke requires target GPU')
    device=torch.device('cuda:0'); bank=Bank(a.data)
    with np.load(a.exposure,allow_pickle=False) as z:
        schedules={name:z[name].copy() for name in ('natural','balanced')};ratio=z['ratio'].copy()
    if a.group=='M_N_ME':
        center_checks(bank,a.artifacts,device,out)
        mask_checks(bank,a.artifacts,device,ratio,out)
    restoration_check(a.group,bank,a.artifacts,device,schedules,ratio,out)
    run(a.group,bank,schedules,ratio,a.artifacts,device,out/'worker',steps=1,candidate_updates=[0,1])
    save_json(out/'smoke_status.json',dict(state='PARTIAL',group=a.group,engineering_updates=6,
        missing=['host_group_aggregation','queue_transition_failure_and_recovery','review']))


if __name__=='__main__':main()
