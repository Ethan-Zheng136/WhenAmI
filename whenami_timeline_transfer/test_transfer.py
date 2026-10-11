import copy
import torch
import pytest

from whenami_timeline_joint.test_joint import fixture
from whenami_timeline_transfer.contracts import make_prefix
from whenami_timeline_transfer.model import CausalStudent,teacher_target,action_draw,objective

torch.set_num_threads(1)


def split(x):
    x=copy.deepcopy(x)
    features={k:x.pop(k) for k in ("patches","vae")}
    x.pop("visible")
    return x,features


def test_prefix_physically_removes_future_and_padding():
    _,batch=fixture();x,f=split(batch)
    p=make_prefix(x,f,1)
    assert p.times.shape==(2,2)
    assert all(v.shape[1]==2 for v in p.pixels.values())
    assert p.input_ids.shape[0]==4 and p.input_ids.shape[1]==3
    assert p.patches.shape[1]==2 and p.vae.shape[1]==2


def test_future_images_language_and_features_cannot_change_student():
    base,batch=fixture();student=CausalStudent(base,4).eval();x,f=split(batch)
    y,g=copy.deepcopy(x),copy.deepcopy(f)
    for v in y["pixels"].values():v[:,2]+=1000
    y["input_ids"].reshape(2,3,-1)[:,2]=31
    y["times"][:,2]+=1000
    for v in g.values():v[:,2]+=1000
    a=student(make_prefix(x,f,1));b=student(make_prefix(y,g,1))
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_privileged_teacher_responds_to_later_evidence():
    base,batch=fixture();x,f=split(batch)
    altered=copy.deepcopy(f);altered["patches"][:,2]*=3
    a=teacher_target(base,x,f,1);b=teacher_target(base,x,altered,1)
    assert not torch.allclose(a,b)


def test_auxiliary_head_is_decoupled_from_action_readout():
    base,batch=fixture();student=CausalStudent(base,4).eval();x,f=split(batch);p=make_prefix(x,f,1)
    noisy,tau,_=action_draw(torch.ones(2,16,7),42)
    a=student(p,noisy,tau)
    with torch.no_grad():student.future_predictor.weight.add_(10)
    b=student(p,noisy,tau)
    torch.testing.assert_close(a["velocity"],b["velocity"],rtol=0,atol=0)
    assert not torch.allclose(a["forecast"],b["forecast"])


def test_teacher_and_label_gradients_blocked_student_gradients_live():
    base,batch=fixture();student=CausalStudent(base,4).train();x,f=split(batch)
    actions=torch.ones(2,16,7,requires_grad=True)
    noisy,tau,target=action_draw(actions,5)
    output=student(make_prefix(x,f,1),noisy,tau)
    teacher=torch.randn_like(output["forecast"],requires_grad=True)
    loss,_=objective(output,target,torch.ones(2,16,dtype=torch.bool),teacher)
    loss.backward()
    assert teacher.grad is None and actions.grad is None
    for module in (student.future_predictor,student.base.vlm.vision_backbone,
                   student.base.vlm.llm_backbone,student.base.timeline,student.base.reader):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0


def test_no_teacher_loss_is_exact_same_action_objective():
    base,batch=fixture();student=CausalStudent(base,4);x,f=split(batch)
    noisy,tau,target=action_draw(torch.ones(2,16,7),42)
    output=student(make_prefix(x,f,1),noisy,tau);mask=torch.ones(2,16,dtype=torch.bool)
    a,_=objective(output,target,mask,None)
    b,m=objective(output,target,mask,torch.randn_like(output["forecast"]),weight=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert float(m["condition_alignment"])==0


def test_student_rejects_unvalidated_full_batch():
    base,batch=fixture();student=CausalStudent(base,4)
    with pytest.raises(TypeError):student(batch)
