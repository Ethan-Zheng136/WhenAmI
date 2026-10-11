import ast
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from stage3.common import BASE
from stage3_perf.checkpoint import CheckpointIO
from stage3_perf.runtime import topology
from whenami_timeline_b.test_b import build
from whenami_timeline_b.model import draw, objective
from whenami_timeline_a.model import ReadEvidence
from whenami_timeline_transfer.contracts import make_prefix
from whenami_b_continue.contracts import PARENTS,START,END,SCHEDULE,continuation_lr,restore
from whenami_b_continue.replay_check import compare_saved

torch.set_num_threads(1)


@pytest.mark.parametrize("rate",(1e-6,1e-5))
def test_terminal_rate_is_held_without_warmup(rate):
    assert [continuation_lr(s,rate) for s in (20000,20001,20100,25000,30000,40000,50000)] == [rate]*7
    for step in (0,19999,50001):
        with pytest.raises(ValueError):continuation_lr(step,rate)


def test_original_model_data_and_objective_reused():
    from whenami_b_continue import train
    from whenami_timeline_b import model,data,evaluate
    assert train.build_student is model.build_student
    assert train.objective is model.objective and train.draw is model.draw
    assert train.batch is data.batch and train.evaluate is evaluate.evaluate
    tree=ast.parse((BASE/"whenami_b_continue/train.py").read_text())
    names=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert "restore" in names and "continuation_lr" in names and "learning_rate" not in names
    text=(BASE/"whenami_b_continue/train.py").read_text()
    assert "load_model_only(student" not in text
    assert "(step*accum+micro)*world+rank" in text


def test_original_teacher_and_student_microstep_unchanged():
    def microstep(path):
        tree=ast.parse(path.read_text())
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For)
                    and isinstance(n.target,ast.Name) and n.target.id=="micro")
    old=microstep(BASE/"whenami_timeline_b50/train.py")
    new=microstep(BASE/"whenami_b_continue/train.py")
    assert ast.dump(old)==ast.dump(new)


@pytest.mark.parametrize("case",tuple(PARENTS))
@pytest.mark.parametrize("phase",("run","admit","full"))
def test_launch_commands(case,phase):
    env=dict(os.environ,WORLD_SIZE="20",RANK="0",NPROC_PER_NODE="16",
             B_CONTINUE_CASE=case,PHASE=phase,DRY_RUN="1")
    output=subprocess.check_output(["bash",str(BASE/"run_b_continue_320.sh")],env=env,text=True)
    assert "--nnodes 20" in output and "--nproc_per_node 16" in output
    assert "--module whenami_b_continue.train --case "+case in output
    assert "/outputs/whenami_b_continue/training/" in output and "_20k_to50k_" in output
    if phase in ("admit","run"):
        assert "--stop-after 20100" in output and "--stop-after 20120" in output
        assert "step-020100" in output
    if phase in ("run","full"):assert "step-020120" in output and "/main" in output
    assert "--lr" not in output and "--warmup" not in output


@pytest.mark.parametrize("nodes,local",(("40","16"),("20","8"),("1","16")))
def test_wrong_topology_refused(nodes,local):
    env=dict(os.environ,WORLD_SIZE=nodes,RANK="0",NPROC_PER_NODE=local,DRY_RUN="1")
    result=subprocess.run(["bash",str(BASE/"run_b_continue_320.sh")],env=env,capture_output=True)
    assert result.returncode==2


def test_no_large_run_without_actual_admission(tmp_path):
    from whenami_b_continue import preflight
    # Missing local/cluster evidence is not silently treated as success.
    with pytest.raises(FileNotFoundError):preflight.check_admission(tmp_path,"old_b0")


def make_replay(root,delta=0):
    root.mkdir()
    (root/"RESULT.json").write_text(json.dumps({"complete":True,"episodes":1,
        "noise_seeds":[42,43,44],"student_observed_nodes":2}))
    np.savez_compressed(root/"rank-00000.npz",indices=np.asarray([0]),
        targets=np.zeros((1,16,7)),masks=np.ones((1,16),bool),
        predictions=np.ones((1,3,16,7))*delta)


def test_replay_checks_actions_not_just_mean_metrics(tmp_path):
    make_replay(tmp_path/"old");make_replay(tmp_path/"same");make_replay(tmp_path/"wrong",.01)
    assert compare_saved(tmp_path/"old",tmp_path/"same")["max_prediction_abs_error"]==0
    with pytest.raises(ValueError,match="mismatch"):compare_saved(tmp_path/"old",tmp_path/"wrong")


def worker(rank,root):
    torch.set_num_threads(1);os.environ["LOCAL_WORLD_SIZE"]="2"
    dist.init_process_group("gloo",init_method="file://"+str(Path(root)/"rdzv"),rank=rank,world_size=4)
    strategy,groups=topology("hybrid_zero2",torch.device("cpu"));io=CheckpointIO("hybrid_zero2",groups)
    def wrap(m):
        allowed={id(m.vlm.vision_backbone),id(m.vlm.projector),id(m.vlm.llm_backbone),id(m.reader)}
        def policy(module,recurse,nonwrapped_numel):
            return True if recurse else id(module) in allowed or isinstance(module,ReadEvidence)
        return FSDP(m,device_id=torch.device("cpu"),use_orig_params=True,
                    auto_wrap_policy=policy,sharding_strategy=strategy,process_group=groups)
    def optimizer(m):
        body,aux=[],[]
        for name,p in m.named_parameters():
            clean=name.replace("_fsdp_wrapped_module.","").replace("_checkpoint_wrapped_module.","")
            if p.requires_grad:(aux if clean.startswith("forecast_head.") else body).append(p)
        return torch.optim.AdamW([{"params":body,"lr":1e-6},{"params":aux,"lr":1e-5}],weight_decay=0.)
    for variant in ("action_only","condition_align"):
        m,x,f=build();m=wrap(m);opt=optimizer(m)
        prefix=make_prefix(x,f,1);teacher=torch.randn(2,3,4)
        valid=torch.ones(2,16,dtype=torch.bool)
        def update(model,optim):
            # Exercise actual restored Torch RNG, not just a pre-built fixed noisy tensor.
            tau=torch.rand(4)
            noisy=torch.randn(4,16,7)
            target=torch.ones_like(noisy)*(rank+1)
            optim.zero_grad()
            out=model(prefix=prefix,noisy_actions=noisy,tau=tau)
            loss,_=objective(out,target,valid,teacher,variant)
            loss.backward();model.clip_grad_norm_(1.);optim.step()
        update(m,opt)
        # Synthetic small-model fixture: establish real moments with one update,
        # then label Adam counters20k. Not a claim of 20k tiny-model training.
        for state in opt.state.values():state["step"].fill_(START)
        parent_recipe={"schema":"toy_original_B","variant":variant,"steps":START}
        path=Path(root)/variant/"parent20k"
        random.seed(876+rank);np.random.seed(876+rank);torch.manual_seed(876+rank)
        io.save(path,m,opt,START,parent_recipe)
        parent={"path":str(path),"committed_sha256":hashlib.sha256((path/"COMMITTED.json").read_bytes()).hexdigest(),
                "parent_terminal_lr":[1e-6,1e-5]}
        recipe={"schema":"toy_continuation","variant":variant,"parent":parent,
                "steps":END,"schedule":SCHEDULE}
        # Parent backend still rejects a changed recipe; continuation must
        # explicitly restore using the original recipe, not weaken CheckpointIO.
        with pytest.raises(ValueError,match="same world/topology/recipe"):
            io.load(path,m,opt,recipe)
        expected_random=(random.random(),np.random.rand(3))
        update(m,opt)
        other,_,_=build();other=wrap(other);newopt=optimizer(other)
        step,proof=restore(io,other,newopt,parent,recipe)
        assert step==START and proof["original_parent_recipe_used"]
        assert random.random()==expected_random[0]
        assert np.array_equal(np.random.rand(3),expected_random[1])
        update(other,newopt)
        for a,b in zip(m.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(opt.state.values(),newopt.state.values()):
            for k in ("step","exp_avg","exp_avg_sq"):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        cont=Path(root)/variant/"continuation20001"
        io.save(cont,other,newopt,START+1,recipe)
        expected_random=(random.random(),np.random.rand(3))
        update(other,newopt)
        third,_,_=build();third=wrap(third);thirdopt=optimizer(third)
        step,proof=restore(io,third,thirdopt,parent,recipe,cont)
        assert step==START+1 and not proof["original_parent_recipe_used"]
        assert random.random()==expected_random[0]
        assert np.array_equal(np.random.rand(3),expected_random[1])
        update(third,thirdopt)
        for a,b in zip(other.parameters(),third.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        bad=copy.deepcopy(recipe);bad["steps"]=60000
        with pytest.raises(ValueError,match="wrong continuation"):
            restore(io,third,thirdopt,parent,bad,cont)
        wrong=copy.deepcopy(parent);wrong["committed_sha256"]="bad"
        with pytest.raises(ValueError,match="wrong original parent"):
            restore(io,third,thirdopt,wrong,recipe)
        dist.barrier()
    dist.destroy_process_group()


def test_cpu4rank_parent_and_continuation_restore_next_update_bitwise(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    ctx=multiprocessing.get_context("spawn")
    workers=[ctx.Process(target=worker,args=(rank,str(tmp_path))) for rank in range(4)]
    for p in workers:p.start()
    for p in workers:
        p.join(240)
        if p.is_alive():p.terminate();p.join()
        assert p.exitcode==0
