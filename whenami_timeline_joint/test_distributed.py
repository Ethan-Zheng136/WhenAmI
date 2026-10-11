"""Four CPU ranks emulate two shard groups using the real joint-model graph."""

import copy
import json
import multiprocessing
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from stage3_perf.checkpoint import CheckpointIO
from stage3_perf.runtime import topology
from whenami_timeline_joint.test_joint import fixture
from whenami_timeline_a.objectives import noisy_labels,velocity_loss


def worker(rank,root):
    torch.set_num_threads(1)
    os.environ["LOCAL_WORLD_SIZE"]="2"
    dist.init_process_group("gloo",init_method="file://"+str(Path(root)/"rdzv"),rank=rank,world_size=4)
    device=torch.device("cpu");strategy,groups=topology("hybrid_zero2",device)
    io=CheckpointIO("hybrid_zero2",groups)
    for mode in ("timeline","pooled_query"):
        core,x=fixture(mode)
        model=FSDP(core,device_id=device,use_orig_params=True,sharding_strategy=strategy,process_group=groups)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        y=torch.ones(2,3,16,7)*(rank+1);mask=torch.ones(2,3,16,dtype=torch.bool)
        noisy,tau,target=noisy_labels(y,42+rank)
        def update(m,o):
            o.zero_grad()
            out=m(**x,noisy_actions=noisy,tau=tau)
            loss,_=velocity_loss(out,target,mask)
            loss.backward();m.clip_grad_norm_(1.);o.step()
        update(model,opt)
        recipe={"mode":mode,"test":"actual_joint_graph_cpu_hybrid_zero2"}
        io.save(Path(root)/mode,model,opt,1,recipe)
        new,_=fixture(mode)
        restored=FSDP(new,device_id=device,use_orig_params=True,sharding_strategy=strategy,process_group=groups)
        op=torch.optim.AdamW(restored.parameters(),lr=.001)
        assert io.load(Path(root)/mode,restored,op,recipe)==1
        update(model,opt);update(restored,op)
        for a,b in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        dist.barrier()
    if rank==0:(Path(root)/"RESULT.json").write_text(json.dumps({"complete":True,"world":4,
        "cpu_emulated_nodes":2,"arms":["timeline","pooled_query"],
        "next_update_bitwise_equal":True,"full_8b_test":False}))
    dist.destroy_process_group()


def test_four_rank_same_graph_hybrid_zero2_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    ctx=multiprocessing.get_context("spawn")
    procs=[ctx.Process(target=worker,args=(rank,str(tmp_path))) for rank in range(4)]
    for p in procs:p.start()
    for p in procs:
        p.join(180)
        if p.is_alive():p.terminate();p.join()
        assert p.exitcode==0
