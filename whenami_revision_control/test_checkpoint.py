import multiprocessing
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from stage3_perf.checkpoint import CheckpointIO
from stage3_perf.runtime import topology
from whenami_timeline_a.model import ReadEvidence
from whenami_revision_control.test_model import fixture, inputs, CFG
from whenami_revision_control.model import draw, objective, VARIANTS
from whenami_revision_control.loading import load_model


def worker(rank, directory):
    torch.set_num_threads(1)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    dist.init_process_group("gloo", init_method="file://"+str(Path(directory)/"rdzv"),
                            rank=rank, world_size=4)
    strategy, groups = topology("hybrid_zero2", torch.device("cpu"))
    io = CheckpointIO("hybrid_zero2", groups)
    def wrap(model):
        allowed = {id(model.vlm.vision_backbone), id(model.vlm.projector),
                   id(model.vlm.llm_backbone), id(model.reader)}
        def policy(module, recurse, nonwrapped_numel):
            return True if recurse else id(module) in allowed or isinstance(module, ReadEvidence)
        return FSDP(model, device_id=torch.device("cpu"), use_orig_params=True,
                    auto_wrap_policy=policy, sharding_strategy=strategy, process_group=groups)
    _, base, _, _ = fixture()
    base = wrap(base)
    io.save(Path(directory)/"B_source", base, torch.optim.AdamW(base.parameters(), lr=.001),
            0, {"source_B": True})
    for variant in VARIANTS:
        core, _, x, window = fixture(variant)
        model = wrap(core)
        loaded = load_model(model, Path(directory)/"B_source", groups[0], True, student=True)
        assert loaded["new_entries"] and all(k.startswith(("memory.", "bridge.")) for k in loaded["new_entries"])
        def optimizer(m):
            return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.001)
        opt = optimizer(model)
        inp=inputs(x,window)
        noise,tau,target=draw(torch.ones(2,16,7)*(rank+1),42+rank)
        before=torch.randn(2,2,CFG.queries,CFG.condition)
        after=torch.randn_like(before)
        mask=torch.ones(2,16,dtype=torch.bool)
        def update(m,o):
            o.zero_grad()
            result=m(inputs=inp,noisy_actions=noise,tau=tau)
            loss,_=objective(result,target,mask,before,after)
            loss.backward();m.clip_grad_norm_(1.);o.step()
        random.seed(120+rank);np.random.seed(120+rank);torch.manual_seed(120+rank)
        update(model,opt)
        path=Path(directory)/variant;recipe={"variant":variant}
        io.save(path,model,opt,1,recipe)
        expected=(random.random(),np.random.rand(4),torch.rand(4))
        fresh,_,_,_=fixture(variant)
        restored=wrap(fresh);new_opt=optimizer(restored)
        assert io.load(path,restored,new_opt,recipe)==1
        assert random.random()==expected[0]
        assert np.array_equal(np.random.rand(4),expected[1])
        assert torch.equal(torch.rand(4),expected[2])
        update(model,opt);update(restored,new_opt)
        for p,q in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        dist.barrier()
    dist.destroy_process_group()


def test_cpu4rank_B_warmstart_and_full_optimizer_rng_restart(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    ctx=multiprocessing.get_context("spawn")
    workers=[ctx.Process(target=worker,args=(rank,str(tmp_path))) for rank in range(4)]
    for p in workers:p.start()
    for p in workers:
        p.join(240)
        if p.is_alive():p.terminate();p.join()
        assert p.exitcode==0
