"""Four-rank CPU nested-FSDP A->C warmstart and C Adam/read-scale restart."""
import multiprocessing
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from stage3_perf.checkpoint import CheckpointIO
from stage3_perf.runtime import topology
from whenami_timeline_a.model import ReadEvidence
from whenami_timeline_joint.test_joint import fixture, CFG
from whenami_timeline_transfer.contracts import make_prefix
from whenami_timeline_b.model import draw
from whenami_timeline_c.model import TimelineStudent, objective
from whenami_timeline_c.loading import load_model


def build(variant):
    a, inputs = fixture()
    model = TimelineStudent(a.vlm, a.local_encoder, a.reader, variant, 4, CFG)
    model.timeline.load_state_dict(a.timeline.state_dict(), strict=True)
    features = {k: inputs.pop(k) for k in ("patches", "vae")}
    inputs.pop("visible")
    return model, make_prefix(inputs, features, 1)


def worker(rank, directory):
    torch.set_num_threads(1)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    root = Path(directory)
    dist.init_process_group("gloo", init_method="file://" + str(root / "rdzv"),
                            rank=rank, world_size=4)
    strategy, groups = topology("hybrid_zero2", torch.device("cpu"))
    io = CheckpointIO("hybrid_zero2", groups)

    def wrapped(module):
        allowed = {id(module.vlm.vision_backbone), id(module.vlm.projector),
                   id(module.vlm.llm_backbone), id(module.reader)}

        def policy(module, recurse, nonwrapped_numel):
            return True if recurse else (id(module) in allowed or isinstance(module, ReadEvidence))
        return FSDP(module, device_id=torch.device("cpu"), use_orig_params=True,
                    auto_wrap_policy=policy, sharding_strategy=strategy, process_group=groups)

    a, _ = fixture()
    a = wrapped(a)
    ao = torch.optim.AdamW(a.parameters(), lr=.001)
    io.save(root / "A", a, ao, 0, {"A": True})
    for variant in ("future_aux", "future_read"):
        core, prefix = build(variant)
        c = wrapped(core)
        loaded = load_model(c, root / "A", groups[0], sharded=True)
        assert loaded["new_entries"] == ["read_scale"]
        optimizer = torch.optim.AdamW(c.parameters(), lr=.001)
        noisy, tau, target = draw(torch.ones(2, 16, 7) * (rank + 1), 42 + rank)
        teacher = torch.randn(2, 2, CFG.queries, CFG.condition)
        valid = torch.ones(2, 16, dtype=torch.bool)

        def update(model, raw, opt, step):
            raw.set_step(step, read_warmup=100)
            opt.zero_grad(set_to_none=True)
            output = model(prefix=prefix, noisy_actions=noisy, tau=tau)
            loss, _ = objective(output, target, valid, teacher)
            loss.backward()
            model.clip_grad_norm_(1.)
            opt.step()

        update(c, core, optimizer, 40)
        recipe = {"C": variant}
        io.save(root / variant, c, optimizer, 1, recipe)
        raw_other, _ = build(variant)
        restored = wrapped(raw_other)
        new_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
        assert io.load(root / variant, restored, new_optimizer, recipe) == 1
        torch.testing.assert_close(core.read_scale, raw_other.read_scale, rtol=0, atol=0)
        update(c, core, optimizer, 41)
        update(restored, raw_other, new_optimizer, 41)
        for x, y in zip(c.parameters(), restored.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        dist.barrier()
    dist.destroy_process_group()


def test_cpu4rank_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, str(tmp_path))) for rank in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(240)
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0
