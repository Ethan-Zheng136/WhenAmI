"""Reuse proven stage3 FSDP topology/checkpoints; add actual original DiTBlock wraps."""

from functools import partial
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from stage3_perf.runtime import topology, recompute, block_classes


def wrap(model,device,strategy,groups,activation="llm"):
    from action_model.models import DiTBlock
    from whenami_timeline_a.model import ReadEvidence
    classes,_=block_classes()
    classes=classes|{DiTBlock,ReadEvidence}
    result=FSDP(model,device_id=device,use_orig_params=True,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=classes),
        sharding_strategy=strategy,process_group=groups,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32),
        limit_all_gathers=True)
    recompute(result,activation)
    return result


def learning_rate(step,steps,peak,warmup=1000):
    import math
    if step<=warmup:return peak*step/max(warmup,1)
    phase=(step-warmup)/max(1,steps-warmup)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*phase)))


def gradient_norms(model,device):
    import torch.distributed as dist
    names=("vlm_vision","projector","llm","qformer","time","dit")
    sums={k:torch.zeros((),device=device) for k in names}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        name=name.replace("_fsdp_wrapped_module.","").replace("_checkpoint_wrapped_module.","")
        if name.startswith("vlm.vision_backbone."):group="vlm_vision"
        elif name.startswith("vlm.projector."):group="projector"
        elif name.startswith("vlm.llm_backbone."):group="llm"
        elif name.startswith("local_encoder."):group="qformer"
        elif name.startswith("timeline."):group="time"
        elif name.startswith("reader."):group="dit"
        else:continue
        sums[group]+=p.grad.detach().float().square().sum()
    vector=torch.stack([sums[k] for k in names])
    if dist.is_initialized():dist.all_reduce(vector)
    return dict(zip(names,vector.sqrt().cpu().tolist()))
