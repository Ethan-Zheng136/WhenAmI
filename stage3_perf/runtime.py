"""Explicit FSDP topology, selective recomputation, and sampled timing."""

from contextlib import contextmanager
from functools import partial
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)


def group_ranks(world, local_world):
    if local_world < 1 or world % local_world:
        raise ValueError("world must be divisible by the physical per-node rank count")
    return (
        [list(range(i, i + local_world)) for i in range(0, world, local_world)],
        [list(range(i, world, local_world)) for i in range(local_world)],
    )


def topology(sharding, device):
    """Construct all groups in identical global order; never assume eight cards."""
    if sharding == "full":
        return ShardingStrategy.FULL_SHARD, None
    if sharding not in ("hybrid", "hybrid_zero2"):
        raise ValueError(sharding)
    world, rank = dist.get_world_size(), dist.get_rank()
    local = int(os.environ["LOCAL_WORLD_SIZE"])
    if world <= local:
        raise ValueError("Hybrid admission requires >=2 physical nodes; use full for a local smoke")
    shards, replicas = group_ranks(world, local)
    shard_group = replica_group = None
    for ranks in shards:
        group = dist.new_group(ranks)
        if rank in ranks:
            shard_group = group
    for ranks in replicas:
        group = dist.new_group(ranks)
        if rank in ranks:
            replica_group = group
    strategy = (ShardingStrategy.HYBRID_SHARD if sharding == "hybrid"
                else ShardingStrategy._HYBRID_SHARD_ZERO2)
    return strategy, (shard_group, replica_group)


def block_classes():
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.bert.modeling_bert import BertLayer
    from timm.models.vision_transformer import Block
    from stage3.reader_adaptation import AdaptedBlock
    return {LlamaDecoderLayer, BertLayer, Block, AdaptedBlock,
            torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer}, LlamaDecoderLayer


def recompute(model, mode):
    classes, llm = block_classes()
    if mode not in ("all", "llm", "none"):
        raise ValueError(mode)
    if mode != "none":
        selected = tuple(classes) if mode == "all" else (llm,)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, selected),
        )


def wrap(model, device, strategy, groups, checkpoint_mode, teacher=False):
    classes, _ = block_classes()
    model = FSDP(
        model, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=classes),
        use_orig_params=True, device_id=device, sharding_strategy=strategy, process_group=groups,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32),
        limit_all_gathers=True,
    )
    if not teacher:
        recompute(model, checkpoint_mode)
    return model


class PhaseTimer:
    """Only profiled steps synchronize phase boundaries; they may run slower."""
    def __init__(self, enabled, device):
        self.enabled, self.device, self.values = enabled, device, {}

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        begin = time.monotonic()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.values[name] = self.values.get(name, 0.) + time.monotonic() - begin

    def global_max(self):
        if not self.enabled:
            return {}
        keys = sorted(self.values)
        values = torch.tensor([self.values[k] for k in keys], device=self.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return dict(zip(keys, values.cpu().tolist()))
