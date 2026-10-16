"""New recipe coverage and existing proven 16-rank HSDP machinery, unchanged."""
from pathlib import Path
import torch
import torch.distributed as dist
from stage3.common import BASE, REFERENCE
from stage3_oxe_eval.prepare import sha
from whenami_timeline_joint.runtime import wrap as proven_wrap, topology, learning_rate
from stage3_perf.checkpoint import CheckpointIO
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from stage3_perf.runtime import recompute

DEPENDENCIES = ("whenami_timeline_force", "whenami_unified", "whenami_timeline_joint",
                "whenami_timeline_a", "whenami_timeline_b", "whenami_revision",
                "stage3", "stage3_scale", "stage3_perf", "stage3_oxe_eval",
                "stage3_oxe_eval50", "evaluation")


def wrap(model, device, strategy, groups, activation="llm"):
    if dist.is_initialized() and dist.get_world_size() == 1:
        # Engineering-only NO_SHARD: nested wrapping of the never-executed
        # final ViT block leaves orig params unregistered during state_dict
        # on this PPU Torch build. Root wrap avoids that unused-child case.
        # Do NOT mistake this for admission of formal nested16rank HSDP.
        result = FSDP(model, device_id=device, use_orig_params=True,
                      sharding_strategy=strategy, process_group=groups,
                      mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                          # Whole-model FP32 gradient buffer would exceed96GiB
                          # after full Adam restore in BF16-storage engineering.
                          # Formal320 nested HSDP continues FP32 reduction.
                          reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32),
                      limit_all_gathers=True)
        recompute(result, activation)
        return result
    return proven_wrap(model, device, strategy, groups, activation)


def fingerprints():
    result = {str(p.relative_to(BASE)): sha(p) for name in DEPENDENCIES
              for p in sorted((BASE/name).glob("*.py"))}
    result["run_whenami_timeline_force_320.sh"] = sha(BASE/"run_whenami_timeline_force_320.sh")
    return result


def reference_fingerprints():
    return {str(p.relative_to(REFERENCE)): sha(p) for p in sorted(REFERENCE.rglob("*.py"))}


def clean_name(name):
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")


def optimizer_groups(model, lr, new_lr):
    old, new = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = clean_name(name)
        (new if clean.startswith(("node_cognition.", "executed_adapter.", "forecast_head.")) else old).append(param)
    return [{"params": old, "lr": lr}, {"params": new, "lr": new_lr}]


def gradients(model, device):
    names = ("vision", "projector", "llm", "local_qformer", "timeline", "node_cognition", "dit")
    if any("executed_adapter." in n for n, _ in model.named_parameters()):
        names += ("executed_adapter", "forecast_head")
    sums = {key: torch.zeros((), device=device) for key in names}
    for name, value in model.named_parameters():
        if value.grad is None:
            continue
        n = clean_name(name)
        if n.startswith("vlm.vision_backbone."): key = "vision"
        elif n.startswith("vlm.projector."): key = "projector"
        elif n.startswith("vlm.llm_backbone."): key = "llm"
        elif n.startswith("local_encoder."): key = "local_qformer"
        elif n.startswith("timeline."): key = "timeline"
        elif n.startswith("node_cognition."): key = "node_cognition"
        elif n.startswith("reader."): key = "dit"
        elif n.startswith("executed_adapter."): key = "executed_adapter"
        elif n.startswith("forecast_head."): key = "forecast_head"
        else: continue
        sums[key] += value.grad.detach().float().square().sum()
    vector = torch.stack(list(sums.values()))
    if dist.is_initialized():
        dist.all_reduce(vector)
    return dict(zip(sums, vector.sqrt().cpu().tolist()))
