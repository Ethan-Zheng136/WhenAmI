"""Reuse proven FSDP machinery; optimizer groups / gradient audit for present-token modules."""
import torch
import torch.distributed as dist
from stage3.common import BASE
from stage3_oxe_eval.prepare import sha
from whenami_timeline_force.runtime import wrap, topology, learning_rate, clean_name, reference_fingerprints  # noqa: F401

DEPENDENCIES = ("whenami_present", "whenami_timeline_force", "whenami_unified", "whenami_timeline_joint",
                "whenami_timeline_a", "stage3", "stage3_scale", "stage3_perf", "stage3_oxe_eval", "stage3_oxe_eval50")


def fingerprints():
    r = {str(p.relative_to(BASE)): sha(p) for n in DEPENDENCIES for p in sorted((BASE/n).glob("*.py"))}
    r["run_whenami_present.sh"] = sha(BASE/"run_whenami_present.sh")
    return r


HEAD_PREFIXES = ("present.precision_", "present.position.", "present.embed")


def optimizer_groups(model, lr, new_lr, new_prefixes, head_lr=None):
    """old (1e-6) / new modules (1e-5) / zero-init small heads (precision log-variance, tau/phase, present embed; 0.45M
    params, from scratch, outputs must travel O(1) units -> separate head LR, both stages)."""
    old, new, head = [], [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = clean_name(name)
            (head if head_lr is not None and n.startswith(HEAD_PREFIXES) else
             new if n.startswith(new_prefixes) else old).append(p)
    groups = [{"params": old, "lr": lr}, {"params": new, "lr": new_lr}]
    if head_lr is not None:
        groups.append({"params": head, "lr": head_lr})
    return groups


GROUPS = (("vlm.vision_backbone.", "vision"), ("vlm.projector.", "projector"), ("vlm.llm_backbone.", "llm"),
          ("local_encoder.", "local_qformer"), ("timeline.", "timeline_U_fwd"), ("node_cognition.", "node_cognition"),
          ("reader.", "dit"), ("present.embed", "present_embed"), ("present.bwd.", "U_bwd"),
          ("present.precision_", "precision_heads"), ("present.position.", "position_head"),
          ("present.imaginer.", "future_imaginer"), ("executed_adapter.", "executed_gru"))


def gradients(model, device):
    names = [g for p, g in GROUPS if any(clean_name(n).startswith(p) for n, _ in model.named_parameters())]
    sums = {k: torch.zeros((), device=device) for k in names}
    for name, v in model.named_parameters():
        if v.grad is None:
            continue
        n = clean_name(name)
        for p, g in GROUPS:
            if n.startswith(p):
                sums[g] += v.grad.detach().float().square().sum()
                break
    vec = torch.stack(list(sums.values()))
    if dist.is_initialized():
        dist.all_reduce(vec)
    return dict(zip(sums, vec.sqrt().cpu().tolist()))
