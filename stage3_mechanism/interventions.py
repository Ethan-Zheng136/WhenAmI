"""Fixed-weight condition interventions. All prediction APIs remain label-free."""

from contextlib import contextmanager
import numpy as np
import torch

VARIANTS = ("full", "no_k", "mean_future", "repeat_history", "repeat_history_no_k")


@contextmanager
def capture_condition_path(policy):
    captured = {}
    def capture_cp(module, args, out):
        captured["cp"] = out
    def capture_k(module, args, out):
        captured["h"], captured["z"], captured["standardized_future"] = args
        captured["k_output"] = out
    handles = [policy.policy.past_qformer.v_proj.register_forward_hook(capture_cp),
               policy.policy.k.register_forward_hook(capture_k)]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def condition_from_path(cp, k_output, scale, enabled=True):
    return cp + k_output * scale if enabled else cp


@torch.no_grad()
def encoded_variants(policy, current, history, instruction):
    with capture_condition_path(policy) as normal:
        full = policy.encode(current, history, instruction)
    required = {"cp", "h", "z", "standardized_future", "k_output"}
    if set(normal) != required:
        raise ValueError("missing condition-path hook capture")
    cp, z = normal["cp"], full["z"]
    if not torch.equal(condition_from_path(cp, normal["k_output"], policy.policy.r.past_scale),
                       full["condition"]):
        raise ValueError("captured path does not reproduce unmodified condition")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        zero_future = torch.zeros_like(normal["standardized_future"])
        mean_delta = policy.policy.k(normal["h"], normal["z"], zero_future)
        mean_condition = condition_from_path(cp, mean_delta, policy.policy.r.past_scale)
    repeated = np.repeat(current[None], 4, axis=0)
    with capture_condition_path(policy) as static:
        repeat = policy.encode(current, repeated, instruction)
    # Current image/language VLM inputs unchanged; no history enters cognition.
    if not torch.equal(z, repeat["z"]):
        raise ValueError("history intervention unexpectedly changed current cognition")
    conditions = {
        "full": full, "no_k": {"z": z, "condition": cp},
        "mean_future": {"z": z, "condition": mean_condition},
        "repeat_history": repeat,
        "repeat_history_no_k": {"z": repeat["z"], "condition": static["cp"]},
    }
    def rms(x):
        return float(x.float().square().mean().sqrt())
    delta = normal["k_output"] * policy.policy.r.past_scale
    diagnostics = {
        "cp_rms": rms(cp), "delta_rms": rms(delta),
        "delta_over_cp_rms": rms(delta) / max(rms(cp), 1e-8),
        "standardized_future_rms": rms(normal["standardized_future"]),
        "mean_future_condition_change_rms": rms(mean_condition - full["condition"]),
        "repeat_history_condition_change_rms": rms(repeat["condition"] - full["condition"]),
    }
    return conditions, diagnostics
