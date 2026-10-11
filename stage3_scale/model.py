"""Joint raw-input WhenAmI policy, explicitly trainable beyond cached adapters.

Student: VLM vision/projector/LLM + Past QFormer + R/K + full Past action DiT.
Frozen foundation visual features are outside the student; Future targets and
optional teacher are separate modules. Student APIs never accept future images.
"""

import copy
from pathlib import Path
import json

import torch
from torch import nn

from stage3.common import DEFAULT_PAST, DEFAULT_FUTURE, STORE, reference_imports
from stage3.connector import SharedConnector, ConnectorConfig
from stage3.refinement import RetrospectiveCorrector, RefinementConfig
from stage3.reader_adaptation import attach_adapters
from stage3.behavior import flow_draw


def tokenize_instruction(instruction, tokenizer, prompt_builder):
    prompt = prompt_builder()
    prompt.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    ids = tokenizer(prompt.get_prompt(), truncation=True, return_tensors="pt").input_ids[0]
    return torch.cat([ids, torch.tensor([29871, 2], dtype=torch.long)])


def cognition(vlm, input_ids, attention_mask, current):
    output = vlm(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=current,
        labels=None, use_cache=False, output_hidden_states=True, return_dict=True,
    )
    hidden = output.hidden_states[-1]
    inserted_patches = hidden.shape[1] - input_ids.shape[1]
    positions = attention_mask.long().sum(1) - 1 + inserted_patches
    return hidden[torch.arange(len(hidden), device=hidden.device), positions][:, None]


class JointPolicy(nn.Module):
    def __init__(self, vlm, past_qformer, action_net, statistics, config=RefinementConfig()):
        super().__init__()
        self.vlm, self.past_qformer, self.net = vlm, past_qformer, action_net
        self.config = config
        self.r = SharedConnector(ConnectorConfig(
            feature_dim=config.feature_dim, cognition_dim=config.cognition_dim,
            tokens=config.tokens, condition_dim=config.condition_dim,
            hidden=config.r_hidden, layers=config.r_layers, heads=config.r_heads,
        ), statistics)
        self.k = RetrospectiveCorrector(config)
        self.requires_grad_(True)
        self.r.freeze_backward_only_adapters()

    def forward(self, input_ids=None, attention_mask=None, current=None, history_features=None,
                history_vae=None, noisy_actions=None, tau=None, condition_only=False, conditioning=None):
        # Sampling reuses freshly computed causal features, never stale training
        # cache. Call through the wrapped root even for decoder-only FSDP passes.
        if conditioning is not None:
            if condition_only or any(x is not None for x in (input_ids, current, history_features)):
                raise ValueError("decoder-only call cannot also encode observations")
            return {"velocity": self.net(noisy_actions, tau, conditioning["z"], per_token=conditioning["condition"])}
        # No detach: action and auxiliary gradients must reach the real policy.
        h = self.past_qformer.encode_queries(history_features, history_vae)
        c_p = self.past_qformer.v_proj(h)
        z = cognition(self.vlm, input_ids, attention_mask, current)
        c_f = self.r.predict(h, z, "forward")
        delta = self.k(h, z, (c_f - self.r.future_mean) / self.r.future_scale) * self.r.past_scale
        c = c_p + delta
        if condition_only:
            return {"z": z, "condition": c, "future_condition": c_f}
        repeat = noisy_actions.shape[0] // input_ids.shape[0]
        if repeat < 1 or len(noisy_actions) % len(input_ids):
            raise ValueError("noisy action batch must be an integer repeat of observations")
        velocity = self.net(
            noisy_actions, tau, z.repeat_interleave(repeat, 0),
            per_token=c.repeat_interleave(repeat, 0),
        )
        return {"velocity": velocity, "future_condition": c_f, "past_condition": c_p}


class FrozenOracle(nn.Module):
    """Fixed adaptive_future5k teacher, re-encoded from raw inputs, own frozen z."""
    def __init__(self, vlm, past_qformer, net, probe):
        super().__init__()
        self.vlm, self.past_qformer, self.net, self.probe = vlm, past_qformer, net, probe
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, input_ids, attention_mask, current, history_features, history_vae,
                true_future_condition, noisy_actions, tau):
        h = self.past_qformer.encode_queries(history_features, history_vae)
        cp = self.past_qformer.v_proj(h)
        z = cognition(self.vlm, input_ids, attention_mask, current)
        c = self.probe(h, z, cp, true_future_condition)
        repeat = noisy_actions.shape[0] // input_ids.shape[0]
        return self.net(noisy_actions, tau, z.repeat_interleave(repeat, 0),
                        per_token=c.repeat_interleave(repeat, 0))


class FrozenVisualTargets(nn.Module):
    """Shared frozen DINO/Wan features, and fixed Future QFormer. No VLM here."""
    def __init__(self, visual_encoder, future_qformer):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.future_qformer = future_qformer
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def encode(self, frames, current, chronological_past=False):
        b, t = next(iter(frames.values())).shape[:2]
        flat = {k: x.reshape(b * t, *x.shape[2:]) for k, x in frames.items()}
        if chronological_past:
            video = {k: torch.cat([frames[k], current[k][:, None]], 1) for k in current}
            features, vae = self.visual_encoder(flat, dict(current), video_pixel_values=video)
        else:
            features, vae = self.visual_encoder(flat, dict(current))
        _, n, dim = features.shape
        size = int(n ** .5)
        if size * size != n:
            raise ValueError("expected square DINO patch grid")
        patches = features.reshape(b * t, size, size, dim).permute(0, 3, 1, 2)
        patches = torch.nn.functional.adaptive_avg_pool2d(patches, (8, 8))
        patches = patches.permute(0, 2, 3, 1).reshape(b, t * 64, dim).contiguous()
        return patches, vae

    @torch.no_grad()
    def forward(self, current, past, future=None):
        features, vae = self.encode(past, current, chronological_past=True)
        out = {"history_features": features, "history_vae": vae}
        if future is not None:
            f, v = self.encode(future, current)
            out["future_condition"] = self.future_qformer(f, v)
        return out


def objective(output, actions, seed, draws=4, future_target=None, future_scale=None,
              teacher_velocity=None, alignment_weight=.1, distill_weight=.3, action_mask=None):
    # This helper expects the same seed/draw used before invoking the policy.
    _, _, target_velocity = flow_draw(actions.detach().repeat_interleave(draws, 0), seed)
    def masked_mse(error):
        if action_mask is None:
            return error.square().mean()
        if action_mask.shape != actions.shape[:2]:
            raise ValueError("valid action mask must match observation/action window")
        mask = action_mask.detach().repeat_interleave(draws, 0).to(error.dtype)
        if not (mask.sum(1) > 0).all():
            raise ValueError("each observation needs a real action target")
        # Each sampled observation contributes equally, regardless of remaining
        # episode length. No padded action is treated as an expert target.
        per_step = error.square().mean(-1)
        return ((per_step * mask).sum(1) / mask.sum(1)).mean()
    action = masked_mse(output["velocity"].float() - target_velocity.float())
    values = {"action_fm": action}
    total = action
    if future_target is not None and alignment_weight:
        align = ((output["future_condition"].float() - future_target.detach().float()) / future_scale).square().mean()
        values["future_alignment"] = align
        total = total + alignment_weight * align
    if teacher_velocity is not None and distill_weight:
        distill = masked_mse(output["velocity"].float() - teacher_velocity.detach().float())
        values["distill_velocity"] = distill
        total = total + distill_weight * distill
    values["total"] = total
    return total, values


def load_joint(seed=42, teacher=False):
    """Build on CPU then let FSDP shard before accelerator allocation.

All warm-starts use adaptive_mean5k; the future teacher is adaptive_future5k.
Frozen DINO/Wan are shared features, but VLM/Past QFormer differ once student learns.
"""
    reference_imports()
    from vla import load_vla
    from vla.qformer import QFormer
    from stage3.oracle_evidence import EvidenceProbe
    readers = STORE / "outputs/stage3_reader_study/runs_01"
    base = readers / f"adaptive_mean_seed{seed}/checkpoints/step-005000.pt"
    if not base.exists():
        raise ValueError("use a seed with a matched ReaderStudy initialization")
    cp = torch.load(base, map_location="cpu", weights_only=True)
    raw = load_vla(DEFAULT_PAST, train_venc=True, action_model_type="DiT-L", load_for_training=True)
    raw.freeze_backbones("full-finetune")
    future = QFormer(1024, 8, 2048, num_layers=6, num_query_tokens=16)
    cf = torch.load(DEFAULT_FUTURE, map_location="cpu", mmap=True, weights_only=True)
    future.load_state_dict(cf["model"]["qformer"], strict=True)
    del cf
    visual = FrozenVisualTargets(raw.v_enc, future)
    # Memory-saving fp32 mmap assignment is safe while unchanged; move/cast by
    # FSDP/device before optimization, never write into the source checkpoint.
    net = attach_adapters(raw.action_model.net, 32, enabled=True)
    stats = {
        "past": {"mean": torch.zeros_like(cp["state"]["probe"]["past_scale"]),
                 "std": cp["state"]["probe"]["past_scale"]},
        "future": {"mean": cp["state"]["probe"]["memory_mean"],
                   "std": cp["state"]["probe"]["memory_scale"]},
    }
    torch.manual_seed(seed)
    policy = JointPolicy(raw.vlm, raw.past_qformer, net, stats)
    kstate = {k.removeprefix("k."): v for k, v in cp["state"]["probe"].items() if k.startswith("k.")}
    policy.k.load_state_dict(kstate, strict=True)
    for block, saved in zip(net.blocks, cp["state"]["adapters"]):
        block.adapter.load_state_dict(saved, strict=True)
    # Frozen-specific flags overridden now: full original DiT and Past QFormer train.
    policy.net.requires_grad_(True)
    policy.vlm.requires_grad_(True)
    policy.past_qformer.requires_grad_(True)
    policy.vlm.vision_backbone_requires_grad = True
    transform = raw.vision_backbone.get_image_transform()
    tokenizer = raw.llm_backbone.get_tokenizer()
    prompt_builder = raw.vlm.get_prompt_builder
    oracle = None
    if teacher:
        fixed = load_vla(DEFAULT_PAST, train_venc=True, action_model_type="DiT-L", load_for_training=False)
        teacher_cp = torch.load(readers / f"adaptive_future_seed{seed}/checkpoints/step-005000.pt",
                                map_location="cpu", weights_only=True)
        probe = EvidenceProbe(RefinementConfig(), stats, stats["future"]["mean"], stats["future"]["std"])
        probe.load_state_dict(teacher_cp["state"]["probe"], strict=True)
        tn = attach_adapters(fixed.action_model.net, 32, enabled=False)
        for block, saved in zip(tn.blocks, teacher_cp["state"]["adapters"]):
            block.adapter.load_state_dict(saved, strict=True)
        oracle = FrozenOracle(fixed.vlm, fixed.past_qformer, tn, probe)
        # fixed.v_enc intentionally not retained; frozen visual features shared.
        del fixed, teacher_cp
    return policy, visual, oracle, transform, tokenizer, prompt_builder
