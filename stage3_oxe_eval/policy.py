"""Strict, read-only inference loaders for the three existing checkpoints.

This is a local research evaluator, not the separately owned NVIDIA release.
No optimizer, RNG shard, future observation, or action label is loaded for inference.
"""

import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

from stage3.common import BASE, STORE, DEFAULT_PAST, REFERENCE, reference_imports
from stage3_oxe_eval.prepare import sha

CHECKPOINT = STORE / "outputs/stage3_joint/oxe27_forecast_zero2_320_50k_save5k_01/checkpoints/step-035000"
WOG = STORE / "ckpts/WoG-A/checkpoints/50k-1229.pt"
KINDS = ("wog_a", "past", "stage3")


def param_bf16(module):
    # Match FSDP param_dtype BF16, buffer_dtype FP32. Do not cast scale buffers.
    for p in module.parameters():
        p.data = p.data.to(dtype=torch.bfloat16)
    return module


def check_production_sources(recipe):
    for field, root in (("reference_source", REFERENCE), ("joint_source", BASE / "stage3_scale"),
                        ("performance_source", BASE / "stage3_perf"), ("reused_source", BASE)):
        for relative, expected in recipe[field].items():
            if sha(root / relative) != expected:
                raise ValueError(f"checkpoint source contract drift: {field}/{relative}")


class Policy:
    def __init__(self, kind, device="cuda:0", checkpoint=CHECKPOINT):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.device = kind, torch.device(device)
        reference_imports()
        from stage3_scale.model import load_joint, FrozenVisualTargets
        from vla import load_vla
        started = time.monotonic()
        self.checkpoint = Path(checkpoint if kind == "stage3" else (WOG if kind == "wog_a" else DEFAULT_PAST))
        self.record = {"kind": kind, "checkpoint": str(self.checkpoint),
                       "inference_source_sha256": sha(__file__)}
        if kind == "stage3":
            import torch.distributed.checkpoint as dcp
            meta = json.loads((self.checkpoint / "COMMITTED.json").read_text())
            if not meta["complete"] or meta["step"] != 35000:
                raise ValueError("this protocol pins committed35k, not an automatically selected checkpoint")
            check_production_sources(meta["recipe"])
            policy, visual, _, transform, tokenizer, prompt = load_joint(seed=42, teacher=False)
            # Do not load into mmap-backed warm-start tensors. Reconstruct an
            # independent CPU state dict, then strictly replace every live tensor.
            print("EVAL_DCP_LOADING", str(self.checkpoint), flush=True)
            state = {"model": {
                name: torch.empty(value.shape, dtype=value.dtype, device="cpu")
                for name, value in policy.state_dict().items()
            }}
            metadata = dcp.FileSystemReader(self.checkpoint / "model_dcp").read_metadata()
            if set(metadata.state_dict_metadata) != {"model." + name for name in state["model"]}:
                raise ValueError("DCP metadata and inference architecture keys differ")
            for name, value in state["model"].items():
                source = metadata.state_dict_metadata["model." + name]
                if source.size != value.shape or source.properties.dtype != value.dtype:
                    raise ValueError("DCP inference shape/dtype mismatch: " + name)
            dcp.load(state, checkpoint_id=self.checkpoint / "model_dcp", no_dist=True)
            result = policy.load_state_dict(state["model"], strict=True, assign=True)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError("incomplete Stage3 inference state")
            # Full finite check once on CPU before conversion; preserve original weights.
            for name, value in state["model"].items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError("nonfinite checkpoint tensor: " + name)
            self.record.update(
                step=meta["step"], manifest_sha256=sha(self.checkpoint / "COMMITTED.json"),
                strict_keys=len(state["model"]), all_loaded_tensors_finite=True,
            )
            del state
            visual.future_qformer = nn.Identity()
            self.policy = param_bf16(policy.requires_grad_(False).eval()).to(self.device)
            self.vlm, self.net, self.past = self.policy.vlm, self.policy.net, self.policy.past_qformer
            self.visual = visual.requires_grad_(False).eval().to(self.device)
            self.transform, self.tokenizer, self.prompt = transform, tokenizer, prompt
            stats_path = DEFAULT_PAST.parents[1] / "dataset_statistics.json"
        else:
            raw = load_vla(self.checkpoint, train_venc=kind == "past",
                           action_model_type="DiT-L", load_for_training=False)
            saved = torch.load(self.checkpoint, map_location="cpu", mmap=True, weights_only=True)
            # The legacy loader permits some unused-state mismatches; ALL modules
            # actually used in our inference are independently checked strictly.
            for group in ("vision_backbone", "projector", "llm_backbone"):
                getattr(raw.vlm, group).load_state_dict(saved["model"][group], strict=True, assign=True)
            raw.action_model.load_state_dict(saved["model"]["action_model"], strict=True, assign=True)
            if kind == "past":
                raw.past_qformer.load_state_dict(saved["model"]["past_qformer"], strict=True, assign=True)
                if raw.whenami_stage != 2 or raw.obs_offsets != [-15, -11, -7, -3, 0]:
                    raise ValueError("wrong Past contract")
            elif raw.action_model.net.use_per_attn:
                raise ValueError("WoG-A must be current-only, not Future oracle")
            del saved
            self.transform = raw.vision_backbone.get_image_transform()
            self.tokenizer = raw.llm_backbone.get_tokenizer()
            self.prompt = raw.vlm.get_prompt_builder
            self.vlm = param_bf16(raw.vlm.requires_grad_(False).eval()).to(self.device)
            self.net = param_bf16(raw.action_model.net.requires_grad_(False).eval()).to(self.device)
            self.past = None
            self.visual = None
            if kind == "past":
                self.past = param_bf16(raw.past_qformer.requires_grad_(False).eval()).to(self.device)
                self.visual = FrozenVisualTargets(raw.v_enc, nn.Identity()).eval().to(self.device)
                self.reference_past = raw
            self.record.update(
                checkpoint_bytes=self.checkpoint.stat().st_size,
                checkpoint_mtime_ns=self.checkpoint.stat().st_mtime_ns,
                config_sha256=sha(self.checkpoint.parents[1] / "config.json"),
                strict_live_modules=True,
            )
            stats_path = self.checkpoint.parents[1] / "dataset_statistics.json"
            del raw
        self.vlm.vision_backbone_requires_grad = False
        self.statistics = json.loads(stats_path.read_text())
        self.record["statistics_sha256"] = sha(stats_path)
        self.record["load_seconds"] = time.monotonic() - started
        self.record["precision"] = "BF16 live policy parameters, FP32 buffers; frozen historical encoder unchanged"
        gc.collect()
        torch.cuda.synchronize(self.device)
        print("EVAL_POLICY_READY", self.record, flush=True)

    def inputs(self, current, history, instruction):
        from stage3_scale.model import tokenize_instruction
        if current.shape != (224, 224, 3) or current.dtype != np.uint8:
            raise ValueError("expected preprocessed RGB uint8 image")
        now = {k: x[None].to(self.device) for k, x in self.transform(Image.fromarray(current)).items()}
        ids = tokenize_instruction(instruction, self.tokenizer, self.prompt)[None].to(self.device)
        hist = None
        if self.kind != "wog_a":
            if history.shape != (4, 224, 224, 3):
                raise ValueError("actual causal history required")
            values = [self.transform(Image.fromarray(image)) for image in history]
            hist = {k: torch.stack([x[k] for x in values])[None].to(self.device) for k in now}
        return now, hist, ids

    @torch.no_grad()
    def encode(self, current, history, instruction):
        """No targets, future images, or action prefixes accepted."""
        from stage3_scale.model import cognition
        now, hist, ids = self.inputs(current, history, instruction)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "stage3":
                features = self.visual(now, hist)
                return self.policy(
                    input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                    current=now, history_features=features["history_features"],
                    history_vae=features["history_vae"], condition_only=True,
                )
            z = cognition(self.vlm, ids, torch.ones_like(ids, dtype=torch.bool), now)
            condition = None
            if self.kind == "past":
                features = self.visual(now, hist)
                h = self.past.encode_queries(features["history_features"], features["history_vae"])
                condition = self.past.v_proj(h)
            return {"z": z, "condition": condition}

    @torch.no_grad()
    def sample(self, encoded, initial_noise):
        if initial_noise.ndim != 3 or initial_noise.shape[1:] != (16, 7):
            raise ValueError("noise shape must be [N,16,7]")
        if initial_noise.dtype != torch.float32:
            raise ValueError("pure FP32 Gaussian initial state required")
        x = initial_noise.to(self.device).clone()
        z = encoded["z"].expand(len(x), -1, -1)
        c = encoded["condition"]
        c = None if c is None else c.expand(len(x), -1, -1)
        ts = torch.linspace(0, 1, 10, device=self.device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for j in range(9):
                velocity = self.net(x, ts[j].expand(len(x)), z, per_token=c)
                x = x + velocity.float() * (ts[j + 1] - ts[j])
        if not torch.isfinite(x).all():
            raise FloatingPointError("nonfinite generated actions")
        return x.cpu().numpy()

    @torch.no_grad()
    def check_cognition(self, current, history, instruction):
        """Check direct-forward cognition against the original generate interface."""
        from prismatic.models.vlms import PrismaticVLM
        from stage3_scale.model import cognition
        now, _, ids = self.inputs(current, history, instruction)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            direct = cognition(self.vlm, ids, torch.ones_like(ids, dtype=torch.bool), now)
            generated = super(PrismaticVLM, self.vlm).generate(
                input_ids=ids, pixel_values=now, max_new_tokens=1, do_sample=False,
                output_hidden_states=True, return_dict_in_generate=True,
            ).hidden_states[0][-1][:, -1, :][:, None]
        error = (direct.float() - generated.float()).abs()
        result = {"max_abs": error.max().item(), "mean_abs": error.mean().item()}
        if not torch.allclose(direct.float(), generated.float(), rtol=.005, atol=.005):
            raise ValueError("direct cognition differs from original generation: " + str(result))
        return result

    @torch.no_grad()
    def check_past_condition(self, current, history, instruction):
        if self.kind != "past":
            raise ValueError("Past-only validation")
        direct = self.encode(current, history, instruction)["condition"]
        now, _, _ = self.inputs(current, history, instruction)
        frames = [Image.fromarray(x) for x in history] + [Image.fromarray(current)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            original = self.reference_past._condition_from_frames(frames, None, now, torch.bfloat16)
        error = (direct.float() - original.float()).abs()
        result = {"max_abs": error.max().item(), "mean_abs": error.mean().item()}
        if not torch.allclose(direct.float(), original.float(), rtol=.005, atol=.005):
            raise ValueError("Past condition differs from original deployment: " + str(result))
        return result
