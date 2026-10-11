"""Memory-bounded local DCP inference loader; target tensors are private GPU memory."""

import gc
import json
import time

import torch
from torch import nn
import torch.distributed.checkpoint as dcp

from stage3.common import DEFAULT_PAST, reference_imports
from stage3_oxe_eval50.policy import Policy, CHECKPOINT, check_production_sources, param_bf16
from stage3_oxe_eval.prepare import sha


class DiagnosticPolicy(Policy):
    def __init__(self, device="cuda:0"):
        self.kind, self.device, self.checkpoint = "stage3", torch.device(device), CHECKPOINT
        reference_imports()
        from stage3_scale.model import load_joint
        began = time.monotonic()
        meta = json.loads((CHECKPOINT / "COMMITTED.json").read_text())
        if not meta["complete"] or meta["step"] != 50000:
            raise ValueError("fixed50k required")
        check_production_sources(meta["recipe"])
        policy, visual, _, transform, tokenizer, prompt = load_joint(seed=42, teacher=False)
        visual.future_qformer = nn.Identity()
        # Crucial: .to(device) after casting makes every destination independent
        # of mmap-loaded warm-start files. Inference never loads into source views.
        self.policy = param_bf16(policy.requires_grad_(False).eval()).to(self.device)
        self.vlm, self.net, self.past = self.policy.vlm, self.policy.net, self.policy.past_qformer
        self.visual = visual.requires_grad_(False).eval().to(self.device)
        self.transform, self.tokenizer, self.prompt = transform, tokenizer, prompt
        del policy, visual
        gc.collect()
        state = self.policy.state_dict()
        metadata = dcp.FileSystemReader(CHECKPOINT / "model_dcp").read_metadata()
        if set(metadata.state_dict_metadata) != {"model." + k for k in state}:
            raise ValueError("DCP key mismatch")
        for key, value in state.items():
            source = metadata.state_dict_metadata["model." + key]
            if value.device != self.device or source.size != value.shape:
                raise ValueError("DCP destination is not private accelerator storage: " + key)
            if source.properties.dtype != torch.float32 or value.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("unexpected source or destination dtype: " + key)
        print("MECHANISM_DCP_PRIVATE_DEVICE_LOAD", len(state), flush=True)
        dcp.load({"model": state}, checkpoint_id=CHECKPOINT / "model_dcp", no_dist=True)
        for key, value in state.items():
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite DCP tensor: " + key)
        self.vlm.vision_backbone_requires_grad = False
        self.statistics = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
        self.record = {
            "checkpoint": str(CHECKPOINT), "step": 50000,
            "manifest_sha256": sha(CHECKPOINT / "COMMITTED.json"),
            "strict_state_entries": len(state), "all_loaded_tensors_finite": True,
            "private_gpu_dcp_destinations": True,
            "statistics_sha256": sha(DEFAULT_PAST.parents[1] / "dataset_statistics.json"),
            "load_seconds": time.monotonic() - began,
            "loader_sha256": sha(__file__),
        }
        del state
        torch.cuda.synchronize(self.device)
        print("MECHANISM_POLICY_READY", self.record, flush=True)
