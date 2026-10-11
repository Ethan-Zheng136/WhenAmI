"""Cold-build B for inference without an A/Stage1/base-weight checkpoint.

Does not implement robot action postprocessing, control timing, or a SIMPLER
wrapper. Those must match the receiving server's pinned evaluation protocol.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image
import torch

from stage3.common import reference_imports
from stage3_scale.model import tokenize_instruction
from whenami_timeline_b.model import Student
from whenami_timeline_joint.data import frozen_features
from whenami_timeline_transfer.contracts import Prefix


SCHEMA = "whenami_timeline_b_student_inference_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def build(config, device="cuda:0"):
    """All body weights remain unusable until a strict complete load follows."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("this inference helper requires a CUDA/PPU accelerator")
    # WanVAE is a plain Python object, not a registered child nn.Module.
    # Its internal model/scale tensors initialize on the current CUDA device.
    torch.cuda.set_device(device)
    reference_imports()
    from prismatic.conf import ModelConfig
    from prismatic.models.materialize import (
        get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform,
    )
    from prismatic.models.vlms import PrismaticVLM
    from prismatic.models.backbones.vision.dinosiglip_vit import FutureEnc
    from vla.qformer import QFormer
    from action_model.action_model import ActionModel

    cfg = ModelConfig.get_choice_class(config["vla"]["base_vlm"])()
    vision, transform = get_vision_backbone_and_transform(
        cfg.vision_backbone_id, cfg.image_resize_strategy)
    llm, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id, llm_max_length=cfg.llm_max_length, inference_mode=True)
    vlm = PrismaticVLM(cfg.model_id, vision, llm, arch_specifier=cfg.arch_specifier)
    local = QFormer(hidden_dim=1024, nheads=8, dim_feedforward=2048,
                    num_layers=6, num_query_tokens=16)
    reader = ActionModel(token_size=4096, model_type="DiT-L", in_channels=7,
                         Ta=16, train_venc=True).net
    student = Student(vlm, local, reader, mode="timeline").requires_grad_(False).eval()
    # Preserve FP32 buffers; match the training mixed-precision parameter policy.
    for parameter in student.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    student = student.to(device)
    student.vlm.vision_backbone_requires_grad = False
    visual = FutureEnc().requires_grad_(False).eval().to(device)
    return student, visual, transform, tokenizer, student.vlm.get_prompt_builder


def load_dcp(student, checkpoint):
    """Private accelerator destinations: never mutate mmap-backed training files."""
    import torch.distributed.checkpoint as dcp
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "COMMITTED.json").read_text())
    if not meta.get("complete") or meta["recipe"]["settings"]["variant"] not in (
        "action_only", "condition_align"
    ):
        raise ValueError("not a committed B checkpoint")
    state = student.state_dict()
    if any(t.device.type != "cuda" for t in state.values()):
        raise ValueError("load_dcp requires independent accelerator destinations")
    source = dcp.FileSystemReader(checkpoint / "model_dcp").read_metadata().state_dict_metadata
    if set(source) != {"model." + k for k in state}:
        raise ValueError("B architecture/DCP keys do not match exactly")
    for k, t in state.items():
        if tuple(source["model." + k].size) != tuple(t.shape):
            raise ValueError("DCP shape mismatch: " + k)
    dcp.load({"model": state}, checkpoint_id=checkpoint / "model_dcp", no_dist=True)
    for k, t in state.items():
        if not torch.isfinite(t).all():
            raise ValueError("nonfinite checkpoint tensor: " + k)
    return meta


def load_release(student, release):
    from safetensors.torch import load_file
    release = Path(release)
    meta = json.loads((release / "MODEL.json").read_text())
    if meta["schema"] != SCHEMA or not meta["complete"]:
        raise ValueError("incomplete inference release")
    expected = student.state_dict()
    seen = set()
    for item in meta["shards"]:
        path = release / item["file"]
        if sha(path) != item["sha256"]:
            raise ValueError("weight checksum mismatch")
        data = load_file(path, device="cpu")
        if set(data) != set(item["keys"]) or seen.intersection(data):
            raise ValueError("duplicate/missing shard keys")
        for k, tensor in data.items():
            if k not in expected or tensor.shape != expected[k].shape or tensor.dtype != expected[k].dtype:
                raise ValueError("inference key/shape/dtype mismatch: " + k)
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite inference tensor: " + k)
        student.load_state_dict(data, strict=False)
        seen.update(data)
        del data
    if seen != set(expected) or len(seen) != meta["state_entries"]:
        raise ValueError("incomplete full-model load")
    return meta


@torch.no_grad()
def prefix_inputs(images, instructions, times, visual, transform, tokenizer, prompt, device):
    """Accept OBSERVED frames only. No future frame or action-label argument."""
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("RGB uint8 [observed_nodes,H,W,3] required")
    k = len(images)
    if k not in (1, 2) or len(instructions) != k:
        raise ValueError("B expects history+current; one node is an explicit cold-start extension")
    if len(times) != k or any(int(x) != x for x in times):
        raise ValueError("integer observation indices required")
    # Match RawAnchors: RGB -> PIL resize(224,224) -> checkpoint image transform.
    frames = [transform(Image.fromarray(image).convert("RGB").resize((224, 224))) for image in images]
    pixels = {key: torch.stack([frame[key] for frame in frames])[None].to(device)
              for key in frames[0]}
    ids = [tokenize_instruction(s, tokenizer, prompt) for s in instructions]
    width = max(len(x) for x in ids)
    input_ids = torch.full((k, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((k, width), dtype=torch.bool, device=device)
    for i, value in enumerate(ids):
        input_ids[i, :len(value)] = value.to(device)
        attention[i, :len(value)] = True
    times = torch.tensor([times], dtype=torch.long, device=device)
    times = times - times[:, :1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features = frozen_features(visual, pixels)
    return Prefix(pixels, input_ids, attention, features["patches"],
                  features["vae"], times).validate()


@torch.no_grad()
def sample(student, prefix, initial_noise):
    """Training-DEV sampler: FP32 x/t, BF16 compute, 10 times / 9 Euler updates."""
    device = next(student.parameters()).device
    x = torch.as_tensor(initial_noise, dtype=torch.float32, device=device).clone()
    if x.shape != (1, 16, 7) or not torch.isfinite(x).all():
        raise ValueError("explicit finite FP32 Gaussian initial noise [1,16,7] required")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = student(prefix=prefix, condition_only=True)
        conditioning = {k: encoded[k] for k in ("conditions", "reader_z")}
        times = torch.linspace(0, 1, 10, dtype=torch.float32, device=device)
        first_velocity = None
        for i in range(9):
            v = student(noisy_actions=x, tau=times[i].expand(1),
                        conditioning=conditioning)["velocity"]
            if first_velocity is None:
                first_velocity = v.float().cpu().numpy()
            x = x + v.float() * (times[i + 1] - times[i])
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite sampled action")
    return {
        "conditions": conditioning["conditions"].float().cpu().numpy(),
        "reader_z": conditioning["reader_z"].float().cpu().numpy(),
        "first_velocity": first_velocity,
        "normalized_action_unclipped": x.cpu().numpy(),
    }
