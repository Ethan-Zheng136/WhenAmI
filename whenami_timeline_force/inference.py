"""New student-only deployment loader; no changes to external NVIDIA releases.

Input commands are already OXE-standardized+normalized. They must be actual
executed commands, never previous prediction's unexecuted suffix. The caller owns
robot-specific inverse/forward command transforms and observation timestamps.
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from stage3.common import DEFAULT_FUTURE
from stage3_oxe_eval50.policy import param_bf16
from stage3_scale.model import tokenize_instruction
from whenami_unified.contracts import Frame, Evidence, Executed
from whenami_timeline_joint.data import frozen_features
from whenami_timeline_force.loading import load_weights, SCHEMA
from whenami_timeline_force.model import build_policy


class Policy:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        # Read strict source hashes, but this must be a STUDENT not teacher.
        from whenami_unified.loading import checkpoint_record
        record = checkpoint_record(self.checkpoint)
        recipe = record["recipe"]
        if recipe.get("schema") != SCHEMA or recipe["settings"]["stage"] != "student":
            raise ValueError("deployable student checkpoint required")
        model, visual, transform, tokenizer, prompt = build_policy(True, recipe["settings"]["variant"])
        self.model = param_bf16(model).to(self.device).requires_grad_(False).eval()
        self.model.vlm.vision_backbone_requires_grad = False
        self.visual = visual.to(self.device).requires_grad_(False).eval()
        loaded = load_weights(self.model, self.checkpoint)
        if loaded["new_entries"]:
            raise ValueError("inference requires complete student weights")
        self.transform, self.tokenizer, self.prompt = transform, tokenizer, prompt
        self.statistics = json.loads((DEFAULT_FUTURE.parents[1]/"dataset_statistics.json").read_text())
        self.record = {k: record[k] for k in ("path", "step", "sha256")}
        self.record.update(loaded=loaded, interface="finite3node_rebuild_same_as_train", ensemble=False)

    @torch.no_grad()
    def predict(self, images, instruction, times, normalized_executed, command_masks, seed=42):
        """Images chronological≤3; intervalj equals priorframe->framej commands."""
        if not 1 <= len(images) <= 3:
            raise ValueError("one to three physical observations")
        if len(times) != len(images):
            raise ValueError("one immutable event ID per frame")
        frames = []
        ids = tokenize_instruction(instruction, self.tokenizer, self.prompt).to(self.device)[None]
        for image in images:
            image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pixels = {key: x[None].to(self.device) for key, x in self.transform(image.convert("RGB")).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = frozen_features(self.visual, {key: x[:, None] for key, x in pixels.items()})
            frames.append(Frame(pixels, ids, torch.ones_like(ids, dtype=torch.bool),
                                features["patches"][:, 0], features["vae"][:, 0]))
        ev = Evidence(tuple(frames), torch.as_tensor(times, dtype=torch.long, device=self.device)[None],
                      torch.ones(1, len(frames), dtype=torch.bool, device=self.device)).validate()
        commands = Executed(torch.as_tensor(normalized_executed, device=self.device, dtype=torch.float32)[None],
                            torch.as_tensor(command_masks, device=self.device, dtype=torch.bool)[None]).validate(ev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = self.model(evidence=ev, executed=commands, condition_only=True)
        x = torch.randn((1, 16, 7), device=self.device,
                        generator=torch.Generator(device=self.device).manual_seed(seed))
        ts = torch.linspace(0, 1, 10, device=self.device)
        for i in range(9):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self.model(conditioning={"conditions": enc["conditions"], "reader_z": enc["current_z"]},
                               noisy_actions=x, tau=ts[i].expand(1))["velocity"]
            x = x+v.float()*(ts[i+1]-ts[i])
        if not torch.isfinite(x).all():
            raise FloatingPointError("nonfinite policy action")
        return x[0].cpu().numpy()

    def denormalize(self, prediction, dataset):
        stats = self.statistics[dataset]["action"]
        value = np.asarray(prediction).copy()
        if value.shape != (16, 7):
            raise ValueError("action chunk shape")
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        mask = np.asarray(stats.get("mask", [True]*6+[False]), dtype=bool)
        value = np.where(mask, .5*(np.clip(value, -1, 1)+1)*(high-low)+low, value)
        unused = np.asarray(stats["min"]) == np.asarray(stats["max"])
        return np.where(unused, 0., value).astype(np.float32)
