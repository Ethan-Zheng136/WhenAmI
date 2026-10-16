"""All DiT conditions originate from a single event node, never raw VLM z.

S_i^e has stable target ID i and evidence cutoff e. The feature-space node keeps
four cognition quarters across sixteen slots instead of squeezing all cognition
into the old 256-dimensional temporal hidden or 32-dimensional action condition.
"""
import math
import torch
from torch import nn

from whenami_timeline_a.model import Config
from whenami_unified.model import UnifiedUpdater, ExecutedAdapter, VARIANTS
from whenami_unified.contracts import Evidence, Executed
from stage3_scale.model import cognition


class NodeCognition(nn.Module):
    """Wide information enters the node; reader cognition is decoded from node ONLY."""
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.pieces = math.ceil(config.cognition / config.feature)
        if config.queries % self.pieces:
            raise ValueError("query count must support ordered cognition quarters")
        self.lift = nn.Linear(config.feature, config.feature, bias=False)
        self.readout = nn.Linear(config.cognition, config.cognition)
        nn.init.eye_(self.lift.weight)
        nn.init.eye_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def seed(self, local, z):
        c = self.config
        flat = z.squeeze(-2)
        flat = torch.nn.functional.pad(flat, (0, self.pieces*c.feature-c.cognition))
        pieces = flat.reshape(*flat.shape[:-1], self.pieces, c.feature)
        repeated = pieces.repeat_interleave(c.queries//self.pieces, -2)
        return local + self.lift(repeated.to(local.dtype))

    def forward(self, state):
        c = self.config
        grouped = state.reshape(*state.shape[:-2], self.pieces,
                                c.queries//self.pieces, c.feature).mean(-2)
        packed = grouped.flatten(-2)[..., :c.cognition]
        return self.readout(packed).unsqueeze(-2)


class TimelinePolicy(nn.Module):
    def __init__(self, vlm, local_encoder, reader, student=False, variant="revise", config=Config()):
        super().__init__()
        self.vlm, self.local_encoder, self.reader = vlm, local_encoder, reader
        self.timeline = UnifiedUpdater(config, variant)
        self.node_cognition = NodeCognition(config)
        self.student, self.config = student, config
        if student:
            self.executed_adapter = ExecutedAdapter(config.feature)
            self.forecast_head = nn.Linear(config.condition, config.condition)
            nn.init.eye_(self.forecast_head.weight)
            nn.init.zeros_(self.forecast_head.bias)
        self.requires_grad_(True)
        self.vlm.vision_backbone_requires_grad = True
        self.local_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.local_encoder.eval()
        return self

    def encode(self, evidence, executed=None, stale_current=False):
        if not isinstance(evidence, Evidence):
            raise TypeError("physically sliced Evidence required")
        evidence.validate()
        if self.student:
            if not isinstance(executed, Executed):
                raise TypeError("explicit executed intervals")
            executed.validate(evidence)
        elif executed is not None:
            raise ValueError("teacher is visual-only; no action inputs")
        hs, zs = [], []
        for j, frame in enumerate(evidence.frames):
            mask = evidence.visible[:, j]
            pixels = {key: torch.where(mask[:, None, None, None], x, 0.) for key, x in frame.pixels.items()}
            zj = cognition(self.vlm, frame.input_ids, frame.attention_mask, pixels)
            hj = self.local_encoder.encode_queries(
                torch.where(mask[:, None, None], frame.patches, 0.),
                torch.where(mask[:, None, None], frame.vae, 0.))
            hs.append(torch.where(mask[:, None, None], hj, 0.))
            zs.append(torch.where(mask[:, None, None], zj, 0.))
        h, raw_z = torch.stack(hs, 1), torch.stack(zs, 1)
        # Cognition enters BEFORE any temporal evidence exchange, not the reader.
        seeded = self.node_cognition.seed(h, raw_z)
        if executed is not None:
            seeded = seeded + self.executed_adapter(executed).to(seeded.dtype)
        output = self.timeline(seeded, raw_z, evidence.times, evidence.visible)
        if stale_current and len(evidence.frames) > 1:
            # Controlled SAME-input diagnostic, never used for normal training.
            j = len(evidence.frames)-1
            previous = output["versions"][-2]
            memory = torch.cat((previous, seeded[:, j:j+1]), 1)
            current = self.timeline.update(
                seeded[:, j:j+1], memory, raw_z, evidence.times[:, j:j+1],
                evidence.times, evidence.times[:, j], evidence.visible)
            versions = list(output["versions"])
            versions[-1] = torch.cat((versions[-1][:, :-1], current), 1)
            output["versions"] = tuple(versions)
        conditions = tuple(self.local_encoder.v_proj(v.to(h.dtype)) for v in output["versions"])
        node_z = tuple(self.node_cognition(v.to(h.dtype)) for v in output["versions"])
        output.update(node_versions=conditions, node_z_versions=node_z,
                      conditions=conditions[-1][:, -1], current_z=node_z[-1][:, -1],
                      raw_z=raw_z, seeded=seeded)
        if self.student:
            output["projected_versions"] = tuple(self.forecast_head(c) for c in conditions)
            output["forecast"] = output["projected_versions"][-1][:, -1]
        return output

    def forward(self, evidence=None, executed=None, conditioning=None, noisy_actions=None, tau=None,
                requests=None, condition_only=False, stale_current=False):
        if conditioning is not None:
            if evidence is not None or executed is not None or condition_only or requests is not None:
                raise ValueError("decoder-only call")
            return {"velocity": self.reader(noisy_actions, tau, conditioning["reader_z"],
                                             per_token=conditioning["conditions"])}
        result = self.encode(evidence, executed, stale_current)
        if condition_only:
            return result
        k = len(evidence.frames)
        requests = ((k-1, k-1),) if requests is None else tuple(requests)
        if not requests or any(not 0 <= i <= e < k for i, e in requests):
            raise ValueError("requests are (target_event_index,evidence_version_index)")
        if self.student and any(i != e for i, e in requests):
            raise ValueError("student FM predicts only legal prefix-current, not retrospective executed labels")
        conditions = torch.stack([result["node_versions"][e][:, i] for i, e in requests], 1).flatten(0, 1)
        reader_z = torch.stack([result["node_z_versions"][e][:, i] for i, e in requests], 1).flatten(0, 1)
        b = len(conditions)
        if noisy_actions is None or tau is None or len(noisy_actions) % b:
            raise ValueError("action draw ordering B×requests×draws")
        draws = len(noisy_actions)//b
        result["velocity"] = self.reader(noisy_actions, tau, reader_z.repeat_interleave(draws, 0),
                                         per_token=conditions.repeat_interleave(draws, 0))
        return result


def build_policy(student=False, variant="revise", seed=42):
    from whenami_timeline_joint.model import load_joint
    old, visual, transform, tokenizer, prompt = load_joint("timeline", seed)
    torch.manual_seed(seed)
    model = TimelinePolicy(old.vlm, old.local_encoder, old.reader, student, variant)
    model.timeline.load_state_dict(old.timeline.state_dict(), strict=True)
    return model, visual, transform, tokenizer, prompt
