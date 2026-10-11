"""Same A parameter space, but query times are separate from observed sources.

No future image is synthesized or passed to the student. A future node is a
prediction from a causal prefix. It is not independent evidence or a dynamics
rollout. C-aux/C-read differ only in whether action reads this predicted node.
"""
import torch
from torch import nn
import torch.nn.functional as F

from whenami_timeline_a.model import Config
from whenami_timeline_joint.model import AddressedConditioner, JointTimeline
from whenami_timeline_transfer.contracts import Prefix
from stage3_scale.model import cognition

VARIANTS = ("future_aux", "future_read")


class TimeQueryConditioner(AddressedConditioner):
    def query(self, h, z, source_times, query_times, visible, cutoffs):
        c = self.config
        b, k = source_times.shape
        n = query_times.shape[1]
        if h.shape != (b, k, c.queries, c.feature) or z.shape != (b, k, 1, c.cognition):
            raise ValueError("one local encoding per observed source")
        if query_times.shape[0] != b or visible.shape != source_times.shape or cutoffs.shape != query_times.shape:
            raise ValueError("source/query/cutoff shapes mismatch")
        if visible.dtype != torch.bool or any(t.dtype not in (torch.int32, torch.int64)
                                               for t in (source_times, query_times, cutoffs)):
            raise ValueError("explicit integer times and boolean source visibility required")
        if (k > 1 and (source_times.diff(dim=1) <= 0).any()) or n < 1:
            raise ValueError("chronological observed sources and nonempty queries required")
        allowed = visible[:, None, :] & (source_times[:, None, :] <= cutoffs[:, :, None])
        matches = (query_times[:, :, None] == source_times[:, None, :]) & allowed
        own = matches.any(-1)
        selected = matches.long().argmax(-1)
        hb = h.gather(1, selected[:, :, None, None].expand(-1, -1, c.queries, c.feature))
        zb = z.gather(1, selected[:, :, None, None].expand(-1, -1, 1, c.cognition))
        base = torch.where(own[:, :, None, None], hb, self.missing_node[:, None])
        safe_h = torch.where(visible[:, :, None, None], h, torch.zeros_like(h))
        safe_z = torch.where(visible[:, :, None, None], z, torch.zeros_like(z))
        source = self.input(safe_h) + self.cognition_input(safe_z) + self.query_position[:, None]
        relative = source_times[:, None, :] - query_times[:, :, None]
        memory = source[:, None] + self.encode_time(relative)[:, :, :, None]
        if self.mode == "pooled_query":
            weight = allowed.to(memory.dtype)[:, :, :, None, None]
            count = allowed.sum(-1).clamp_min(1).to(memory.dtype)[:, :, None, None]
            pooled = (memory * weight).sum(2) / count
            memory = pooled[:, :, None].expand(-1, -1, k, -1, -1)
        query = (self.input(base) + self.query_position[:, None]
                 + self.encode_time(query_times - source_times[:, :1])[:, :, None])
        memory = memory.reshape(b * n, k * c.queries, c.hidden)
        memory = torch.cat((memory, self.null_memory.expand(b * n, -1, -1)), 1)
        prohibited = ~allowed.repeat_interleave(c.queries, -1)
        prohibited = torch.cat((prohibited.reshape(b * n, -1), allowed.any(-1).reshape(b * n, 1)), 1)
        query = query.reshape(b * n, c.queries, c.hidden)
        for block in self.blocks:
            query = block(query, memory, prohibited)
        delta = self.output(query).reshape(b, n, c.queries, c.feature)
        return {
            "state": base + delta,
            "reader_z": torch.where(own[:, :, None, None], zb, self.missing_cognition[:, None]),
            "allowed_sources": allowed, "target_observed": own, "residual": delta,
        }

    def forward(self, h, z, times, visible, cutoffs=None):
        cutoffs = times[:, -1:].expand_as(times) if cutoffs is None else cutoffs
        return self.query(h, z, times, times, visible, cutoffs)


class TimelineStudent(JointTimeline):
    def __init__(self, vlm, local_encoder, reader, variant="future_aux", future_gap=4, config=Config()):
        if variant not in VARIANTS or future_gap < 1:
            raise ValueError("C variant/future horizon")
        super().__init__(vlm, local_encoder, reader, "timeline", config)
        self.timeline = TimeQueryConditioner(config, "timeline")
        self.variant, self.future_gap = variant, int(future_gap)
        self.register_buffer("read_scale", torch.zeros((), dtype=torch.float32))

    def set_step(self, step, read_warmup=1000):
        if step < 0 or read_warmup < 1:
            raise ValueError("invalid read-in schedule")
        value = min(1., step / read_warmup) if self.variant == "future_read" else 0.
        self.read_scale.fill_(value)

    def forward(self, prefix=None, noisy_actions=None, tau=None,
                condition_only=False, conditioning=None):
        if conditioning is not None:
            if prefix is not None or condition_only:
                raise ValueError("decoder-only call")
            return {"velocity": self.reader(noisy_actions, tau, conditioning["reader_z"],
                                             per_token=conditioning["conditions"])}
        if not isinstance(prefix, Prefix):
            raise TypeError("C student accepts only a physical observed prefix")
        prefix.validate()
        b, k = prefix.times.shape
        if k not in (1, 2):
            raise ValueError("one history+current; K=1 is an explicit cold-start extension")
        pixels = {key: value.flatten(0, 1) for key, value in prefix.pixels.items()}
        z = cognition(self.vlm, prefix.input_ids, prefix.attention_mask, pixels).reshape(b, k, 1, -1)
        h = self.local_encoder.encode_queries(prefix.patches.flatten(0, 1), prefix.vae.flatten(0, 1))
        h = h.reshape(b, k, *h.shape[1:])
        now = prefix.times[:, -1:]
        query_times = torch.cat((now, now + self.future_gap), 1)
        encoded = self.timeline.query(
            h, z, prefix.times, query_times, torch.ones_like(prefix.times, dtype=torch.bool),
            now.expand_as(query_times),
        )
        nodes = self.local_encoder.v_proj(encoded["state"].to(h.dtype))
        current, future = nodes[:, 0], nodes[:, 1]
        # Both arms have 32 condition tokens and identical compute shapes.
        # At scale=0 this is duplicated current context, mathematically equivalent
        # to the original 16-token attention (numerical tolerance is tested).
        extra = current + self.read_scale.to(current.dtype) * (future - current)
        action_condition = torch.cat((current, extra), 1)
        result = {
            "conditions": action_condition,
            "reader_z": encoded["reader_z"][:, 0],
            "node_conditions": nodes, "current_condition": current,
            "future_condition": future, "query_times": query_times,
            "target_observed": encoded["target_observed"],
        }
        if condition_only:
            return result
        if noisy_actions is None or tau is None or len(noisy_actions) % b:
            raise ValueError("current-action decoder states required")
        draws = len(noisy_actions) // b
        result["velocity"] = self.reader(
            noisy_actions, tau, result["reader_z"].repeat_interleave(draws, 0),
            per_token=action_condition.repeat_interleave(draws, 0))
        return result


def objective(output, flow_target, valid, teacher_nodes, weight=.1, draws=2):
    if weight < 0 or valid.ndim != 2 or valid.shape[-1] != 16 or not valid.any(-1).all():
        raise ValueError("loss configuration")
    if teacher_nodes.shape != output["node_conditions"].shape:
        raise ValueError("teacher must provide current AND future node conditions")
    mask = valid.detach().repeat_interleave(draws, 0)
    error = (output["velocity"].float() - flow_target.detach().float()).square().mean(-1)
    action = ((error * mask).sum(-1) / mask.sum(-1)).mean()
    distances = 1 - F.cosine_similarity(output["node_conditions"].float(),
                                         teacher_nodes.detach().float(), dim=-1, eps=1e-6)
    current, future = distances[:, 0].mean(), distances[:, 1].mean()
    align = (current + future) / 2
    total = action + weight * align
    return total, {"action_fm": action.detach(), "current_alignment": current.detach(),
                   "future_alignment": future.detach(), "weighted_alignment": weight * align.detach(),
                   "total": total.detach()}
