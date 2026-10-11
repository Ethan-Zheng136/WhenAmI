"""Explicit time nodes with matched pooled/independent/cross-node controls."""

from dataclasses import dataclass
import math

import torch
from torch import nn

MODES = ("pooled", "independent", "timeline")


@dataclass(frozen=True)
class Config:
    feature: int = 1024
    cognition: int = 4096
    queries: int = 16
    hidden: int = 256
    heads: int = 8
    layers: int = 2
    condition: int = 32

    def __post_init__(self):
        if min(self.feature, self.cognition, self.queries, self.hidden,
               self.heads, self.layers, self.condition) < 1 or self.hidden % self.heads:
            raise ValueError("invalid model dimensions")


class ReadEvidence(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden
        self.q_norm, self.m_norm = nn.LayerNorm(h), nn.LayerNorm(h)
        self.attention = nn.MultiheadAttention(h, config.heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Linear(4*h, h))

    def forward(self, query, memory, prohibited):
        # True means NOT visible. No cross-target query attention exists.
        msg, _ = self.attention(self.q_norm(query), self.m_norm(memory),
                                self.m_norm(memory), key_padding_mask=prohibited,
                                need_weights=False)
        query = query + msg
        return query + self.ffn(self.norm(query))


class TimelineConditioner(nn.Module):
    def __init__(self, config=Config(), mode="timeline"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.config, self.mode = config, mode
        c = config
        self.input = nn.Sequential(nn.LayerNorm(c.feature), nn.Linear(c.feature, c.hidden))
        self.cognition_input = nn.Sequential(nn.LayerNorm(c.cognition), nn.Linear(c.cognition, c.hidden))
        self.time = nn.Sequential(nn.Linear(6, c.hidden), nn.Tanh(), nn.Linear(c.hidden, c.hidden))
        self.query_position = nn.Parameter(torch.randn(1, c.queries, c.hidden) * .02)
        self.missing_node = nn.Parameter(torch.zeros(1, c.queries, c.feature))
        self.missing_cognition = nn.Parameter(torch.zeros(1, 1, c.cognition))
        self.null_memory = nn.Parameter(torch.zeros(1, 1, c.hidden))
        self.blocks = nn.ModuleList([ReadEvidence(c) for _ in range(c.layers)])
        self.output = nn.Sequential(nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.feature))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def encode_time(self, delta):
        x = delta.float()
        value = torch.stack((x/32, torch.sign(x)*torch.log1p(x.abs())/4,
                             torch.sin(x/4), torch.cos(x/4), torch.sin(x/16), torch.cos(x/16)), -1)
        return self.time(value)

    def forward(self, h, z, times, visible, cutoffs=None):
        c = self.config
        if h.ndim != 4 or h.shape[2:] != (c.queries, c.feature):
            raise ValueError("h must have shape [B,K,query,feature]")
        b, k = h.shape[:2]
        if z.shape != (b,k,1,c.cognition) or times.shape != (b,k) or visible.shape != (b,k):
            raise ValueError("misaligned anchor tensors")
        if visible.dtype != torch.bool or times.dtype not in (torch.int32, torch.int64):
            raise ValueError("boolean visibility and integer event times required")
        if k > 1 and (times.diff(dim=1) <= 0).any():
            raise ValueError("anchors must be distinct and chronological")
        if cutoffs is None:
            cutoffs = times[:, -1:].expand(-1,k)  # explicit Phase-A full evidence
        if cutoffs.shape != times.shape:
            raise ValueError("one evidence cutoff per target node")
        # source j is accessible for target i iff it is visible and at/before cutoff_i.
        allowed = visible[:, None, :] & (times[:, None, :] <= cutoffs[:, :, None])
        own_visible = visible & (times <= cutoffs)
        safe_h = torch.where(visible[:,:,None,None], h, torch.zeros_like(h))
        safe_z = torch.where(visible[:,:,None,None], z, torch.zeros_like(z))
        source = self.input(safe_h.float()) + self.cognition_input(safe_z.float()) + self.query_position[:,None]

        # Different cutoffs may produce different admissible pooled contexts.
        count = allowed.sum(-1).clamp_min(1).to(h.dtype)
        if self.mode == "pooled":
            pool_h = torch.einsum("bij,bjqf->biqf", allowed.to(h.dtype), safe_h) / count[:,:,None,None]
            base = torch.where(allowed.any(-1)[:,:,None,None], pool_h, self.missing_node[:,None])
            origin = times[:, :1]
            addressed = source + self.encode_time(times-origin)[:,:,None]
            pooled = torch.einsum("bij,bjqf->biqf", allowed.to(addressed.dtype), addressed) / count[:,:,None,None]
            # Duplicate pooled bank to retain the exact same attention sequence
            # lengths/computation shape as the other arms. Repetition adds no info.
            memory = pooled[:,:,None].expand(-1,-1,k,-1,-1)
            query = self.input(base.float()) + self.query_position[:,None]
        else:
            base = torch.where(own_visible[:,:,None,None], h, self.missing_node[:,None])
            delta = times[:,None,:] - times[:,:,None]
            memory = source[:,None] + self.encode_time(delta)[:,:,:,None]
            query = self.input(base.float()) + self.query_position[:,None]
            if self.mode == "independent":
                allowed = allowed & torch.eye(k,device=h.device,dtype=torch.bool)[None]
        # A null slot is ONLY visible for a target with no usable observations.
        mask = ~allowed.repeat_interleave(c.queries,dim=-1)
        memory = memory.reshape(b*k,k*c.queries,c.hidden)
        memory = torch.cat((memory,self.null_memory.expand(b*k,-1,-1)),dim=1)
        mask = torch.cat((mask.reshape(b*k,-1), allowed.any(-1).reshape(b*k,1)),dim=1)
        query = query.reshape(b*k,c.queries,c.hidden)
        for block in self.blocks:
            query = block(query,memory,mask)  # immutable evidence at EVERY layer
        residual = self.output(query).reshape(b,k,c.queries,c.feature)
        state = base.float() + residual.float()
        reader_z = torch.where(own_visible[:,:,None,None], z,
                               self.missing_cognition[:,None]).float()
        return {"state":state, "reader_z":reader_z, "base_state":base,
                "allowed_sources":allowed, "residual":residual, "target_visible":own_visible}


class TimelinePolicy(nn.Module):
    """Shared trainable local QFormer, conditioner, and FULL trainable original DiT."""
    def __init__(self, local_encoder, action_net, config=Config(), mode="timeline"):
        super().__init__()
        self.local_encoder, self.reader = local_encoder, action_net
        self.timeline = TimelineConditioner(config,mode)
        self.requires_grad_(True)
        self.local_encoder.eval()  # deterministic evidence encoding; gradients still enabled

    def train(self, mode=True):
        super().train(mode)
        self.local_encoder.eval()  # do not introduce mask-dependent dropout RNG
        return self

    def encode(self, patches, vae, z, times, visible, cutoffs=None):
        if patches.ndim != 4 or vae.ndim != 4 or patches.shape[:2] != times.shape or vae.shape[:2] != times.shape:
            raise ValueError("one independent local feature set per actual node required")
        b,k=times.shape
        # Mask BEFORE local transformer; hidden observations never enter memory.
        p = torch.where(visible[:,:,None,None], patches, torch.zeros_like(patches))
        v = torch.where(visible[:,:,None,None], vae, torch.zeros_like(vae))
        h = self.local_encoder.encode_queries(p.flatten(0,1),v.flatten(0,1))
        h = h.reshape(b,k,*h.shape[1:])
        out = self.timeline(h,z,times,visible,cutoffs)
        out["conditions"] = self.local_encoder.v_proj(out["state"].to(h.dtype))
        return out

    def forward(self, patches, vae, z, times, visible, cutoffs=None,
                noisy_actions=None, tau=None):
        out=self.encode(patches,vae,z,times,visible,cutoffs)
        if (noisy_actions is None)!=(tau is None):
            raise ValueError("action reader needs both noisy state and flow time")
        if noisy_actions is not None:
            n=times.numel()
            if len(noisy_actions)%n:raise ValueError("reader batch is not an integer repeat of anchors")
            repeat=len(noisy_actions)//n
            out["velocity"]=self.reader(
                noisy_actions,tau,out["reader_z"].flatten(0,1).repeat_interleave(repeat,0),
                per_token=out["conditions"].flatten(0,1).repeat_interleave(repeat,0))
        return out
