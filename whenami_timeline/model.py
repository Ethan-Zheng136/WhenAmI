"""Multi-anchor token timeline with one shared condition interface, no future API."""

from dataclasses import dataclass
import math

import torch
from torch import nn


MODES = ("independent", "forward", "direct_history", "revision")


@dataclass(frozen=True)
class TimelineConfig:
    feature_dim: int = 1024
    cognition_dim: int = 4096
    tokens: int = 16
    condition_dim: int = 32
    hidden: int = 128
    heads: int = 4


class Timeline(nn.Module):
    def __init__(self, config=TimelineConfig(), mode="revision"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.config, self.mode = config, mode
        c = config
        self.h_input = nn.Sequential(nn.LayerNorm(c.feature_dim), nn.Linear(c.feature_dim, c.hidden))
        self.z_input = nn.Sequential(nn.LayerNorm(c.cognition_dim), nn.Linear(c.cognition_dim, c.hidden))
        self.position = nn.Parameter(torch.randn(1, c.tokens, c.hidden) * .02)  # query ID, not time
        self.action_encoder = nn.GRUCell(7, c.hidden)
        self.time_encoder = nn.Sequential(nn.Linear(3, c.hidden), nn.Tanh())
        self.transition = nn.Sequential(nn.Linear(3*c.hidden, 2*c.hidden), nn.GELU(),
                                        nn.Linear(2*c.hidden, c.hidden))
        self.prior_condition = nn.Linear(c.hidden, c.condition_dim)
        self.filter_update = nn.GRUCell(2*c.hidden, c.hidden)
        self.backward_attention = nn.MultiheadAttention(c.hidden, c.heads, batch_first=True, dropout=0.)
        self.backward_update = nn.Sequential(nn.Linear(3*c.hidden, c.hidden), nn.GELU(),
                                             nn.Linear(c.hidden, c.hidden), nn.Tanh())
        self.relevance_gate = nn.Sequential(nn.Linear(3*c.hidden, c.hidden), nn.GELU(),
                                            nn.Linear(c.hidden, 1), nn.Sigmoid())
        nn.init.zeros_(self.relevance_gate[-2].weight)
        nn.init.constant_(self.relevance_gate[-2].bias, -2.)
        self.bank_attention = nn.MultiheadAttention(c.hidden, c.heads, batch_first=True, dropout=0.)
        self.readout = nn.Sequential(nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.condition_dim))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def dt(self, gap):
        x = gap.to(dtype=self.position.dtype)
        return self.time_encoder(torch.stack((torch.log1p(x), x/16., torch.sqrt(x+1)-1), -1))

    def executed(self, seq, mask):
        if seq.ndim != 3 or seq.shape[-1] != 7 or mask.shape != seq.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("executed interval shape")
        if ((~mask[:, :-1]) & mask[:, 1:]).any() or not torch.isfinite(seq[mask]).all():
            raise ValueError("executed interval mask/values")
        value = seq.new_zeros(len(seq), self.config.hidden, dtype=self.position.dtype)
        for j in range(seq.shape[1]):
            x = torch.where(mask[:, j, None], seq[:, j].to(value.dtype), torch.zeros_like(seq[:, j], dtype=value.dtype))
            updated = self.action_encoder(x, value)
            value = torch.where(mask[:, j, None], updated, value)
        return value

    def forward(self, h, z, base_condition, times, executed_actions, executed_mask):
        c = self.config
        if h.ndim != 4 or h.shape[2:] != (c.tokens, c.feature_dim):
            raise ValueError("h must be [B,K,queries,feature]")
        b, k = h.shape[:2]
        if k < 1 or z.shape != (b,k,1,c.cognition_dim) or base_condition.shape != (b,k,c.tokens,c.condition_dim):
            raise ValueError("anchor features misaligned")
        if times.shape != (b,k) or times.dtype not in (torch.int32,torch.int64):
            raise ValueError("actual integer anchor times required")
        if executed_actions.ndim != 4 or executed_actions.shape[:2] != (b,k) or executed_mask.shape != executed_actions.shape[:-1]:
            raise ValueError("intervals misaligned")
        if executed_mask[:,0].any():
            raise ValueError("first anchor cannot have an unspecified predecessor")
        if k > 1 and ((torch.diff(times,dim=1)<=0).any() or
                      not torch.equal(torch.diff(times,dim=1),executed_mask[:,1:].sum(-1))):
            raise ValueError("actual gaps must equal complete executed-action interval lengths")
        # Frozen per-anchor encodings are independent. No temporal attention over
        # full h before the prefix scan, so future encodings cannot leak backward.
        observed = self.h_input(h.float()) + self.z_input(z.float()) + self.position[:,None]
        filtered, bank, conditions, priors, versions, gates = [], [], [], [], [], []
        for t in range(k):
            current = observed[:,t]
            u = self.executed(executed_actions[:,t],executed_mask[:,t])
            if t == 0:
                now = current
                prior = current
            else:
                gap = self.dt(times[:,t]-times[:,t-1])[:,None].expand(-1,c.tokens,-1)
                prior = filtered[-1] + self.transition(torch.cat(
                    (filtered[-1],u[:,None].expand(-1,c.tokens,-1),gap),-1))
                now = self.filter_update(
                    torch.cat((current,u[:,None].expand(-1,c.tokens,-1)),-1).reshape(-1,2*c.hidden),
                    prior.reshape(-1,c.hidden)).reshape(b,c.tokens,c.hidden)
            priors.append(self.prior_condition(prior))
            if self.mode == "independent":
                now = current
            if t and self.mode in ("revision","direct_history"):
                old = torch.stack(bank if self.mode == "revision" else filtered,1)
                original = torch.stack(filtered,1)
                age = self.dt(times[:,t,None]-times[:,:t])[:,:,None]
                query = (old+age).reshape(b,t*c.tokens,c.hidden)
                message = current-prior if self.mode=="revision" else current
                content,_ = self.backward_attention(query,message,message,need_weights=False)
                content = content.reshape(b,t,c.tokens,c.hidden)
                signals = torch.cat((old,content,original),-1)
                gate = self.relevance_gate(signals)
                corrected = old + gate*self.backward_update(signals)
                bank = list(corrected.unbind(1))
                gates.append(gate)
            filtered.append(now)
            bank.append(now)
            if self.mode in ("revision","direct_history"):
                memory = torch.stack(bank,1)
                age = self.dt(times[:,t,None]-times[:,:t+1])[:,:,None]
                context,_ = self.bank_attention(now,(memory+age).flatten(1,2),memory.flatten(1,2),need_weights=False)
                state = now+context
            else:
                state = now
            conditions.append(base_condition[:,t].float()+self.readout(state))
            versions.append(torch.stack(bank,1))
        return {
            "conditions":torch.stack(conditions,1),  # b[t|t] only for each action window
            "predicted_conditions":torch.stack(priors,1),
            "belief_versions":tuple(versions),       # b[k|t] lower triangle for diagnostic use
            "revision_gates":tuple(gates),
        }
