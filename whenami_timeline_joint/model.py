"""True joint VLM->temporal condition->DiT graph; no cached cognition."""

import torch
from torch import nn

from whenami_timeline_a.model import Config, TimelineConditioner
from stage3_scale.model import cognition

MODES = ("timeline", "pooled_query")


class AddressedConditioner(TimelineConditioner):
    """Same parameter set; explicit target-time query in BOTH arms."""
    def __init__(self, config=Config(), mode="timeline"):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(config, "timeline")
        self.mode = mode

    def forward(self, h, z, times, visible, cutoffs=None):
        c = self.config
        if h.ndim != 4 or h.shape[2:] != (c.queries, c.feature):
            raise ValueError("h shape")
        b, k = h.shape[:2]
        if z.shape != (b,k,1,c.cognition) or times.shape != (b,k) or visible.shape != (b,k):
            raise ValueError("anchor tensors misaligned")
        if visible.dtype != torch.bool or times.dtype not in (torch.int32, torch.int64):
            raise ValueError("boolean mask and actual integer event times required")
        if k > 1 and (times.diff(dim=1) <= 0).any():
            raise ValueError("anchors must be chronological and distinct")
        if cutoffs is None:
            cutoffs = times[:, -1:].expand_as(times)
        if cutoffs.shape != times.shape:
            raise ValueError("one cutoff per target")
        allowed = visible[:,None,:] & (times[:,None,:] <= cutoffs[:,:,None])
        own = visible & (times <= cutoffs)
        safe_h = torch.where(visible[:,:,None,None], h, torch.zeros_like(h))
        safe_z = torch.where(visible[:,:,None,None], z, torch.zeros_like(z))
        source = self.input(safe_h) + self.cognition_input(safe_z) + self.query_position[:,None]
        relative = times[:,None,:] - times[:,:,None]
        memory = source[:,None] + self.encode_time(relative)[:,:,:,None]
        if self.mode == "pooled_query":
            weights = allowed.to(memory.dtype)[:,:,:,None,None]
            count = allowed.sum(-1).clamp_min(1).to(memory.dtype)[:,:,None,None]
            pooled = (memory * weights).sum(2) / count
            memory = pooled[:,:,None].expand(-1,-1,k,-1,-1)
        base = torch.where(own[:,:,None,None], h, self.missing_node[:,None])
        query = (self.input(base) + self.query_position[:,None]
                 + self.encode_time(times - times[:,:1])[:,:,None])
        memory = memory.reshape(b*k,k*c.queries,c.hidden)
        memory = torch.cat((memory,self.null_memory.expand(b*k,-1,-1)),1)
        forbidden = ~allowed.repeat_interleave(c.queries,-1)
        forbidden = torch.cat((forbidden.reshape(b*k,-1),allowed.any(-1).reshape(b*k,1)),1)
        query = query.reshape(b*k,c.queries,c.hidden)
        for block in self.blocks:
            query = block(query,memory,forbidden)
        delta = self.output(query).reshape(b,k,c.queries,c.feature)
        return {
            "state":base+delta,
            "reader_z":torch.where(own[:,:,None,None],z,self.missing_cognition[:,None]),
            "allowed_sources":allowed,"residual":delta,
        }


class JointTimeline(nn.Module):
    def __init__(self, vlm, local_encoder, reader, mode="timeline", config=Config()):
        super().__init__()
        self.vlm,self.local_encoder,self.reader=vlm,local_encoder,reader
        self.timeline=AddressedConditioner(config,mode)
        self.requires_grad_(True)
        self.vlm.vision_backbone_requires_grad=True
        self.local_encoder.eval()  # stable Q evidence path, not frozen

    def train(self, mode=True):
        super().train(mode);self.local_encoder.eval()
        return self

    def forward(self, pixels=None, input_ids=None, attention_mask=None, patches=None,
                vae=None, times=None, visible=None, cutoffs=None,
                noisy_actions=None, tau=None, condition_only=False, conditioning=None):
        if conditioning is not None:
            if pixels is not None or condition_only:
                raise ValueError("decoder-only FSDP call cannot also encode observations")
            return {"velocity":self.reader(noisy_actions,tau,conditioning["reader_z"],
                                           per_token=conditioning["conditions"])}
        if times is None or visible is None or times.shape != visible.shape:
            raise ValueError("explicit actual times/visibility required")
        b,k=times.shape
        if input_ids.shape[0] != b*k:
            raise ValueError("one independent VLM input per anchor")
        if patches.shape[:2] != (b,k) or vae.shape[:2] != (b,k):
            raise ValueError("one frozen visual feature set per anchor")
        # No video/sequence VLM attention between frames: each is a batch item.
        safe_pixels={key:torch.where(visible[:,:,None,None,None],v,torch.zeros_like(v)).flatten(0,1)
                     for key,v in pixels.items()}
        z=cognition(self.vlm,input_ids,attention_mask,safe_pixels)
        z=z.reshape(b,k,1,-1)  # attached to VLM graph, NEVER detach/cpu/no_grad
        p=torch.where(visible[:,:,None,None],patches,torch.zeros_like(patches))
        v=torch.where(visible[:,:,None,None],vae,torch.zeros_like(vae))
        h=self.local_encoder.encode_queries(p.flatten(0,1),v.flatten(0,1))
        h=h.reshape(b,k,*h.shape[1:])
        enc=self.timeline(h,z,times,visible,cutoffs)
        conditions=self.local_encoder.v_proj(enc["state"].to(h.dtype))
        result={"conditions":conditions.flatten(0,1),"reader_z":enc["reader_z"].flatten(0,1)}
        if condition_only:
            return result
        if noisy_actions is None or tau is None or len(noisy_actions)%(b*k):
            raise ValueError("noisy action batch must repeat anchors")
        repeat=len(noisy_actions)//(b*k)
        result["velocity"]=self.reader(
            noisy_actions,tau,result["reader_z"].repeat_interleave(repeat,0),
            per_token=result["conditions"].repeat_interleave(repeat,0))
        return result


def load_joint(mode="timeline", seed=42):
    from stage3.common import DEFAULT_FUTURE,reference_imports
    reference_imports()
    from vla import load_vla
    raw=load_vla(DEFAULT_FUTURE,train_venc=True,action_model_type="DiT-L",load_for_training=True)
    raw.freeze_backbones("full-finetune")
    torch.manual_seed(seed)
    student=JointTimeline(raw.vlm,raw.qformer,raw.action_model.net,mode)
    visual=raw.v_enc.requires_grad_(False).eval()
    transform=raw.vision_backbone.get_image_transform()
    tokenizer=raw.llm_backbone.get_tokenizer()
    prompt=raw.vlm.get_prompt_builder
    return student,visual,transform,tokenizer,prompt
