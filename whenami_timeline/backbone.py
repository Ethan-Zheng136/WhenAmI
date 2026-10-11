"""Real frozen Stage2 VLM/Past/DINO/Wan and a SINGLE shared original action DiT."""

import json
import torch
from torch import nn

from stage3.common import DEFAULT_PAST, reference_imports, endpoint_record
from stage3_scale.model import FrozenVisualTargets, cognition, tokenize_instruction


class FrozenPastBackbone:
    def __init__(self, device):
        reference_imports()
        from vla import load_vla
        raw = load_vla(DEFAULT_PAST, train_venc=True, action_model_type="DiT-L", load_for_training=False)
        self.transform = raw.vision_backbone.get_image_transform()
        self.tokenizer = raw.llm_backbone.get_tokenizer()
        self.prompt_builder = raw.vlm.get_prompt_builder
        self.statistics = json.loads((DEFAULT_PAST.parents[1]/"dataset_statistics.json").read_text())
        self.endpoint = endpoint_record(DEFAULT_PAST)
        self.device = torch.device(device)
        self.vlm = raw.vlm.requires_grad_(False).eval().to(self.device,dtype=torch.bfloat16)
        self.vlm.vision_backbone_requires_grad = False
        self.past = raw.past_qformer.requires_grad_(False).eval().to(self.device,dtype=torch.bfloat16)
        self.reader = raw.action_model.net.requires_grad_(False).eval().to(self.device,dtype=torch.bfloat16)
        # No future endpoint is constructed/used. This reuses only the exact
        # chronological historical feature path from the admitted implementation.
        self.visual = FrozenVisualTargets(raw.v_enc,nn.Identity()).eval().to(self.device)
        del raw

    @torch.no_grad()
    def encode(self, sample):
        current={k:v.to(self.device) for k,v in sample["current"].items()}
        past={k:v.to(self.device) for k,v in sample["past"].items()}
        tokens=[tokenize_instruction(s,self.tokenizer,self.prompt_builder) for s in sample["instructions"]]
        size=max(len(t) for t in tokens)
        ids=torch.full((len(tokens),size),self.tokenizer.pad_token_id,dtype=torch.long,device=self.device)
        mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,t in enumerate(tokens):
            ids[i,:len(t)]=t.to(self.device);mask[i,:len(t)]=True
        with torch.autocast("cuda",dtype=torch.bfloat16):
            f=self.visual(current,past)
            h=self.past.encode_queries(f["history_features"],f["history_vae"])
            cp=self.past.v_proj(h)
            z=cognition(self.vlm,ids,mask,current)
        return {
            "h":h.float().cpu(),"z":z.float().cpu(),"base_condition":cp.float().cpu(),
            "times":sample["times"].clone(),"executed_actions":sample["executed_actions"].clone(),
            "executed_mask":sample["executed_mask"].clone(),
        }

    def assert_frozen(self):
        for module in (self.vlm,self.past,self.reader,self.visual):
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError("original endpoint accidentally trained")
