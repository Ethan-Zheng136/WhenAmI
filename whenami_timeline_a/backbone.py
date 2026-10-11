"""Reuse the action-grounded Future Stage1 endpoint without altering its files."""

import gc
import json
from pathlib import Path
import time

import torch
from PIL import Image

from stage3.common import DEFAULT_FUTURE,endpoint_record,check_endpoint,reference_imports
from stage3_scale.model import cognition,tokenize_instruction
from whenami_timeline_a.model import Config,TimelinePolicy


class Foundation:
    def __init__(self,device="cuda:0"):
        reference_imports()
        from vla import load_vla
        self.device=torch.device(device)
        self.endpoint=endpoint_record(DEFAULT_FUTURE)
        self.statistics=self.endpoint["statistics"]
        began=time.monotonic()
        raw=load_vla(DEFAULT_FUTURE,train_venc=True,action_model_type="DiT-L",load_for_training=False)
        if raw.past_spec is not None or raw.use_temporal:raise ValueError("expected original Future Stage1")
        self.transform=raw.vision_backbone.get_image_transform()
        self.tokenizer=raw.llm_backbone.get_tokenizer()
        self.prompt=raw.vlm.get_prompt_builder
        # VLM and single-frame visual targets remain frozen, no optimizer state.
        self.vlm=raw.vlm.requires_grad_(False).eval().to(self.device,dtype=torch.bfloat16)
        self.vlm.vision_backbone_requires_grad=False
        self.visual=raw.v_enc.requires_grad_(False).eval().to(self.device)
        self.local=raw.qformer.requires_grad_(True).eval().to(self.device)
        self.reader=raw.action_model.net.requires_grad_(True).to(self.device)
        del raw
        gc.collect()
        print("TIMELINE_A_FOUNDATION_READY",round(time.monotonic()-began,2),
              "seconds",str(DEFAULT_FUTURE),flush=True)

    @torch.no_grad()
    def encode_frame(self,image,instruction):
        """No temporal batch/video; future pixels cannot leak into earlier features."""
        now={k:v[None].to(self.device) for k,v in self.transform(Image.fromarray(image)).items()}
        ids=tokenize_instruction(instruction,self.tokenizer,self.prompt)[None].to(self.device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            # DINO sees this frame only; Wan receives T=1 (a supported causal VAE length).
            patches,vae=self.visual(now,now,video_pixel_values={k:v[:,None] for k,v in now.items()})
            size=int(patches.shape[1]**.5)
            if size*size!=patches.shape[1]:raise ValueError("non-square patch grid")
            p=patches.reshape(1,size,size,-1).permute(0,3,1,2)
            p=torch.nn.functional.adaptive_avg_pool2d(p,(8,8)).flatten(2).transpose(1,2)
            z=cognition(self.vlm,ids,torch.ones_like(ids,dtype=torch.bool),now)
        return {"patches":p[0].float().cpu(),"vae":vae[0].float().cpu(),"z":z[0].float().cpu()}

    @torch.no_grad()
    def encode(self,observations):
        values=[self.encode_frame(image,text) for image,text in
                zip(observations["images"],observations["instructions"])]
        return {**{k:torch.stack([v[k] for v in values]) for k in ("patches","vae","z")},
                "times":observations["times"].clone()}

    def model(self,mode,seed=42,config=Config()):
        # Always reset BOTH pretrained learned modules from immutable CPU mmap.
        state=torch.load(DEFAULT_FUTURE,weights_only=True,mmap=True,map_location="cpu")["model"]
        self.local.load_state_dict(state["qformer"],strict=True)
        self.reader.load_state_dict({k.removeprefix("net."):v for k,v in state["action_model"].items()
                                    if k.startswith("net.")},strict=True)
        del state
        torch.manual_seed(seed)
        model=TimelinePolicy(self.local,self.reader,config,mode).to(self.device)
        return model

    def check_frozen(self):
        for module in (self.vlm,self.visual):
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError("foundation accidentally trained")
        check_endpoint(self.endpoint)
