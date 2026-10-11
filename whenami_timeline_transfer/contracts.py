"""Student only receives a physical prefix, not masked copies of future inputs."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Prefix:
    pixels: dict
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    patches: torch.Tensor
    vae: torch.Tensor
    times: torch.Tensor

    def validate(self):
        if self.times.ndim != 2 or self.times.dtype not in (torch.int32, torch.int64):
            raise ValueError("actual integer prefix times required")
        b,k=self.times.shape
        if k<1 or (k>1 and (self.times.diff(dim=1)<=0).any()):
            raise ValueError("nonempty chronological prefix required")
        if self.input_ids.ndim!=2 or self.input_ids.shape[0]!=b*k or self.attention_mask.shape!=self.input_ids.shape:
            raise ValueError("one language sequence per prefix image")
        if self.attention_mask.dtype!=torch.bool or not self.attention_mask.any(-1).all():
            raise ValueError("invalid language mask")
        if self.patches.shape[:2]!=(b,k) or self.vae.shape[:2]!=(b,k):
            raise ValueError("one local visual feature per prefix image")
        if not self.pixels or any(x.shape[:2]!=(b,k) for x in self.pixels.values()):
            raise ValueError("pixel prefix mismatch")
        return self

    def kwargs(self):
        self.validate()
        return {
            "pixels":self.pixels,"input_ids":self.input_ids,
            "attention_mask":self.attention_mask,"patches":self.patches,"vae":self.vae,
            "times":self.times,"visible":torch.ones_like(self.times,dtype=torch.bool),
            "cutoffs":self.times[:,-1:].expand_as(self.times),
        }


def make_prefix(inputs,features,current_index):
    times=inputs["times"];b,k=times.shape
    if not isinstance(current_index,int) or not 0<=current_index<k:
        raise ValueError("valid current anchor index required")
    n=current_index+1
    ids=inputs["input_ids"].reshape(b,k,-1)[:,:n].clone()
    attention=inputs["attention_mask"].reshape(b,k,-1)[:,:n].clone()
    # Remove padding whose length could have depended on future language.
    positions=torch.arange(attention.shape[-1],device=attention.device)+1
    width=int(torch.where(attention,positions,0).max())
    ids=ids[...,:width].reshape(b*n,width)
    attention=attention[...,:width].reshape(b*n,width)
    return Prefix(
        pixels={name:x[:,:n].clone() for name,x in inputs["pixels"].items()},
        input_ids=ids,attention_mask=attention,
        patches=features["patches"][:,:n].clone(),vae=features["vae"][:,:n].clone(),
        times=times[:,:n].clone(),
    ).validate()


def select_anchor(encoded,batch,nodes,index):
    if not 0<=index<nodes:raise ValueError("target index")
    result={}
    for key in ("conditions","reader_z"):
        x=encoded[key]
        if x.shape[0]!=batch*nodes:raise ValueError("encoded shape mismatch")
        result[key]=x.reshape(batch,nodes,*x.shape[1:])[:,index]
    return result
