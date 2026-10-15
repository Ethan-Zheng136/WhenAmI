"""WhenAmI present-token pipeline (panel b): locate the PRESENT between past and future.

Timeline = past nodes S_<t | learnable present token p_t | future nodes S_>t.
  U_fwd : existing A50k timeline updater, query p_t^0, memory past nodes + current obs.
  U_bwd : new updater (A50k-initialised copy), query p_t^0, memory future nodes.
  fusion: per-token precision weights from predicted log-variances (heteroscedastic).
  revise: existing revise update of past nodes, now conditioned on the fused p_t.
  readout: position head (progress tau, K phase prototypes) on p_t.
  action: DiT reads ONLY v_proj(p_t) and NodeCognition readout of p_t (raw-z bypass = ablation flag).
Stage I future nodes = real future frames through QFormer(+NodeCognition seed) = WoG Future Enc.
Stage II future nodes = Future Imaginer (OFormer-style learnable queries over past/current nodes).
No backbone/DiT/QFormer change; all new modules live under `present.` (+ GRU executed_adapter).
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F

from whenami_timeline_a.model import Config
from whenami_unified.model import UnifiedUpdater, ExecutedAdapter
from whenami_unified.contracts import Evidence, Executed
from whenami_timeline_force.model import NodeCognition
from stage3_scale.model import cognition


@dataclass(frozen=True)
class PresentConfig:
    backward: bool = True          # U_bwd branch (off => forward-only ablation)
    fusion: str = "precision"      # precision | mean | forward
    revise: bool = True            # p_t-conditioned revision of past nodes
    raw_z_bypass: bool = False     # ablation: DiT reader_z = raw VLM z instead of p_t readout
    phases: int = 8                # K phase prototypes
    position_hidden: int = 256
    proto_dim: int = 128
    precision_hidden: int = 64
    temperature: float = .1
    bwd_logvar_init: float = 0.0   # equal start (w_fwd=.5). Measured NLL optimum on real batches
                                   # w*=e_b/(e_f+e_b)≈.55-.57; old init 2.0 (w=.88) was unreachable in 50k steps

    def __post_init__(self):
        if self.fusion not in ("precision", "mean", "forward") or self.phases < 2:
            raise ValueError("present config")


def fp32(seq, x):
    """Run a small head in true FP32 even inside the BF16 autocast/FSDP region.

    Bug fixed (job 30368): under autocast these Linear layers ran in BF16, so tau
    (sigmoid near .5, BF16 step 2^-8=.0039) and log-variances were quantised -> tau ties,
    strict tau-order accuracy ~0 and constant per-token fusion weights."""
    with torch.autocast("cuda", enabled=False):
        x = x.float()
        for m in seq:
            if isinstance(m, nn.LayerNorm):
                x = F.layer_norm(x, m.normalized_shape, m.weight.float(), m.bias.float(), m.eps)
            elif isinstance(m, nn.Linear):
                x = F.linear(x, m.weight.float(), None if m.bias is None else m.bias.float())
            elif isinstance(m, nn.GELU):
                x = F.gelu(x)
            else:
                raise TypeError(type(m))
        return x


class Precision(nn.Module):
    """Per-token log-variance s; precision = exp(-s)."""
    def __init__(self, feature, hidden, bias=0.):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature), nn.Linear(feature, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, x):
        return fp32(self.net, x).clamp(-8, 8)            # [B,1,Q,1] FP32


class PositionHead(nn.Module):
    def __init__(self, feature, p):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(feature), nn.Linear(feature, p.position_hidden), nn.GELU())
        self.tau = nn.Linear(p.position_hidden, 1)
        nn.init.zeros_(self.tau.weight); nn.init.zeros_(self.tau.bias)
        self.proj = nn.Linear(p.position_hidden, p.proto_dim)
        self.prototypes = nn.Parameter(torch.randn(p.phases, p.proto_dim) * .5)
        self.temperature = p.temperature

    def forward(self, node):                               # node [B,1,Q,F]
        with torch.autocast("cuda", enabled=False):
            g = fp32(self.trunk, node.float().mean(-2))[:, 0]                  # [B,H] FP32
            tau = torch.sigmoid(fp32([self.tau], g))[:, 0]                       # [B]
            logits = (F.normalize(fp32([self.proj], g), dim=-1) @
                      F.normalize(self.prototypes.float(), dim=-1).T / self.temperature)
        return tau, logits                                                       # [B], [B,K]


class Imaginer(nn.Module):
    """WoG-OFormer-style: learnable future queries read the causal past/current nodes."""
    def __init__(self, config):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, config.queries, config.feature))
        self.updater = UnifiedUpdater(config, "revise")    # A50k timeline weights copied at load


class PresentModules(nn.Module):
    def __init__(self, config, p, student):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(1, 1, config.queries, config.feature))
        self.bwd = UnifiedUpdater(config, "revise")        # A50k timeline weights copied at load
        self.precision_fwd = Precision(config.feature, p.precision_hidden, 0.)
        self.precision_bwd = Precision(config.feature, p.precision_hidden, p.bwd_logvar_init)
        self.position = PositionHead(config.feature, p)
        if student:
            self.imaginer = Imaginer(config)


class PresentPolicy(nn.Module):
    def __init__(self, vlm, local_encoder, reader, student=False, present=PresentConfig(), config=Config()):
        super().__init__()
        self.vlm, self.local_encoder, self.reader = vlm, local_encoder, reader
        self.timeline = UnifiedUpdater(config, "revise")
        self.node_cognition = NodeCognition(config)
        self.present = PresentModules(config, present, student)
        self.student, self.config, self.pcfg = student, config, present
        if student:
            self.executed_adapter = ExecutedAdapter(config.feature)
        self.requires_grad_(True)
        self.vlm.vision_backbone_requires_grad = True
        self.local_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.local_encoder.eval()
        return self

    # ---------------- per-frame evidence -> nodes ----------------
    def frames(self, evidence, executed):
        evidence.validate()
        if self.student:
            if not isinstance(executed, Executed):
                raise TypeError("student needs executed intervals")
            executed.validate(evidence)
        elif executed is not None:
            raise ValueError("teacher is visual-only")
        hs, zs = [], []
        for j, frame in enumerate(evidence.frames):
            m = evidence.visible[:, j]
            pixels = {k: torch.where(m[:, None, None, None], x, 0.) for k, x in frame.pixels.items()}
            zj = cognition(self.vlm, frame.input_ids, frame.attention_mask, pixels)
            hj = self.local_encoder.encode_queries(torch.where(m[:, None, None], frame.patches, 0.),
                                                   torch.where(m[:, None, None], frame.vae, 0.))
            hs.append(torch.where(m[:, None, None], hj, 0.))
            zs.append(torch.where(m[:, None, None], zj, 0.))
        h, raw_z = torch.stack(hs, 1), torch.stack(zs, 1)
        seeded = self.node_cognition.seed(h, raw_z)
        if executed is not None:
            seeded = seeded + self.executed_adapter(executed).to(seeded.dtype)
        tl, vis = self.timeline, evidence.visible
        nodes = torch.where(vis[:, :, None, None], seeded, tl.missing_node[:, None].to(seeded.dtype))
        zm = torch.where(vis[:, :, None, None], raw_z, tl.missing_cognition[:, None].to(raw_z.dtype))
        causal = tl(seeded, raw_z, evidence.times, vis)      # versions[j] = nodes after evidence<=j
        return {"h": h, "raw_z": raw_z, "nodes": nodes, "z": zm, "versions": causal["versions"],
                "times": evidence.times, "visible": vis}

    def imagine(self, enc, c, future_times):
        """Imagined future nodes for present c at the given (timestamp-only) future times."""
        im = self.present.imaginer
        b, m = future_times.shape
        past = enc["versions"][c-1] if c else enc["nodes"][:, :0]
        memory = torch.cat((past, enc["nodes"][:, c:c+1]), 1)
        query = (enc["nodes"][:, c:c+1] + im.query.to(memory.dtype)).expand(-1, m, -1, -1)
        return im.updater.update(query, memory, enc["z"][:, :c+1], future_times,
                                 enc["times"][:, :c+1], enc["times"][:, c], enc["visible"][:, :c+1])

    def locate(self, enc, c, future=None, future_times=None, future_valid=None):
        p = self.pcfg
        times, vis = enc["times"], enc["visible"]
        b = len(times)
        p0 = enc["nodes"][:, c:c+1] + self.present.embed.to(enc["nodes"].dtype)
        past = enc["versions"][c-1] if c else enc["nodes"][:, :0]
        memory = torch.cat((past, enc["nodes"][:, c:c+1]), 1)
        # U_fwd = the existing (A50k) timeline updater, present-token query.
        p_fwd = self.timeline.update(p0, memory, enc["z"][:, :c+1], times[:, c:c+1],
                                     times[:, :c+1], times[:, c], vis[:, :c+1])
        s_fwd = self.present.precision_fwd(p_fwd)
        out = {"p0": p0, "p_fwd": p_fwd, "s_fwd": s_fwd}
        has_future = (future is not None and p.backward and future.shape[1] > 0)
        if has_future:
            fz = self.node_cognition(future)
            p_bwd = self.present.bwd.update(p0, future, fz, times[:, c:c+1], future_times,
                                            future_times[:, -1], future_valid)
            s_bwd = self.present.precision_bwd(p_bwd)
            usable = future_valid.any(-1)                  # [B]
            if p.fusion == "precision":
                w = torch.sigmoid(s_bwd - s_fwd)           # exp(-s_f)/(exp(-s_f)+exp(-s_b))
            elif p.fusion == "mean":
                w = torch.full_like(s_fwd, .5)
            else:
                w = torch.ones_like(s_fwd)
            w = torch.where(usable[:, None, None, None], w, torch.ones_like(w))
            fused = w*p_fwd.float() + (1-w)*p_bwd.float()
            out.update(p_bwd=p_bwd, s_bwd=s_bwd, w_fwd=w, usable=usable)
        else:
            fused = p_fwd.float()
            out.update(w_fwd=torch.ones_like(s_fwd), usable=torch.zeros(b, dtype=torch.bool, device=times.device))
        present = fused.to(p_fwd.dtype)
        zp = self.node_cognition(present)                  # [B,1,1,C]
        out.update(present=present, present_z=zp, condition=self.local_encoder.v_proj(present))
        out["reader_z"] = enc["raw_z"][:, c] if p.raw_z_bypass else zp[:, 0]
        out["tau"], out["logits"] = self.present.position(present)
        _, out["logits_fwd"] = self.present.position(p_fwd)
        if has_future:
            _, out["logits_bwd"] = self.present.position(out["p_bwd"])
        if c and p.revise:
            revised = self.timeline.update(past, torch.cat((past, present), 1),
                                           torch.cat((enc["z"][:, :c], zp.to(enc["z"].dtype)), 1),
                                           times[:, :c], times[:, :c+1], times[:, c], vis[:, :c+1])
            out["revised"] = revised
            out["revised_condition"] = self.local_encoder.v_proj(revised)
            out["revised_z"] = self.node_cognition(revised)
        return out

    # ---------------- full stage forward ----------------
    def forward(self, evidence=None, executed=None, presents=(0, 1, 2), future_times=None,
                future_keep=None, noisy_actions=None, tau=None, fm_requests=None, conditioning=None,
                imagine_times=None):
        if conditioning is not None:
            return {"velocity": self.reader(noisy_actions, tau, conditioning["reader_z"],
                                             per_token=conditioning["conditions"])}
        if not isinstance(evidence, Evidence):
            raise TypeError("Evidence required")
        enc = self.frames(evidence, executed)
        k = len(evidence.frames)
        result = {"enc": enc, "present": {}}
        for c in presents:
            if self.student:
                ft = imagine_times[c] if imagine_times is not None else None
                if ft is not None and ft.shape[1] > 0:
                    fut = self.imagine(enc, c, ft)
                    out = self.locate(enc, c, fut, ft, torch.ones(ft.shape, dtype=torch.bool, device=ft.device))
                    out["imagined"] = fut
                    out["imagined_condition"] = self.local_encoder.v_proj(fut)
                else:
                    out = self.locate(enc, c)
            else:
                if c+1 < k and future_keep is not None:
                    fut = enc["nodes"][:, c+1:]
                    fv = enc["visible"][:, c+1:] & future_keep[:, None]
                    out = self.locate(enc, c, fut, enc["times"][:, c+1:], fv)
                    out["future_nodes"] = fut
                    out["future_condition"] = self.local_encoder.v_proj(fut)
                else:
                    out = self.locate(enc, c)
            result["present"][c] = out
        if not self.student and k == 4:
            hs = enc["versions"][3]
            result["hindsight"] = hs
            result["hindsight_condition"] = self.local_encoder.v_proj(hs)
            result["hindsight_z"] = self.node_cognition(hs)
        if fm_requests is not None:
            conds, zs = [], []
            for kind, c, i in fm_requests:
                o = result["present"].get(c)
                if kind == "present":
                    conds.append(o["condition"][:, 0]); zs.append(o["reader_z"])
                elif kind == "revised":
                    conds.append(o["revised_condition"][:, i]); zs.append(o["revised_z"][:, i])
                elif kind == "hindsight":
                    conds.append(result["hindsight_condition"][:, i]); zs.append(result["hindsight_z"][:, i])
                else:
                    raise ValueError(kind)
            cond = torch.stack(conds, 1).flatten(0, 1)
            zz = torch.stack(zs, 1).flatten(0, 1)
            draws = len(noisy_actions)//len(cond)
            result["velocity"] = self.reader(noisy_actions, tau, zz.repeat_interleave(draws, 0),
                                             per_token=cond.repeat_interleave(draws, 0))
        return result


def build_policy(student=False, present=PresentConfig(), seed=42):
    from whenami_timeline_joint.model import load_joint
    old, visual, transform, tokenizer, prompt = load_joint("timeline", seed)
    torch.manual_seed(seed)
    model = PresentPolicy(old.vlm, old.local_encoder, old.reader, student, present)
    model.timeline.load_state_dict(old.timeline.state_dict(), strict=True)
    return model, visual, transform, tokenizer, prompt


def new_parameter_report(model):
    groups = {}
    for n, q in model.named_parameters():
        n = n.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        top = n.split(".")[0] if not n.startswith("present.") else ".".join(n.split(".")[:2])
        groups[top] = groups.get(top, 0) + q.numel()
    new = {k: v for k, v in groups.items() if k.startswith(("present.", "node_cognition", "executed_adapter"))}
    return {"groups": groups, "new_vs_A50k": new, "new_total": sum(new.values()), "total": sum(groups.values())}
