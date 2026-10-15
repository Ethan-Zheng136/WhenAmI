"""Present-token losses. All position losses are label-free except the optional weak t/T term."""
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from whenami_unified.losses import draw, fm   # noqa: F401  (re-export)

TEACHER_FM = (("present", 0, 0), ("present", 1, 1), ("present", 2, 2),
              ("revised", 2, 0), ("revised", 2, 1), ("hindsight", 3, 0), ("hindsight", 3, 1))
STUDENT_FM = (("present", 0, 0), ("present", 1, 1), ("present", 2, 2))


def fm_labels(labels, requests):
    a = torch.stack([labels.actions[:, i if kind != "present" else c] for kind, c, i in requests], 1).flatten(0, 1)
    v = torch.stack([labels.valid[:, i if kind != "present" else c] for kind, c, i in requests], 1).flatten(0, 1)
    return a, v


def cos(a, b, mask=None):
    d = 1 - F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-6)   # per token
    d = d.flatten(1).mean(-1)                                               # per sample
    if mask is None:
        return d.mean()
    m = mask.float()
    return (d*m).sum()/m.sum().clamp_min(1.)


def _err(x, y):
    """Scale-free per-token squared error [B,1,Q,1]."""
    return ((x.float()-y.float()).square().mean(-1, keepdim=True) /
            (y.float().square().mean(-1, keepdim=True).detach()+1e-6))


def branch_errors(out, target):
    """Diagnostics: mean scale-free error of each branch to the target (usable samples) and the
    NLL-optimal forward weight e_b/(e_f+e_b) the precision heads should converge to."""
    with torch.no_grad():
        if "p_bwd" not in out or not out["usable"].any():
            return None
        u = out["usable"]
        ef = _err(out["p_fwd"][u], target[u]).mean()
        eb = _err(out["p_bwd"][u], target[u]).mean()
        return ef, eb, eb/(ef+eb+1e-12)


def precision_nll(out, target):
    """Heteroscedastic Gaussian NLL of each branch against the (stop-grad) target present.

    Residuals are detached: this trains ONLY the log-variance heads to be calibrated, so the
    fusion weights follow each branch's actual error instead of collapsing to uniform.
    """
    y = target.detach()
    e = _err(out["p_fwd"].detach(), y)
    loss = .5*(e*torch.exp(-out["s_fwd"]) + out["s_fwd"]).mean()
    if "p_bwd" in out:
        m = out["usable"].float()[:, None, None, None]
        e = _err(out["p_bwd"].detach(), y)
        lb = .5*(e*torch.exp(-out["s_bwd"]) + out["s_bwd"])
        loss = loss + (lb*m).sum()/(m.sum()*lb[0].numel()).clamp_min(1.)
    return loss


def agreement(out):
    """NLL of fwd-vs-bwd disagreement under combined variance (branches pulled together
    where both are confident; disagreement absorbed by variance where truly ambiguous)."""
    if "p_bwd" not in out:
        z = out["p_fwd"].float().sum()*0
        return z, z.detach()
    m = out["usable"].float()[:, None, None, None]
    d = _err(out["p_fwd"], out["p_bwd"])   # gradients reach both branches
    var = torch.exp(out["s_fwd"]) + torch.exp(out["s_bwd"])
    nll = .5*(d/var + torch.log(var))
    norm = (m.sum()*nll[0].numel()).clamp_min(1.)
    return (nll*m).sum()/norm, ((d*m).sum()/norm).detach()


def monotonic(taus, margin=.01):
    """Intra-episode ranking: progress of later presents must exceed earlier ones."""
    loss, n = taus[:, 0].sum()*0, 0
    for i in range(taus.shape[1]):
        for j in range(i+1, taus.shape[1]):
            loss = loss + F.relu(margin-(taus[:, j]-taus[:, i])).mean(); n += 1
    return loss/max(n, 1)


@torch.no_grad()
def sinkhorn(logits, iters=3, eps=.05):
    """Log-domain Sinkhorn-Knopp (SwAV). The previous linear-domain version computed
    exp((l-max)/eps); with logits in [-1/T, 1/T]=[-10,10] and eps .05 a prototype that no sample
    favours underflows to an all-zero row -> 0/0 -> NaN codes on EVERY rank (64-PPU job 30368,
    main step 451). Log-domain normalisation cannot underflow."""
    log_q = (logits.float() / eps).T                     # K x N
    k, n = log_q.shape
    for _ in range(iters):
        log_q = log_q - torch.logsumexp(log_q, 1, keepdim=True) - math.log(k)
        log_q = log_q - torch.logsumexp(log_q, 0, keepdim=True) - math.log(n)
    q = torch.exp(log_q + math.log(n)).T                  # N x K, rows sum to 1
    if not torch.isfinite(q).all():
        raise FloatingPointError("sinkhorn codes")
    return q


def global_codes(logits):
    """Equipartitioned codes over the GLOBAL batch (cross-episode alignment)."""
    flat = logits.detach().float().flatten(0, -2)
    if dist.is_initialized() and dist.get_world_size() > 1:
        bank = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
        dist.all_gather(bank, flat)
        r = dist.get_rank()
        q = sinkhorn(torch.cat(bank))
        return q[r*len(flat):(r+1)*len(flat)].reshape(logits.shape)
    return sinkhorn(flat).reshape(logits.shape)


def phase_alignment(outs):
    """Cross-episode prototype consistency (SwAV swapped prediction between the fwd and bwd
    views of the SAME present, codes equipartitioned across all episodes of the global batch)
    plus non-decreasing expected phase index along the episode."""
    lf = torch.stack([o["logits_fwd"] for o in outs], 1)
    if all("logits_bwd" in o for o in outs):
        lb = torch.stack([o["logits_bwd"] for o in outs], 1)
        m = torch.stack([o["usable"] for o in outs], 1).float()
    else:
        lb, m = torch.stack([o["logits"] for o in outs], 1), torch.ones(lf.shape[:2], device=lf.device)
    qf, qb = global_codes(lf), global_codes(lb)
    swap = -(qb*F.log_softmax(lf, -1)).sum(-1) - (qf*F.log_softmax(lb, -1)).sum(-1)
    swap = (swap*m).sum()/m.sum().clamp_min(1.)/2
    fused = torch.stack([o["logits"] for o in outs], 1)
    k = fused.shape[-1]
    index = (F.softmax(fused, -1)*torch.arange(k, device=fused.device)).sum(-1)/(k-1)
    order = sum(F.relu(index[:, i]-index[:, j]).mean() for i in range(index.shape[1])
                for j in range(i+1, index.shape[1]))
    return swap, order


def progress_label(cases, presents, device):
    """WEAK free label t/(T-1); episode length known. Optional (--tau-label-weight)."""
    rows = [[c["provenance"]["source_times"][j]/max(1, c["provenance"]["episode_length"]-1)
             for j in presents] for c in cases]
    return torch.tensor(rows, device=device, dtype=torch.float32)
