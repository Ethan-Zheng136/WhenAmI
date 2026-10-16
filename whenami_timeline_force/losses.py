"""Actual current-prefix control, paired versions, and frozen-reader input gradients."""
import torch
import torch.nn.functional as F
from whenami_unified.losses import draw, fm

TEACHER_REQUESTS = ((0, 0), (1, 1), (2, 2), (0, 3), (1, 3), (2, 3))
STUDENT_REQUESTS = ((0, 0), (1, 1), (2, 2))


def requested_labels(labels, requests):
    actions = torch.stack([labels.actions[:, i] for i, _ in requests], 1).flatten(0, 1)
    valid = torch.stack([labels.valid[:, i] for i, _ in requests], 1).flatten(0, 1)
    return actions, valid


def current_loss(output, teacher):
    predicted = torch.stack([output["projected_versions"][i][:, i] for i in range(3)], 1)
    target = teacher["node_versions"][3][:, :3].detach()
    condition = (1-F.cosine_similarity(predicted.float(), target.float(), dim=-1, eps=1e-6)).mean()
    z = torch.stack([output["node_z_versions"][i][:, i] for i in range(3)], 1)
    tz = teacher["node_z_versions"][3][:, :3].detach()
    cognition = (1-F.cosine_similarity(z.float(), tz.float(), dim=-1, eps=1e-6)).mean()
    return condition, cognition


def revision_loss(output, teacher_output, teacher, noisy, tau, valid, event=1, beta=.5):
    """Match SAME old event at cutoffs1/2, with BOTH reader inputs from its version.

    Executed historical commands may make this auxiliary metric easier. It is not
    evidence of physical event understanding; only causal current FM/closed loop are.
    """
    if not 0 <= event < 2:
        raise ValueError("old event identity")
    sb, sa = (output["projected_versions"][v][:, event] for v in (1, 2))
    szb, sza = (output["node_z_versions"][v][:, event] for v in (1, 2))
    tb, ta = (teacher_output["node_versions"][v][:, event].detach() for v in (1, 2))
    tzb, tza = (teacher_output["node_z_versions"][v][:, event].detach() for v in (1, 2))
    b = len(sb)
    draws = len(noisy)//b
    def decode(c0, c1, z0, z1):
        c = torch.stack((c0, c1), 1).flatten(0, 1).repeat_interleave(draws, 0)
        z = torch.stack((z0, z1), 1).flatten(0, 1).repeat_interleave(draws, 0)
        x = noisy.reshape(b, draws, *noisy.shape[1:])[:, None].expand(-1, 2, -1, -1, -1).reshape(b*2*draws, *noisy.shape[1:])
        times = tau.reshape(b, draws)[:, None].expand(-1, 2, -1).reshape(-1)
        value = teacher(conditioning={"conditions": c, "reader_z": z},
                        noisy_actions=x, tau=times)["velocity"].reshape(b, 2, draws, *noisy.shape[1:])
        return value[:, 0].flatten(0, 1), value[:, 1].flatten(0, 1)
    with torch.no_grad():
        vtb, vta = decode(tb, ta, tzb, tza)
    # Frozen teacher parameters, LIVE student inputs: no no_grad on this call.
    vsb, vsa = decode(sb, sa, szb, sza)
    mask = valid.detach().repeat_interleave(draws, 0)
    after = (vsa.float()-vta.float()).square().mean(-1)
    delta = ((vsa.float()-vsb.float())-(vta.float()-vtb.float())).square().mean(-1)
    loss = (((after+beta*delta)*mask).sum(-1)/mask.sum(-1)).mean()
    diagnostics = {
        "teacher_revision_velocity_rms": (vta.float()-vtb.float()).square().mean().sqrt().detach(),
        "student_revision_velocity_rms": (vsa.float()-vsb.float()).square().mean().sqrt().detach()}
    return loss, diagnostics
