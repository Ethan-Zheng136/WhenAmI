"""Only labeled diagnostics here. No fabricated contact-state labels for OXE."""

import torch
from torch.nn import functional as F


def diagnostic_loss(output, observations, state_labels, decision_labels):
    """Targets never enter model.forward. Prefix cutoff controls all evidence."""
    b, t, _ = observations.shape
    if state_labels.shape != (b, t) or decision_labels.shape != (b, t):
        raise ValueError("labels must match this evidence prefix")
    action = F.cross_entropy(output["decision_logits"], decision_labels[:, -1])
    state = F.cross_entropy(output["event_logits"].flatten(0, 1), state_labels.flatten())
    # Only real transitions: initial anchor has no previous state.
    dynamics = (F.mse_loss(output["observation_predictions"][:, 1:], observations[:, 1:].detach())
                if t > 1 else action.new_zeros(()))
    return action + .5 * state + .1 * dynamics, {
        "decision_ce": action.detach(), "event_ce": state.detach(), "prediction_mse": dynamics.detach(),
    }
