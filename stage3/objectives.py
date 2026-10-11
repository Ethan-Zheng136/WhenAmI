"""Paired, directional behavior/condition losses with explicit target ownership."""

import torch


def connector_inputs(batch, mode):
    if mode not in ("forward", "bidirectional"):
        raise ValueError("mode must be forward or bidirectional")
    inputs = [batch["past_h"].detach(), batch["past_z"].detach()]
    if mode == "bidirectional":
        inputs += [batch["future_h"].detach(), batch["future_z"].detach()]
    return inputs


def directional_losses(connector, predictions, batch, readers, seed, alignment_weight=1., action_weight=1.):
    """Only the condition has gradients through a frozen destination reader.

    Forward: Past inputs -> C_hat_F -> Future reader(z_F).
    Backward: Future inputs -> C_hat_P -> Past reader(z_P).
    z_F/z_P both come from current image+language, not from future observations.
    Actions are labels here; they never enter connector_inputs().
    """
    if min(alignment_weight, action_weight) < 0 or alignment_weight + action_weight <= 0:
        raise ValueError("nonnegative weights with at least one active objective required")
    total = None
    values = {}
    for direction, target, offset in [("forward", "future", 0), ("backward", "past", 1000003)]:
        if direction not in predictions:
            continue
        condition = predictions[direction]
        align = connector.alignment_loss(condition, batch[f"{target}_c"], target)
        action = (
            readers[target].loss(
                batch[f"{target}_actions"].detach(), batch[f"{target}_z"].detach(), condition, seed + offset,
            ) if action_weight else condition.sum() * 0
        )
        weighted = alignment_weight * align + action_weight * action
        total = weighted if total is None else total + weighted
        values[f"{direction}_alignment"] = align
        values[f"{direction}_action_fm"] = action
        values[f"{direction}_total"] = weighted
    if total is None:
        raise ValueError("no directional prediction")
    values["total"] = total
    return total, values


@torch.no_grad()
def validate_conditions(connector, readers, cache, device, batch_size=8, max_samples=256):
    """Fixed validation draws, including actual-target and train-mean controls."""
    connector.eval()
    indices = cache.valid_indices[:max_samples]
    accum = {}
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        batch = cache.batch(ids, device)
        for direction, source, target, offset in [
            ("forward", "past", "future", 0), ("backward", "future", "past", 1000003)
        ]:
            if target not in readers:
                continue
            predicted = connector.predict(batch[f"{source}_h"], batch[f"{source}_z"], direction)
            truth = batch[f"{target}_c"]
            mean = getattr(connector, f"{target}_mean").unsqueeze(0).expand_as(truth)
            measures = {
                f"{direction}_alignment": connector.alignment_loss(predicted, truth, target),
                f"{direction}_mean_alignment": connector.alignment_loss(mean, truth, target),
            }
            # Same epsilon/tau per corresponding batch, for every condition variant.
            seed = 424242 + start + offset
            for name, condition in [("predicted", predicted), ("true", truth), ("mean", mean)]:
                measures[f"{direction}_fm_{name}"] = readers[target].loss(
                    batch[f"{target}_actions"], batch[f"{target}_z"], condition, seed
                )
            for name, value in measures.items():
                accum[name] = accum.get(name, 0.) + float(value) * len(ids)
    return {"n": len(indices), **{k: v / len(indices) for k, v in accum.items()}}
