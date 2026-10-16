"""Compare actual uninterrupted/resumed next update and selected DCP tensors.

This proves local engineering restart for tested tensors/losses, not bitwise
identity of every 8B tensor or formal multi-node FP32-state HSDP.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from stage3_oxe_eval.prepare import write_json, sha

PROBES = (
    "node_cognition.readout.bias",
    "timeline.output.1.bias",
    "local_encoder.v_proj.bias",
    "reader.final_layer.linear.weight",
    "reader.final_layer.linear.bias",
    "timeline.query_position",
)


def read_probes(checkpoint):
    reader = dcp.FileSystemReader(Path(checkpoint)/"model_dcp")
    metadata = reader.read_metadata().state_dict_metadata
    values = {}
    for name in PROBES:
        key = "model."+name
        if key not in metadata:
            continue
        info = metadata[key]
        values[name] = torch.empty(info.size, dtype=info.properties.dtype)
    if len(values) < 4:
        raise ValueError("restart probes absent")
    dcp.load({"model": values}, storage_reader=reader, no_dist=True)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uninterrupted", type=Path, required=True)
    p.add_argument("--resumed", type=Path, required=True)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    u, r = a.uninterrupted, a.resumed
    rows = []
    for path in (u, r):
        matching = [json.loads(line) for line in (path/"train.jsonl").read_text().splitlines()
                    if json.loads(line)["step"] == a.step]
        if len(matching) != 1:
            raise ValueError("missing next update")
        rows.append(matching[0])
    keys = ("action_fm", "current_alignment", "cognition_alignment", "revision", "total")
    if any(rows[0][key] != rows[1][key] for key in keys) or rows[0]["lr"] != rows[1]["lr"]:
        raise ValueError("actual next update differs")
    checkpoints = [p/"checkpoints"/f"step-{a.step:06d}" for p in (u, r)]
    meta = [json.loads((p/"COMMITTED.json").read_text()) for p in checkpoints]
    if meta[0]["recipe"] != meta[1]["recipe"]:
        raise ValueError("restart recipe differs")
    tensors = [read_probes(path) for path in checkpoints]
    if set(tensors[0]) != set(tensors[1]):
        raise ValueError("restart probe coverage")
    for key in tensors[0]:
        if not torch.equal(tensors[0][key], tensors[1][key]):
            raise ValueError("next update tensor differs: "+key)
    resume = json.loads((r/"RESUME_VERIFIED.json").read_text())
    if not resume["complete"] or resume["restored_step"] != a.step-1:
        raise ValueError("new-process restore proof absent")
    write_json(a.output, {"complete": True, "engineering_only": True,
        "step": a.step, "losses_bitwise_equal": True, "probed_tensors_bitwise_equal": list(tensors[0]),
        "probed_elements": sum(x.numel() for x in tensors[0].values()),
        "uninterrupted_committed_sha256": sha(checkpoints[0]/"COMMITTED.json"),
        "resumed_committed_sha256": sha(checkpoints[1]/"COMMITTED.json"),
        "scope": "actual single-card fullAdamRNG restart; selected tensors/losses, not allmodel or320rank",
        "optimizer_precision": meta[0]["recipe"]["optimizer_precision"]})
    print("ACTUAL_LOCAL_RESTART_PASS", a.output, flush=True)


if __name__ == "__main__":
    main()
