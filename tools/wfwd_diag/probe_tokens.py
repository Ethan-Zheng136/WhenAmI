"""Offline per-token fusion probe on a committed present-teacher checkpoint (read-only; no training).
torchrun --nproc_per_node 1 -m probe_tokens --ckpt <.../checkpoints/step-XXXXXX> --out <dir> [--batches 200]
Writes diag_tokens.jsonl in the same format as train --diag-tokens (no grads); analyse with analyze.py."""
import argparse, json, os
from datetime import timedelta
from pathlib import Path
import torch, torch.distributed as dist
from whenami_present.train import arguments, present_config
from whenami_present.model import build_policy
from whenami_present.runtime import topology, wrap
from whenami_present.loading import load_weights
from whenami_present.data import Inputs, batch
from whenami_present.losses import token_record

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
p.add_argument("--batches", type=int, default=200); p.add_argument("--micro", type=int, default=2)
p.add_argument("--seed", type=int, default=4242)
q = p.parse_args()
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))); torch.cuda.set_device(device)
dist.init_process_group("nccl", timeout=timedelta(minutes=40))
q.out.mkdir(parents=True, exist_ok=False)
a = arguments().parse_args(["--stage", "teacher", "--output", str(q.out)])
core, visual, transform, tokenizer, prompt = build_policy(False, present_config(a), 42)
strategy, groups = topology("full", device)
model = wrap(core, device, strategy, groups, "none")
loaded = load_weights(model, q.ckpt, False, True, groups[0] if groups else None)
visual = visual.to(device).requires_grad_(False).eval()
model.eval(); data = Inputs()
with (q.out/"diag_tokens.jsonl").open("w") as f, torch.no_grad():
    for s in range(1, q.batches+1):
        cases = data.batch(q.micro, q.seed, s, 0, 0)
        evidence, labels = batch(cases, visual, transform, tokenizer, prompt, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = model(evidence=evidence, presents=(0, 1, 2),
                        future_keep=torch.ones(len(cases), dtype=torch.bool, device=device))
        outs = [res["present"][c] for c in (0, 1, 2)]
        rec = token_record(outs, [res["hindsight"][:, c:c+1] for c in (0, 1, 2)])
        f.write(json.dumps({"step": s, "micro": 0, "hide": False, "rec": rec, "grads": []})+"\n")
        if s % 20 == 0: print("PROBE", s, flush=True)
print("PROBE_DONE", json.dumps({"ckpt": str(q.ckpt)}), flush=True)
dist.destroy_process_group()
