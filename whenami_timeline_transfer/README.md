# Parallel preparation for Phase B

This package is independent of the actively training Phase-A source tree. It
does not modify its trainer, checkpoint, model classes, or resume recipe.

## What is implemented

```text
Pinned A5k teacher:
full three-node observations -> frozen A model -> current privileged condition
                                                        |
                                                        | auxiliary target only
                                                        v
Online student:
physical observed prefix -> same causal model -> current condition -> action DiT
                                      |
                                      +-> auxiliary condition predictor -> loss
```

Future prediction output is NOT fed into the action readout. The alignment loss
may improve shared student representations, but target accuracy alone does not
establish better actions or closed-loop control.

- `contracts.py`: physically truncate pixels, features, times, and per-frame
  language; remove future-dependent padding. Student accepts only `Prefix`.
- `model.py`: current-decision causal student, detached frozen-teacher target,
  action FM and optional cosine alignment. Same action path for action-only
  control and aligned student. No new algorithm claim.
- `loader.py`: strict read-only A5k DCP model load, no optimizer resume.
- `test_transfer.py`:7 CPU tests covering prefix boundaries, teacher detach,
  live gradients, auxiliary/action-path separation.
- `smoke.py`: real TRAIN input and loaded teacher/student forward/backward
  admission, one SGD engineering update. It is NOT fullB training.

## What is NOT implemented/proven

- Final B large-training runner and full sharded Adam/save/resume.
- Full multi-node/horizon distillation.
- Final teacher selection or causal-student effectiveness.
- NVIDIA deployment or closed-loop integration (other server's responsibility).
- Novelty of decoupled auxiliary supervision.

Teacher is pinned at5000 only to develop the interface while A continues;
we do not overwrite/update its files. Final experiments must freeze a declared
teacher and compare action-only vs aligned students from the same initialization.

## Commands

```bash
cd /home/zhengguantian/WoG_fresh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/pfs/zhengguantian/WoG_v2/pylibs-train:/home/zhengguantian/WoG_fresh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/pfs/zhengguantian/WoG_v2/hf_home
python -m pytest -q whenami_timeline_transfer/test_transfer.py
```

Smoke requires a fresh output directory and a free local PPU:

```bash
python -m whenami_timeline_transfer.smoke --output <new-output>
```

Actual records and intermediate A result audit:

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_transfer/
  A_PROGRESS_AUDIT.json
  test_transfer.log
  real_interface_smoke_01.log
  real_interface_smoke_01/RESULT.json   # only exists after successful completion
```
