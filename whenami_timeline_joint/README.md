# Full-VLM Timeline-A joint teacher

This isolated package implements the large model discussed on2026-10-05.
It is NOT the earlier32-PPU frozen-VLM teacher and does NOT resume its optimizer.

## Actual trainable path

```text
raw frame nodes + instructions -> TRAINABLE VLM -> per-node z
raw frame nodes -> frozen standalone DINO/Wan -> TRAINABLE local QFormer -> H
H,z,actual time addresses,visibility -> TRAINABLE temporal condition model
node conditions,z + noisy actions -> TRAINABLE full original DiT -> action FM
```

All VLM vision/projector/LLM parameters are inside FSDP and receive action gradients.
No cached/detached VLM features are used in training or development evaluation.
Hidden images are blocked at the VLM pixel input and local/source/reader branches.
Different target evidence cutoffs do not relay through mutable target tokens.

This learns privileged Phase-A action conditions, not a physical transition model
or deployed Phase-B policy. No claim of novel architecture or performance is made
from the local smoke.

## Modes

- `timeline`: preserve visible nodes as individually time-addressed source tokens.
- `pooled_query`: compress the same sources to a pooled bank, but retain each
  target's own local base and target-time query. It does not force identical
  conditions at every target. Same parameters/source visibility/initialization/
  attention shapes/supervision budget as timeline.

This control is stronger than the old frozen-VLM `pooled`; it is not a published
WoG reimplementation or a substitute for all eventual scientific baselines.

## Allocation

Two independent jobs, **20nodes x16PPUs each** =320PPUs per arm,640 total.
Do not submit one40-node job to this launcher.

Per arm:50k updates,global320 trajectories,3anchors/trajectory (960 observations),
2 flow draws,pretrained parameters LR1e-5/new temporal LR1e-4,1000warmup/cosine.
Micro1,hybrid ZeRO2,LLM activation checkpointing,save5k/eval2.5k.
Constant no-augmentation data protocol in both arms.
Reuses complete OXE27 training index and410 independent dev episodes; no download.

## One submission with mandatory admission before automatic continuation

`PHASE=run` first performs the mandatory actual320-rank admission below. Only
after100/120 full-model save/eval/new-process restore pass does it automatically
launch the remaining updates to50k. This avoids a second allocation/submit while
still refusing to continue after an engineering failure.

```bash
JOINT_MODE=timeline PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh

# Separate20-node allocation for the comparator:
JOINT_MODE=pooled_query PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh
```

## Alternatively, stop after admission for manual inspection

```bash
JOINT_MODE=timeline PHASE=admit NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh
```

Second independent320PPU job:

```bash
JOINT_MODE=pooled_query PHASE=admit NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh
```

The platform provides WORLD_SIZE=20 (physical node count before torchrun),
RANK and MASTER_ADDR. Do not set WORLD_SIZE=320.

Admission executes:

1. Tiny actual320-rank hybrid topology/model-Adam-RNG restore proof.
2. Full raw-image8B-class model100 updates,save and27-domain development.
3. New process resumes all optimizer/RNG to120,save/development,then stops.

The full50k scientific schedule is used from update1. Local proof does not certify
actual320-rank optimizer/memory/speed; admission is the gate.

## Continue only after actual admission passed

```bash
JOINT_MODE=timeline PHASE=full NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh
```

Use `JOINT_MODE=pooled_query` for the other arm. Continues from120, no reset.
`PHASE=full` checks committed checkpoints, exact recipe, all-module gradients,
new-process resume and updated-VLM27-domain evaluation. An existing main output
is rejected. Do not overwrite/relaunch into failed directories.

Output roots:

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_joint/training/
  timeline_joint_timeline_320_50k_01/
  timeline_joint_pooled_query_320_50k_01/
```

Each has `topology_probe/`, `admission1/`, `admission2/`, then `main/`.
Each segment records `progress.json`, `train.jsonl`, `recipe.json`,
`checkpoints/step-*/COMMITTED.json`, `development/step-*/RESULT.json`.

## Verification boundaries

`LOCAL_VERIFICATION.json` is generated only after the local tests actually pass.
Source hashes include this package and all reused model/trainer dependencies.
Do not edit Python code while a submitted job is using this recipe.

- CPU tests: raw hidden-input invariance, causal cutoff, full VLM gradients,
  same initial weights, distinct target-time queries, decoder-only FSDP API.
- CPU4-rank emulated two-shard-group joint graph: model/Adam/RNG save/reload,
  same next update for both arms.
- Actual full pretrained model FSDP backward on onePPU: all six learned groups,
  mask behavior, memory. Single rank necessarily falls back to NO_SHARD.
- A96GiB PPU cannot hold allFP32 master weights,gradients,and fullAdam states of
  this8B-class model. Local smoke does NOT claim a full8B Adam restart passed;
  that is what actual320-rank admission tests.

Prior32-rank frozen-VLM experiments were all independently audited at120 steps.
None of their code, weights or logs is modified.
