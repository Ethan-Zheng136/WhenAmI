# Timeline-A training integration

Keeps `whenami_timeline_a/` model code and all prior smoke checkpoints immutable.
Full explanation and pipeline figures:
`../WHENAMI_TIMELINE_A_TRAIN_AND_PIPELINES_20261004.md`.

## Data

Existing OXE27 raw train index, no new training preprocessing.
Development: `outputs/whenami_timeline_a/dev27_16ep_01`,410 existing-dev episodes,
27 domains,3 real anchors/episode, zero overlap with actual train index.
Base pretraining may have seen these episodes; this is not sealed test.

## Admission / full train

Request2 physical nodes,16PPUs/node. The platform supplies node WORLD_SIZE/RANK.

```bash
TL_MODE=timeline PHASE=admit NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_a_32.sh
```

It trains1–100, saves/evaluates, launches a new process, resumes101–120,
and stops. Same settings allow later continuation to10k:

```bash
TL_MODE=timeline PHASE=full NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_a_32.sh
```

`full` is refused until that mode's32-PPU admission completes. `TL_MODE` may also
be `pooled` or `independent`; each gets its own default run directory.
Three simultaneous modes require96PPUs; no automatic submission is performed.

## Recipe

Global64 trajectories,3anchors,2 FM draws; local QFormer/full DiT LR1e-5,
temporal LR1e-4, constant;10k total updates including admission.
Save1000/evaluate500; also evaluate initial and segment-final models.
Development reports full-visible and middle-hidden observations, true Gaussian
sampled-action errors and FM losses, per-domain and macro. Tqdm/ETA included.

No Phase-B student or closed-loop deployment is produced by Phase A alone.
The pooled condition control is not an exact implementation of WoG.
The independent routing arm intentionally denies cross-node evidence; it is not
an equal-information strong recurrent memory baseline.

Local integration verification is recorded in `LOCAL_VERIFICATION.json`.
8 new integration tests pass; actual train/save/dev-eval/new-process-resume passes
on a single PPU, with exact final model/Adam and matching dev results.
Actual32-PPU admission has NOT yet run.
Duplicate continuous-run checkpoint files were cleaned after equality checking;
the resumed final4 checkpoint and all comparison logs remain.
