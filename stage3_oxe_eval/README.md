# OXE27 offline comparison

Independent of production training and the other server's NVIDIA/SIMPLER work.
Full protocol: `PLAN.md`. Created 2026-10-03.

## Prepared data, actually completed

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_oxe_eval/dev27_64ep_01/
```

- 27 domains, 1,348 unique existing-DEV episodes, 2,694 windows.
- Actual Stage3 training-index overlap: **0**.
- Target64 episodes/domain, up to2 windows/episode. Small domains keep all eligible
  existing-dev cases; four domains have fewer than16 episodes.
- Original pretraining exposure is not excluded. This is development evaluation,
  not sealed-test or closed-loop task success.
- Only causal history/current pixels are retained; action targets are separately
  passed to the scorer, never to model prediction.
- Manifest pins the actual train-index SHA256 to the checkpoint's training recipe.

## Models/protocol

- Local WoG-A `50k-1229.pt`, current-only.
- Original cumulative20k Past, history/current.
- Full Stage3 committed35k, history/current.
- Same three actual Gaussian noise arrays per observation, BF16 policy parameters,
  FP32 buffers/integration,9 Euler intervals,16x7 actions.
- Masked MSE/MAE/first-action/gripper + per-domain raw r/R²/RMSE.
- Episode-equal within domain, domain-macro primary. All27 domains reported.
- Inference-seed variance is not training-seed variance.
- Frozen history visual encoder precision is unchanged from the trained path.
- No model weights exported/copied; no optimizer or rank RNG loaded.

## Commands

Environment:

```bash
cd /home/zhengguantian/WoG_fresh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/pfs/zhengguantian/WoG_v2/pylibs-train:/home/zhengguantian/WoG_fresh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/pfs/zhengguantian/WoG_v2/hf_home
ROOT=/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_oxe_eval
```

Tests:

```bash
python -m unittest stage3_oxe_eval.test_protocol -v
```

Evaluation (one model at a time; do not start a duplicate while it is running):

```bash
python -m stage3_oxe_eval.evaluate \
  --kind stage3 \
  --data "$ROOT/dev27_64ep_01" \
  --output "$ROOT/compare35k_01/stage3" --resume
```

Use `--kind wog_a`/`past` with the matching output subdirectory for the baselines.
Per-episode atomic files survive interruption; resume validates identity/protocol
and skips completed episodes. Source-code changes invalidate a run's protocol.

Independent aggregation only after all models finish:

```bash
python -m stage3_oxe_eval.summarize \
  --data "$ROOT/dev27_64ep_01" \
  --runs "$ROOT/compare35k_01" \
  --output "$ROOT/compare35k_01/summary"
```

`finish_local.sh` is already launched for this specific local run: it waits for
baseline admission, runs both baselines, then waits for the separately started
Stage3 job and summarizes. Do not launch a second copy.

## Check progress/results

```text
outputs/stage3_oxe_eval/
  prepare64_01.log
  protocol_tests.log
  admit_stage3_02/RESULT.json
  admit_wog_a_01/RESULT.json
  admit_past_01/RESULT.json          # exists only after successful admission
  evaluate_stage3_01.log
  evaluate_wog_a_01.log
  evaluate_past_01.log
  finish_local_02.log
  compare35k_01/
    {stage3,wog_a,past}/progress-rank0000.json
    {stage3,wog_a,past}/per_episode/
    {stage3,wog_a,past}/COMPLETE-rank0000-of0001.json
    summary/RESULT.json
    summary/RESULTS.md
```

Admission's two old Bridge/Fractal samples are **loader replay fixtures only**;
they are not this experiment's performance dataset.

Stage3 initial loader attempt using an empty DCP planner failed with KeyError;
no predictions were produced. The corrected loader preallocates an independent
CPU state, checks all1664 keys/shapes/dtypes, strictly assigns weights, checks
finite tensors, then casts policy parameters. No writes to source checkpoints.

Stage3 replay passed the tolerance set in admission code. Normalized action RMSE
versus saved distributed predictions was0.002477 and0.001339 for the two fixtures;
maximum errors0.008138 and0.003472. This is **not bitwise equality**.
Direct cognition versus the original generation interface was identical.
The new comparison recomputes every model under its common local protocol;
do not splice old distributed numbers into the new table.

The separate remote inference release is NOT implemented by this evaluator:
`Policy` still uses local warm-start assets to construct the architecture before
strictly replacing all Stage3 inference tensors.

The first detached shell attempt did not remain alive and produced no baseline
predictions. The active baseline/summary sequence is the attached exec session
writing `finish_local_02.log`; this does not change any scientific protocol.
