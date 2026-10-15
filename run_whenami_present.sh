#!/usr/bin/env bash
# WhenAmI present-token launcher, same conventions as run_whenami_timeline_force_{320,160}.sh:
#   platform injects WORLD_SIZE = NODE count, RANK = node rank, MASTER_ADDR/MASTER_PORT;
#   run once per node; output $STORE/outputs/whenami_present/training/$RUN_ID/<phase dir>, logs <dir>.node$RANK.log.
# Topology is free (formal target 4 nodes x16 = 64 PPU); accum = GLOBAL_BATCH/(nodes*NPROC_PER_NODE).
#   STAGE   teacher (Stage I, from A50k) | student (Stage II, needs TEACHER_CKPT = completed Stage I step-050000)
#   VARIANT present (full pipeline) | no_backward | mean_fusion | forward_fusion | no_revise | raw_z   (ablations)
#   PHASE   run (main from scratch) | admit (admission1 100 steps + admission2 resume ->120, then stop)
#           | full (admit then main resumed from admission2) | resume (needs RESUME_FROM, fresh RUN_ID, same topology)
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
STAGE=${STAGE:-teacher}
VARIANT=${VARIANT:-present}
PHASE=${PHASE:-run}
RUN_ID=${RUN_ID:-present_${STAGE}_${VARIANT}_$((${WORLD_SIZE:-1}*${NPROC_PER_NODE:-16}))_50k_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
GLOBAL_BATCH=${GLOBAL_BATCH:-320}
STEPS=${STEPS:-50000}
WORLD=$((NNODES*NPROC_PER_NODE))
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || { echo "invalid platform node RANK" >&2; exit 2; }
case "$STAGE" in teacher|student) ;; *) echo "invalid STAGE" >&2; exit 2;; esac
case "$VARIANT" in
  present) VFLAGS=();; no_backward) VFLAGS=(--no-backward);; mean_fusion) VFLAGS=(--fusion mean);;
  forward_fusion) VFLAGS=(--fusion forward);; no_revise) VFLAGS=(--no-revise);; raw_z) VFLAGS=(--raw-z-bypass);;
  *) echo "invalid VARIANT" >&2; exit 2;;
esac
case "$PHASE" in run|admit|full|resume) ;; *) echo "invalid PHASE" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
(( GLOBAL_BATCH % WORLD == 0 )) || { echo "GLOBAL_BATCH $GLOBAL_BATCH not divisible by $WORLD ranks" >&2; exit 2; }
if [[ "$STAGE" == student && -z ${TEACHER_CKPT:-} ]]; then
  echo "Set TEACHER_CKPT to completed present-token Stage I step-050000." >&2; exit 2; fi
if [[ "$PHASE" == resume && -z ${RESUME_FROM:-} ]]; then
  echo "PHASE=resume requires explicit RESUME_FROM and fresh RUN_ID." >&2; exit 2; fi
SHARDING=hybrid_zero2; (( NNODES == 1 )) && SHARDING=full   # hybrid needs >=2 nodes
ROOT=$STORE/outputs/whenami_present/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29911}")
common=(--module whenami_present.train --stage "$STAGE" --steps "$STEPS" --micro 1
  --global-batch "$GLOBAL_BATCH" --seed 42 --lr 1e-6 --new-lr 1e-5 --head-lr 1e-4 --warmup 1000 --draws 2
  --sharding "$SHARDING" --activation llm --save-every 5000 --eval-every 5000 --first-check 100
  "${VFLAGS[@]}")
# shellcheck disable=SC2206
common+=(${EXTRA_ARGS:-})          # engineering/override only, e.g. "--stop-after 5 --eval-max 2"
preflight=(-m whenami_present.preflight --stage "$STAGE" --world "$WORLD" --global-batch "$GLOBAL_BATCH")
if [[ "$STAGE" == student ]]; then common+=(--teacher "$TEACHER_CKPT"); preflight+=(--teacher "$TEACHER_CKPT"); fi
if [[ "$PHASE" == resume ]]; then preflight+=(--resume-from "$RESUME_FROM"); fi
A1=(--output "$ROOT/admission1" --stop-after 100)
A2=(--output "$ROOT/admission2" --stop-after 120 --resume-from "$ROOT/admission1/checkpoints/step-000100")
case "$PHASE" in
  run) MAIN=(--output "$ROOT/main");;
  full) MAIN=(--output "$ROOT/main" --resume-from "$ROOT/admission2/checkpoints/step-000120");;
  resume) MAIN=(--output "$ROOT/main" --resume-from "$RESUME_FROM");;
  admit) MAIN=();;
esac
if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "# world=$WORLD nodes=$NNODES accum=$((GLOBAL_BATCH/WORLD)) sharding=$SHARDING root=$ROOT"
  printf '%q ' python "${preflight[@]}"; printf '\n'
  if [[ "$PHASE" == admit || "$PHASE" == full ]]; then
    printf '%q ' "${prefix[@]}" "${common[@]}" "${A1[@]}"; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" "${A2[@]}"; printf '\n'
  fi
  if (( ${#MAIN[@]} )); then printf '%q ' "${prefix[@]}" "${common[@]}" "${MAIN[@]}"; printf '\n'; fi
  exit 0
fi
# Every node verifies sources/data/teacher; same as old launchers (rank0 full, others local).
python "${preflight[@]}"
if [[ "$NODE_RANK" == 0 && -e "$ROOT/main" && "$PHASE" != admit ]]; then
  echo "$ROOT/main exists; use a fresh RUN_ID (resume writes a NEW run's main)" >&2; exit 2; fi
mkdir -p "$ROOT"
if [[ "$PHASE" == admit || "$PHASE" == full ]]; then
  "${prefix[@]}" "${common[@]}" "${A1[@]}" 2>&1 | tee "$ROOT/admission1.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" "${A2[@]}" 2>&1 | tee "$ROOT/admission2.node${NODE_RANK}.log"
  python - "$ROOT" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
for d, end in (("admission1", 100), ("admission2", 120)):
    meta = json.loads((root/d/"checkpoints"/f"step-{end:06d}"/"COMMITTED.json").read_text())
    rows = [json.loads(l) for l in (root/d/"train.jsonl").read_text().splitlines()]
    assert meta["complete"] and rows[-1]["step"] == end and all(math.isfinite(r["total"]) for r in rows)
    assert all(v > 0 for r in rows if r["gradients"] for v in r["gradients"].values())
    assert json.loads((root/d/"development"/f"step-{end:06d}"/"RESULT.json").read_text())["complete"]
print("PRESENT_ADMISSION_PASS", root, flush=True)
PY
fi
if (( ${#MAIN[@]} )); then
  "${prefix[@]}" "${common[@]}" "${MAIN[@]}" 2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
