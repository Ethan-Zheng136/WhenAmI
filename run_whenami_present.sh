#!/usr/bin/env bash
# WhenAmI present-token launcher. Any nodes x NPROC_PER_NODE (target 4x16=64 PPU).
# Platform convention (same as previous launchers): WORLD_SIZE = NODE count, RANK = node rank.
# STAGE=teacher (Stage I, from A50k) | student (Stage II, needs TEACHER_CKPT = completed Stage I).
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
STAGE=${STAGE:-teacher}
RUN_ID=${RUN_ID:-present_${STAGE}_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
GLOBAL_BATCH=${GLOBAL_BATCH:-320}
STEPS=${STEPS:-50000}
EXTRA_ARGS=${EXTRA_ARGS:-}          # ablation flags, e.g. "--no-backward" / "--fusion mean" / "--raw-z-bypass"
WORLD=$((NNODES*NPROC_PER_NODE))
case "$STAGE" in teacher|student) ;; *) echo "invalid STAGE" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
(( GLOBAL_BATCH % WORLD == 0 )) || { echo "GLOBAL_BATCH $GLOBAL_BATCH not divisible by $WORLD ranks" >&2; exit 2; }
if [[ "$STAGE" == student && -z ${TEACHER_CKPT:-} ]]; then echo "set TEACHER_CKPT (Stage I step-$STEPS)" >&2; exit 2; fi
SHARDING=hybrid_zero2; (( NNODES == 1 )) && SHARDING=full
ROOT=$STORE/outputs/whenami_present/training/$RUN_ID
OUT=$ROOT/main
[[ -n ${RESUME_FROM:-} ]] && OUT=$ROOT/resume_$(basename "$RESUME_FROM")
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
args=(--module whenami_present.train --stage "$STAGE" --output "$OUT" --steps "$STEPS" --micro 1
  --global-batch "$GLOBAL_BATCH" --seed 42 --lr 1e-6 --new-lr 1e-5 --warmup 1000 --draws 2
  --sharding "$SHARDING" --activation llm --save-every 5000 --eval-every 5000 --first-check 100)
pre=(-m whenami_present.preflight --stage "$STAGE" --world "$WORLD" --global-batch "$GLOBAL_BATCH")
if [[ "$STAGE" == student ]]; then args+=(--teacher "$TEACHER_CKPT"); pre+=(--teacher "$TEACHER_CKPT"); fi
if [[ -n ${RESUME_FROM:-} ]]; then args+=(--resume-from "$RESUME_FROM"); pre+=(--resume-from "$RESUME_FROM"); fi
# shellcheck disable=SC2206
args+=($EXTRA_ARGS)
prefix=(torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29911}")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' python "${pre[@]}"; echo; printf '%q ' "${prefix[@]}" "${args[@]}"; echo; exit 0; fi
if [[ "$NODE_RANK" == 0 ]]; then python "${pre[@]}"; fi
mkdir -p "$ROOT"
"${prefix[@]}" "${args[@]}" 2>&1 | tee "$ROOT/$(basename "$OUT").node${NODE_RANK}.log"
