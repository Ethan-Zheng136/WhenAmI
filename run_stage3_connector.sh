#!/usr/bin/env bash
# Stage-3 pilot ONLY. Fixed episode-disjoint feature cache; no Stage1/2 updates.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
[[ ${WORLD_SIZE:-1} == 1 && ${RANK:-0} == 0 ]] || {
  echo "Stage-3 pilot launcher currently supports one platform node." >&2; exit 2;
}
S3_MODE=${S3_MODE:-bidirectional}
[[ $S3_MODE == forward || $S3_MODE == bidirectional ]] || { echo "Invalid S3_MODE" >&2; exit 2; }
CACHE=${CACHE:-$STORE/outputs/stage3_pilot/cache_pilot_01}
RUN_ID=${RUN_ID:-s3_${S3_MODE}_pilot_01}
OUTPUT=$STORE/outputs/stage3_pilot/$RUN_ID
[[ -f "$CACHE/manifest.json" ]] || {
  [[ ${DRY_RUN:-0} == 1 ]] || { echo "Prepare a complete endpoint cache first" >&2; exit 2; }
}
[[ ! -e "$OUTPUT" ]] || { echo "Output exists; choose new RUN_ID" >&2; exit 2; }
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cmd=(
  torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE:-1}"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29552}"
  --module stage3.train --cache "$CACHE" --output "$OUTPUT" --mode "$S3_MODE"
  --max-steps "${MAX_STEPS:-1000}" --batch-size "${PER_DEVICE_BS:-8}"
  --lr "${LR:-1e-4}" --seed "${SEED:-42}"
  --hidden "${HIDDEN:-256}" --layers "${LAYERS:-3}"
  --alignment-weight "${ALIGN_WEIGHT:-1.0}" --action-weight "${ACTION_WEIGHT:-1.0}"
  --save-every "${SAVE_INTERVAL:-500}" --eval-every "${EVAL_INTERVAL:-100}"
  --eval-limit "${EVAL_LIMIT:-256}" --precision "${PRECISION:-bf16}"
)
if [[ ${CHECKPOINT_READERS:-0} == 1 ]]; then
  cmd+=(--checkpoint-readers)
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
cd "$BASE"
mkdir -p "$(dirname "$OUTPUT")"
exec "${cmd[@]}"
