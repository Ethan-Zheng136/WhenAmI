#!/usr/bin/env bash
# ONE node. Prepare real standardized RLDS once, then evaluate across GPUs.
# No training or checkpoint modification. Keep old compare outputs untouched.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
if [[ ${WORLD_SIZE:-1} != 1 || ${RANK:-0} != 0 ]]; then
  echo "This evaluation launcher requires ONE platform node." >&2; exit 2
fi
RUN_ID=${RUN_ID:-past_s2_20k_eval_01}
OUT=$STORE/outputs/past_eval/$RUN_ID
PAST_CKPT=${PAST_CKPT:-$STORE/outputs/stage2_16rank_probe/past_s2_16rank_20node_from5k_01--image_aug/checkpoints/step-015000-epoch-01-loss=0.2112.pt}
FUTURE_CKPT=${FUTURE_CKPT:-$STORE/outputs/stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt}
[[ -f "$PAST_CKPT" && -f "$FUTURE_CKPT" ]] || { echo "Missing checkpoint" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output already exists; choose a NEW RUN_ID: $OUT" >&2; exit 2; }
export PYTHONPATH="$STORE/pylibs-train:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2
read -r -a datasets <<< "${DATASETS:-fractal20220817_data bridge_orig}"
read -r -a variants <<< "${VARIANTS:-past past_static future_oracle}"
read -r -a seeds <<< "${NOISE_SEEDS:-42}"
prepare=(
  python3 -u "$BASE/evaluation/eval_past.py" prepare
  --past "$PAST_CKPT" --future "$FUTURE_CKPT" --output "$OUT/samples"
  --datasets "${datasets[@]}" --limit "${N:-256}" --per-episode "${PER_EPISODE:-4}"
)
evaluate=(
  torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29533}"
  "$BASE/evaluation/eval_past.py" evaluate --samples "$OUT/samples"
  --output "$OUT/results.json" --variants "${variants[@]}" --seeds "${seeds[@]}"
)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${prepare[@]}"; printf '\n'
  printf '%q ' "${evaluate[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$OUT"
"${prepare[@]}" 2>&1 | tee "$OUT/prepare.log"
"${evaluate[@]}" 2>&1 | tee "$OUT/evaluate.log"
