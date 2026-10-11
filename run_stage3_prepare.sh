#!/usr/bin/env bash
# One node: fresh RLDS subset -> frozen endpoint cache. No model updates.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
[[ ${WORLD_SIZE:-1} == 1 && ${RANK:-0} == 0 ]] || {
  echo "Stage-3 cache preparation supports one platform node." >&2; exit 2;
}
CACHE=${CACHE:-$STORE/outputs/stage3_pilot/cache_pilot_01}
PAST_CKPT=${PAST_CKPT:-$STORE/outputs/stage2_16rank_probe/past_s2_16rank_20node_from5k_01--image_aug/checkpoints/step-015000-epoch-01-loss=0.2112.pt}
FUTURE_CKPT=${FUTURE_CKPT:-$STORE/outputs/stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt}
[[ ! -e "$CACHE" ]] || { echo "Cache exists; choose a fresh CACHE path" >&2; exit 2; }
[[ -f "$PAST_CKPT" && -f "$FUTURE_CKPT" ]] || { echo "Missing endpoint checkpoint" >&2; exit 2; }
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MALLOC_ARENA_MAX=2
read -r -a datasets <<< "${DATASETS:-fractal20220817_data bridge_orig}"
sample=(
  python3 -u -m stage3.prepare samples --output "$CACHE" --past "$PAST_CKPT" --future "$FUTURE_CKPT"
  --limit "${N:-512}" --per-episode "${PER_EPISODE:-4}" --datasets "${datasets[@]}" --seed "${SEED:-42}"
)
encode=(
  torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE:-16}"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29551}"
  --module stage3.prepare encode --cache "$CACHE"
)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${sample[@]}"; printf '\n'
  printf '%q ' "${encode[@]}"; printf '\n'; exit 0
fi
cd "$BASE"
mkdir -p "$(dirname "$CACHE")"
"${sample[@]}" 2>&1 | tee "$CACHE.prepare.log"
"${encode[@]}" 2>&1 | tee "$CACHE.encode.log"
