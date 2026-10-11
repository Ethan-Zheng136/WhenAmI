#!/usr/bin/env bash
# User submits 1 node x16 cards. One independent student experiment per card.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
CACHE=${CACHE:-$STORE/outputs/stage3_reader_study/data_01/train_dev}
READERS=${READERS:-$STORE/outputs/stage3_reader_study/runs_01}
RUNS=${RUNS:-$STORE/outputs/stage3_causal_student/runs_20k_01}
[[ ${WORLD_SIZE:-1} == 1 && ${RANK:-0} == 0 && ${NPROC_PER_NODE:-16} == 16 ]] || {
  echo "Submit ONE node with 16 cards. These are independent experiments, not one DDP model." >&2; exit 2;
}
if [[ ${RESUME:-0} != 1 && -e "$RUNS" ]]; then
  echo "RUNS exists. Use RESUME=1 only for interrupted runs, or choose a fresh RUNS." >&2; exit 2
fi
[[ -f "$CACHE/manifest.json" && ! -f "$CACHE/SEALED_DO_NOT_EVALUATE" ]] || {
  echo "Requires the complete non-sealed train_dev feature cache." >&2; exit 2;
}
for seed in 42 43 44 45; do
  for evidence in mean future; do
    [[ -f "$READERS/adaptive_${evidence}_seed${seed}/checkpoints/step-005000.pt" ]] || {
      echo "Missing matched5k teacher/base: $evidence seed$seed" >&2; exit 2;
    }
  done
done
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cmd=(
  torchrun --nnodes 1 --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29571}"
  --module stage3.causal_student_sweep --cache "$CACHE" --readers "$READERS" --output "$RUNS"
  --max-steps "${MAX_STEPS:-20000}" --batch-size "${BATCH_SIZE:-32}" --fm-draws "${FM_DRAWS:-4}"
  --lr "${LR:-1e-4}" --warmup "${WARMUP:-500}"
)
if [[ ${RESUME:-0} == 1 ]]; then cmd+=(--resume); fi
compare=(python -m stage3.compare_causal_student --runs "$RUNS" --output "$RUNS/development_comparison.json")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  printf '%q ' "${compare[@]}"; printf '\n'; exit 0
fi
cd "$BASE"
mkdir -p "$(dirname "$RUNS")"
"${cmd[@]}" 2>&1 | tee -a "${RUNS}.launcher.log"
if [[ ! -f "$RUNS/development_comparison.json" ]]; then "${compare[@]}"; fi
echo "20k causal students complete; sealed_test untouched."
