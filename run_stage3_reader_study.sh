#!/usr/bin/env bash
# 1 node x16 cards: feature extraction then 16 independent factorial experiments.
# User submits the cluster job. This script never touches sealed_test.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
ROOT=${STUDY_ROOT:-$STORE/outputs/stage3_reader_study}
DATA_ROOT=${DATA_ROOT:-$ROOT/data_01}
CACHE=$DATA_ROOT/train_dev
RUNS=${RUNS:-$ROOT/runs_01}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ ${WORLD_SIZE:-1} == 1 && ${RANK:-0} == 0 && $NPROC_PER_NODE == 16 ]] || {
  echo "Submit exactly one node with 16 cards; workers are independent experiments." >&2; exit 2;
}
[[ ! -e "$RUNS" ]] || { echo "RUNS exists: choose a fresh path; never overwrite experiments" >&2; exit 2; }
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false WAI_BOUNDED_INPUT=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
encode=(
  torchrun --nnodes 1 --nproc_per_node 16 --master_addr "${MASTER_ADDR:-127.0.0.1}"
  --master_port "${MASTER_PORT:-29561}" --module stage3.prepare encode --cache "$CACHE"
)
train=(
  torchrun --nnodes 1 --nproc_per_node 16 --master_addr "${MASTER_ADDR:-127.0.0.1}"
  --master_port "${SWEEP_MASTER_PORT:-29562}" --module stage3.reader_study_sweep
  --cache "$CACHE" --output "$RUNS" --max-steps "${MAX_STEPS:-5000}"
  --batch-size "${BATCH_SIZE:-32}" --lr "${LR:-1e-4}" --adapter-rank "${ADAPTER_RANK:-32}"
)
compare=(python -m stage3.compare_reader_study --runs "$RUNS" --output "$RUNS/development_comparison.json")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${encode[@]}"; printf '\n'
  printf '%q ' "${train[@]}"; printf '\n'
  printf '%q ' "${compare[@]}"; printf '\n'
  exit 0
fi
cd "$BASE"
[[ -f "$DATA_ROOT/split_audit.json" && -f "$CACHE/samples_manifest.json" ]] || {
  echo "Fresh raw data still preparing or incomplete: $DATA_ROOT" >&2; exit 2;
}
[[ ! -f "$CACHE/SEALED_DO_NOT_EVALUATE" ]] || { echo "Refusing sealed test" >&2; exit 2; }
if [[ ! -f "$CACHE/manifest.json" ]]; then
  "${encode[@]}" 2>&1 | tee "$ROOT/reader_study_encode_01.log"
fi
"${train[@]}" 2>&1 | tee "$ROOT/reader_study_sweep_01.log"
"${compare[@]}"
echo "COMPLETE. Only development evaluated; sealed_test untouched."
