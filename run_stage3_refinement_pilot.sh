#!/usr/bin/env bash
# Explicit local single-device pilot, never a cluster job submission.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
[[ ${WORLD_SIZE:-1} == 1 ]] || { echo "Local single-device pilot only" >&2; exit 2; }
ROOT=${OUTPUT_ROOT:-$STORE/outputs/stage3_refine}
CACHE=${CACHE:-$STORE/outputs/stage3_pilot/cache_pilot_01}
export PYTHONPATH="$STORE/pylibs-train:$BASE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home"
cd "$BASE"
mkdir -p "$ROOT"
for name in rk_adapter_01 rk_predict_01 rk_teacher_01 rk_teacher_roll4_01; do
  [[ ! -e "$ROOT/$name" && ! -e "$ROOT/${name}_eval" ]] || {
    echo "Existing output $name; choose a fresh OUTPUT_ROOT" >&2; exit 2;
  }
done
for name in rk_adapter_01 rk_predict_01 rk_teacher_01 rk_teacher_roll4_01; do
  case "$name" in
    rk_adapter_01) role=adapter; extra=() ;;
    rk_predict_01) role=predict; extra=() ;;
    rk_teacher_01) role=teacher; extra=() ;;
    rk_teacher_roll4_01) role=teacher; extra=(--rollout-steps 4 --rollout-weight .25) ;;
  esac
  echo "TRAIN_START $(date -u +%FT%TZ) $name"
  python -m stage3.train_refinement \
    --cache "$CACHE" --output "$ROOT/$name" --mode "$role" \
    --max-steps 1000 --batch-size 8 --lr 1e-4 --seed 42 --alignment-weight .1 \
    "${extra[@]}" > "$ROOT/$name.log" 2>&1
  echo "TRAIN_DONE $(date -u +%FT%TZ) $name"
done
for name in rk_adapter_01 rk_predict_01 rk_teacher_01 rk_teacher_roll4_01; do
  echo "EVAL_START $(date -u +%FT%TZ) $name"
  python -m stage3.evaluate_refinement \
    --checkpoint "$ROOT/$name/checkpoints/step-001000.pt" --output "$ROOT/${name}_eval" \
    --seed 42 > "$ROOT/${name}_eval.log" 2>&1
  echo "EVAL_DONE $(date -u +%FT%TZ) $name"
done
python -m stage3.compare_refinement --root "$ROOT" --output "$ROOT/comparison_01.json" \
  > "$ROOT/comparison_01.log" 2>&1
echo "ALL_DONE $(date -u +%FT%TZ)"
