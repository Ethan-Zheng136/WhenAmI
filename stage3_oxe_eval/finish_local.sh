#!/usr/bin/env bash
# Finish the two baselines after their admissions; Stage3 is already launched separately.
set -euo pipefail
cd /home/zhengguantian/WoG_fresh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/pfs/zhengguantian/WoG_v2/pylibs-train:/home/zhengguantian/WoG_fresh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/pfs/zhengguantian/WoG_v2/hf_home
ROOT=/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_oxe_eval
trap 'echo "LOCAL_COMPARE_FAILED line=$LINENO utc=$(date -u +%FT%TZ)" >&2' ERR

for kind in wog_a past; do
  while [[ ! -f "$ROOT/admit_${kind}_01/RESULT.json" ]]; do
    if [[ -f "$ROOT/admit_${kind}_01.log" ]] && grep -q '^Traceback (most recent call last):' "$ROOT/admit_${kind}_01.log"; then
      echo "Admission failed for $kind; preserve logs and fix before evaluation." >&2
      exit 1
    fi
    sleep 10
  done
  python -m stage3_oxe_eval.evaluate --kind "$kind" \
    --data "$ROOT/dev27_64ep_01" --output "$ROOT/compare35k_01/$kind" --resume \
    > "$ROOT/evaluate_${kind}_01.log" 2>&1
done

while [[ ! -f "$ROOT/compare35k_01/stage3/COMPLETE-rank0000-of0001.json" ]]; do
  if grep -q '^Traceback (most recent call last):' "$ROOT/evaluate_stage3_01.log"; then
    echo "Stage3 evaluation failed; summary withheld." >&2
    exit 1
  fi
  sleep 10
done
python -m stage3_oxe_eval.summarize --data "$ROOT/dev27_64ep_01" \
  --runs "$ROOT/compare35k_01" --output "$ROOT/compare35k_01/summary" \
  > "$ROOT/summary35k_01.log" 2>&1
echo "LOCAL_COMPARE_COMPLETE utc=$(date -u +%FT%TZ)"
