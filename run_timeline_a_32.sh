#!/usr/bin/env bash
# Admission is a real32-PPU trial; full training is gated by its committed results.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
TL_MODE=${TL_MODE:-timeline}
PHASE=${PHASE:-admit}
RUN_ID=${RUN_ID:-timeline_a_${TL_MODE}_32_10k_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ "$NNODES" == 2 && "$NPROC_PER_NODE" == 16 ]] || {
  echo "Requires2 physical nodes x16 PPUs. WORLD_SIZE is node count before torchrun." >&2; exit 2;
}
case "$TL_MODE" in pooled|independent|timeline) ;; *) echo "Invalid TL_MODE" >&2; exit 2;; esac
case "$PHASE" in admit|full) ;; *) echo "PHASE must be admit or full" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe RUN_ID" >&2; exit 2; }
ROOT=$STORE/outputs/whenami_timeline_a/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29713}")
common=(--module whenami_timeline_a_training.train --mode "$TL_MODE"
  --steps 10000 --micro 2 --nodes 3 --seed 42
  --save-every 1000 --eval-every 500 --first-check 100)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  if [[ "$PHASE" == admit ]]; then
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 100; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 120 \
      --resume-from "$ROOT/admission1/checkpoints/step-000100"; printf '\n'
  else
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
      --resume-from "$ROOT/admission2/checkpoints/step-000120"; printf '\n'
  fi
  exit 0
fi
python -m whenami_timeline_a_training.preflight --root "$ROOT" --mode "$TL_MODE" --phase "$PHASE"
mkdir -p "$ROOT"
if [[ "$PHASE" == admit ]]; then
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 100 \
    2>&1 | tee "$ROOT/admission1.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 120 \
    --resume-from "$ROOT/admission1/checkpoints/step-000100" \
    2>&1 | tee "$ROOT/admission2.node${NODE_RANK}.log"
  python - "$ROOT" "$TL_MODE" <<'PY'
import sys
from whenami_timeline_a_training.preflight import check_admission
print("TIMELINE_A_32PPU_ADMISSION_COMPLETE", check_admission(sys.argv[1], sys.argv[2]), flush=True)
PY
else
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
    --resume-from "$ROOT/admission2/checkpoints/step-000120" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
