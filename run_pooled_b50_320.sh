#!/usr/bin/env bash
# Independent matched Pooled-A50k -> pooled-B controls; do not modify existing A/B/C jobs.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
POOLED_B_VARIANT=${POOLED_B_VARIANT:-condition_align}
PHASE=${PHASE:-run}
RUN_ID=${RUN_ID:-pooled_b50_${POOLED_B_VARIANT}_A50k_320_20k_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ "$NNODES" == 20 && "$NPROC_PER_NODE" == 16 ]] || {
  echo "Each B arm requires20 physical nodes x16PPUs=320; two jobs need640 additional PPUs." >&2; exit 2;
}
case "$POOLED_B_VARIANT" in action_only|condition_align) ;; *) echo "invalid POOLED_B_VARIANT" >&2; exit 2;; esac
case "$PHASE" in admit|run|full) ;; *) echo "invalid PHASE" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
ROOT=$STORE/outputs/whenami_pooled_b50/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29863}")
common=(--module whenami_pooled_b50.train --variant "$POOLED_B_VARIANT"
  --steps 20000 --micro 1 --global-batch 320 --seed 42 --lr 1e-5 --aux-lr 1e-4
  --alignment-weight 0.1 --warmup 1000 --sharding hybrid_zero2 --activation llm
  --save-every 2500 --eval-every 1000 --first-check 100)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  if [[ "$PHASE" == admit || "$PHASE" == run ]]; then
    printf '%q ' "${prefix[@]}" --module stage3_perf.probe_fsdp --sharding hybrid_zero2 --output "$ROOT/topology_probe"; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 100; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 120 \
      --resume-from "$ROOT/admission1/checkpoints/step-000100"; printf '\n'
  fi
  if [[ "$PHASE" == run || "$PHASE" == full ]]; then
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
      --resume-from "$ROOT/admission2/checkpoints/step-000120"; printf '\n'
  fi
  exit 0
fi
if [[ "$NODE_RANK" == 0 ]]; then
  python -m whenami_pooled_b50.preflight --root "$ROOT" --variant "$POOLED_B_VARIANT" --phase "$PHASE"
else
  python - <<'PY'
from whenami_pooled_b50.preflight import local_preflight
local_preflight()
PY
fi
mkdir -p "$ROOT"
if [[ "$PHASE" == admit || "$PHASE" == run ]]; then
  "${prefix[@]}" --module stage3_perf.probe_fsdp --sharding hybrid_zero2 --output "$ROOT/topology_probe" \
    2>&1 | tee "$ROOT/topology.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 100 \
    2>&1 | tee "$ROOT/admission1.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 120 \
    --resume-from "$ROOT/admission1/checkpoints/step-000100" \
    2>&1 | tee "$ROOT/admission2.node${NODE_RANK}.log"
  python - "$ROOT" "$POOLED_B_VARIANT" <<'PY'
import sys
from whenami_pooled_b50.preflight import check_admission
print("POOLED_B50_320_ADMISSION_COMPLETE",check_admission(sys.argv[1],sys.argv[2]),flush=True)
PY
fi
if [[ "$PHASE" == run || "$PHASE" == full ]]; then
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
    --resume-from "$ROOT/admission2/checkpoints/step-000120" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
