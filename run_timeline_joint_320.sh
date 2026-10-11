#!/usr/bin/env bash
# One full-VLM arm on20 physical nodes x16PPUs. Two arms use640PPUs total.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
JOINT_MODE=${JOINT_MODE:-timeline}
PHASE=${PHASE:-admit}
RUN_ID=${RUN_ID:-timeline_joint_${JOINT_MODE}_320_50k_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ "$NNODES" == 20 && "$NPROC_PER_NODE" == 16 ]] || {
  echo "One arm requires20 physical nodes x16PPUs (320). Two separate arms use640." >&2; exit 2;
}
case "$JOINT_MODE" in timeline|pooled_query) ;; *) echo "invalid JOINT_MODE" >&2; exit 2;; esac
case "$PHASE" in admit|run|full) ;; *) echo "PHASE must be admit, run, or full" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
ROOT=$STORE/outputs/whenami_timeline_joint/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29783}")
common=(--module whenami_timeline_joint.train --mode "$JOINT_MODE"
  --steps 50000 --micro 1 --global-batch 320 --nodes 3 --seed 42
  --lr 1e-5 --new-lr 1e-4 --warmup 1000 --sharding hybrid_zero2 --activation llm
  --save-every 5000 --eval-every 2500 --first-check 100)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  if [[ "$PHASE" == admit || "$PHASE" == run ]]; then
    printf '%q ' "${prefix[@]}" --module stage3_perf.probe_fsdp --sharding hybrid_zero2 --output "$ROOT/topology_probe"; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 100; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 120 \
      --resume-from "$ROOT/admission1/checkpoints/step-000100"; printf '\n'
    if [[ "$PHASE" == run ]]; then
      printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
        --resume-from "$ROOT/admission2/checkpoints/step-000120"; printf '\n'
    fi
  else
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
      --resume-from "$ROOT/admission2/checkpoints/step-000120"; printf '\n'
  fi
  exit 0
fi
# Per-node preflight precedes creating the common root. Serialize only the filesystem
# decision on rank0; other nodes independently verify sources but do not race on existence.
if [[ "$NODE_RANK" == 0 ]]; then
  python -m whenami_timeline_joint.preflight --root "$ROOT" --mode "$JOINT_MODE" --phase "$PHASE"
else
  python - <<'PY'
from whenami_timeline_joint.preflight import local_preflight
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
  python - "$ROOT" "$JOINT_MODE" <<'PY'
import sys
from whenami_timeline_joint.preflight import check_large_admission
print("FULL_TIMELINE_320_ADMISSION_COMPLETE",check_large_admission(sys.argv[1],sys.argv[2]),flush=True)
PY
  if [[ "$PHASE" == run ]]; then
    # Long training only starts after actual full-model save/eval/restart passes.
    # Any prior command failure stops because of set -e/pipefail.
    "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
      --resume-from "$ROOT/admission2/checkpoints/step-000120" \
      2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
  fi
else
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
    --resume-from "$ROOT/admission2/checkpoints/step-000120" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
