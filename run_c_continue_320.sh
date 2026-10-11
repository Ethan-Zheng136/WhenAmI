#!/usr/bin/env bash
# Two independent C continuations, own20k->total50k; keep own teacher and Adam/RNG. No original job changes.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
C_CONTINUE_CASE=${C_CONTINUE_CASE:-c_read}
PHASE=${PHASE:-run}
RUN_ID=${RUN_ID:-c_continue_${C_CONTINUE_CASE}_20k_to50k_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ "$NNODES" == 20 && "$NPROC_PER_NODE" == 16 ]] || {
  echo "Each continuation requires20 physical nodes x16PPUs=320. Two simultaneous C jobs require640 cards." >&2; exit 2;
}
case "$C_CONTINUE_CASE" in c_aux|c_read) ;; *) echo "invalid C_CONTINUE_CASE" >&2; exit 2;; esac
case "$PHASE" in admit|run|full) ;; *) echo "invalid PHASE" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
ROOT=$STORE/outputs/whenami_c_continue/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29893}")
common=(--module whenami_c_continue.train --case "$C_CONTINUE_CASE")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  if [[ "$PHASE" == admit || "$PHASE" == run ]]; then
    printf '%q ' "${prefix[@]}" --module stage3_perf.probe_fsdp --sharding hybrid_zero2 --output "$ROOT/topology_probe"; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 20100; printf '\n'
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 20120 \
      --resume-from "$ROOT/admission1/checkpoints/step-020100"; printf '\n'
  fi
  if [[ "$PHASE" == run || "$PHASE" == full ]]; then
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
      --resume-from "$ROOT/admission2/checkpoints/step-020120"; printf '\n'
  fi
  exit 0
fi
if [[ "$NODE_RANK" == 0 ]]; then
  python -m whenami_c_continue.preflight --root "$ROOT" --case "$C_CONTINUE_CASE" --phase "$PHASE"
else
  python - "$C_CONTINUE_CASE" <<'PY'
import sys
from whenami_c_continue.preflight import local_preflight
local_preflight(sys.argv[1])
PY
fi
mkdir -p "$ROOT"
if [[ "$PHASE" == admit || "$PHASE" == run ]]; then
  "${prefix[@]}" --module stage3_perf.probe_fsdp --sharding hybrid_zero2 --output "$ROOT/topology_probe" \
    2>&1 | tee "$ROOT/topology.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 20100 \
    2>&1 | tee "$ROOT/admission1.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 20120 \
    --resume-from "$ROOT/admission1/checkpoints/step-020100" \
    2>&1 | tee "$ROOT/admission2.node${NODE_RANK}.log"
  python - "$ROOT" "$C_CONTINUE_CASE" <<'PY'
import sys
from whenami_c_continue.preflight import check_admission
print("C_CONTINUATION_320_ADMISSION_COMPLETE",check_admission(sys.argv[1],sys.argv[2]),flush=True)
PY
fi
if [[ "$PHASE" == run || "$PHASE" == full ]]; then
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
    --resume-from "$ROOT/admission2/checkpoints/step-020120" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
