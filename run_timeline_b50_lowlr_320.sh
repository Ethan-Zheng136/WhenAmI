#!/usr/bin/env bash
# Control for the warmup-disruption hypothesis: A50k -> B with LOWER peak LR.
# Copy of run_timeline_b50_320.sh; ONLY differences: (1) --lr/--aux-lr/--warmup come from env vars,
# (2) outputs go to a separate family dir whenami_timeline_b50_lowlr/, (3) run name carries the LR tag.
# Training code, teacher (A50k), data, steps, batch, seed, eval/save cadence are unchanged.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
B50_VARIANT=${B50_VARIANT:-condition_align}
B50_LR=${B50_LR:-1e-6}          # body peak LR (original 1e-5)
B50_AUX_LR=${B50_AUX_LR:-1e-5}  # forecast_head peak LR (original 1e-4); keep 10x body ratio
B50_WARMUP=${B50_WARMUP:-1000}  # keep original warmup length so only the peak LR changes
PHASE=${PHASE:-run}
RUN_ID=${RUN_ID:-timeline_b50_${B50_VARIANT}_A50k_320_20k_lr${B50_LR}_wu${B50_WARMUP}_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ "$NNODES" == 20 && "$NPROC_PER_NODE" == 16 ]] || {
  echo "Each B arm requires20 physical nodes x16PPUs=320." >&2; exit 2;
}
case "$B50_VARIANT" in action_only|condition_align) ;; *) echo "invalid B50_VARIANT" >&2; exit 2;; esac
case "$PHASE" in admit|run|full) ;; *) echo "invalid PHASE" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
[[ "$B50_LR" =~ ^[0-9.eE+-]+$ && "$B50_AUX_LR" =~ ^[0-9.eE+-]+$ && "$B50_WARMUP" =~ ^[0-9]+$ ]] || { echo "bad LR/warmup" >&2; exit 2; }
ROOT=$STORE/outputs/whenami_timeline_b50_lowlr/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29873}")
common=(--module whenami_timeline_b50.train --variant "$B50_VARIANT"
  --steps 20000 --micro 1 --global-batch 320 --seed 42 --lr "$B50_LR" --aux-lr "$B50_AUX_LR"
  --alignment-weight 0.1 --warmup "$B50_WARMUP" --sharding hybrid_zero2 --activation llm
  --save-every 2500 --eval-every 1000 --first-check 100)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "ROOT=$ROOT"
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
# Same preflight as the original B50 job (checks code fingerprints, pinned A50k teacher, DEV split,
# and that ROOT is fresh). It also checks the ORIGINAL launcher hash, which this copy leaves untouched.
if [[ "$NODE_RANK" == 0 ]]; then
  python -m whenami_timeline_b50.preflight --root "$ROOT" --variant "$B50_VARIANT" --phase "$PHASE"
else
  python - <<'PY'
from whenami_timeline_b50.preflight import local_preflight
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
  python - "$ROOT" "$B50_VARIANT" <<'PY'
import sys
from whenami_timeline_b50.preflight import check_admission
print("TIMELINE_B50_LOWLR_320_ADMISSION_COMPLETE",check_admission(sys.argv[1],sys.argv[2]),flush=True)
PY
fi
if [[ "$PHASE" == run || "$PHASE" == full ]]; then
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
    --resume-from "$ROOT/admission2/checkpoints/step-000120" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
