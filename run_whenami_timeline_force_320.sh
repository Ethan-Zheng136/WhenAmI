#!/usr/bin/env bash
# Independent A+ / causal U pipeline. Each formal task:20nodes×16=320,50k.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
STAGE=${STAGE:-teacher}
VARIANT=${VARIANT:-revise}
PHASE=${PHASE:-run}
RUN_ID=${RUN_ID:-timeline_force_${STAGE}_${VARIANT}_320_50k_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ "$NNODES" == 20 && "$NPROC_PER_NODE" == 16 ]] || {
  echo "Require20 physical nodes×16PPUs=320; WORLD_SIZE is platform NODE count." >&2; exit 2;
}
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || {
  echo "invalid platform node RANK" >&2; exit 2;
}
case "$STAGE" in teacher|student) ;; *) echo "invalid STAGE" >&2; exit 2;; esac
case "$VARIANT" in direct|revise) ;; *) echo "invalid VARIANT" >&2; exit 2;; esac
case "$PHASE" in admit|run|full|resume) ;; *) echo "invalid PHASE" >&2; exit 2;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
if [[ "$STAGE" == teacher && "$VARIANT" != revise ]]; then
  echo "Main teacher uses revise. Matched direct/revise comparison is Stage2." >&2; exit 2;
fi
if [[ "$STAGE" == student && -z ${TEACHER_CKPT:-} ]]; then
  echo "Set TEACHER_CKPT to completed NEW Stage1 step-050000; old A50k is not Stage1." >&2; exit 2;
fi
if [[ "$PHASE" == resume && -z ${RESUME_FROM:-} ]]; then
  echo "PHASE=resume requires explicit RESUME_FROM and fresh RUN_ID." >&2; exit 2;
fi
ROOT=$STORE/outputs/whenami_timeline_force/training/$RUN_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29887}")
common=(--module whenami_timeline_force.train --stage "$STAGE" --variant "$VARIANT"
  --steps 50000 --micro 1 --global-batch 320 --seed 42 --lr 1e-6 --new-lr 1e-5
  --draws 2 --current-weight 0.1 --cognition-weight 0.1 --revision-weight 0.1
  --hidden-every 4 --warmup 1000 --sharding hybrid_zero2 --activation llm
  --save-every 5000 --eval-every 5000 --first-check 100)
preflight=(-m whenami_timeline_force.preflight --root "$ROOT"
  --stage "$STAGE" --variant "$VARIANT" --phase "$PHASE")
if [[ "$STAGE" == student ]]; then
  common+=(--teacher "$TEACHER_CKPT")
  preflight+=(--teacher "$TEACHER_CKPT")
fi
if [[ "$PHASE" == resume ]]; then preflight+=(--resume-from "$RESUME_FROM"); fi
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
  elif [[ "$PHASE" == resume ]]; then
    printf '%q ' "${prefix[@]}" "${common[@]}" --output "$ROOT/main" --resume-from "$RESUME_FROM"; printf '\n'
  fi
  exit 0
fi
if [[ "$NODE_RANK" == 0 ]]; then python "${preflight[@]}"
else python -c 'from whenami_timeline_force.preflight import local_preflight; local_preflight()'
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
  python - "$ROOT" "$STAGE" "$VARIANT" <<'PY'
import sys
from whenami_timeline_force.preflight import check_admission
print("TIMELINE_FORCE_320_ADMISSION_PASS", check_admission(*sys.argv[1:]), flush=True)
PY
fi
if [[ "$PHASE" == run || "$PHASE" == full ]]; then
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" \
    --resume-from "$ROOT/admission2/checkpoints/step-000120" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
elif [[ "$PHASE" == resume ]]; then
  "${prefix[@]}" "${common[@]}" --output "$ROOT/main" --resume-from "$RESUME_FROM" \
    2>&1 | tee "$ROOT/main.node${NODE_RANK}.log"
fi
