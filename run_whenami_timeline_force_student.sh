#!/usr/bin/env bash
# Stage II causal U student from INTERMEDIATE A+ teacher step-040000 (user decision 2026-10-16).
# Same recipe as run_whenami_timeline_force_320.sh STAGE=student (50k, gb320, micro1, draws2,
# LR 1e-6/1e-5, warmup1000, weights 0.1/0.1/0.1, save/eval 5k, admission 100/120).
# ONLY deviations: (1) teacher = step-040000 not completed step-050000 (gate.py, pinned);
# (2) 10 nodes x16=160 allowed (whenami_timeline_force_160.train, accum2, virtual world 320),
#     20 nodes x16=320 uses the original whenami_timeline_force.train.
# Platform WORLD_SIZE = NODE count (10 or 20), RANK = node rank.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
VARIANT=${VARIANT:-revise}
PHASE=${PHASE:-run}
TEACHER_CKPT=${TEACHER_CKPT:-$STORE/outputs/whenami_timeline_force/training/timeline_force_Aplus_160_resume5k_01/main/checkpoints/step-040000}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
case "$NNODES" in 10) WORLD=160;; 20) WORLD=320;; *)
  echo "Require 10 or 20 physical nodes x16 PPUs; WORLD_SIZE is platform NODE count." >&2; exit 2;; esac
[[ "$NPROC_PER_NODE" == 16 ]] || { echo "Require 16 PPUs per node" >&2; exit 2; }
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || { echo "invalid platform node RANK" >&2; exit 2; }
case "$VARIANT" in direct|revise) ;; *) echo "invalid VARIANT" >&2; exit 2;; esac
case "$PHASE" in admit|run|full|resume) ;; *) echo "invalid PHASE" >&2; exit 2;; esac
RUN_ID=${RUN_ID:-timeline_force_student_${VARIANT}_from40k_${WORLD}_01}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
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
prefix=(torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29887}")
common=(--module whenami_timeline_force_student.train --base "$WORLD" --stage student --variant "$VARIANT"
  --teacher "$TEACHER_CKPT"
  --steps 50000 --micro 1 --global-batch 320 --seed 42 --lr 1e-6 --new-lr 1e-5
  --draws 2 --current-weight 0.1 --cognition-weight 0.1 --revision-weight 0.1
  --hidden-every 4 --warmup 1000 --sharding hybrid_zero2 --activation llm
  --save-every 5000 --eval-every 5000 --first-check 100)
preflight=(-m whenami_timeline_force_student.preflight --root "$ROOT" --variant "$VARIANT"
  --phase "$PHASE" --world "$WORLD" --teacher "$TEACHER_CKPT")
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
if [[ "$NODE_RANK" == 0 && ! -e "$ROOT/DEVIATION.json" ]]; then python - "$ROOT" "$TEACHER_CKPT" "$WORLD" <<'PY2'
import json, sys
from whenami_timeline_force_student.gate import DEVIATION
json.dump({"teacher": sys.argv[2], "world": int(sys.argv[3]), "deviations": [DEVIATION,
    "160-card runs use whenami_timeline_force_160.train (accum2, virtual world 320) instead of 20x16"
    if sys.argv[3] == "160" else "none beyond teacher step"]}, open(sys.argv[1]+"/DEVIATION.json", "w"), indent=1)
PY2
fi
if [[ "$PHASE" == admit || "$PHASE" == run ]]; then
  "${prefix[@]}" --module stage3_perf.probe_fsdp --sharding hybrid_zero2 --output "$ROOT/topology_probe" \
    2>&1 | tee "$ROOT/topology.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission1" --stop-after 100 \
    2>&1 | tee "$ROOT/admission1.node${NODE_RANK}.log"
  "${prefix[@]}" "${common[@]}" --output "$ROOT/admission2" --stop-after 120 \
    --resume-from "$ROOT/admission1/checkpoints/step-000100" \
    2>&1 | tee "$ROOT/admission2.node${NODE_RANK}.log"
  python - "$ROOT" "$VARIANT" "$WORLD" "$TEACHER_CKPT" <<'PY'
import sys
from whenami_timeline_force_student.preflight import check_admission
print("TIMELINE_FORCE_STUDENT_ADMISSION_PASS", check_admission(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]), flush=True)
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
