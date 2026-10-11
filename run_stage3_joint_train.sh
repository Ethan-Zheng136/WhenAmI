#!/usr/bin/env bash
# Full raw policy trainer. Guarded until the same code passes 2-node admission.
# Suggested deployment: 16 nodes x16 =256 cards, micro2 x accumulation2 => global1024.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
VARIANT=${VARIANT:-forecast}
RUN_ID=${RUN_ID:-joint_${VARIANT}_50k_01}
DATA_INDEX=${DATA_INDEX:-$STORE/outputs/stage3_joint/raw_full_01/train_index.json}
ADMISSION=${ADMISSION:-$STORE/outputs/stage3_joint/admission_32_01/ADMISSION_COMPLETE.json}
OUTPUT=$STORE/outputs/stage3_joint/$RUN_ID
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cmd=(
  torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29583}"
  --module stage3_scale.train_joint --data-index "$DATA_INDEX" --output "$OUTPUT"
  --variant "$VARIANT" --steps "${MAX_STEPS:-50000}" --global-batch "${GLOBAL_BATCH:-1024}"
  --micro-batch "${MICRO_BATCH:-2}" --fm-draws "${FM_DRAWS:-4}"
  --lr "${LR:-2e-5}" --new-lr "${NEW_LR:-1e-4}" --warmup "${WARMUP:-1000}"
  --save-every "${SAVE_EVERY:-2500}"
)
if [[ ${RESUME:-0} == 1 ]]; then cmd+=(--resume); fi
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
cd "$BASE"
python - "$ADMISSION" "$DATA_INDEX" <<'PY'
from pathlib import Path
import hashlib,json,sys
admission_path,data=map(Path,sys.argv[1:])
if not admission_path.exists():raise SystemExit("Run 2-node admission first. No validation bypass is provided.")
a=json.loads(admission_path.read_text());d=json.loads(data.read_text())
if not a.get("complete") or not a.get("full_policy_train_save_resume_passed"):raise SystemExit("Admission incomplete")
if not d.get("complete") or d.get("scope")!="full_source_train_filtered":raise SystemExit("Full raw corpus not complete")
root=Path("/home/zhengguantian/WoG_fresh/stage3_scale")
current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}
if current!=a["joint_source"]:raise SystemExit("Joint training code changed since admission; revalidate")
base=root.parent
for name,expected in a["reused_source"].items():
    if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit(f"Dependency changed since admission: {name}")
PY
mkdir -p "$(dirname "$OUTPUT")"
"${cmd[@]}" 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
