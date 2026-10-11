#!/usr/bin/env bash
# Main policy training: 20 nodes x16 =320 PPUs, real FSDP,27-domain original OXE mix.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ $NNODES == 20 && $NPROC_PER_NODE == 16 ]] || {
  echo "This recipe is explicitly20 nodes x16 cards (320). Do not reinterpret worker counts." >&2; exit 2;
}
VARIANT=${VARIANT:-forecast}
RUN_ID=${RUN_ID:-oxe27_${VARIANT}_320_50k_01}
DATA_INDEX=${DATA_INDEX:-$STORE/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json}
ADMISSION=${ADMISSION:-$STORE/outputs/stage3_joint/oxe27_admission_32_v2_01/ADMISSION_COMPLETE.json}
DOMAIN_AUDIT=${DOMAIN_AUDIT:-$STORE/outputs/stage3_joint/oxe27_domain_audit_02/results.json}
OUTPUT=$STORE/outputs/stage3_joint/$RUN_ID
GLOBAL_BATCH=${GLOBAL_BATCH:-1280}
MICRO_BATCH=${MICRO_BATCH:-2}
FM_DRAWS=${FM_DRAWS:-4}
[[ $((GLOBAL_BATCH % (320 * MICRO_BATCH))) == 0 ]] || {
  echo "Global batch must be divisible by320*microbatch" >&2; exit 2;
}
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cmd=(
  torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29593}"
  --module stage3_scale.train_joint --data-index "$DATA_INDEX" --output "$OUTPUT"
  --variant "$VARIANT" --steps "${MAX_STEPS:-50000}"
  --global-batch "$GLOBAL_BATCH" --micro-batch "$MICRO_BATCH" --fm-draws "$FM_DRAWS"
  --lr "${LR:-2e-5}" --new-lr "${NEW_LR:-1e-4}" --warmup "${WARMUP:-1000}"
  --save-every "${SAVE_EVERY:-2500}" --development-every "${DEV_EVERY:-2500}"
)
if [[ ${RESUME:-0} == 1 ]]; then cmd+=(--resume); fi
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
cd "$BASE"
python - "$ADMISSION" "$DATA_INDEX" "$DOMAIN_AUDIT" "$MICRO_BATCH" "$FM_DRAWS" <<'PY'
from pathlib import Path
import json,sys
from stage3.common import sha256,REFERENCE
from stage3_scale.oxe_data import oxe_source_hashes
ap,dp,qp=map(Path,sys.argv[1:4]);micro,draws=map(int,sys.argv[4:])
if not ap.exists():raise SystemExit("Corrected32-card admission must complete before main training")
a=json.loads(ap.read_text());d=json.loads(dp.read_text());q=json.loads(qp.read_text())
if not a.get("complete") or not a.get("full_policy_train_save_resume_passed"):
    raise SystemExit("Admission incomplete")
if a.get("checkpoint_schema")!="fsdp_model_dcp_local_adam_v2":
    raise SystemExit("Previous optimizer schema failed; cannot reuse old admission")
if a.get("micro_batch")!=micro or a.get("fm_draws")!=draws:
    raise SystemExit("Microbatch/FM shape differs from validated admission")
if not d.get("complete") or d.get("data_mix")!="oxe_magic_soup_plus" or d.get("domain_count")!=27:
    raise SystemExit("Complete27-domain prepared index required (not two-domain/partial fixture)")
if not q.get("complete") or q.get("passed")!=27:
    raise SystemExit("Real27-domain compatibility audit required")
root=Path("/home/zhengguantian/WoG_fresh/stage3_scale")
if {p.name:sha256(p) for p in root.glob("*.py")}!=a["joint_source"]:
    raise SystemExit("Training code changed since admission")
for name,expected in a["reused_source"].items():
    if sha256(root.parent/name)!=expected:raise SystemExit(f"Dependency changed: {name}")
for name,expected in a["reference_source"].items():
    if sha256(REFERENCE/name)!=expected:raise SystemExit(f"Reference changed: {name}")
if d["oxe_source_sha256"]!=oxe_source_hashes() or q["decoder_sha256"]!=sha256(root/"oxe_data.py"):
    raise SystemExit("OXE standardization/decoder drift")
PY
mkdir -p "$(dirname "$OUTPUT")"
"${cmd[@]}" 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
