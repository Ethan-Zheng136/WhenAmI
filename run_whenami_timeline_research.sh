#!/usr/bin/env bash
# New multi-anchor real-OXE research, independent of the running full-policyStage3.
# 2nodes x16PPU. Frozen real backbone + trainable timeline DDP.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ $NNODES == 2 && $NPROC_PER_NODE == 16 ]] || {
  echo "Real timeline research requires2 physical nodes x16 distinct PPUs (32)." >&2; exit 2;
}
# MODE is overwritten to CUDA by the platform's BASH_ENV. Never use that key
# for a scientific experiment selection.
TIMELINE_MODE=${TIMELINE_MODE:-revision}
case "$TIMELINE_MODE" in independent|forward|direct_history|revision) ;; *) echo "Unknown TIMELINE_MODE" >&2; exit 2;; esac
SEED=${SEED:-42}
[[ $SEED =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
RUN_ID=${RUN_ID:-${TIMELINE_MODE}_seed${SEED}_32ppu_5k_01}
OUTPUT=$STORE/outputs/whenami_timeline/$RUN_ID
INDEX=$STORE/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$BASE"
prefix=(torchrun --nnodes 2 --node_rank "$NODE_RANK" --nproc_per_node 16
        --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29643}")
cmd=("${prefix[@]}" --module whenami_timeline.train_real --index "$INDEX" --output "$OUTPUT"
     --mode "$TIMELINE_MODE" --seed "$SEED" --steps 5000 --micro 2 --draws 2 --lr 1e-4
     --save-every 500 --dev-every 250)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  if [[ ${RESUME:-0} != 1 ]]; then printf '%q ' "${cmd[@]}" --stop-after 2; printf '\n'; fi
  printf '%q ' "${cmd[@]}" --resume; printf '\n'; exit 0
fi
python - "$BASE" "$INDEX" "$OUTPUT" "${RESUME:-0}" <<'PY'
from pathlib import Path
import hashlib,json,sys
from stage3.common import check_endpoint
base,index,output=map(Path,sys.argv[1:4]);resume=sys.argv[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
verified=json.loads((base/"whenami_timeline/LOCAL_VERIFICATION.json").read_text())
actual={p.name:sha(p) for p in (base/"whenami_timeline").glob("*.py")}
if not verified["local_checks_passed"] or actual!=verified["source"]:
    raise SystemExit("Timeline source not locally verified")
check_endpoint(verified["frozen_endpoint"])
if sha(index)!=verified["index_sha256"]:
    raise SystemExit("Full OXE index changed")
for name,expected in verified["protected_sources"].items():
    if sha(base/name)!=expected:raise SystemExit(f"Dependency changed: {name}")
if resume=="0":
    if output.exists():raise SystemExit("Output exists; choose fresh RUN_ID or inspect an explicit resume")
elif resume=="1":
    recipe=json.loads((output/"recipe.json").read_text())
    if recipe["world_size"]!=32 or recipe["source"]!=actual:
        raise SystemExit("Resume requires same32 rank timeline source/recipe")
    if not list((output/"checkpoints").glob("step-*/COMMITTED.json")):
        raise SystemExit("No committed timeline checkpoint")
else:raise SystemExit("RESUME must be0 or1")
print("TIMELINE_PREFLIGHT_OK: original OXE, endpoint, source checks; first2-step multihost admission follows")
PY
mkdir -p "$(dirname "$OUTPUT")"
if [[ ${RESUME:-0} != 1 ]]; then
  "${cmd[@]}" --stop-after 2 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
fi
# First multihost save/resume is checked before consuming the rest of the budget.
"${cmd[@]}" --resume 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
