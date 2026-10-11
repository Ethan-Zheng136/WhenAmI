#!/usr/bin/env bash
# Raw full-training-partition preparation. No model/GPU compute, no dev/test output.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
INVENTORY=${INVENTORY:-$STORE/outputs/stage3_scale/inventory_01/inventory.json}
RAW_ROOT=${RAW_ROOT:-$STORE/outputs/stage3_joint/raw_full_01}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
export PYTHONPATH="$STORE/pylibs-train:$BASE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cmd=(
  torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29581}"
  --module stage3_scale.raw_data --inventory "$INVENTORY" --output "$RAW_ROOT"
)
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
cd "$BASE"
mkdir -p "$RAW_ROOT"
"${cmd[@]}" 2>&1 | tee -a "$RAW_ROOT/prepare-node${NODE_RANK}.log"
if [[ $NODE_RANK == 0 ]]; then
  # No distributed group in preparation workers. Wait explicitly for each
  # immutable source completion before rank0 merges, rather than relying on
  # node-local torchrun exit timing.
  python - "$RAW_ROOT" "$INVENTORY" <<'PY'
import json,sys,time
from pathlib import Path
root,inv=Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text())
deadline=time.monotonic()+1800
while True:
    missing=[r["id"] for r in inv["source_shards"] if not (
        root/"shards"/r["dataset"]/(r["id"].split("/")[-1]+".json")).exists()]
    if not missing:break
    if time.monotonic()>deadline:raise RuntimeError(f"Missing {len(missing)} source completions")
    time.sleep(5)
PY
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    python -m stage3_scale.raw_data --inventory "$INVENTORY" --output "$RAW_ROOT" --finalize-only
fi
echo "Raw conversion complete on node $NODE_RANK; main-training readiness still requires distributed admission."
