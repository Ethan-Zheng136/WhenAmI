#!/usr/bin/env bash
# Complete27-domain raw conversion. CPU/I/O only; no model feature cache.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
INVENTORY=${INVENTORY:-$STORE/outputs/stage3_joint/oxe27_inventory_01/inventory.json}
RAW_ROOT=${RAW_ROOT:-$STORE/outputs/stage3_joint/oxe27_raw_full_01}
AUDIT=${AUDIT:-$STORE/outputs/stage3_joint/oxe27_domain_audit_02/results.json}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
export PYTHONPATH="$STORE/pylibs-train:$BASE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cmd=(
  torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29591}"
  --module stage3_scale.oxe_data --inventory "$INVENTORY" --output "$RAW_ROOT"
)
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
cd "$BASE"
python - "$AUDIT" "$INVENTORY" <<'PY'
import json,sys
from pathlib import Path
from stage3.common import sha256
from stage3_scale.oxe_data import oxe_source_hashes
p,inv=map(Path,sys.argv[1:])
a=json.loads(p.read_text())
if not a["complete"] or a["passed"]!=27:raise SystemExit("27-domain audit required")
if a["inventory_sha256"]!=sha256(inv):raise SystemExit("Inventory differs from audited version")
if a["decoder_sha256"]!=sha256(Path("stage3_scale/oxe_data.py")) or a["oxe_source_sha256"]!=oxe_source_hashes():
    raise SystemExit("Decoder/standardization changed since audit")
PY
mkdir -p "$RAW_ROOT"
"${cmd[@]}" 2>&1 | tee -a "$RAW_ROOT/prepare-node${NODE_RANK}.log"
if [[ $NODE_RANK == 0 ]]; then
  python - "$RAW_ROOT" "$INVENTORY" <<'PY'
from pathlib import Path
import json,sys,time
root,inv=Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text())
deadline=time.monotonic()+7200
while True:
    missing=[r["id"] for r in inv["source_shards"] if not (
        root/"shards"/r["dataset"]/(r["id"].split("/")[-1]+".json")).exists()]
    if not missing:break
    if time.monotonic()>deadline:raise RuntimeError(f"Missing {len(missing)} OXE source completions")
    print("WAIT_OXE_SOURCES",len(missing),flush=True)
    time.sleep(30)
PY
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    python -m stage3_scale.oxe_data --inventory "$INVENTORY" --output "$RAW_ROOT" --finalize-only
fi
echo "OXE raw conversion finished on node $NODE_RANK; inspect oxe27_train_index.json on rank0."
