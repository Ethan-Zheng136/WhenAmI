#!/usr/bin/env bash
# Main320 throughput candidate: retain full parameters between forward/backward.
# SAME scientific code, global1280,50k steps. More memory, potentially less communication.
# New output/initialization, not a recipe-bypassing resume of the prior HYBRID_SHARD run.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ $NNODES == 20 && $NPROC_PER_NODE == 16 ]] || {
  echo "This recipe requires20 physical nodes x16 PPUs (320)." >&2; exit 2;
}
[[ ${RESUME:-0} == 0 || ${RESUME:-0} == 1 ]] || {
  echo "RESUME must be0 (fresh run) or1 (same zero2 run only)." >&2; exit 2;
}
VARIANT=${VARIANT:-forecast}
RUN_ID=${RUN_ID:-oxe27_${VARIANT}_zero2_320_50k_save5k_01}
OUTPUT=$STORE/outputs/stage3_joint/$RUN_ID
DATA_INDEX=${DATA_INDEX:-$STORE/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json}
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 20 --node_rank "$NODE_RANK" --nproc_per_node 16
        --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29633}")
cmd=("${prefix[@]}" --module stage3_perf.train --data-index "$DATA_INDEX" --output "$OUTPUT"
     --variant "$VARIANT" --steps 50000 --global-batch 1280 --micro-batch 4 --fm-draws 4 --sharding hybrid_zero2
     --lr 2e-5 --new-lr 1e-4 --warmup 1000 --activation-checkpoint none
     --prefetch-batches 2 --log-every 10 --profile-every 100
     --first-save 100 --first-eval 100 --save-every 5000 --development-every 2500
     --admission-gradient-audit)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  if [[ ${RESUME:-0} != 1 ]]; then
    printf '%q ' "${prefix[@]}" --module stage3_perf.probe_fsdp --output "${OUTPUT}.tiny" --sharding hybrid_zero2; printf '\n'
    printf '%q ' "${cmd[@]}" --stop-after 100; printf '\n'
  fi
  printf '%q ' "${cmd[@]}" --resume; printf '\n'; exit 0
fi
pre=(python -m stage3_perf.preflight --data-index "$DATA_INDEX" --output "$OUTPUT")
if [[ ${RESUME:-0} == 1 ]]; then pre+=(--resume); fi
"${pre[@]}"
mkdir -p "$(dirname "$OUTPUT")"
if [[ ${RESUME:-0} == 1 ]]; then
  python - "$OUTPUT" "$NODE_RANK" <<'PY'
import json, re, sys, uuid
from pathlib import Path
root = Path(sys.argv[1])
recipe = json.loads((root / "recipe.json").read_text())
s = recipe["settings"]
if (recipe["world_size"], s["global_batch"], s["micro_batch"], s["steps"],
        s["save_every"], s["development_every"], s["sharding"]) != (
        320, 1280, 4, 50000, 5000, 2500, "hybrid_zero2"):
    raise SystemExit("This launcher only resumes its own320/zero2/global1280/save5k recipe.")
# A resumed checkpoint may precede an evaluation-only step. Preserve those
# predictions outside the live path so replay cannot collide with old output.
if int(sys.argv[2]) == 0:
    checkpoints = sorted((root / "checkpoints").glob("step-*/COMMITTED.json"))
    restored = json.loads(checkpoints[-1].read_text())["step"]
    candidates = [p for p in (root / "development").glob("step-*")
                  if p.is_dir() and re.fullmatch(r"step-\d+", p.name)
                  and int(p.name.split("-")[1]) > restored]
    if candidates:
        archive = root / "replayed_development" / uuid.uuid4().hex
        archive.mkdir(parents=True, exist_ok=False)
        for p in candidates:
            destination = archive / p.name
            p.rename(destination)
            print(f"PRESERVED_REPLAYED_EVAL {p} -> {destination}", flush=True)
PY
fi
if [[ ${RESUME:-0} != 1 ]]; then
  "${prefix[@]}" --module stage3_perf.probe_fsdp --output "${OUTPUT}.tiny" --sharding hybrid_zero2 \
    2>&1 | tee -a "${OUTPUT}.tiny-node${NODE_RANK}.log"
  "${cmd[@]}" --stop-after 100 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
  python -m stage3_perf.check_first_segment "$OUTPUT" --step 100 \
    2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
fi
# Intentional new-process restart proves same-recipe state restoration; do not
# silently start over if optimizer/RNG/layout checking fails.
"${cmd[@]}" --resume 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
