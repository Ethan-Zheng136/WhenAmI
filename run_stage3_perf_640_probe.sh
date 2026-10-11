#!/usr/bin/env bash
# Independent strong-scaling measurement. Does NOT stop/migrate the running320.
# Same global1280 and 50k LR schedule; micro2 on640 instead of micro4 on320.
# Stops after100 updates/save/eval. This is NOT a second automatic50k main run.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
[[ $NNODES == 40 && $NPROC_PER_NODE == 16 ]] || {
  echo "Scaling probe needs40 physical nodes x16 cards (640)." >&2; exit 2;
}
[[ ${RESUME:-0} == 0 ]] || {
  echo "Fresh scaling measurement only; do not resume a320 checkpoint here." >&2; exit 2;
}
RUN_ID=${RUN_ID:-oxe27_forecast_perf640_probe_01}
OUTPUT=$STORE/outputs/stage3_joint/$RUN_ID
DATA_INDEX=$STORE/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
cd "$BASE"
prefix=(torchrun --nnodes 40 --node_rank "$NODE_RANK" --nproc_per_node 16
        --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29613}")
cmd=("${prefix[@]}" --module stage3_perf.train --data-index "$DATA_INDEX" --output "$OUTPUT"
     --variant forecast --steps 50000 --global-batch 1280 --micro-batch 2 --fm-draws 4
     --lr 2e-5 --new-lr 1e-4 --warmup 1000 --sharding hybrid
     --activation-checkpoint none --prefetch-batches 2 --log-every 10 --profile-every 100
     --first-save 100 --first-eval 100 --save-every 2500 --development-every 2500
     --admission-gradient-audit --stop-after 100)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${prefix[@]}" --module stage3_perf.probe_fsdp --output "${OUTPUT}.tiny" --sharding hybrid; printf '\n'
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
python -m stage3_perf.preflight --data-index "$DATA_INDEX" --output "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")"
"${prefix[@]}" --module stage3_perf.probe_fsdp --output "${OUTPUT}.tiny" --sharding hybrid \
  2>&1 | tee -a "${OUTPUT}.tiny-node${NODE_RANK}.log"
"${cmd[@]}" 2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
python -m stage3_perf.check_first_segment "$OUTPUT" --step 100 \
  2>&1 | tee -a "${OUTPUT}.node${NODE_RANK}.log"
echo "640 scaling measurement complete. Compare FIRST_SEGMENT_COMPLETE.json with320 before choosing main topology."
