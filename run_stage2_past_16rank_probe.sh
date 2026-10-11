#!/usr/bin/env bash
# EXPERIMENTAL: isolated snapshot, NOT the production 8-rank launcher.
# Start with 1 node, 16 processes, 250 steps; user submits the cluster job.
set -euo pipefail
CODE=/home/zhengguantian/WoG_fresh/WoG_16rank_probe
STORE=/mnt/pfs/zhengguantian/WoG_v2
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_GPUS=$((NPROC_PER_NODE * NNODES))
GLOBAL_BS=$((PER_DEVICE_BS * TOTAL_GPUS))
RUN_ID=${RUN_ID:-past_s2_16rank_probe_01}
CKPT=${CKPT:-$STORE/outputs/stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt}
MAX_STEPS=${MAX_STEPS:-250}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}
# Conservative diagnostic setting. Restore 32000 only after the 16-rank run
# passes: 16*32000 = 8*64000 buffered windows/node. Smaller buffers alter
# shuffle correlations, so this is NOT the final scientific comparison recipe.
SHUFFLE_BUFFER=${SHUFFLE_BUFFER:-4096}
DIAG=$STORE/outputs/runtime16/$RUN_ID
[[ ! -e "$STORE/outputs/stage2_16rank_probe/$RUN_ID--image_aug" ]] || {
  echo "Run already exists; use a NEW RUN_ID" >&2; exit 2;
}
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
export PYTHONPATH="$STORE/pylibs-train:$CODE"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
export WAI_BOUNDED_INPUT=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export WAI_MEMORY_LOG_DIR="$DIAG/ranks"
export WAI_DATASET_STATISTICS=${WAI_DATASET_STATISTICS:-$STORE/outputs/stage1/baseline_v2--image_aug/dataset_statistics.json}

cmd=(
  torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK"
  --nproc_per_node "$NPROC_PER_NODE" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT"
  scripts/train.py --pretrained_checkpoint "$CKPT"
  --whenami_stage 2 --past_init checkpoint --train_venc True --use_temporal False
  --obs_offsets "[-15,-11,-7,-3,0]"
  --vla.type prism-dinosiglip-224px+oxe+diffusion
  --vla.data_mix "${DATA_MIX:-oxe_magic_soup_plus}"
  --vla.expected_world_size "$TOTAL_GPUS" --vla.global_batch_size "$GLOBAL_BS"
  --vla.per_device_batch_size "$PER_DEVICE_BS" --vla.learning_rate "${LR:-3.2e-5}"
  --vla.shuffle_buffer_size "$SHUFFLE_BUFFER" --vla.max_steps "$MAX_STEPS"
  --data_root_dir "$STORE/datasets/open-x-embodiment"
  --run_root_dir "$STORE/outputs/stage2_16rank_probe" --run_id "$RUN_ID"
  --image_aug True --save_interval "$SAVE_INTERVAL"
  --repeated_diffusion_steps 4 --Ta 16 --action_model_type DiT-L --is_resume False
)
printf 'EXPERIMENTAL: nodes=%s ranks/node=%s global_batch=%s shuffle/rank=%s\n' \
  "$NNODES" "$NPROC_PER_NODE" "$GLOBAL_BS" "$SHUFFLE_BUFFER"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
cd "$CODE"
mkdir -p "$MPLCONFIGDIR" "$DIAG"
python3 scripts/check_past_stage2.py --checkpoint "$CKPT" --nproc "$NPROC_PER_NODE"
# Keep the node monitor outside torchrun so worker deaths retain cgroup evidence.
exec python3 scripts/monitor_node.py --out "$DIAG/node-$NODE_RANK.jsonl" -- "${cmd[@]}"
