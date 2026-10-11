#!/usr/bin/env bash
# WhenAmI Stage 2 = Past. Not the original WoG Stage II (oformer distillation).
# Default is a 250-step cluster smoke run; set MAX_STEPS=20000 for a full run.
# No jobs are submitted by this file; the user supplies it as a platform command.
set -euo pipefail

CODE=/home/zhengguantian/WoG_fresh/WoG
STORE=/mnt/pfs/zhengguantian/WoG_v2
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
PER_DEVICE_BS=${PER_DEVICE_BS:-8}
TOTAL_GPUS=$((NPROC_PER_NODE * NNODES))
GLOBAL_BS=$((PER_DEVICE_BS * TOTAL_GPUS))
OBS_OFFSETS=${OBS_OFFSETS:-"[-15,-11,-7,-3,0]"}
INIT_MODE=${INIT_MODE:-stage1}

case "$INIT_MODE" in
  stage1)
    DEFAULT_CKPT=$STORE/outputs/stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt
    PAST_INIT=checkpoint
    ;;
  openvla)
    DEFAULT_CKPT=$STORE/ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt
    PAST_INIT=random
    ;;
  *) echo "INIT_MODE must be stage1 or openvla" >&2; exit 2 ;;
esac
CKPT=${CKPT:-$DEFAULT_CKPT}
RUN_ID=${RUN_ID:-past_s2_${INIT_MODE}_smoke}
MAX_STEPS=${MAX_STEPS:-250}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}

# Never reuse an existing run: old flow logs were ambiguous after auto-restarts.
if [[ -e "$STORE/outputs/stage2/$RUN_ID--image_aug" ]]; then
  echo "Run already exists; choose a new RUN_ID: $RUN_ID--image_aug" >&2
  exit 2
fi
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }

export PYTHONPATH="$STORE/pylibs-train:$CODE"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}

cmd=(
  torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK"
  --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT"
  scripts/train.py
  --pretrained_checkpoint "$CKPT"
  --whenami_stage 2 --past_init "$PAST_INIT"
  --train_venc True --use_temporal False --obs_offsets "$OBS_OFFSETS"
  --vla.type prism-dinosiglip-224px+oxe+diffusion
  --vla.data_mix "${DATA_MIX:-oxe_magic_soup_plus}"
  --vla.expected_world_size "$TOTAL_GPUS"
  --vla.global_batch_size "$GLOBAL_BS"
  --vla.per_device_batch_size "$PER_DEVICE_BS"
  --vla.learning_rate "${LR:-3.2e-5}"
  --vla.shuffle_buffer_size "${SHUFFLE_BUFFER:-64000}"
  --vla.max_steps "$MAX_STEPS"
  --data_root_dir "$STORE/datasets/open-x-embodiment"
  --run_root_dir "$STORE/outputs/stage2" --run_id "$RUN_ID"
  --image_aug True --save_interval "$SAVE_INTERVAL"
  --repeated_diffusion_steps "${REPEATED_DIFFUSION_STEPS:-4}"
  --Ta 16 --action_model_type DiT-L --is_resume False
)

printf 'Stage 2: init=%s, offsets=%s, processes=%s, global_batch=%s\n' \
  "$INIT_MODE" "$OBS_OFFSETS" "$TOTAL_GPUS" "$GLOBAL_BS"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
cd "$CODE"
mkdir -p "$MPLCONFIGDIR"
[[ -f .hf_token ]] || { echo "Missing offline token placeholder .hf_token" >&2; exit 2; }
python3 scripts/check_past_stage2.py \
  --checkpoint "$CKPT" --past-init "$PAST_INIT" --obs-offsets "$OBS_OFFSETS" \
  --nproc "$NPROC_PER_NODE"
exec "${cmd[@]}"
