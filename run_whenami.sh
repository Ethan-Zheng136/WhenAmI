#!/bin/bash
# When Am I? —— Stage 1 训练启动脚本
#
# 与 run_stage1.sh（原版 WoG baseline）的唯一区别是三个参数：
#     --use_temporal True
#     --obs_offsets "[-4,-3,-2,-1,0,1,2,3,4]"
#     future_action_window_size 由 obs_offsets 自动推导为 19
# 其余完全一致，故两者可直接对比。
#
# 平台: Neolix 训练平台 / PPU-DLC-QUEUE / PyTorch 框架
# 镜像: torchengine-torch280-cu129-py312-v103-tmp5
#
# NPROC_PER_NODE=8（关键，实测结论）
#   平台每节点 16 卡，但用满 16 个进程会触发系统内存 OOM
#   （exitcode -9 / SIGKILL，不是 CUDA OOM）。用 8 个进程可稳定运行。
#   吃内存的是每进程的 TF 数据管线（27 个数据集的 reader + prefetch），
#   而非模型权重 —— 模型侧已用 mmap+assign 优化到 11.6 GB/进程
#   （原版 61.8 GB，见 vla/vla.py 的 torch.load 改动）。
#   代价是每节点闲置 8 张卡，用节点数补回 global batch 即可。
#
# 断点续训（原版已支持，恢复 模型+优化器+调度器+步数）:
#   每次 save 落盘 87 GB（model 29 + optimizer 58），故 save_interval 不宜过密。
#   默认 5000 步一次 = 30000 步存 6 次 = 523 GB。
#   中断后续训:
#     CKPT=<run_dir>/checkpoints/step-XXXXXX-epoch-YY-loss=Z.pt \
#     IS_RESUME=True RESUME_STEP=XXXXXX RESUME_EPOCH=YY \
#     bash run_whenami.sh
#   注意 RESUME_STEP/EPOCH 必须与文件名中的数字一致（train.py 有 assert 校验）。
#
# 已验证可跑（2026-09-25，27 个数据集全量）:
#   NPROC_PER_NODE=8 PER_DEVICE_BS=8 -> 4.2 s/step
#   loss 1.20 (step5) -> 0.73 (step73)
set -x

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_GPUS=$(( NPROC_PER_NODE * NNODES ))

# per_device 默认 4（而非 WoG 原版的 32）：时间线上 9 个节点各跑一次 DiT，
# 激活全部保留到 backward。PPU 实测（rds=8）：
#     全节点 + per_device=16  ->  43.5 GB   OOM
#     采样3  + per_device=16  ->  15.6 GB   牺牲了监督完整性
#     全节点 + per_device=4   ->  12.6 GB   <- 更省，且不牺牲方法
# 故选择降 batch 而非降节点，global batch 靠卡数补足：
#     per_device=4 x 256卡 = 1024，与论文完全一致。
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
GLOBAL_BS=$(( PER_DEVICE_BS * TOTAL_GPUS ))

CODE=/home/zhengguantian/WoG_fresh/WoG
STORE=/mnt/pfs/zhengguantian/WoG_v2
PKGS=$STORE/pylibs-train

export PYTHONPATH=$PKGS:$CODE
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=/tmp/mpl
export HF_HOME=$STORE/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}

cd $CODE
mkdir -p $STORE/outputs/whenami /tmp/mpl
[ -f "$CODE/.hf_token" ] || printf 'hf_placeholder_offline' > "$CODE/.hf_token"

python3 - <<'PY' || { echo "!!! preflight failed"; exit 1; }
import sys
import torch, numpy, tensorflow as tf, tensorflow_datasets as tfds, dlimp
import transformers, timm, draccus
print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
print(f"  numpy {numpy.__version__} | tf {tf.__version__} | transformers {transformers.__version__}")
ok = transformers.__version__.startswith("4.40") and numpy.__version__.startswith("1.26")
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")
from vla.temporal import TemporalConverge, StateSpec
print(f"  TemporalConverge importable, StateSpec={StateSpec()}")
from pathlib import Path
ck = Path("/mnt/pfs/zhengguantian/WoG_v2/ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt")
print(f"  {'OK  ' if ck.exists() else 'MISS'} openvla init ckpt")
sys.exit(0 if (ok and ck.exists()) else 1)
PY

torchrun \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --nproc_per_node $NPROC_PER_NODE \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  scripts/train.py \
  --pretrained_checkpoint ${CKPT:-$STORE/ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX:-oxe_magic_soup_plus} \
  --vla.expected_world_size $TOTAL_GPUS \
  --vla.global_batch_size $GLOBAL_BS \
  --vla.per_device_batch_size $PER_DEVICE_BS \
  --vla.learning_rate ${LR:-2e-5} \
  --vla.shuffle_buffer_size ${SHUFFLE_BUFFER:-64000} \
  --vla.max_steps ${MAX_STEPS:-100000} \
  --data_root_dir $STORE/datasets/open-x-embodiment \
  --run_root_dir $STORE/outputs/whenami \
  --run_id ${RUN_ID:-whenami_stage1} \
  --image_aug True \
  --save_interval ${SAVE_INTERVAL:-5000} \
  --repeated_diffusion_steps 8 \
  --Ta 16 \
  --action_model_type DiT-L \
  --is_resume ${IS_RESUME:-False} \
  ${RESUME_STEP:+--resume_step $RESUME_STEP} \
  ${RESUME_EPOCH:+--resume_epoch $RESUME_EPOCH} \
  --train_venc True \
  --use_temporal True \
  --obs_offsets "${OBS_OFFSETS:-[-4,-3,-2,-1,0,1,2,3,4]}" \
  --distill_weight ${DISTILL_WEIGHT:-0.0} \
  --nodes_per_step ${NODES_PER_STEP:-0} \
  --converge_type ${CONVERGE_TYPE:-recurrent} \
  --calib_recon_weight ${CALIB_RECON:-0.0} \
  --calib_act_weight ${CALIB_ACT:-0.0} \
  --flow_fwd_weight ${FLOW_FWD:-1.0} \
  --flow_bwd_weight ${FLOW_BWD:-1.0} \
  --flow_cycle_weight ${FLOW_CYCLE:-0.1} \
  --flow_enc_layers ${FLOW_ENC_L:-3} \
  --flow_cross_layers ${FLOW_CROSS_L:-2} \
  --flow_enc_hidden ${FLOW_ENC_H:-768} \
  --flow_enc_use_ffn ${FLOW_ENC_FFN:-True}
