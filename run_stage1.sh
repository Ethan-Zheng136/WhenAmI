#!/bin/bash
# WoG Stage I —— PPU-DLC 集群启动脚本
#
# 平台: Neolix 训练平台 / PPU-DLC-QUEUE / PyTorch 框架
#       多机模式下 PAI 后端不支持碎卡，每节点锁定 16 卡
# 镜像: torchengine-torch280-cu129-py312-v103-tmp5
#
# 依赖: 先跑一次 setup_train_deps.sh 把依赖装到 $PKGS（已装好，在 PFS 上，集群可直接读）
# 已实测验证(2026-09-21): 集群镜像 python3 + PFS 依赖能完整读出 RLDS 数据
#   ACTION (16,7) / IMAGE (224,224,3) / 真实语言指令
set -x

# ============ 分布式参数 ============
# 平台在 PyTorch 框架模式下注入 MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK
NPROC_PER_NODE=${NPROC_PER_NODE:-8}       # 平台锁定每节点 16 卡
NNODES=${WORLD_SIZE:-1}                     # 节点数（截图里 Worker 实例数）
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_GPUS=$(( NPROC_PER_NODE * NNODES ))

# 论文官方 Stage I: 32 卡 / global_batch 1024 / per_device 32
# per_device 固定 32（显存决定），global 随卡数走
PER_DEVICE_BS=${PER_DEVICE_BS:-32}
GLOBAL_BS=$(( PER_DEVICE_BS * TOTAL_GPUS ))

CODE=/home/zhengguantian/WoG_fresh/WoG
STORE=/mnt/pfs/zhengguantian/WoG_v2
PKGS=$STORE/pylibs-train

# ============ 环境 ============
# PKGS 必须在最前：镜像自带 transformers 4.57 / numpy 2.3，
# 而 WoG 要 transformers 4.40.1 + timm 0.9.10 + numpy 1.26.4（TF 2.16 要求）
export PYTHONPATH=$PKGS:$CODE
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=/tmp/mpl
export HF_HOME=$STORE/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}

cd $CODE
mkdir -p $STORE/outputs/stage1 /tmp/mpl

# train.py:62 默认从 ./.hf_token 读 HF token。权重全在本地且 HF_HUB_OFFLINE=1，
# 内容无所谓，但文件必须存在，否则 read_text() 抛 FileNotFoundError。
# （该文件在官方 .gitignore:135 里，不会污染仓库）
[ -f "$CODE/.hf_token" ] || printf 'hf_placeholder_offline' > "$CODE/.hf_token"

# ============ 启动前自检（失败就别浪费卡时）============
python3 - <<'PY' || { echo "!!! 自检失败，终止"; exit 1; }
import sys
ok = True
import torch, numpy, tensorflow as tf, tensorflow_datasets as tfds, dlimp
import transformers, timm, draccus
print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
print(f"  numpy {numpy.__version__} (需 1.26.x) | tf {tf.__version__} | tfds {tfds.__version__}")
print(f"  transformers {transformers.__version__} (需 4.40.1) | timm {timm.__version__} (需 0.9.10)")
if not transformers.__version__.startswith("4.40"):
    print("  !! transformers 版本不对 —— PYTHONPATH 顺序有问题"); ok = False
if not numpy.__version__.startswith("1.26"):
    print("  !! numpy 版本不对 —— TF 2.16 需要 1.26.x"); ok = False
from pathlib import Path
ck = Path("/mnt/pfs/zhengguantian/WoG_v2/ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt")
print(f"  {'OK  ' if ck.exists() else 'MISS'} openvla 起点 ({ck.stat().st_size/2**30:.1f} GB)" if ck.exists() else f"  MISS {ck}")
if not ck.exists(): ok = False
sys.exit(0 if ok else 1)
PY

# ============ 启动 ============
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
  --run_root_dir $STORE/outputs/stage1 \
  --run_id ${RUN_ID:-wog_stage1} \
  --image_aug True \
  --save_interval ${SAVE_INTERVAL:-5000} \
  --repeated_diffusion_steps 8 \
  --Ta 16 \
  --action_model_type DiT-L \
  --is_resume ${IS_RESUME:-False} \
  ${RESUME_STEP:+--resume_step $RESUME_STEP} \
  ${RESUME_EPOCH:+--resume_epoch $RESUME_EPOCH} \
  --train_venc True
