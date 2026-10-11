#!/bin/bash
# [zgt] Stage I 条件质量评测 —— 集群多卡
#
# 对比 WoG(QFormer 5 帧) 与 WhenAmI(时间线 9 帧) 在同一批样本、
# 同样喂真实未来的前提下，条件注入 DiT 后的动作预测精度。
#
# 提交方式与训练任务相同（PPU-DLC / PyTorch 框架），Worker 实例数填 1。
# 16 卡跑全量约几分钟。
set -x

NPROC_PER_NODE=${NPROC_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29533}

CODE=/home/zhengguantian/WoG_fresh/WoG
STORE=/mnt/pfs/zhengguantian/WoG_v2
PKGS=$STORE/pylibs-train

# PKGS 必须在最前：镜像自带 transformers 4.57 / numpy 2.3，
# 而 WoG 要 transformers 4.40.1 + timm 0.9.10 + numpy 1.26.4
export PYTHONPATH=$PKGS:$CODE
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=/tmp/mpl
export HF_HOME=$STORE/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $CODE
mkdir -p /tmp/mpl
[ -f "$CODE/.hf_token" ] || printf 'hf_placeholder_offline' > "$CODE/.hf_token"

WOG=${WOG_CKPT:-$STORE/outputs/stage1/baseline_v2--image_aug/checkpoints/step-010000-epoch-01-loss=0.1111.pt}
WAI=${WAI_CKPT:-$STORE/outputs/whenami/whenami_v3--image_aug/checkpoints/step-010000-epoch-01-loss=0.1325.pt}

for f in "$WOG" "$WAI"; do
  [ -f "$f" ] || { echo "!!! checkpoint 不存在: $f"; exit 1; }
done

torchrun \
  --nnodes 1 \
  --nproc_per_node $NPROC_PER_NODE \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  /home/zhengguantian/WoG_fresh/eval_conditioned_dist.py \
  --ckpt "WoG-10k=$WOG" \
  --ckpt "WhenAmI-10k=$WAI" \
  --temporal-for WhenAmI-10k \
  --datasets ${DATASETS:-fractal20220817_data bridge_orig} \
  --limit ${N:-0} \
  --out ${OUT_JSON:-$STORE/outputs/oxe/eval_cond_10k.json}
