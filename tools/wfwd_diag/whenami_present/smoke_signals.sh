#!/usr/bin/env bash
# Signal smoke: FP32 storage, only NEW modules train (backbone frozen so FP32 Adam fits 1 PPU),
# real LR (1e-5 new), NO warmup, real changing batches, future-drop .25, hidden-every 4.
BASE=/home/zhengguantian/WoG_fresh; STORE=/mnt/pfs/zhengguantian/WoG_v2; L=$STORE/outputs/whenami_present/local/${TAG:-signals_01}; mkdir -p $L
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $BASE
torchrun --nproc_per_node 1 --master_port 29933 --module whenami_present.train --stage teacher --output $L/teacher \
  --steps ${STEPS:-300} --micro 2 --global-batch 2 --warmup 0 --sharding full --engineering --freeze-old \
  --save-every 100000 --eval-every ${EVAL_EVERY:-150} --first-check 2 --eval-max 8 --probe-episodes 4 --no-initial-eval ${EXTRA:-} > $L/teacher.log 2>&1
echo EXIT $? >> $L/teacher.log
