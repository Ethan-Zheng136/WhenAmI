#!/usr/bin/env bash
# fixed-batch overfit check (engineering): losses must decrease
BASE=/home/zhengguantian/WoG_fresh; STORE=/mnt/pfs/zhengguantian/WoG_v2; L=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_present/local/overfit_01; mkdir -p $L
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $BASE
C="--steps 40 --global-batch 1 --warmup 2 --sharding full --engineering --bf16-storage --fixed-data --lr 1e-5 --new-lr 1e-3 --save-every 1000 --eval-every 1000 --first-check 2 --eval-max 1 --probe-episodes 1 --no-initial-eval"
torchrun --nproc_per_node 1 --master_port 29932 --module whenami_present.train --stage teacher --output $L/teacher $C > $L/teacher.log 2>&1; echo teacher $?
torchrun --nproc_per_node 1 --master_port 29932 --module whenami_present.train --stage student --teacher /mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_present/local/smoke_03/teacher1/checkpoints/step-000021 --output $L/student $C > $L/student.log 2>&1; echo student $?
echo OVERFIT_DONE
