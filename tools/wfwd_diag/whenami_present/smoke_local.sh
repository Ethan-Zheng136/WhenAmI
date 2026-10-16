#!/usr/bin/env bash
# Single-PPU engineering smoke: Stage I 21 steps + resume, Stage II 21 steps + resume.
set -uo pipefail
BASE=/home/zhengguantian/WoG_fresh; STORE=/mnt/pfs/zhengguantian/WoG_v2
L=$STORE/outputs/whenami_present/local/${TAG:-smoke_01}; mkdir -p $L
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $BASE
T="torchrun --nnodes 1 --nproc_per_node 1 --master_port 29931 --module whenami_present.train"
C="--steps 40 --global-batch 1 --micro 1 --warmup 5 --sharding full --activation llm --engineering --bf16-storage --save-every 20 --eval-every 1000 --first-check 2 --eval-max 2 --probe-episodes 2 ${SMOKE_EXTRA:-}"
run(){ name=$1; shift; echo "=== $name $(date -u)"; "$@" > $L/$name.log 2>&1; echo "=== $name exit $? $(date -u)"; }
run teacher1 $T --stage teacher --output $L/teacher1 $C --stop-after 21
run teacher2 $T --stage teacher --output $L/teacher2 $C --stop-after 21 --resume-from $L/teacher1/checkpoints/step-000020 --no-initial-eval
run student1 $T --stage student --teacher $L/teacher1/checkpoints/step-000021 --output $L/student1 $C --stop-after 21
run student2 $T --stage student --teacher $L/teacher1/checkpoints/step-000021 --output $L/student2 $C --stop-after 21 --resume-from $L/student1/checkpoints/step-000020 --no-initial-eval
echo SMOKE_ALL_DONE
