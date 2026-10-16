#!/usr/bin/env bash
# Unfrozen (except VLM) replication of the 64-card dynamics on 1 PPU: real LR groups 1e-6/1e-5/1e-4, warmup 300, 800 steps.
source /etc/profile.d/ppu-sdk-env.sh >/dev/null 2>&1
BASE=/home/zhengguantian/WoG_fresh; D=/home/zhengguantian/wfwd_diag; STORE=/mnt/pfs/zhengguantian/WoG_v2
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$D:$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $BASE
for spec in "A_base:" "C_fwdonly:--fusion forward" "B_agreedetach:--agree-detach-var" "D_mean:--fusion mean"; do
  name=${spec%%:*}; flags=${spec#*:}
  L=$STORE/outputs/whenami_present/local/wfwd_diag2_$name; mkdir -p $L
  [ -f $L/teacher/COMPLETE.json ] && continue
  rm -rf $L/teacher
  torchrun --nproc_per_node 1 --master_port 29936 --module whenami_present.train --stage teacher --output $L/teacher \
    --steps 50000 --stop-after 800 --warmup 300 --micro 2 --global-batch 2 --sharding full --engineering --freeze-vlm \
    --save-every 100000 --eval-every 100000 --first-check 2 --eval-max 64 --probe-episodes 4 --no-initial-eval \
    --diag-tokens --no-final-save $flags > $L/teacher.log 2>&1
  echo EXIT $? >> $L/teacher.log
done
echo ALLDONE > $D/ALLDONE2
