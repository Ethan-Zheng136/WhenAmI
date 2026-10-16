#!/usr/bin/env bash
# wfwd diagnosis: signals_04 config (FP32 heads, head-lr 1e-4, bwd init 0, freeze-old, 1 PPU) + real-like LR schedule.
source /etc/profile.d/ppu-sdk-env.sh >/dev/null 2>&1
BASE=/home/zhengguantian/WoG_fresh; D=/home/zhengguantian/wfwd_diag; STORE=/mnt/pfs/zhengguantian/WoG_v2
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$D:$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $BASE
for spec in "base:" "h3_fusion_detach:--fusion-detach-w" "h2_agree_detach:--agree-detach-var"; do
  name=${spec%%:*}; flags=${spec#*:}
  L=$STORE/outputs/whenami_present/local/wfwd_diag_$name; mkdir -p $L
  [ -f $L/teacher/COMPLETE.json ] && continue
  rm -rf $L/teacher
  torchrun --nproc_per_node 1 --master_port 29934 --module whenami_present.train --stage teacher --output $L/teacher \
    --steps 50000 --stop-after ${STOP:-700} --warmup 300 --micro 2 --global-batch 2 --sharding full --engineering --freeze-old \
    --save-every 100000 --eval-every 100000 --first-check 2 --eval-max 8 --probe-episodes 4 --no-initial-eval \
    --diag-tokens --no-final-save $flags > $L/teacher.log 2>&1
  echo EXIT $? >> $L/teacher.log
done
echo ALLDONE > $D/ALLDONE
