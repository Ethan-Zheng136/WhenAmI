#!/usr/bin/env bash
# usage: probe.sh <ckpt dir> <tag>   (waits for the diag runs to finish; single PPU)
D=/home/zhengguantian/wfwd_diag; while [ ! -f $D/ALLDONE ]; do sleep 60; done
source /etc/profile.d/ppu-sdk-env.sh >/dev/null 2>&1
BASE=/home/zhengguantian/WoG_fresh; STORE=/mnt/pfs/zhengguantian/WoG_v2
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$D:$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
O=$STORE/outputs/whenami_present/local/wfwd_diag_probe_$2; cd $BASE
torchrun --nproc_per_node 1 --master_port 29935 --module probe_tokens --ckpt $1 --out $O --batches ${BATCHES:-150} > $O.log 2>&1; echo EXIT $? >> $O.log
