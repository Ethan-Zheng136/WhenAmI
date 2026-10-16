#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/ppu-sdk-env.sh >/dev/null 2>&1 || true
set -u
BASE=/home/zhengguantian/WoG_fresh; STORE=/mnt/pfs/zhengguantian/WoG_v2
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$BASE"
V=$STORE/outputs/whenami_timeline_force/validation_25k
python $V/counterfactual_hide.py "$@"
