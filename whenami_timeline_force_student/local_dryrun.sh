#!/usr/bin/env bash
# Single-PPU ENGINEERING dry run of the Stage II entry with the REAL A+ step-040000 teacher.
# Same as local/train_student_01 (gb1, BF16 storage, full shard) but draws2 and real teacher.
set -euo pipefail
source /etc/profile.d/ppu-sdk-env.sh >/dev/null 2>&1 || true
BASE=/home/zhengguantian/WoG_fresh; STORE=/mnt/pfs/zhengguantian/WoG_v2
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
OUT=$STORE/outputs/whenami_timeline_force/local/${1:-student_from40k_dryrun_01}
DRAWS=${DRAWS:-2}
cd "$BASE"
torchrun --nnodes 1 --nproc_per_node 1 --master_port 29911 --module whenami_timeline_force_student.train --base 160 \
  --stage student --variant revise \
  --teacher $STORE/outputs/whenami_timeline_force/training/timeline_force_Aplus_160_resume5k_01/main/checkpoints/step-040000 \
  --steps 3 --micro 1 --global-batch 1 --seed 42 --lr 1e-6 --new-lr 1e-5 --warmup 1 --draws "$DRAWS" \
  --current-weight 0.1 --cognition-weight 0.1 --revision-weight 0.1 --sharding full --activation llm \
  --save-every 3 --eval-every 3 --first-check 3 --dev-limit 1 --eval-max 1 --engineering --bf16-storage \
  --output "$OUT"
echo DRYRUN_EXIT_OK
