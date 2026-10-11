#!/usr/bin/env bash
# 2 nodes x16 distinct PPUs. Real8B train -> DCP save -> new process resume.
# Admission runs are correctness tests, NOT scientific performance claims.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
MICRO_BATCH=${MICRO_BATCH:-1}
FM_DRAWS=${FM_DRAWS:-1}
[[ $NNODES == 2 && $NPROC_PER_NODE == 16 ]] || {
  echo "Admission requires exactly 2 nodes x16 distinct PPUs." >&2; exit 2;
}
ROOT=${ADMISSION_ROOT:-$STORE/outputs/stage3_joint/admission_32_01}
DATA_INDEX=${DATA_INDEX:-$STORE/outputs/stage3_joint/raw_smoke_01/train_index.json}
# Every node checks before torchrun. These artifacts cannot be created by a
# fresh run until all nodes have joined, so this does not race normal startup.
# Never delete/overwrite an old checkpoint to make a correctness test pass.
if [[ ${DRY_RUN:-0} != 1 ]]; then
  for artifact in tiny_fsdp latent forecast distill_future ADMISSION_COMPLETE.json; do
    if [[ -e "$ROOT/$artifact" || -L "$ROOT/$artifact" ]]; then
      echo "Admission output already exists: $ROOT/$artifact" >&2
      echo "Stopped BEFORE torchrun; existing checkpoints/logs are unchanged." >&2
      echo "Choose a fresh ADMISSION_ROOT, identical on BOTH nodes." >&2
      echo "For OXE27 use run_stage3_oxe27_admission.sh with its fresh v2 directory." >&2
      echo "Do not add RESUME=1: this wrapper tests save/resume internally." >&2
      exit 2
    fi
  done
  [[ -f "$DATA_INDEX" ]] || { echo "Data index not found: $DATA_INDEX" >&2; exit 2; }
  printf '[admission] node=%s root=%s data=%s micro_batch=%s fm_draws=%s\n' \
    "$NODE_RANK" "$ROOT" "$DATA_INDEX" "$MICRO_BATCH" "$FM_DRAWS"
fi
export PYTHONPATH="$STORE/pylibs-train:$BASE:$BASE/WoG_16rank_probe"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export HF_HOME="$STORE/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$BASE"
prefix=(torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE"
        --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29582}")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${prefix[@]}" --module stage3_scale.test_fsdp_ppu --output "$ROOT/tiny_fsdp"; printf '\n'
fi
if [[ ${DRY_RUN:-0} != 1 ]]; then
  mkdir -p "$ROOT"
  # First test FSDP collectives+DCP across DISTINCT devices, before loading8B.
  "${prefix[@]}" --module stage3_scale.test_fsdp_ppu --output "$ROOT/tiny_fsdp" \
    2>&1 | tee -a "$ROOT/tiny-node${NODE_RANK}.log"
fi
for variant in latent forecast distill_future; do
  cmd=("${prefix[@]}" --module stage3_scale.train_joint --data-index "$DATA_INDEX"
       --output "$ROOT/$variant" --variant "$variant" --steps 4 --warmup 1
       --global-batch "$((32 * MICRO_BATCH))" --micro-batch "$MICRO_BATCH" --fm-draws "$FM_DRAWS" --save-every 2
       --allow-partial-data --admission-gradient-audit --development-limit 16 --development-every 4)
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${cmd[@]}" --stop-after 2; printf '\n'
    printf '%q ' "${cmd[@]}" --resume; printf '\n'
  else
    "${cmd[@]}" --stop-after 2 2>&1 | tee -a "$ROOT/${variant}-node${NODE_RANK}.log"
    "${cmd[@]}" --resume 2>&1 | tee -a "$ROOT/${variant}-node${NODE_RANK}.log"
  fi
done
if [[ ${DRY_RUN:-0} != 1 && $NODE_RANK == 0 ]]; then
  python - "$ROOT" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]); recipes=[]
for variant in ("latent","forecast","distill_future"):
    run=root/variant
    done=json.loads((run/"complete.json").read_text())
    assert done["complete"] and done["steps"]==4 and done["final_development_action_eval"]
    assert json.loads((run/"development/step-000004/results.json").read_text())["complete"]
    for step in (2,4):
        cp=json.loads((run/"checkpoints"/f"step-{step:06d}"/"COMMITTED.json").read_text())
        assert cp["complete"] and cp["world_size"]==32
    for step in (1,3):
        assert json.loads((run/f"gradient-audit-step{step:06d}.json").read_text())["complete"]
    recipes.append(json.loads((run/"recipe.json").read_text()))
assert all(r["joint_source"]==recipes[0]["joint_source"] for r in recipes)
(root/"ADMISSION_COMPLETE.json").write_text(json.dumps({
    "complete":True,"world_size":32,"variants":["latent","forecast","distill_future"],
    "full_policy_train_save_resume_passed":True,"joint_source":recipes[0]["joint_source"],
    "checkpoint_schema":"fsdp_model_dcp_local_adam_v2",
    "micro_batch":recipes[0]["settings"]["micro_batch"],
    "fm_draws":recipes[0]["settings"]["fm_draws"],
    "data_index_sha256":recipes[0]["data_index_sha256"],
    "reference_source":recipes[0]["reference_source"],
    "reused_source":recipes[0]["reused_source"],
    "note":"Four-step admission on partial raw shards, not a performance evaluation."},indent=2))
PY
fi
