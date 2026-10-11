#!/usr/bin/env bash
# Retry corrected optimizer checkpoint on distinct32 PPUs + actual27-domain raw inputs.
set -euo pipefail
BASE=/home/zhengguantian/WoG_fresh
STORE=/mnt/pfs/zhengguantian/WoG_v2
export ADMISSION_ROOT=${ADMISSION_ROOT:-$STORE/outputs/stage3_joint/oxe27_admission_32_v2_01}
export DATA_INDEX=${DATA_INDEX:-$STORE/outputs/stage3_joint/oxe27_raw_smoke_01/oxe27_train_index.json}
export MICRO_BATCH=${MICRO_BATCH:-2}
export FM_DRAWS=${FM_DRAWS:-4}
export MASTER_PORT=${MASTER_PORT:-29592}
if [[ ${DRY_RUN:-0} != 1 ]]; then
  python - "$DATA_INDEX" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get("data_mix")!="oxe_magic_soup_plus" or d.get("domain_count")!=27:
    raise SystemExit("Need the actual27-domain raw admission fixture")
if d.get("window_policy")!="episode_boundary_repeat_with_valid_action_mask":
    raise SystemExit("Boundary mask contract missing")
PY
fi
exec bash "$BASE/run_stage3_joint_admission.sh"
