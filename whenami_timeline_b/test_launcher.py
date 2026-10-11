import os
import subprocess
import pytest
from stage3.common import BASE
from whenami_timeline_b.preflight import check_admission


@pytest.mark.parametrize("variant",("action_only","condition_align"))
@pytest.mark.parametrize("phase",("admit","run","full"))
def test_commands(variant,phase):
    env=dict(os.environ,WORLD_SIZE="20",RANK="0",NPROC_PER_NODE="16",DRY_RUN="1",
             B_VARIANT=variant,PHASE=phase)
    out=subprocess.check_output(["bash",str(BASE/"run_timeline_b_320.sh")],env=env,text=True)
    assert "--nnodes 20" in out and "--nproc_per_node 16" in out
    assert "--steps 20000" in out and "--global-batch 320" in out
    assert "--variant "+variant in out
    assert "--save-every 2500" in out and "--eval-every 1000" in out
    if phase in ("run","admit"):assert "--stop-after 100" in out and "--stop-after 120" in out
    if phase in ("run","full"):assert "/main" in out and "step-000120" in out


def test_no_large_training_without_proof(tmp_path):
    with pytest.raises(FileNotFoundError):check_admission(tmp_path,"condition_align")
