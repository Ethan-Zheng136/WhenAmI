import os
import subprocess
import pytest
from stage3.common import BASE
from whenami_timeline_joint.preflight import check_large_admission
from whenami_timeline_joint.runtime import learning_rate


@pytest.mark.parametrize("mode",("timeline","pooled_query"))
@pytest.mark.parametrize("phase",("admit","run","full"))
def test_dryrun(mode,phase):
    env=dict(os.environ,WORLD_SIZE="20",RANK="0",NPROC_PER_NODE="16",DRY_RUN="1",JOINT_MODE=mode,PHASE=phase)
    out=subprocess.check_output(["bash",str(BASE/"run_timeline_joint_320.sh")],env=env,text=True)
    assert "--nnodes 20" in out and "--nproc_per_node 16" in out
    assert "--steps 50000" in out and "--global-batch 320" in out and "--micro 1" in out
    assert "--mode "+mode in out
    if phase in ("admit","run"):
        assert "--stop-after 100" in out and "--stop-after 120" in out
        if phase=="run":assert "/main" in out and "step-000120" in out
    else:assert "step-000120" in out and "--stop-after" not in out


def test_long_run_rejected_without_admission(tmp_path):
    with pytest.raises(FileNotFoundError):check_large_admission(tmp_path,"timeline")


def test_schedule_does_not_restart_at_admission_boundary():
    assert learning_rate(100,50000,1e-5)==1e-6
    assert learning_rate(101,50000,1e-5)>learning_rate(100,50000,1e-5)
    assert learning_rate(1000,50000,1e-5)==1e-5
    assert abs(learning_rate(50000,50000,1e-5)-1e-6)<1e-14
