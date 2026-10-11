import os
import subprocess
import pytest
from stage3.common import BASE
from whenami_pooled_b50.preflight import check_admission


@pytest.mark.parametrize("variant",("action_only","condition_align"))
@pytest.mark.parametrize("phase",("admit","run","full"))
def test_commands(variant,phase):
    env=dict(os.environ,WORLD_SIZE="20",RANK="0",NPROC_PER_NODE="16",DRY_RUN="1",
             POOLED_B_VARIANT=variant,PHASE=phase)
    out=subprocess.check_output(["bash",str(BASE/"run_pooled_b50_320.sh")],env=env,text=True)
    assert "--nnodes 20" in out and "--nproc_per_node 16" in out
    assert "--steps 20000" in out and "--global-batch 320" in out
    assert "--variant "+variant in out
    assert "--save-every 2500" in out and "--eval-every 1000" in out
    assert "--lr 1e-5" in out and "--aux-lr 1e-4" in out and "--warmup 1000" in out
    assert "--micro 1" in out and "--alignment-weight 0.1" in out
    assert "--module whenami_pooled_b50.train" in out and "A50k_320_20k" in out
    assert "/outputs/whenami_pooled_b50/training/" in out
    assert "whenami_timeline_c" not in out and "A10k" not in out
    if phase in ("run","admit"):assert "--stop-after 100" in out and "--stop-after 120" in out
    if phase in ("run","full"):assert "/main" in out and "step-000120" in out


def test_no_large_training_without_proof(tmp_path):
    with pytest.raises(FileNotFoundError):check_admission(tmp_path,"condition_align")


@pytest.mark.parametrize("nodes,local", (("40","16"),("20","8"),("1","16")))
def test_wrong_topology_rejected(nodes,local):
    env=dict(os.environ,WORLD_SIZE=nodes,RANK="0",NPROC_PER_NODE=local,DRY_RUN="1")
    result=subprocess.run(["bash",str(BASE/"run_pooled_b50_320.sh")],env=env,capture_output=True)
    assert result.returncode==2
