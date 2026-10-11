import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from stage3.common import BASE
from whenami_timeline_a_training.preflight import check_admission


@pytest.mark.parametrize("mode", ("pooled", "independent", "timeline"))
@pytest.mark.parametrize("phase", ("admit", "full"))
def test_launcher_budget_and_mode(mode, phase):
    env = dict(os.environ, WORLD_SIZE="2", RANK="0", NPROC_PER_NODE="16", DRY_RUN="1",
               TL_MODE=mode, PHASE=phase)
    out = subprocess.check_output(["bash", str(BASE / "run_timeline_a_32.sh")], env=env, text=True)
    assert "--nproc_per_node 16" in out and "--nnodes 2" in out
    assert "--mode " + mode in out
    assert "--steps 10000" in out and "--micro 2" in out
    assert "--eval-every 500" in out and "--save-every 1000" in out
    if phase == "admit":
        assert "--stop-after 100" in out and "--stop-after 120" in out
    else:
        assert "step-000120" in out and "--stop-after" not in out


def test_full_is_rejected_without_multinode_admission(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_admission(tmp_path, "timeline")


def test_development_manifest_has_correct_shapes_and_no_duplicate_ids():
    from whenami_timeline_a_training.train import DEFAULT_DEV
    path = DEFAULT_DEV / "MANIFEST.json"
    if not path.exists():
        pytest.skip("development preparation not finished")
    m = json.loads(path.read_text())
    assert m["complete"] and m["domains"] == 27 and m["train_overlap"] == 0
    assert len({(r["dataset"],r["episode_key"]) for r in m["clips"]}) == m["episodes"]
    for row in m["clips"]:
        assert row["role"] == "dev"
        with np.load(DEFAULT_DEV / row["file"]) as a:
            assert a["images"].shape == (3,224,224,3)
            assert a["raw_actions"].shape == (3,16,7)
            assert a["action_mask"].shape == (3,16)
            assert (np.diff(a["times"])>0).all()
