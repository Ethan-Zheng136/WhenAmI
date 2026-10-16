"""Launcher/CLI contract tests never allocate hardware or submit a job."""
import os
from pathlib import Path
import subprocess
import unittest

BASE = Path(__file__).resolve().parents[1]
SCRIPT = BASE/"run_whenami_timeline_force_320.sh"


class Tests(unittest.TestCase):
    def launch(self, **extra):
        env = dict(os.environ, WORLD_SIZE="20", RANK="0", NPROC_PER_NODE="16", DRY_RUN="1")
        for key in ("STAGE", "VARIANT", "PHASE", "TEACHER_CKPT", "RESUME_FROM"):
            env.pop(key, None)
        env.update(extra)
        return subprocess.run(["bash", str(SCRIPT)], env=env, cwd=BASE,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_teacher_50k_320_save5k_resume_admission(self):
        p = self.launch()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--steps 50000", p.stdout)
        self.assertIn("--nproc_per_node 16", p.stdout)
        self.assertIn("--global-batch 320", p.stdout)
        self.assertIn("--save-every 5000", p.stdout)
        self.assertIn("--stop-after 100", p.stdout)
        self.assertIn("--stop-after 120", p.stdout)
        self.assertIn("--resume-from", p.stdout)
        self.assertNotIn("--bf16-storage", p.stdout)
        self.assertNotIn("--engineering", p.stdout)

    def test_student_requires_pinned_teacher(self):
        fail = self.launch(STAGE="student")
        self.assertNotEqual(fail.returncode, 0)
        ok = self.launch(STAGE="student", TEACHER_CKPT="/example/Aplus/checkpoints/step-050000")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("--teacher /example/Aplus/checkpoints/step-050000", ok.stdout)

    def test_invalid_topology_and_unsafe_output(self):
        self.assertNotEqual(self.launch(WORLD_SIZE="320").returncode, 0)
        self.assertNotEqual(self.launch(NPROC_PER_NODE="8").returncode, 0)
        self.assertNotEqual(self.launch(RUN_ID="../overwrite").returncode, 0)
        self.assertNotEqual(self.launch(RANK="20").returncode, 0)

    def test_resume_is_explicit_new_output(self):
        self.assertNotEqual(self.launch(PHASE="resume").returncode, 0)
        ok = self.launch(PHASE="resume", RESUME_FROM="/example/step-005000", RUN_ID="fresh_resume")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("--resume-from /example/step-005000", ok.stdout)
        self.assertNotIn("--stop-after 100", ok.stdout)


if __name__ == "__main__":
    unittest.main()
