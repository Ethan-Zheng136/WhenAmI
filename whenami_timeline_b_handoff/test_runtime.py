"""CPU serialization safeguards; full-size numerical verification is separate."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from whenami_timeline_b_handoff.runtime import SCHEMA, load_release, load_dcp, sha


class Tests(unittest.TestCase):
    def make(self, root, change=None):
        source = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
        source.register_buffer("scale", torch.ones(2, dtype=torch.float32))
        state = {k: v.clone() for k, v in source.state_dict().items()}
        if change:
            change(state)
        file = root / "model.safetensors"
        save_file(state, str(file))
        manifest = {"schema": SCHEMA, "complete": True, "state_entries": 3,
                    "shards": [{"file": file.name, "keys": list(state), "sha256": sha(file)}]}
        (root / "MODEL.json").write_text(json.dumps(manifest))
        target = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
        target.register_buffer("scale", torch.zeros(2, dtype=torch.float32))
        return source, target

    def test_complete_mixed_dtype_load(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = self.make(Path(d))
            load_release(target, d)
            self.assertTrue(all(torch.equal(t, target.state_dict()[k]) for k, t in source.state_dict().items()))

    def test_missing_key(self):
        with tempfile.TemporaryDirectory() as d:
            _, target = self.make(Path(d), lambda s: s.pop("bias"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_release(target, d)

    def test_dtype_change(self):
        with tempfile.TemporaryDirectory() as d:
            _, target = self.make(Path(d), lambda s: s.update(weight=s["weight"].float()))
            with self.assertRaisesRegex(ValueError, "dtype"):
                load_release(target, d)

    def test_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            _, target = self.make(Path(d), lambda s: s["weight"].fill_(float("nan")))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_release(target, d)

    def test_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, target = self.make(root)
            with (root / "model.safetensors").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_release(target, d)

    def test_duplicate_shard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, target = self.make(root)
            meta = json.loads((root / "MODEL.json").read_text())
            meta["shards"] *= 2
            (root / "MODEL.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_release(target, d)

    def test_reject_cpu_dcp_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "COMMITTED.json").write_text(json.dumps({
                "complete": True, "recipe": {"settings": {"variant": "action_only"}},
            }))
            with self.assertRaisesRegex(ValueError, "accelerator"):
                load_dcp(torch.nn.Linear(2, 2), root)


if __name__ == "__main__":
    unittest.main()
