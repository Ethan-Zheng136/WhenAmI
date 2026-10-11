"""Same-FSDP-backend checkpoint replay gate, without rerunning extra model work."""
import json
from pathlib import Path
import numpy as np


def compare_saved(expected, actual, tolerance=1e-6):
    expected, actual = Path(expected), Path(actual)
    a = json.loads((expected/"RESULT.json").read_text())
    b = json.loads((actual/"RESULT.json").read_text())
    if any(a[k] != b[k] for k in ("complete", "episodes", "noise_seeds", "student_observed_nodes")):
        raise ValueError("restored model replay protocol differs from checkpoint evaluation")
    files = sorted(expected.glob("rank-*.npz"))
    if len(files) != len(list(actual.glob("rank-*.npz"))) or not files:
        raise ValueError("restored replay ranks missing")
    maximum, examples = 0., 0
    for p in files:
        with np.load(p) as x, np.load(actual/p.name) as y:
            for k in ("indices", "targets", "masks"):
                if not np.array_equal(x[k], y[k]):
                    raise ValueError("restored replay examples/labels changed: "+p.name)
            if x["predictions"].shape != y["predictions"].shape:
                raise ValueError("restored prediction shape differs")
            delta = np.abs(x["predictions"].astype(np.float64)-y["predictions"].astype(np.float64))
            if delta.size:
                if not np.isfinite(delta).all():
                    raise ValueError("restored predictions are nonfinite")
                maximum = max(maximum, float(delta.max()))
            examples += len(x["indices"])
    if examples != a["episodes"] or maximum > tolerance:
        raise ValueError(f"restored checkpoint replay mismatch: episodes={examples}, max_abs={maximum}")
    return {"complete": True, "episodes": examples, "noise_seeds": a["noise_seeds"],
            "max_prediction_abs_error": maximum, "tolerance": tolerance,
            "expected": str(expected), "actual": str(actual),
            "scope": "same FSDP backend checkpoint replay; not cross-device native parity"}
