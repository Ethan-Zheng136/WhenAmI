import json, sys, numpy as np
from pathlib import Path
V = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/validation_25k")
T = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/timeline_force_Aplus_160_resume5k_01/main/development")
for step in sys.argv[1:]:
    d = json.loads((V/f"cf_step{step}.json").read_text())
    ref = json.loads((T/f"step-{step}"/"RESULT.json").read_text())["macro"]
    m = d["macro"]
    print(f"== step {step}  episodes {d['episodes']}  ({d['seconds']:.0f}s)")
    print("train-time RESULT (FP32 FSDP):", {k: (round(v["mse6"], 5), round(v["first_action_mse6"], 5)) for k, v in ref.items()})
    for k in ["true/causal_last", "true/causal_middle", "true/hindsight_middle", "swap/hindsight_middle", "repeat/hindsight_middle", "swap/causal_middle", "repeat/causal_middle"]:
        print(f"{k:28s} mse6 {m[k]['mse6']:.5f}  first {m[k]['first_action_mse6']:.5f}  grip {m[k]['gripper_accuracy']:.4f}")
    cm = m["true/causal_middle"]
    for c in ("true", "swap", "repeat"):
        h = m[c+"/hindsight_middle"]
        print(f"  {c}: hindsight vs causal_middle mse6 {100*(h['mse6']/cm['mse6']-1):+.2f}%  first {100*(h['first_action_mse6']/cm['first_action_mse6']-1):+.2f}%")
    rows = d["rows"]
    t = np.array([r["true"]["hindsight_middle"]["mse6"] for r in rows])
    rng = np.random.default_rng(0)
    for c in ("swap", "repeat"):
        x = np.array([r[c]["hindsight_middle"]["mse6"] for r in rows])
        diff = x - t
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        print(f"  paired {c}-true (episode-mean): {diff.mean():+.5f} 95%CI [{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] frac eps true better {np.mean(diff>0):.2f}")
    print("  leak check max|causal_middle swap-true|:", max(abs(r["swap"]["causal_middle"]["mse6"]-r["true"]["causal_middle"]["mse6"]) for r in rows),
          " repeat:", max(abs(r["repeat"]["causal_middle"]["mse6"]-r["true"]["causal_middle"]["mse6"]) for r in rows))
    for dom in ("bridge_orig", "fractal20220817_data", "droid", "language_table", "kuka", "taco_play", "berkeley_autolab_ur5"):
        x = d["domains"][dom]
        print(f"  {dom:24s} C:{x['true/causal_middle']['mse6']:.5f} H-true:{x['true/hindsight_middle']['mse6']:.5f} H-swap:{x['swap/hindsight_middle']['mse6']:.5f} H-rep:{x['repeat/hindsight_middle']['mse6']:.5f}")
    for c in ("swap", "repeat"):
        n = sum(d["domains"][k][c+"/hindsight_middle"]["mse6"] > d["domains"][k]["true/hindsight_middle"]["mse6"] for k in d["domains"])
        print(f"  domains where {c} worse than true: {n}/{len(d['domains'])}")
