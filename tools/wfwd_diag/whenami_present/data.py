"""Training batches with episode length (weak progress label) and a dense progress PROBE set."""
import numpy as np
import torch
from whenami_unified.data import Inputs as UnifiedInputs, batch, executed   # noqa: F401
from whenami_timeline_joint.data import INDEX
from whenami_timeline_a.data import RawAnchors


class Inputs(UnifiedInputs):
    def batch(self, count, seed, step, rank, micro=0):
        cases = []
        for i in range(count):
            e, t = self.raw.select(seed, step, rank, micro*count+i)
            obs, labels, prov = self.raw.materialize(e, t, self.stats)
            prov["episode_length"] = int(e["length"])
            cases.append({"observations": obs, "labels": labels, "provenance": prov})
        return cases


def future_times(cases, presents, device):
    """Timestamp-only future targets for the student imaginer (teacher anchors' times)."""
    t = torch.stack([c["observations"]["times"] for c in cases]).to(device)
    return {c: t[:, c+1:] for c in presents}


def canonical_times(times, presents, gap=4, count=3):
    """Deployment imaginer targets: fixed history_gap offsets after the present."""
    return {c: times[:, c:c+1] + gap*torch.arange(1, count+1, device=times.device)[None] for c in presents}


class ProbeSet:
    """Fixed train-index episodes (NOT held-out; DEV lacks episode lengths), 3-anchor causal
    windows (f-2g, f-g, f) at `frames` evenly spaced f per episode."""
    def __init__(self, stats, episodes=48, frames=8, gap=4, seed=7):
        self.raw = RawAnchors(INDEX, 3)
        self.stats, self.gap = stats, gap
        rng = np.random.default_rng(seed)
        pools = {d: [e for e in self.raw.episodes[d] if e["length"] >= 2*gap+frames+4] for d in self.raw.domains}
        pools = {d: r for d, r in pools.items() if r}
        order, self.items = sorted(pools), []
        while len(self.items) < episodes*frames:
            for d in order:
                if len(self.items) >= episodes*frames:
                    break
                e = pools[d][int(rng.integers(len(pools[d])))]
                L = int(e["length"])
                fs = np.unique(np.linspace(2*gap, L-2, frames).round().astype(int))
                for f in fs[:frames]:
                    self.items.append((e, int(f), len(self.items)//frames))

    def __len__(self):
        return len(self.items)

    def case(self, i):
        e, f, ep = self.items[i]
        times = np.array([f-2*self.gap, f-self.gap, f], dtype=np.int64)
        obs, labels, prov = self.raw.materialize(e, times, self.stats)
        prov.update(episode_length=int(e["length"]), probe_frame=f, probe_episode=ep,
                    progress=f/(int(e["length"])-1))
        return {"observations": obs, "labels": labels, "provenance": prov}
