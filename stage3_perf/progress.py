"""Rank-zero tqdm PLUS permanent plain-text ETA for platform/tee logs."""

from collections import deque
from datetime import datetime, timedelta, timezone
import json
import statistics
import time

from tqdm import tqdm


def duration(seconds):
    value = int(max(seconds, 0))
    days, rem = divmod(value, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours:02}:{minutes:02}:{secs:02}" if days else f"{hours:02}:{minutes:02}:{secs:02}"


class Progress:
    def __init__(self, output, total, start, batch, save_every, eval_every, enabled, log_every=10,
                 first_save=0):
        self.output, self.total, self.batch = output, total, batch
        self.save_every, self.eval_every = save_every, eval_every
        self.enabled, self.log_every = enabled, log_every
        self.first_save = first_save
        self.recent = deque(maxlen=100)
        self.overheads = {"save": [], "eval": []}
        self.first_step = start + 1
        if enabled and start:
            # An intentional save/restart must not throw away the measured ETA.
            history = output / "train.jsonl"
            if history.exists():
                for line in history.read_text().splitlines():
                    row = json.loads(line)
                    if row["step"] <= start:
                        self.recent.append(row["seconds"])
            history = output / "phase_events.jsonl"
            if history.exists():
                for line in history.read_text().splitlines():
                    row = json.loads(line)
                    if row["step"] <= start and row["phase"] in self.overheads:
                        self.overheads[row["phase"]].append(row["seconds"])
        self.bar = tqdm(total=total, initial=start, desc="Stage3", unit="step",
                        mininterval=5, dynamic_ncols=False, ncols=150, disable=not enabled)

    def update(self, record):
        if not self.enabled:
            return
        self.recent.append(record["seconds"])
        step = record["step"]
        average = statistics.mean(self.recent)
        remaining = max(self.total - step, 0) * average
        self.bar.update(step - self.bar.n)
        self.bar.set_postfix_str(
            f"L={record['total']:.4f} lr={record['lr']:.1e} "
            f"{average:.2f}s/step ETA_train={duration(remaining)}", refresh=False)
        if step == self.first_step or step % self.log_every == 0 or step == self.total:
            extra = 0.
            known = True
            for name, frequency in (("save", self.save_every), ("eval", self.eval_every)):
                if self.overheads[name]:
                    events = self.total // frequency - step // frequency
                    if self.total % frequency and step < self.total:
                        events += 1
                    extra += events * statistics.mean(self.overheads[name])
                else:
                    known = False
            checkpoint = min((step // self.save_every + 1) * self.save_every, self.total)
            if step < self.first_save <= self.total:
                checkpoint = min(checkpoint, self.first_save)
            summary = {
                "step": step, "total_steps": self.total, "percent": 100 * step / self.total,
                "rolling_seconds_per_step": average, "observations_per_second": self.batch / average,
                "eta_training_seconds": remaining,
                "eta_total_seconds": remaining + extra if known else None,
                "eta_note": "total includes measured save/eval estimates" if known else
                            "training-only estimate; save/evaluation overhead not measured yet",
                "next_checkpoint_step": checkpoint,
                "seconds_to_next_checkpoint_training_only": max(checkpoint - step, 0) * average,
                "estimated_finish_utc_training_only":
                    (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat(),
                "latest": record,
            }
            with (self.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(summary, allow_nan=False) + "\n")
            self.bar.write(
                f"PROGRESS {step:,}/{self.total:,} ({summary['percent']:.2f}%) "
                f"| {average:.2f}s/step | {summary['observations_per_second']:.1f} obs/s "
                f"| ETA_train {duration(remaining)} "
                f"| ETA_total {duration(remaining + extra) if known else 'pending save/eval timing'} "
                f"| next_ckpt {checkpoint:,} ~{duration(summary['seconds_to_next_checkpoint_training_only'])} "
                f"| loss {record['total']:.4f} action {record['action_fm']:.4f} "
                f"| LR {record['lr']:.2e} R/K {record['new_lr']:.2e} "
                f"| peak_max {record['peak_allocated_gib_max']:.1f}GiB"
            )

    def begin(self, phase, step):
        if self.enabled:
            self.bar.write(f"PHASE_BEGIN {phase} step={step} (training paused for this phase)")
        return time.monotonic()

    def end(self, phase, step, started):
        if self.enabled:
            elapsed = time.monotonic() - started
            self.overheads[phase].append(elapsed)
            with (self.output / "phase_events.jsonl").open("a") as stream:
                stream.write(json.dumps({"phase": phase, "step": step, "seconds": elapsed}) + "\n")
            self.bar.write(f"PHASE_END {phase} step={step} elapsed={duration(elapsed)}")

    def close(self):
        self.bar.close()
