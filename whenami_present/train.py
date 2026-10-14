"""Present-token WhenAmI: Stage I teacher (real future) / Stage II student (imagined future).

Arbitrary topology: world = nodes x NPROC_PER_NODE, accum = global_batch/(world*micro).
"""
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from stage3_oxe_eval.prepare import write_json, sha
from stage3_oxe_eval50.policy import param_bf16
from stage3_perf.checkpoint import CheckpointIO
from whenami_unified.loading import A50
from whenami_unified.data import DEV, INDEX
from whenami_present.data import Inputs, ProbeSet, batch, executed, future_times
from whenami_present.model import build_policy, PresentConfig, new_parameter_report
from whenami_present.loading import SCHEMA, source_record, load_weights, load_frozen_teacher, new_prefixes
from whenami_present.losses import (TEACHER_FM, STUDENT_FM, fm_labels, draw, fm, cos, precision_nll,
                                    agreement, monotonic, phase_alignment, progress_label)
from whenami_present.runtime import (fingerprints, reference_fingerprints, topology, wrap, learning_rate,
                                     optimizer_groups, gradients)
from whenami_present.evaluate import evaluate

PRESENTS = (0, 1, 2)
LOSS_KEYS = ("fm_present", "fm_generation", "revision", "precision_nll", "agreement", "monotonic",
             "phase_swap", "phase_order", "tau_label", "distill_condition", "distill_z", "distill_tau",
             "distill_phase", "imagine_condition", "imagine_node", "total")
DIAG_KEYS = ("w_fwd_mean", "w_fwd_std", "fwd_bwd_disagreement", "logvar_fwd", "logvar_bwd",
             "tau0", "tau1", "tau2", "tau_order_acc", "tau_err_vs_t_over_T", "phase_entropy_batch")


def arguments():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--stage", choices=("teacher", "student"), required=True)
    a("--output", type=Path, required=True)
    a("--teacher", type=Path, help="Stage II: committed Stage I checkpoint")
    a("--steps", type=int, default=50000); a("--micro", type=int, default=1)
    a("--global-batch", type=int, default=320); a("--seed", type=int, default=42)
    a("--lr", type=float, default=1e-6); a("--new-lr", type=float, default=1e-5)
    a("--warmup", type=int, default=1000); a("--draws", type=int, default=2)
    a("--hidden-every", type=int, default=4, help="Stage I: hide anchor1 every N steps (0=off)")
    a("--future-drop", type=float, default=.25, help="Stage I: P(drop all real future) so fwd-only works")
    a("--imagine-gap", type=int, default=4, help="deploy/eval imaginer offsets = gap*{1,2,3}")
    # pipeline component flags (ablations)
    a("--no-backward", action="store_true"); a("--fusion", choices=("precision", "mean", "forward"), default="precision")
    a("--no-revise", action="store_true"); a("--raw-z-bypass", action="store_true")
    a("--phases", type=int, default=8)
    # loss weights
    for name, v in (("gen", 1.), ("revision", .1), ("precision", .1), ("agree", .1), ("mono", .1),
                    ("phase", .05), ("phase-order", .05), ("tau-label", 0.), ("distill", .1),
                    ("distill-tau", .1), ("distill-phase", .05), ("imagine", .1)):
        a(f"--{name}-weight", type=float, default=v)
    a("--sharding", choices=("full", "hybrid", "hybrid_zero2"), default="hybrid_zero2")
    a("--activation", choices=("none", "llm", "all"), default="llm")
    a("--save-every", type=int, default=5000); a("--eval-every", type=int, default=5000)
    a("--first-check", type=int, default=100); a("--stop-after", type=int, default=0)
    a("--resume-from", type=Path); a("--dev-limit", type=int, default=0); a("--eval-max", type=int, default=0)
    a("--probe-episodes", type=int, default=48); a("--no-initial-eval", action="store_true")
    a("--fixed-data", action="store_true", help="engineering overfit check: same batch/noise every step")
    a("--engineering", action="store_true")
    a("--bf16-storage", action="store_true", help="engineering single-card only")
    return p


def present_config(a):
    return PresentConfig(backward=not a.no_backward, fusion=a.fusion, revise=not a.no_revise,
                         raw_z_bypass=a.raw_z_bypass, phases=a.phases)


def fm_split(velocity, target, valid, requests, draws):
    R = len(requests)
    b = len(valid)//R
    v = velocity.reshape(b, R, draws, *velocity.shape[1:])
    t = target.reshape(b, R, draws, *target.shape[1:])
    m = valid.reshape(b, R, *valid.shape[1:])
    return [fm(v[:, r].flatten(0, 1), t[:, r].flatten(0, 1), m[:, r]) for r in range(R)]


def diagnostics(outs, prog):
    w = torch.cat([o["w_fwd"].flatten() for o in outs]).float()
    taus = torch.stack([o["tau"] for o in outs], 1).float()
    usable = [o for o in outs if "p_bwd" in o and o["usable"].any()]
    dis = (torch.stack([((o["p_fwd"].float()-o["p_bwd"].float()).square().mean(-1) /
                         (o["p_fwd"].float().square().mean(-1)+1e-6)).mean() for o in usable]).mean()
           if usable else w.new_zeros(()))
    sb = torch.stack([o["s_bwd"].float().mean() for o in usable]).mean() if usable else w.new_zeros(())
    p = torch.cat([F.softmax(o["logits"].float(), -1) for o in outs]).mean(0)
    return [w.mean(), w.std() if w.numel() > 1 else w.new_zeros(()), dis,
            torch.stack([o["s_fwd"].float().mean() for o in outs]).mean(), sb,
            taus[:, 0].mean(), taus[:, 1].mean(), taus[:, 2].mean(),
            ((taus[:, 0] < taus[:, 1]) & (taus[:, 1] < taus[:, 2])).float().mean(),
            (taus-prog).abs().mean(), -(p*torch.log(p+1e-9)).sum()]


def teacher_requests(a):
    return tuple(r for r in TEACHER_FM if not (a.no_revise and r[0] == "revised"))


def teacher_step(a, model, evidence, labels, cases, gen, noisy, tau, target, valid, device):
    req = teacher_requests(a)
    keep = torch.rand(len(cases), generator=gen, device="cpu") >= a.future_drop
    res = model(evidence=evidence, presents=PRESENTS, future_keep=keep.to(device),
                noisy_actions=noisy, tau=tau, fm_requests=req)
    parts = fm_split(res["velocity"], target, valid, req, a.draws)
    fm_present, fm_gen = sum(parts[:3])/3, sum(parts[3:])/len(parts[3:])
    outs = [res["present"][c] for c in PRESENTS]
    hs, hc = res["hindsight"], res["hindsight_condition"]
    rev = sum(cos(res["present"][c]["revised_condition"], hc[:, :c]) for c in (1, 2))/2 if a.revision_weight and not a.no_revise else fm_present*0
    prec = sum(precision_nll(o, hs[:, c:c+1]) for c, o in zip(PRESENTS, outs))/3
    agree_vals = [agreement(o) for o in outs]
    agree = sum(x for x, _ in agree_vals)/3
    taus = torch.stack([o["tau"] for o in outs], 1)
    mono = monotonic(taus)
    swap, order = phase_alignment(outs)
    prog = progress_label(cases, PRESENTS, device)
    tau_l = F.mse_loss(taus.float(), prog)
    z = fm_present*0
    L = dict(fm_present=fm_present, fm_generation=fm_gen, revision=rev, precision_nll=prec, agreement=agree,
             monotonic=mono, phase_swap=swap, phase_order=order, tau_label=tau_l, distill_condition=z,
             distill_z=z, distill_tau=z, distill_phase=z, imagine_condition=z, imagine_node=z)
    L["total"] = (fm_present + a.gen_weight*fm_gen + a.revision_weight*rev + a.precision_weight*prec +
                  a.agree_weight*agree + a.mono_weight*mono + a.phase_weight*swap +
                  a.phase_order_weight*order + a.tau_label_weight*tau_l)
    return L, diagnostics(outs, prog)


def student_step(a, model, teacher, evidence, labels, cases, noisy, tau, target, valid, device):
    with torch.no_grad():
        tres = teacher(evidence=evidence, presents=PRESENTS,
                       future_keep=torch.ones(len(cases), dtype=torch.bool, device=device))
    cmd = executed(cases, 3, device)
    ft = future_times(cases, PRESENTS, device)
    res = model(evidence=evidence.prefix(3), executed=cmd, presents=PRESENTS, imagine_times=ft,
                noisy_actions=noisy, tau=tau, fm_requests=STUDENT_FM)
    fm_present = sum(fm_split(res["velocity"], target, valid, STUDENT_FM, a.draws))/3
    outs = [res["present"][c] for c in PRESENTS]
    touts = [tres["present"][c] for c in PRESENTS]
    dc = sum(cos(o["condition"], t["condition"].detach()) for o, t in zip(outs, touts))/3
    dz = sum(cos(o["present_z"], t["present_z"].detach()) for o, t in zip(outs, touts))/3
    dt = sum(F.mse_loss(o["tau"].float(), t["tau"].detach().float()) for o, t in zip(outs, touts))/3
    dp = sum(F.kl_div(F.log_softmax(o["logits"].float(), -1), F.softmax(t["logits"].detach().float(), -1),
                      reduction="batchmean") for o, t in zip(outs, touts))/3
    ic = sum(cos(o["imagined_condition"], t["future_condition"].detach()) for o, t in zip(outs, touts))/3
    inode = sum(cos(o["imagined"], t["future_nodes"].detach()) for o, t in zip(outs, touts))/3
    hc = tres["hindsight_condition"].detach()
    rev = sum(cos(res["present"][c]["revised_condition"], hc[:, :c]) for c in (1, 2))/2 if not a.no_revise else fm_present*0
    prec = sum(precision_nll(o, t["present"]) for o, t in zip(outs, touts))/3
    agree = sum(agreement(o)[0] for o in outs)/3
    taus = torch.stack([o["tau"] for o in outs], 1)
    mono = monotonic(taus)
    swap, order = phase_alignment(outs)
    prog = progress_label(cases, PRESENTS, device)
    tau_l = F.mse_loss(taus.float(), prog)
    L = dict(fm_present=fm_present, fm_generation=fm_present*0, revision=rev, precision_nll=prec,
             agreement=agree, monotonic=mono, phase_swap=swap, phase_order=order, tau_label=tau_l,
             distill_condition=dc, distill_z=dz, distill_tau=dt, distill_phase=dp,
             imagine_condition=ic, imagine_node=inode)
    L["total"] = (fm_present + a.distill_weight*(dc+dz) + a.distill_tau_weight*dt + a.distill_phase_weight*dp +
                  a.imagine_weight*(ic+inode) + a.revision_weight*rev + a.precision_weight*prec +
                  a.agree_weight*agree + a.mono_weight*mono + a.phase_weight*swap +
                  a.phase_order_weight*order + a.tau_label_weight*tau_l)
    return L, diagnostics(outs, prog)


def main():
    parser = arguments()
    a = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if a.global_batch % (world*a.micro) or min(a.steps, a.micro, a.draws, a.save_every, a.eval_every, a.first_check) < 1:
        parser.error("global batch must be divisible by world*micro; positive counts")
    if not 0 <= a.stop_after <= a.steps or not 0 <= a.warmup < a.steps or not 0 <= a.future_drop < 1:
        parser.error("schedule")
    if a.bf16_storage and not a.engineering:
        parser.error("BF16 storage is engineering only")
    if a.fixed_data and not a.engineering:
        parser.error("--fixed-data is engineering only")
    if a.stage == "student" and a.teacher is None:
        parser.error("Stage II needs --teacher")
    if a.sharding != "full" and world <= local_world:
        parser.error("hybrid sharding needs >=2 nodes; use --sharding full on one node")
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=40))
    err = [None]
    if rank == 0:
        try:
            a.output.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            err[0] = str(e)
    dist.broadcast_object_list(err, src=0)
    if err[0]:
        raise ValueError(err[0])
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    student = a.stage == "student"
    src_path = a.teacher if student else A50
    source = source_record(src_path, "teacher" if student else "a50", a.engineering)
    pc = present_config(a)
    data = Inputs(dev_limit=a.dev_limit)
    probe = ProbeSet(data.stats, a.probe_episodes) if a.probe_episodes else None
    core, visual, transform, tokenizer, prompt = build_policy(student, pc, a.seed)
    report = new_parameter_report(core)
    trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if a.bf16_storage:
        core = param_bf16(core)
    strategy, groups = topology(a.sharding, device)
    model = wrap(core, device, strategy, groups, a.activation)
    group = groups[0] if groups else None
    loaded = {"resume": str(a.resume_from)} if a.resume_from else load_weights(model, src_path, student, not student, group)
    teacher, tloaded = None, None
    if student:
        tc, _, _, _, _ = build_policy(False, PresentConfig(**source["recipe"]["present_config"]), a.seed)
        tc.requires_grad_(False).eval()
        tc.vlm.vision_backbone_requires_grad = False
        if a.bf16_storage:
            tc = param_bf16(tc)
        teacher = wrap(tc, device, strategy, groups, "none").eval()
        tloaded = load_frozen_teacher(teacher, a.teacher, group)
    visual = visual.to(device).requires_grad_(False).eval()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                if k not in ("output", "resume_from", "stop_after", "no_initial_eval")}
    recipe = {"schema": SCHEMA, "settings": settings, "present_config": asdict(pc),
              "world_size": world, "local_world_size": local_world, "source": fingerprints(),
              "warmstart": {"path": str(src_path), "step": source["step"]},
              "train_index_sha256": sha(INDEX), "dev_manifest_sha256": sha(DEV/"MANIFEST.json"),
              "trainable_parameters": trainable, "parameter_report": report,
              "optimizer_precision": "BF16-storage ENGINEERING" if a.bf16_storage else "FP32 params/Adam, BF16 compute"}
    if rank == 0:
        write_json(a.output/"recipe.json", recipe)
        write_json(a.output/"LOAD_VERIFIED.json", {"model": loaded, "teacher": tloaded})
        print("PRESENT_PARAMS", json.dumps(report["new_vs_A50k"]), "new_total", report["new_total"],
              "trainable", trainable, flush=True)
    optimizer = torch.optim.AdamW(optimizer_groups(model, a.lr, a.new_lr, new_prefixes(student, not student)),
                                  weight_decay=0., foreach=False)
    io = CheckpointIO(a.sharding, groups)
    random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank)
    start = io.load(a.resume_from, model, optimizer, recipe) if a.resume_from else 0
    if a.stop_after and a.stop_after <= start:
        raise ValueError("stop-after must exceed restored step")
    if rank == 0:
        print("PRESENT_READY", a.stage, world, "start", start, flush=True)
    ev = lambda s: evaluate(model, visual, data, transform, tokenizer, prompt, device,
                            a.output/"development"/f"step-{s:06d}", student, limit=a.eval_max,
                            probe=probe, gap=a.imagine_gap)
    if not a.no_initial_eval:
        ev(start)
    accum = a.global_batch//(world*a.micro)
    vworld = world*accum
    recent = deque(maxlen=100)
    optimizer.zero_grad(set_to_none=True)
    step = start
    for step in range(start+1, a.steps+1):
        begin = time.monotonic()
        model.train()
        for g, peak in zip(optimizer.param_groups, (a.lr, a.new_lr)):
            g["lr"] = learning_rate(step, a.steps, peak, a.warmup)
        numeric = []
        hide = 1 if (not student and a.hidden_every and step % a.hidden_every == 0 and not a.fixed_data) else None
        for micro in range(accum):
            vrank = rank+micro*world
            dstep = 1 if a.fixed_data else step
            cases = data.batch(a.micro, a.seed, dstep, vrank, 0)
            evidence, labels = batch(cases, visual, transform, tokenizer, prompt, device, hide=hide)
            req = STUDENT_FM if student else teacher_requests(a)
            acts, valid = fm_labels(labels, req)
            noisy, tau, target = draw(acts, a.seed+dstep*vworld+vrank, a.draws)
            gen = torch.Generator().manual_seed(a.seed*7919+dstep*vworld+vrank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if student:
                    L, D = student_step(a, model, teacher, evidence, labels, cases, noisy, tau, target, valid, device)
                else:
                    L, D = teacher_step(a, model, evidence, labels, cases, gen, noisy, tau, target, valid, device)
            if not torch.isfinite(L["total"]):
                raise FloatingPointError("present loss "+str({k: float(v) for k, v in L.items()}))
            (L["total"]/accum).backward()
            numeric.append([float(L[k].detach()) for k in LOSS_KEYS] + [float(x.detach()) for x in D])
            del L, D
        audited = step in (start+1, a.first_check)
        grad = gradients(model, device) if audited else None
        if audited and not all(np.isfinite(v) and v > 0 for v in grad.values()):
            raise ValueError("missing gradient group: "+str(grad))
        norm = model.clip_grad_norm_(1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("grad norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise ValueError("frozen teacher got gradients")
        values = torch.tensor(np.mean(numeric, 0), device=device)
        dist.all_reduce(values); values /= world
        sec = torch.tensor(time.monotonic()-begin, device=device)
        dist.all_reduce(sec, op=dist.ReduceOp.MAX)
        peak = torch.tensor(torch.cuda.max_memory_allocated()/2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        recent.append(float(sec))
        if rank == 0:
            row = {"step": step, **dict(zip(LOSS_KEYS+DIAG_KEYS, values.cpu().tolist())),
                   "seconds_max": float(sec), "peak_gib": float(peak), "grad_norm": float(norm),
                   "gradients": grad, "lr": [g["lr"] for g in optimizer.param_groups],
                   "teacher_hidden": hide is not None}
            with (a.output/"train.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            if step == start+1 and a.resume_from:
                write_json(a.output/"RESUME_VERIFIED.json", {"complete": True, "restored_step": start,
                                                             "first_update": step, "world": world})
            if step % 10 == 0 or step in (start+1, a.first_check):
                speed = float(np.mean(recent))
                print(f"PROGRESS {step}/{a.steps} | {speed:.2f}s/step | ETA {timedelta(seconds=int((a.steps-step)*speed))} | "
                      f"total {row['total']:.4f} fm {row['fm_present']:.4f} w_fwd {row['w_fwd_mean']:.3f} "
                      f"dis {row['fwd_bwd_disagreement']:.4f} tau_ord {row['tau_order_acc']:.2f} peak {float(peak):.1f}GiB", flush=True)
        last = step == a.stop_after or step == a.steps
        if step % a.save_every == 0 or step == a.first_check or last:
            io.save(a.output/"checkpoints"/f"step-{step:06d}", model, optimizer, step, recipe)
        if step % a.eval_every == 0 or step == a.first_check or last:
            ev(step)
        if step == a.stop_after:
            break
    if rank == 0:
        write_json(a.output/"COMPLETE.json", {"complete": True, "step": step, "stage": a.stage,
            "formal_training_complete": step == a.steps and not a.engineering,
            "utc": datetime.now(timezone.utc).isoformat()})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
