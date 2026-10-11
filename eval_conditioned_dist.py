"""[zgt] Stage I 条件质量评测 —— 集群多卡全量版。

与 eval_conditioned.py 的差别只在并行：样本按 rank 切分，各卡独立推理，
最后 rank0 汇总。单卡跑 150 个样本约 25 分钟，16 卡跑全量约 2 分钟。

评测口径（与单卡版一致）：
    两条路径都喂真实未来帧，各自算出条件后注入 DiT。
      WoG     : 5 帧 -> QFormer -> O^c
      WhenAmI : 9 帧 -> alpha/beta 汇聚 -> 当下节点 s*_t
    两边共用同一批样本（都要求同 episode 内能凑齐 9 帧），
    保证唯一变量是条件机制本身。

    这评的是 Stage I 的**条件质量**，不是可部署的策略能力 ——
    后者要等 Stage II 撤掉未来之后才谈。

用法（torchrun）：
    torchrun --nproc_per_node 16 eval_conditioned_dist.py \
        --ckpt WoG-10k=/path/a.pt \
        --ckpt WhenAmI-10k=/path/b.pt --temporal-for WhenAmI-10k
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

OUT = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe")
DIMS = ["dx", "dy", "dz", "roll", "pitch", "yaw", "grip"]
OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
STEP = 4          # .npz 内相邻采样点的 tstep 间隔，实测恒为 4


def collect_samples(d, limit=None):
    """挑出能凑齐完整时间线的样本。

    过去帧必须来自同一 episode 且间隔正确 —— 否则喂给 alpha 的是
    别的轨迹，测出来的东西没有意义。
    """
    idx = {(int(e), int(t)): i
           for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"]))}
    out = []
    for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"])):
        e, t = int(e), int(t)
        rows = [idx.get((e, t + o * STEP)) for o in OFFSETS]
        if all(r is not None for r in rows):
            out.append((i, rows))
            if limit and len(out) >= limit:
                break
    return out


@torch.no_grad()
def run_one(ckpt_path, use_temporal, ds, samples, rank, world):
    from vla import load_vla

    # converge_type 必须与训练一致：masked 与 recurrent 的模块结构不同，
    # 传错会让 state_dict 的 key 对不上（静默加载失败比报错更糟）。
    # 从 checkpoint 同目录的 config.json 读，而不是靠调用方记得传。
    extra = {}
    if use_temporal:
        # 从**权重本身**判断，而不是 config.json。config 记录的是提交时的
        # 配置，未必等于实际构造的模型 —— 参数透传曾经断链，config 写着
        # masked 而权重是 recurrent，两个 320 卡任务因此白跑。
        # 键名是结构的事实，配置只是意图。
        import torch as _t
        _sd = _t.load(str(ckpt_path), map_location="cpu", mmap=True,
                      weights_only=False)["model"].get("converge", {})
        # 三种实现的键名互斥，用它判定。新增 flow 时必须同步更新这里 ——
        # 漏掉就会被误判成 recurrent，加载时 key 全对不上。
        if any(k.startswith("flow.") for k in _sd):
            ct = "flow"
        elif any(k.startswith("q_past.") for k in _sd):
            ct = "masked"
        elif any("obs_encoder" in k for k in _sd):
            ct = "recurrent"
        else:
            raise ValueError(
                f"cannot infer converge_type from checkpoint keys: "
                f"{sorted(_sd)[:5]}... (expected flow./q_past./obs_encoder.)"
            )
        extra = dict(use_temporal=True, obs_offsets=OFFSETS, converge_type=ct)
        if ct == "flow":
            # flow 的编码器容量是可配的，不同 run 结构不同（v2 无 FFN 4.5M、
            # v3 有 FFN 44.3M）。同样从权重本身推断，不靠调用方记得传环境
            # 变量 —— 配置漏传导致 key 对不上，正是此前白跑 20 小时的那类事故。
            enc = {k: v for k, v in _sd.items() if k.startswith("encoder.")}
            n_enc = len({k.split(".")[2] for k in enc if k.startswith("encoder.blocks.")})
            n_cross = len({k.split(".")[2] for k in enc
                           if k.startswith("encoder.cross_frame.")})
            extra.update(
                flow_enc_layers=n_enc,
                flow_cross_layers=n_cross,
                flow_enc_hidden=int(enc["encoder.query"].shape[-1]),
                flow_enc_use_ffn=any("ffn" in k for k in enc),
            )
            print(f"    flow enc: {n_enc}+{n_cross} layers "
                  f"h={extra['flow_enc_hidden']} "
                  f"ffn={extra['flow_enc_use_ffn']} (inferred from weights)", flush=True)
        del _sd
        print(f"    converge_type={ct}（据权重键名判定）", flush=True)
    # train_venc 必须与训练时一致：它决定 ActionModel 是否构建 per_attn
    #     use_per_attn = True if train_venc else False
    # load_vla 默认 False，那样加载出来的 DiT 没有 per_attn，
    # 条件无论怎么传都不影响输出（实测 |out(cond)-out(none)| = 0）。
    vla = load_vla(str(ckpt_path), load_for_training=False, train_venc=True,
                   action_model_type="DiT-L", future_action_window_size=15, **extra)
    vla = vla.to(torch.float32).to("cuda").eval()    # bf16 下 .numpy() 会炸
    # 诊断开关：把条件放大到指定倍数，用来判定幅度是否是评测落后的主因
    vla.cond_scale = float(os.environ.get("COND_SCALE", "1.0"))
    if rank == 0 and vla.cond_scale != 1.0:
        print(f"    cond_scale={vla.cond_scale}", flush=True)

    d = np.load(OUT / f"{ds}.npz", allow_pickle=True)
    mine = samples[rank::world]                      # 按 rank 切分

    preds, gts = [], []
    for cnt, (i, rows) in enumerate(mine):
        cur = Image.fromarray(d["frames_cur"][i])
        instr = str(d["lang"][i]) or "do the task"
        if use_temporal:
            frames = [Image.fromarray(d["frames_cur"][r]) for r in rows]
            acts = np.stack([d["action"][r][0] for r in rows[:-1]])
            kw = dict(cond_frames=frames, cond_actions=acts)
        else:
            kw = dict(cond_frames=[Image.fromarray(f) for f in d["frames_fut"][i]])
        act, _ = vla.predict_action(image=cur, instruction=instr, unnorm_key=ds, **kw)
        preds.append(act)
        gts.append(d["action"][i][:16])
        if rank == 0 and cnt and cnt % 10 == 0:
            print(f"    rank0 {cnt}/{len(mine)}", flush=True)

    del vla
    torch.cuda.empty_cache()
    return np.stack(preds), np.stack(gts)


def gather(arr, world):
    """把各 rank 的结果收到 rank0。形状不齐，用 all_gather_object。"""
    buf = [None] * world
    dist.all_gather_object(buf, arr)
    return np.concatenate(buf, axis=0)


def report(P, G, tag, ds):
    rs = []
    lines = [f"  {'维度':<8}{'预测std':>10}{'真实std':>10}{'相关r':>9}", "  " + "-" * 40]
    for j in range(6):
        p, g = P[:, :, j].ravel(), G[:, :, j].ravel()
        r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-9 and g.std() > 1e-9 else float("nan")
        rs.append(r)
        lines.append(f"  {DIMS[j]:<8}{p.std():>10.4f}{g.std():>10.4f}{r:>9.3f}")
    P6, G6 = P[:, :, :6], G[:, :, :6]
    r2 = float(1 - ((P6 - G6) ** 2).sum() / (((G6 - G6.mean((0, 1))) ** 2).sum() + 1e-12))
    r6 = float(np.nanmean(rs))
    lines.append("  " + "-" * 40)
    lines.append(f"  前6维 r = {r6:.3f} | R² = {r2:.3f} | n = {len(P)}")
    print("\n".join(lines), flush=True)
    return dict(tag=tag, ds=ds, r_6=r6, r2_6=r2, n=int(len(P)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--temporal-for", action="append", default=[],
                    help="checkpoint names that use the temporal head")
    ap.add_argument("--datasets", nargs="+", default=["fractal20220817_data"])
    # --limit 而非 --n：torchrun 会把 --n 当成 --nnodes/--nproc_per_node 的前缀
    # 而报 "ambiguous option"，命令行参数名必须避开它的选项空间。
    ap.add_argument("--limit", type=int, default=0, help="0 = all available samples")
    ap.add_argument("--out", default="/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe/eval_cond.json")
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    results = []
    for ds in a.datasets:
        d = np.load(OUT / f"{ds}.npz", allow_pickle=True)
        samples = collect_samples(d, a.limit or None)
        if rank == 0:
            print(f"\n{'='*66}\n{ds}: {len(samples)} 个可用样本 / {len(d['ep_id'])} 总样本"
                  f"  ({world} 卡并行)\n{'='*66}", flush=True)

        for spec in a.ckpt:
            name, path = spec.split("=", 1)
            if not Path(path).exists():
                raise SystemExit(f"checkpoint not found: {path}")
            use_temporal = name in a.temporal_for
            if rank == 0:
                print(f"\n--- {name}  ({'时间线 9 帧' if use_temporal else 'QFormer 5 帧'}) ---",
                      flush=True)
            t0 = time.time()
            P, G = run_one(path, use_temporal, ds, samples, rank, world)
            P, G = gather(P, world), gather(G, world)
            if rank == 0:
                print(f"    用时 {time.time()-t0:.0f}s", flush=True)
                results.append(report(P, G, name, ds))
            dist.barrier()

    if rank == 0:
        print(f"\n{'='*66}\n汇总\n{'='*66}")
        print(f"{'ckpt':<20}{'数据集':<26}{'r':>8}{'R²':>9}{'n':>6}")
        print("-" * 66)
        for x in results:
            print(f"{x['tag']:<20}{x['ds']:<26}{x['r_6']:>8.3f}{x['r2_6']:>9.3f}{x['n']:>6}")
        Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"\n  结果已写入 {a.out}")

    dist.destroy_process_group()
