"""[zgt] 在 PPU 上复现训练失败：逐步测量内存，定位到底是谁吃的。

前五次诊断全是推测，没有一次拿到真实数据。本脚本直接跑真实训练的
前几步，在每个阶段打点 RSS（系统内存）和显存，把爆点钉死。

测量点：
  0 起始
  1 模型构造完（7B VLM + converge）
  2 数据管线构造完（27 个数据集 interleave）
  3 第一个 batch 到手
  4 forward 完
  5 backward 完          <- 梯度分配
  6 第一次 optimizer.step() 完   <- AdamW 状态分配，怀疑是这里
  7 第二步 forward+backward+step

单进程跑，显存和内存都归一个进程，比 16 rank 更容易观察。
若单进程就爆，说明和分布式无关。

结果实时写 /tmp/ppu_probe.json（每步都写，中途被杀也能看到走到哪）
"""
import sys, os, json, time, resource
sys.path.insert(0, "/home/zhengguantian/WoG_fresh")
import stub_tf
stub_tf.patch(verbose=False)
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch

OUT = "/tmp/ppu_probe.json"
marks = []


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def vram_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def mark(tag):
    m = {"stage": tag, "rss_gb": round(rss_gb(), 2), "vram_gb": round(vram_gb(), 2),
         "t": round(time.time() - T0, 1)}
    marks.append(m)
    with open(OUT, "w") as f:
        json.dump(marks, f, indent=1)
    print(f"[{m['t']:7.1f}s] {tag:<34} RSS {m['rss_gb']:7.2f} GB   VRAM {m['vram_gb']:6.2f} GB", flush=True)


T0 = time.time()
mark("0_start")

# ---- 模型 ----
from vla import load_vla

CKPT = ("/mnt/pfs/zhengguantian/WoG_v2/ckpts/openvla-7b-prismatic/"
        "checkpoints/step-295000-epoch-40-loss=0.2200.pt")
OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
N_PAST, N_FUT, Ta = 4, 4, 16

vla = load_vla(CKPT, load_for_training=True, action_model_type="DiT-L",
               future_action_window_size=Ta - 1, train_venc=True,
               use_temporal=True, obs_offsets=OFFSETS)
mark("1_model_built")

vla = vla.to(torch.float32).to("cuda").train()
torch.cuda.reset_peak_memory_stats()
mark("2_model_on_gpu")

# ---- 优化器（真实训练里 AdamW 状态在第一次 step 时分配）----
opt = torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad], lr=2e-5)
mark("3_optimizer_created")

# ---- 数据：直接用真实 RLDS 管线，27 个数据集 ----
from prismatic.vla import get_vla_dataset_and_collator
from torch.utils.data import DataLoader

DATA_MIX = os.environ.get("PROBE_DATA_MIX", "oxe_magic_soup_plus")
PER_DEV = int(os.environ.get("PROBE_BS", "2"))
past_w = max(0, -min(OFFSETS))
fut_w = max(Ta - 1, max(OFFSETS) + Ta - 1)

ds, _, collator = get_vla_dataset_and_collator(
    "/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment",
    DATA_MIX,
    image_transform=vla.vision_backbone.get_image_transform(),
    tokenizer=vla.llm_backbone.get_tokenizer(),
    prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
    default_image_resolution=vla.vision_backbone.default_image_resolution,
    shuffle_buffer_size=int(os.environ.get("PROBE_SHUFFLE", "16000")),
    image_aug=True,
    future_action_window_size=fut_w,
    past_action_window_size=past_w,
    obs_offsets=OFFSETS,
)
mark("4_dataset_built")

dl = DataLoader(ds, batch_size=PER_DEV, collate_fn=collator, num_workers=0)
it = iter(dl)

for step in range(1, 4):
    batch = next(it)
    mark(f"5_step{step}_batch_fetched")

    batch = {k: (v.to("cuda") if torch.is_tensor(v) else
                 {kk: vv.to("cuda") for kk, vv in v.items()} if isinstance(v, dict) else v)
             for k, v in batch.items()}

    loss, _, act_loss, obs_loss = vla(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        future_pixel_values=batch["future_pixel_values"],
        past_pixel_values=batch.get("past_pixel_values"),
        actions=batch["actions"], labels=batch["labels"],
        output_hidden_states=True, repeated_diffusion_steps=8,
    )
    mark(f"6_step{step}_forward  loss={float(loss):.4f}")

    loss.backward()
    mark(f"7_step{step}_backward")

    opt.step()
    opt.zero_grad(set_to_none=True)
    mark(f"8_step{step}_optimizer_step")

print("\nPROBE COMPLETE -- no OOM in single process", flush=True)
