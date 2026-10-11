"""完整 WoG 模型加载 + 真实推理测试。

这是决定性验证：8B 全模型能否在 PPU 上加载并产出动作。
之前只验证过 QFormer(109M) 这一小块。
"""
import sys, time, torch
from pathlib import Path

REPO = Path("/home/zhengguantian/WoG_fresh/WoG")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
import os
os.chdir(REPO)

import stub_tf; stub_tf.patch()      # 必须在 import vla 之前

from PIL import Image
import numpy as np

t0 = time.time()
print("=== 1) 导入 load_vla ===", flush=True)
from vla import load_vla
print(f"  import 耗时 {time.time()-t0:.1f}s", flush=True)

# load_vla 要的是 .pt 文件完整路径（不是目录），且必须位于 <run_dir>/checkpoints/ 下
CKPT = "/mnt/pfs/zhengguantian/WoG_v2/ckpts/WoG-V/checkpoints/step-100000-epoch-07-loss=0.1041.pt"
print(f"\n=== 2) 加载完整模型 (30GB, 需数分钟) ===", flush=True)
t = time.time()
vla = load_vla(CKPT, load_for_training=False, action_model_type="DiT-L", future_action_window_size=15)
print(f"  加载耗时 {time.time()-t:.1f}s", flush=True)

n = sum(p.numel() for p in vla.parameters())
print(f"  参数量 {n/1e9:.2f} B", flush=True)

print("\n=== 3) 转 fp32 并上卡 ===", flush=True)
t = time.time()
vla = vla.to(torch.float32).to("cuda").eval()   # numpy 不支持 bf16，官方 predict_action 假设 fp32
print(f"  上卡耗时 {time.time()-t:.1f}s | 显存 {torch.cuda.memory_allocated()/2**30:.1f} GB", flush=True)

print("\n=== 4) 真实推理 ===", flush=True)
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
t = time.time()
with torch.no_grad():
    actions, norm_actions = vla.predict_action(
        image=img, instruction="pick up the coke can",
        unnorm_key="fractal20220817_data",
    )
dt = time.time() - t
print(f"  推理耗时 {dt:.2f}s", flush=True)
print(f"  action shape {actions.shape}  (期望 (16,7): Ta=16 步, 7 维 EEF)", flush=True)
print(f"  action range [{actions.min():.3f}, {actions.max():.3f}]", flush=True)
print(f"  gripper 维(第7列) 取值: {np.unique(norm_actions[:,6])}", flush=True)

assert actions.shape == (16, 7), f"形状错误 {actions.shape}"
assert np.isfinite(actions).all(), "输出含 NaN/Inf"

print(f"\n峰值显存 {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
print("\n=== 完整 WoG 在 PPU 上推理成功 ===", flush=True)
