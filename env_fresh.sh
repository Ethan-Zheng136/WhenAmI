# WoG_fresh 环境 — 官方代码零改动跑通
# 用法: source /home/zhengguantian/WoG_fresh/env_fresh.sh
#
# 与 WoG_v2/env.sh 的关键区别:**不注入 PYTHONPATH**。
# WoG_v2/pylibs 里装的是 timm 1.0.11,会盖掉 venv 的 timm 0.9.10;
# timm 1.x 的 forward_features 返回 list 而非 tensor,直接导致
# dinosiglip_vit.py:151 `torch.cat` 报 "expected Tensor, got list"。
# 官方 pyproject.toml:44 要求 timm==0.9.10,必须让 venv 的版本生效。

# --- 网络代理(hf.co 直连超时,GitHub/arXiv 可直连) ---
export http_proxy=http://10.10.64.15:18000
export https_proxy=http://10.10.64.15:18000
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
export NO_PROXY=$no_proxy

# --- 关键:清掉 PYTHONPATH,防止 pylibs 的 timm 1.x 抢占 ---
unset PYTHONPATH

# --- 解释器:必须用 .venv-wog(transformers 4.40.1 + timm 0.9.10) ---
# 系统 python 的 transformers 是 4.57.0,BertModel 导入即失败。
export WOG_PY=/home/zhengguantian/WoG_v2/.venv-wog/bin/python

# --- 路径 ---
export WOG_REPO=/home/zhengguantian/WoG_fresh/WoG
export WOG_STORE=/mnt/pfs/zhengguantian/WoG_v2
export WOG_CKPTS=$WOG_STORE/ckpts
export WOG_DATA=$WOG_STORE/datasets
export WOG_OUT=$WOG_STORE/outputs

# --- 自检 ---
wog_check() {
    $WOG_PY - <<'EOF'
import sys
ok = True
import transformers, timm, torch
print(f"  transformers {transformers.__version__}  (需 4.40.1)")
print(f"  timm         {timm.__version__}  (需 0.9.10)")
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
if transformers.__version__ != "4.40.1": print("  !! transformers 版本不对"); ok = False
if timm.__version__ != "0.9.10":        print("  !! timm 版本不对 —— 检查 PYTHONPATH 是否被注入"); ok = False
from pathlib import Path
b = Path("/home/zhengguantian/WoG_fresh/WoG/prismatic/models/backbones/vision")
for p in ["../../../../pretrained/vision/dinov2_weights.pth",
          "../../../../pretrained/vision/Wan2.1_VAE.pth"]:
    r = (b / p).resolve()
    print(f"  {'OK  ' if r.exists() else 'MISS'} {r}")
    if not r.exists(): ok = False
print("  => " + ("环境就绪" if ok else "环境有问题,见上面 !!"))
sys.exit(0 if ok else 1)
EOF
}

echo "WoG_fresh 环境已加载。运行 wog_check 自检。"
