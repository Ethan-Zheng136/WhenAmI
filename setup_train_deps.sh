#!/bin/bash
# WoG 训练依赖 —— 集群 py312 镜像专用
#
# 镜像: torchengine-torch280-cu129-py312-v103-tmp5  (torch 2.8.0 / py3.12)
#
# 核心矛盾与解法（2026-09-21 实测验证）:
#   WoG 官方要求 python 3.10 + TF 2.15.0，但 TF 2.15 无 py312 wheel（最低 2.16）。
#   实测 TF 2.16.1 可完整跑通 WoG 的 RLDS 数据管线，读出
#   action (16,7) / image (224,224,3) / 真实语言指令 —— 见 _rlds_probe.py
#
# 踩过的 4 个坑（顺序不能乱，否则版本会互相顶掉）:
#   1. dlimp 硬钉 tensorflow==2.15.0 -> 必须 --no-deps 装
#   2. protobuf 太新(7.x) -> tfds 报 FieldDescriptor has no attribute 'label'
#   3. protobuf 5.29 + tensorflow-metadata 新版 -> gencode 6.31 vs runtime 5.29 冲突
#      -> 必须把 tensorflow-metadata 降到 1.15.0
#   4. droid_utils.py 要 tensorflow_graphics -> 也得 --no-deps 装
#
# 用法（在集群容器内）:
#   bash setup_train_deps.sh /path/to/pkgs
#   export PYTHONPATH=/path/to/pkgs:$PYTHONPATH

set -e
TARGET=${1:?用法: bash setup_train_deps.sh <安装目录>}

export http_proxy=http://10.10.64.15:18000
export https_proxy=http://10.10.64.15:18000
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy

mkdir -p "$TARGET"
PIP="python3 -m pip install -q --target=$TARGET"

echo "[1/4] tensorflow-cpu 2.16.1 + tfds 4.9.3"
# 用 -cpu 版：数据管线只在 CPU 上跑，避免 TF 抢显存与 torch 冲突
$PIP "tensorflow-cpu==2.16.1" "tensorflow_datasets==4.9.3"

echo "[2/4] dlimp (--no-deps，它硬钉 TF 2.15)"
$PIP --no-deps "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972"

echo "[3/4] tensorflow-graphics (--no-deps，droid_utils.py 需要)"
$PIP --no-deps tensorflow-graphics

echo "[4/5] 锁定 protobuf / tensorflow-metadata 版本"
$PIP --upgrade "protobuf==5.29.5"
$PIP --no-deps --upgrade "tensorflow-metadata==1.15.0"

echo "[5/5] setuptools（py3.12 移除了 distutils，TF 2.16 仍依赖它）"
$PIP --upgrade "setuptools<81"
# --target 安装不会触发 setuptools 的 .pth 钩子，需要显式建一个 sitecustomize
# 让 _distutils_hack 在解释器启动时自动装上 distutils 垫片
cat > "$TARGET/sitecustomize.py" <<'PY'
# py3.12 移除了 stdlib distutils，TF 2.16 的 __init__ 仍 `import distutils`。
# setuptools 自带垫片，但 --target 安装不会触发它的 .pth，这里显式激活。
try:
    import _distutils_hack
    _distutils_hack.add_shim()
except Exception:
    pass
PY

echo ""
echo "=== 自检 ==="
PYTHONPATH="$TARGET" CUDA_VISIBLE_DEVICES="" MPLCONFIGDIR=/tmp/mpl python3 - <<'PY'
import tensorflow as tf, tensorflow_datasets as tfds, dlimp
import google.protobuf as pb
import tensorflow_graphics.geometry.transformation  # noqa
print(f"  tensorflow {tf.__version__}   (需 2.16.1)")
print(f"  tfds       {tfds.__version__} (需 4.9.3)")
print(f"  protobuf   {pb.__version__}   (需 5.29.5)")
print("  dlimp / tensorflow_graphics  OK")
PY
echo ""
echo "安装完成。使用前 export PYTHONPATH=$TARGET:\$PYTHONPATH"
