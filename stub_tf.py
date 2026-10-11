"""在 import WoG 之前调用 patch()，屏蔽推理用不到的数据管线依赖。

背景：
    `vla/load.py` → `prismatic.conf` → `prismatic/__init__.py`
      → `prismatic.models.load` → `prismatic.models.vlas.OpenVLA`
      → `prismatic.vla.action_tokenizer` → `prismatic.vla.__init__`
      → `materialize` → `datasets.rlds` → `import dlimp` → `tensorflow`

    推理**完全用不到** RLDS 数据管线，但这条 import 链会强行拉进
    tensorflow==2.15（py3.12 无 wheel，最低只有 2.16）。

做法：
    往 sys.modules 里塞轻量假模块，让那条链在触及 tensorflow 之前就短路。
    不修改官方代码 —— 保持与上游零 diff，便于后续同步。

注意：这只对**推理**安全。真要跑训练，必须装真的 tensorflow + dlimp
      （那时应换 python3.11 环境，因为 TF 2.15 只到 cp311）。
"""

import importlib.machinery
import sys
import types


class _AnyAttr(types.ModuleType):
    """对任意属性访问都返回一个无害的可调用对象。

    这样就不必逐个补齐 TF 的 API（那是无底洞）——
    数据管线在 import 期会调用 `tf.config.set_visible_devices([], "GPU")`
    之类的语句，让它们静默成功即可；推理路径不会真正跑这些数据变换。
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _AnyAttr(f"{self.__name__}.{name}")
        sub.__spec__ = importlib.machinery.ModuleSpec(sub.__name__, loader=None, is_package=True)
        sub.__path__ = []
        setattr(self, name, sub)
        return sub

    def __call__(self, *a, **kw):
        return None


def _stub(name: str, any_attr: bool = False, **attrs):
    m = _AnyAttr(name) if any_attr else types.ModuleType(name)
    # 必须给 __spec__：transformers 的 import_utils 用
    # `importlib.util.find_spec("tensorflow")` 探测 TF，
    # 而 find_spec 对 __spec__ is None 的已加载模块会抛 ValueError。
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    m.__loader__ = None
    for k, v in attrs.items():
        setattr(m, k, v)
    m.__path__ = []           # 伪装成 package，允许 submodule 导入
    sys.modules[name] = m
    return m


def patch(verbose: bool = True):
    """屏蔽 dlimp / tensorflow / RLDS 数据管线。必须在 import vla 之前调用。"""
    if "dlimp" in sys.modules:
        return

    # 告诉 transformers 别去碰 TF/Flax —— 否则它探测到 stub 后会尝试真导入
    import os
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "NO")
    os.environ.setdefault("USE_TORCH", "YES")
    os.environ.setdefault("USE_JAX", "NO")

    # dlimp：被 rlds/dataset.py 以 `import dlimp as dl` 使用
    dl = _stub("dlimp", any_attr=True)
    dl.DLataset = type("DLataset", (), {})
    _stub("dlimp.transforms", any_attr=True)
    _stub("dlimp.utils", any_attr=True)

    # tensorflow：数据管线在 import 期就会调 tf.config.set_visible_devices 等，
    # 用 any_attr 让任意调用静默成功
    tf = _stub("tensorflow", any_attr=True)
    # 这两个必须是**真正的类型**：einops 的后端探测会执行
    # `isinstance(tensor, (tf.Tensor, tf.Variable))`，
    # 传入非 type 对象会抛 `isinstance() arg 2 must be a type`。
    # 用永不匹配的私有类，保证 einops 判定「不是 TF 张量」后回退到 torch 后端。
    class _NeverTFTensor:
        pass

    class _NeverTFVariable:
        pass

    tf.Tensor = _NeverTFTensor
    tf.Variable = _NeverTFVariable
    tf.__version__ = "0.0.0-stub"
    _stub("tensorflow_datasets", any_attr=True)
    _stub("tensorflow_graphics", any_attr=True)
    # `from tensorflow_graphics.geometry.transformation import ...` 这种写法
    # 要求每一层都在 sys.modules 里真实注册，__getattr__ 兜不住
    for sub in (
        "tensorflow_graphics.geometry",
        "tensorflow_graphics.geometry.transformation",
        "tensorflow.data", "tensorflow.io", "tensorflow.image",
        "tensorflow.math", "tensorflow.random", "tensorflow.config",
        "tensorflow.experimental", "tensorflow.nest",
    ):
        _stub(sub, any_attr=True)

    if verbose:
        print("[stub_tf] 已屏蔽 dlimp/tensorflow（推理不需要数据管线）")


__all__ = ["patch"]
