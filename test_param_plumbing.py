"""[zgt] 参数透传测试 —— 配置必须真的到达模型。

为什么需要：
    `converge_type=masked` 写进了 config.json，训练也正常跑，但 checkpoint
    里存的是 `obs_encoder.*`（32 个键，递推版结构）而非三路分支（416 个键）。
    两个 320 卡任务因此白跑了 10000+ 步。

    根因：load_vla / from_pretrained 的 `**kwargs` 被转发给 PrismaticVLM
    而非 WoG，落进 kwargs 的参数被静默丢弃。签名里显式列出的参数才会到达。

    这类 bug 不会报错 —— 训练照常、loss 照常下降，只有去读 checkpoint 的
    键名才发现跑的是另一个模型。故用结构指纹（键名集合）来验证。
"""
import json
import sys

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

res = {}


def sig(module):
    """结构指纹：用 state_dict 的键名判断实际构造的是哪个实现。"""
    ks = set(module.state_dict())
    return {
        "n_keys": len(ks),
        "has_obs_encoder": any("obs_encoder" in k for k in ks),   # 递推版特征
        "has_q_past": any(k.startswith("q_past.") for k in ks),   # masked 版特征
        "has_frame_embed": "frame_embed" in ks,
    }


# --- 1. 两种实现的指纹可区分 ---
try:
    from vla.temporal import TemporalConverge, StateSpec
    from vla.temporal_masked import MaskedConverge
    r = sig(TemporalConverge(StateSpec(), patch_dim=1024))
    m = sig(MaskedConverge(patch_dim=1024, spec=StateSpec()))
    res["1_signatures_differ"] = {
        "PASS": r["has_obs_encoder"] and not r["has_q_past"]
                and m["has_q_past"] and not m["has_obs_encoder"],
        "recurrent": r,
        "masked": m,
    }
except Exception as e:
    res["1_signatures_differ"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 2. from_pretrained 的签名必须显式列出这三个参数 ---
try:
    import inspect
    from vla.vla import WoG
    params = set(inspect.signature(WoG.from_pretrained).parameters)
    need = {"converge_type", "calib_recon_weight", "calib_act_weight"}
    res["2_from_pretrained_signature"] = {
        "PASS": need <= params,
        "missing": sorted(need - params),
        "note": "**kwargs goes to PrismaticVLM, not WoG - explicit params only",
    }
except Exception as e:
    res["2_from_pretrained_signature"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 3. load_vla 的签名同理 ---
try:
    import inspect
    from vla.load import load_vla
    params = set(inspect.signature(load_vla).parameters)
    need = {"converge_type", "calib_recon_weight", "calib_act_weight"}
    res["3_load_vla_signature"] = {
        "PASS": need <= params,
        "missing": sorted(need - params),
    }
except Exception as e:
    res["3_load_vla_signature"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 4. from_pretrained 内部把它们传给了 cls() ---
try:
    src = inspect.getsource(WoG.from_pretrained)
    body = src[src.index("vlm,"):]      # cls(...) 调用之后的部分
    ok = all(f"{k}={k}" in body for k in
             ("converge_type", "calib_recon_weight", "calib_act_weight"))
    res["4_forwarded_to_cls"] = {
        "PASS": ok,
        "note": "listing a param in the signature is not enough - it must "
                "also be passed on to cls()",
    }
except Exception as e:
    res["4_forwarded_to_cls"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 5. train.py 把 cfg 里的值传给了 load_vla ---
try:
    src = open("/home/zhengguantian/WoG_fresh/WoG/scripts/train.py").read()
    ok = all(f"{k}=cfg.{k}" in src for k in
             ("converge_type", "calib_recon_weight", "calib_act_weight"))
    res["5_train_py_passes"] = {"PASS": ok}
except Exception as e:
    res["5_train_py_passes"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 6. 启动脚本透传环境变量 ---
try:
    sh = open("/home/zhengguantian/WoG_fresh/run_whenami.sh").read()
    ok = ("--converge_type" in sh and "--calib_recon_weight" in sh
          and "--calib_act_weight" in sh)
    res["6_launch_script_passes"] = {"PASS": ok}
except Exception as e:
    res["6_launch_script_passes"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_plumbing.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
