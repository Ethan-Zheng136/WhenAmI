import json, sys, glob, numpy as np
sys.path.insert(0, "/home/zhengguantian/wfwd_diag"); from analyze import load, report
O = "/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_present/local/wfwd_diag2_"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
fm = {}
for n in ("A_base", "B_agreedetach", "C_fwdonly", "D_mean"):
    try:
        rows = [json.loads(l) for l in open(O+n+"/teacher/train.jsonl")]
    except FileNotFoundError:
        continue
    rows = {r["step"]: r for r in rows if lo <= r["step"] <= hi}
    fm[n] = rows
    T = load(O+n+"/teacher/diag_tokens.jsonl", lo, hi)
    o = report(T)
    corr = np.mean(T["C"]/np.sqrt(T["A"]*T["B"]+1e-12))
    keys = ("mean_ef", "mean_eb", "w_learned_mean", "w_iv_tok_mean", "w_logged_ratio", "w_global_opt", "fused_learned", "fwd_only",
            "fused_0.5", "fused_iv_tok", "fused_oracle_tok", "cal_fwd_mean_exp_s/mean_e", "cal_bwd_mean_exp_s/mean_e", "rel_corr(sb-sf, log eb/ef)")
    print(n, "steps", min(rows), max(rows), "tok_err_corr %.3f" % corr, {k: round(float(o[k]), 4) for k in keys})
    print("   ", {k: round(float(np.mean([r[k] for r in rows.values() if r[k] is not None])), 5) for k in ("fm_present", "fm_generation", "total", "err_fwd", "err_bwd", "w_fwd_optimal")})
    ev = sorted(glob.glob(O+n+"/teacher/development/step-*/*.json"))
    print("    dev files:", [e.split("/")[-1] for e in ev][:6])
if "A_base" in fm:
    for n in fm:
        if n == "A_base": continue
        common = sorted(set(fm[n]) & set(fm["A_base"]))
        d = np.array([fm[n][s]["fm_present"]-fm["A_base"][s]["fm_present"] for s in common])
        print(f"paired fm_present {n}-A: mean {d.mean():+.5f} se {d.std()/np.sqrt(len(d)):.5f} n {len(d)}")
