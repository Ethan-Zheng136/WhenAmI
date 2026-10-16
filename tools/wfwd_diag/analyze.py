import json, sys, numpy as np
def load(path, lo, hi):
    T = {k: [] for k in "A B C Y sf sb w c step gw gsf gsb".split()}
    for line in open(path):
        r = json.loads(line)
        if not lo <= r["step"] <= hi: continue
        g = {(c, n): np.array(v) for c, n, v in r["grads"]}
        for x in r["rec"]:
            if "usable" not in x: continue
            u = np.array(x["usable"], bool)
            if not u.any(): continue
            c = x["c"]
            for k, kk in (("A","A"),("B","B"),("C","C"),("Y","Y"),("sf","s_fwd"),("sb","s_bwd"),("w","w")):
                T[k].append(np.array(x[kk])[u].ravel())
            n = int(u.sum()*len(x["A"][0]))
            T["c"].append(np.full(n, c)); T["step"].append(np.full(n, r["step"]))
            for k, nm in (("gw","w"),("gsf","s_fwd"),("gsb","s_bwd")):
                T[k].append(g[(c, nm)][u].ravel() if (c, nm) in g else np.full(n, np.nan))
    return {k: np.concatenate(v) for k, v in T.items()}

def report(T):
    A,B,C,Y,sf,sb,w = (T[k] for k in "A B C Y sf sb w".split())
    Y = Y+1e-6; ef, eb = A/Y, B/Y
    fe = lambda ww: np.mean((ww**2*A + (1-ww)**2*B + 2*ww*(1-ww)*C)/Y)
    w_iv = eb/(ef+eb+1e-12)
    w_or = np.clip((B-C)/np.maximum(A+B-2*C, 1e-12), 0, 1)
    w_glob = np.sum((B-C)/Y)/np.sum((A+B-2*C)/Y)
    w_logged = eb.mean()/(ef.mean()+eb.mean())
    o = {"tokens": len(A), "mean_ef": ef.mean(), "mean_eb": eb.mean(), "frac_tok_eb<ef": np.mean(eb < ef),
         "median_ef": np.median(ef), "median_eb": np.median(eb),
         "w_learned_mean": w.mean(), "w_learned_std": w.std(), "w_iv_tok_mean": w_iv.mean(), "w_iv_tok_median": np.median(w_iv),
         "w_oracle_tok_mean": w_or.mean(), "w_global_opt": w_glob, "w_logged_ratio": w_logged,
         "fused_learned": fe(w), "fused_0.5": fe(.5), "fused_global_opt": fe(w_glob), "fused_logged_ratio": fe(w_logged),
         "fused_iv_tok": fe(w_iv), "fused_oracle_tok": fe(w_or), "fwd_only": fe(1.), "bwd_only": fe(0.)}
    for nm, s, e in (("fwd", sf, ef), ("bwd", sb, eb)):
        le = np.log(e+1e-12)
        o[f"cal_{nm}_corr(s,loge)"] = np.corrcoef(s, le)[0, 1]
        o[f"cal_{nm}_slope(loge~s)"] = np.polyfit(s, le, 1)[0]
        o[f"cal_{nm}_mean_s"] = s.mean(); o[f"cal_{nm}_mean_loge"] = le.mean(); o[f"cal_{nm}_log_mean_e"] = np.log(e.mean())
        o[f"cal_{nm}_mean_exp_s/mean_e"] = np.exp(s).mean()/e.mean()
    ds, de = sb-sf, np.log(eb+1e-12)-np.log(ef+1e-12)
    o["rel_corr(sb-sf, log eb/ef)"] = np.corrcoef(ds, de)[0, 1]
    o["rel_mean(sb-sf)"] = ds.mean(); o["rel_mean(log eb/ef)"] = de.mean()
    o["corr(w_learned, w_oracle_tok)"] = np.corrcoef(w, w_or)[0, 1]
    gw = T["gw"]
    if np.isfinite(gw).any():
        f = gw*w*(1-w)      # downstream-loss gradient on s_b (and minus on s_f)
        o["dL/dw_mean(downstream)"] = np.nanmean(gw)
        o["frac dL/dw>0 (wants w down)"] = np.nanmean(gw > 0)
        o["grad_sb: fusion_path"] = np.nanmean(f); o["grad_sb: nll+agree"] = np.nanmean(T["gsb"]-f)
        o["grad_sf: fusion_path"] = np.nanmean(-f); o["grad_sf: nll+agree"] = np.nanmean(T["gsf"]+f)
        o["|fusion|/|nll+agree| on (sb-sf)"] = np.nanmean(np.abs(2*f))/np.nanmean(np.abs((T["gsb"]-f)-(T["gsf"]+f)))
        o["grad (sb-sf) fusion"] = np.nanmean(2*f); o["grad (sb-sf) nll+agree"] = np.nanmean((T["gsb"]-f)-(T["gsf"]+f))
    else:
        o["grad_sb total"] = np.nanmean(T["gsb"]); o["grad_sf total"] = np.nanmean(T["gsf"])
        o["grad (sb-sf) total"] = np.nanmean(T["gsb"]-T["gsf"])
    return o

if __name__ == "__main__":
    lo, hi = int(sys.argv[2]), int(sys.argv[3])
    o = report(load(sys.argv[1], lo, hi))
    print(json.dumps({k: (round(float(v), 5) if not isinstance(v, int) else v) for k, v in o.items()}, indent=0))
