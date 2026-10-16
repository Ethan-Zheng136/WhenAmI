"""Create diagnostic copy of whenami_present (defaults unchanged; new behaviour only behind flags)."""
import sys, re
from pathlib import Path
d = Path(sys.argv[1])

def sub(f, old, new, count=1):
    p = d/f; s = p.read_text()
    assert s.count(old) == count, (f, old[:60], s.count(old))
    p.write_text(s.replace(old, new))

# ---- losses: H2 flag ----
sub("losses.py", "def agreement(out):", "AGREE_DETACH_VAR = False   # diag flag (H2): no agreement gradient into s\n\n\ndef agreement(out):")
sub("losses.py", "    var = torch.exp(out[\"s_fwd\"]) + torch.exp(out[\"s_bwd\"])\n",
    "    var = torch.exp(out[\"s_fwd\"]) + torch.exp(out[\"s_bwd\"])\n    if AGREE_DETACH_VAR:\n        var = var.detach()\n")
sub("losses.py", "def monotonic(", '''def token_record(outs, targets):
    """Diag: raw per-token quantities so any fusion weight's error is computable offline.
    a=p_fwd-y, b=p_bwd-y; A=|a|^2/n, B=|b|^2/n, C=<a,b>/n, Y=|y|^2/n (same normalisation as _err)."""
    rec = []
    with torch.no_grad():
        for c, (o, y) in enumerate(zip(outs, targets)):
            r = {"c": c, "s_fwd": o["s_fwd"].float().flatten(1).tolist()}
            if "p_bwd" in o:
                y = y.float(); a = o["p_fwd"].float()-y; b = o["p_bwd"].float()-y
                f = lambda t: t.mean(-1).flatten(1).tolist()
                r.update(usable=o["usable"].tolist(), A=f(a*a), B=f(b*b), C=f(a*b), Y=f(y*y),
                         Pb=f(o["p_bwd"].float().square()), s_bwd=o["s_bwd"].float().flatten(1).tolist(),
                         w=o["w_fwd"].float().flatten(1).tolist())
            rec.append(r)
    return rec


def monotonic(''')

# ---- model: H3 flag (fusion weight detached) + gradient hooks ----
sub("model.py", "class Precision(nn.Module):", '''FUSION_DETACH_W = False   # diag flag (H3): downstream losses cannot move the fusion weights
GRAD_HOOKS = None         # diag: list collecting per-token dL/dw, dL/ds_fwd, dL/ds_bwd


def _hook(name, c):
    def f(g):
        if GRAD_HOOKS is not None:
            GRAD_HOOKS.append((c, name, g.detach().float().flatten(1).tolist()))
    return f


class Precision(nn.Module):''')
sub("model.py", "            if p.fusion == \"precision\":\n                w = torch.sigmoid(s_bwd - s_fwd)           # exp(-s_f)/(exp(-s_f)+exp(-s_b))\n",
    "            if p.fusion == \"precision\":\n                w = torch.sigmoid(s_bwd - s_fwd)           # exp(-s_f)/(exp(-s_f)+exp(-s_b))\n"
    "                if FUSION_DETACH_W:\n                    w = w.detach()\n"
    "                if GRAD_HOOKS is not None and torch.is_grad_enabled():\n"
    "                    if w.requires_grad:\n                        w.register_hook(_hook(\"w\", c))\n"
    "                    s_fwd.register_hook(_hook(\"s_fwd\", c)); s_bwd.register_hook(_hook(\"s_bwd\", c))\n")

# ---- train: flags, record dump, optional no final save ----
sub("train.py", '    a("--bf16-storage", action="store_true", help="engineering single-card only")\n',
    '    a("--bf16-storage", action="store_true", help="engineering single-card only")\n'
    '    a("--diag-tokens", action="store_true", help="DIAG: per-token fusion records -> diag_tokens.jsonl")\n'
    '    a("--agree-detach-var", action="store_true", help="DIAG H2: agreement variances detached")\n'
    '    a("--fusion-detach-w", action="store_true", help="DIAG H3: fusion weights detached from downstream losses")\n'
    '    a("--no-final-save", action="store_true", help="DIAG: skip checkpoint at stop/last")\n')
sub("train.py", "    return L, diagnostics(outs, prog, [hs[:, c:c+1] for c in PRESENTS])\n",
    "    if a.diag_tokens:\n        import whenami_present.losses as _l\n"
    "        a._rec = _l.token_record(outs, [hs[:, c:c+1] for c in PRESENTS])\n"
    "    return L, diagnostics(outs, prog, [hs[:, c:c+1] for c in PRESENTS])\n")
sub("train.py", "    torch.set_num_threads(1)\n",
    "    torch.set_num_threads(1)\n    import whenami_present.losses as _l, whenami_present.model as _m\n"
    "    _l.AGREE_DETACH_VAR = a.agree_detach_var; _m.FUSION_DETACH_W = a.fusion_detach_w\n")
sub("train.py", "        for micro in range(accum):\n",
    "        for micro in range(accum):\n"
    "            if a.diag_tokens:\n                import whenami_present.model as _m\n                _m.GRAD_HOOKS = []\n")
sub("train.py", "            (L[\"total\"]/accum).backward()\n",
    "            (L[\"total\"]/accum).backward()\n"
    "            if a.diag_tokens and rank == 0:\n"
    "                with (a.output/\"diag_tokens.jsonl\").open(\"a\") as f:\n"
    "                    f.write(json.dumps({\"step\": step, \"micro\": micro, \"hide\": hide is not None, \"rec\": a._rec,\n"
    "                                        \"grads\": _m.GRAD_HOOKS})+\"\\n\")\n"
    "                _m.GRAD_HOOKS = None\n")
sub("train.py", "        if step % a.save_every == 0 or step == a.first_check or last:\n",
    "        if step % a.save_every == 0 or ((step == a.first_check or last) and not a.no_final_save):\n")
print("patched")
