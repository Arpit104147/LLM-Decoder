"""
Fix answers: a computed weight edit (least squares) in one layer's MLP output matrix.

You pick positions (in saved traces, or in a corrected copy of a reply) and the token the model should have said
there. For each fix:
  key   k = what the edited matrix reads at that position (the layer's neuron activations), times the factor its
            output is scaled by (an expert's router weight, the shared expert's gate; 1 for a dense MLP)
  goal  r = the change the MLP's raw output needs so the wanted token wins by `margin`. Found with the real model:
            a direction toward the wanted token's output row (through the MLP's output norm when the model has
            one), scaled by a line search through the later layers and refined a few times
Then, per edited matrix, the smallest change to its columns S, measured on the activations of other saved chats
(so their outputs move as little as possible), that gives exactly r for every key and exactly 0 for protected keys:
  ΔW_S = R (K A⁻¹ Kᵀ)⁻¹ K A⁻¹,   A = covariance of the kept activations on S (+ a small ridge)

What gets edited in layer L:
  dense MLP              blk.L.ffn_down.weight                (Llama, Qwen, Mistral, Phi, Granite, Command-R…)
  dense MLP + post-norm  blk.L.ffn_down.weight, solved through the norm (Gemma 2/3, OLMo 2, LFM)
  MoE with shared expert blk.L.ffn_down_shexp.weight          (Qwen-MoE, DeepSeek, Llama 4, GLM…)
  MoE, routed only       blk.L.ffn_down_exps.weight, the expert each fixed token was sent to first (OLMoE, Qwen3-MoE,
                         GPT-OSS, Mixtral…): one solve per expert
The edit is saved as computed/<name>.npz (patches: tensor, expert or -1, columns, values to ADD) and applied after
grid / CSV / edit_weights.py edits on export and in Chat & trace.
"""
import copy
import itertools
import json
import re
import time
import uuid
import weakref

import numpy as np

import gguf_core as core
import trace_core as tc

_SOL = {}            # solution id -> unsaved solution
_LAST = None         # the last solve's matrices (for tests)
CHUNK = 64
MAX_FIXES = 512
BIG_EDIT_PCT = 50.0     # an edit bigger than this share of its matrix (by size) is blocked from one-click apply


def computed_dir(ws):
    return core.workspace_dir(ws) / "computed"


# ------------------------------------------------------------------ what a layer's fix edits
def edit_target(model, L, part="mlp"):
    """{"part": mlp | attn, "kind": dense | shared | routed | attn, "tensor", "post": output norm or None, "ff": key length}
    part="attn": the attention (or linear-attention / mixer) output matrix: attn_output, or ssm_out."""
    if not 0 <= L < model.n_layer:
        raise ValueError(f"Layer must be 0–{model.n_layer - 1}")
    if not model.attribution_exact:
        raise ValueError("This architecture merges several residual streams (Gemma 3n); Fix answers can't edit it yet")
    if part == "attn":
        return _attn_target(model, L)
    post = model.post_norm_names(L)[1]
    if model.is_moe(L):
        if post:
            raise ValueError(f"Layer {L}: a mixture-of-experts layer with an output norm isn't supported yet")
        sh = model.w(L, "ffn_down_shexp.weight")
        if sh is not None:
            tg = {"kind": "shared", "tensor": f"blk.{L}.ffn_down_shexp.weight", "post": None, "ff": int(sh.shape[1]), "W": sh}
        else:
            Dn = model.w(L, "ffn_down_exps.weight")
            if Dn is None or getattr(Dn, "ndim", 0) != 3:
                raise ValueError(f"Layer {L} has no expert output matrix to edit")
            tg = {"kind": "routed", "tensor": f"blk.{L}.ffn_down_exps.weight", "post": None, "ff": int(Dn.shape[2]), "W": Dn}
    else:
        W = model.w(L, "ffn_down.weight")
        if W is None or getattr(W, "ndim", 0) != 2:
            raise ValueError(f"Layer {L} has no MLP output matrix (ffn_down) to edit (Mamba / RWKV layers have none)")
        tg = {"kind": "dense", "tensor": f"blk.{L}.ffn_down.weight", "post": post, "ff": int(W.shape[1]), "W": W}
    if not isinstance(tg["W"], np.ndarray):
        raise ValueError("Fix answers needs the model fully in memory (not low-memory mode)")
    tg["part"] = "mlp"
    tg["layer"] = L
    return tg


def _attn_target(model, L):
    if type(model).block is not tc.Model.block:
        raise ValueError(f"Layer {L}: attention edits need the standard layer layout (not hybrid Mamba / RWKV blocks)")
    W = model.out_proj(L)
    if W is None or getattr(W, "ndim", 0) != 2:
        raise ValueError(f"Layer {L} has no attention output matrix to edit")
    if not isinstance(W, np.ndarray):
        raise ValueError("Fix answers needs the model fully in memory (not low-memory mode)")
    name = next((n for n, v in model.T.items() if v is W), None)
    if name is None:
        raise ValueError(f"Layer {L}: can't find the attention output tensor")
    b = model.T.get(name[:-len("weight")] + "bias") if name.endswith("weight") else None
    return {"part": "attn", "kind": "attn", "tensor": name, "post": model.post_norm_names(L)[0], "ff": int(W.shape[1]),
            "W": W, "bias": b, "layer": L}


def editable_layers(model, part="mlp"):
    out = []
    for L in range(model.n_layer):
        try:
            edit_target(model, L, part)
            out.append(L)
        except ValueError:
            pass
    return out


# ------------------------------------------------------------------ running parts of the model
_PFX = {}          # the last prefix read: {"model", "ids", "st"} (token fixes of one answer share most of their text)


def _forget_prefix():
    _PFX.clear()


def _prefix_state(model, ids):
    """Every layer's state after reading ids (positions 0..len-1). Starts from the last prefix read when ids extends
    it (same model, same weights), so the token fixes of one answer don't re-read the whole reply each time."""
    ids = [int(i) for i in ids]
    c = _PFX
    if c.get("model") is not None and c["model"]() is model and len(c["ids"]) <= len(ids) and ids[:len(c["ids"])] == c["ids"]:
        st, start = copy.deepcopy(c["st"]), len(c["ids"])
    else:
        st, start = tc.State(model.n_layer), 0
    rec = tc.NullRecorder()
    for s0 in range(start, len(ids), CHUNK):
        x = model.embed(ids[s0:s0 + CHUNK])
        pos = np.arange(s0, s0 + len(x))
        for L in range(model.n_layer):
            x = model.block(L, x, pos, st, rec)[0].astype(np.float32, copy=False)
    _PFX.update(model=weakref.ref(model), ids=ids, st=copy.deepcopy(st))     # weak: an unloaded model can be freed
    return st


def _moe_rows(rec, L):
    if f"moe{L}_ids" not in rec.cols:
        return None
    return {k: np.concatenate(rec.cols[f"moe{L}_{k}"]) for k in ("ids", "w", "sg")}


def _upto(model, st, tok, pos, L):
    """The token at `pos` through layers 0..L (st's layers 0..L then include pos, the later ones don't yet).
    Returns the parts of layer L: neuron activations, residual after it, the MLP output it added, routing."""
    rec, x, p = tc.Recorder(), model.embed([tok]), np.array([pos])
    x_in = x
    for l in range(L + 1):
        if l == L:
            x_in = x
        out = model.block(l, x, p, st, rec)
        x = out[0].astype(np.float32, copy=False)
    moe = _moe_rows(rec, L)
    return {"act": out[1][0].astype(np.float64), "x": x, "m_out": out[4][0].astype(np.float64),
            "x_in": x_in[0].astype(np.float64), "a_out": out[3][0].astype(np.float64),
            "hv": np.asarray(out[2], np.float64)[0].reshape(-1),
            "moe": {k: v[0] for k, v in moe.items()} if moe else None}


def _finish(model, st, x, pos, L):
    """Residual after layer L at `pos` -> (logits, final residual), on a copy of the state."""
    st, rec, p = copy.deepcopy(st), tc.NullRecorder(), np.array([pos])
    for l in range(L + 1, model.n_layer):
        x = model.block(l, x, p, st, rec)[0].astype(np.float32, copy=False)
    return model.logits(model.final(x))[0].astype(np.float64), x[0].astype(np.float64)


def _keys(model, tg, parts):
    """[(group, key)] for one position: group = expert index for routed experts, else None."""
    act, moe = parts["act"], parts["moe"]
    if tg["kind"] == "attn":
        return [(None, parts["hv"][:tg["ff"]])]
    if tg["kind"] == "dense":
        return [(None, act[:tg["ff"]])]
    ids = np.atleast_1d(moe["ids"])
    k = len(ids)
    if tg["kind"] == "shared":
        ffe = (act.shape[-1] - tg["ff"]) // k
        return [(None, act[k * ffe:k * ffe + tg["ff"]] * float(moe["sg"]))]
    ffe = tg["ff"]
    w = np.atleast_1d(moe["w"])
    return [(int(ids[s]), act[s * ffe:(s + 1) * ffe] * (1.0 if model.moe_weight_before else float(w[s]))) for s in range(k)]


def _raw(model, tg, parts, L):
    """The edited matrix's output before the output norm (dense MLP, or attention)."""
    W = tg["W"]
    if tg["part"] == "attn":
        return parts["hv"][:tg["ff"]] @ W.T.astype(np.float64) + (tg["bias"] if tg["bias"] is not None else 0)
    b = model.w(L, "ffn_down.bias")
    return parts["act"][:tg["ff"]] @ W.T.astype(np.float64) + (b if b is not None else 0)


def _rest_of_layer(model, L, x_mid):
    """The MLP half of a standard layer on a new residual (what the attention output feeds). x_mid: [D] or [n, D]."""
    if np.ndim(x_mid) == 2:
        return np.stack([_rest_of_layer(model, L, r) for r in x_mid]) if model.is_moe(L) else _rest_rows(model, L, x_mid)
    return _rest_rows(model, L, np.asarray(x_mid)[None])[0]


def _rest_rows(model, L, X):
    pa, pf = model.post_norm_names(L)
    if pa and model.w(L, "ffn_norm.weight") is not None:
        fn = f"blk.{L}.ffn_norm.weight"
    elif model.w(L, "attn_norm.weight") is None:
        fn = None
    else:
        fn = model.pre_ffn_norm(L)
    X = np.asarray(X, np.float64)
    h2 = model.norm(X.astype(np.float32), fn) if fn else X.astype(np.float32)
    _act, m = model.ffn(L, h2, None)
    m, _div = model._post(m, pf)
    return X + float(model.res_scale) * m.astype(np.float64)


def _resid(model, tg, parts, dm, L):
    """Residual after layer L at the position when the MLP's raw output changes by dm."""
    x = parts["x"][0].astype(np.float64)
    rs = float(model.res_scale)
    if tg["part"] == "attn":
        if tg["post"]:
            a_new = rs * model._post((parts["raw"] + dm)[None].astype(np.float32), tg["post"])[0][0].astype(np.float64)
        else:
            a_new = parts["a_out"] + rs * dm
        if model.parallel:                             # attention and MLP read the same input: only the sum changes
            return x - parts["a_out"] + a_new
        return _rest_of_layer(model, L, parts["x_in"] + a_new)   # the MLP of this layer reads the new attention output
    if tg["post"]:
        x_mid = x - parts["m_out"]
        y = model._post((parts["raw"] + dm)[None].astype(np.float32), tg["post"])[0][0].astype(np.float64)
        return x_mid + rs * y
    return x + rs * dm


def _margin(lg, t):
    o = lg.copy()
    o[t] = -np.inf
    j = int(o.argmax())
    return lg[t] - lg[j], j


def _probs(lg):
    e = np.exp(lg - lg.max())
    return e / e.sum()


def _score_diff(model, X, t, j):
    """score(t) − score(j) for final residuals X [n, D] (final norm, output rows t and j, bias, scale, softcap)."""
    xf = model.final(np.asarray(X, np.float32)).astype(np.float64)
    yt, yj = xf @ model.U[t].astype(np.float64), xf @ model.U[j].astype(np.float64)
    if model.out_bias is not None:
        yt, yj = yt + float(model.out_bias[t]), yj + float(model.out_bias[j])
    if model.logit_scale != 1.0:
        yt, yj = yt / model.logit_scale, yj / model.logit_scale
    if model.softcap_final:
        c = float(model.softcap_final)
        yt, yj = np.tanh(yt / c) * c, np.tanh(yj / c) * c
    return yt - yj


def _vjp(fn, v, g=None, rel=1e-2):
    """gᵀ·J of a row-wise map fn (R^D -> R^D, or -> R when g is None) at v, by central differences over all D
    coordinates at once (exact for linear maps; a 1% step: smaller ones drown in float32 rounding through an MLP)."""
    v = np.asarray(v, np.float64)
    D = len(v)
    h = rel * (float(np.sqrt(np.mean(v ** 2))) + 1e-6)
    out = np.zeros(D)
    for i0 in range(0, D, 1024):
        n = min(1024, D - i0)
        E = np.zeros((n, D))
        E[np.arange(n), i0 + np.arange(n)] = h
        yp, ym = np.asarray(fn(v + E), np.float64), np.asarray(fn(v - E), np.float64)
        out[i0:i0 + n] = ((yp - ym) if g is None else (yp - ym) @ g) / (2 * h)
    return out


def L_of(tg):
    return int(tg["layer"])


def _direction(model, tg, parts, t, j, x_final):
    """Steepest direction, per unit of change, in the edited matrix's raw output that raises token t's score over
    token j's: the exact gradient of score(t) − score(j) along the direct path (final norm, output rows, softcap;
    through the layer's output norm when it has one). Later layers are then taken into account by the line search."""
    g = _vjp(lambda X: _score_diff(model, X, t, j), x_final)            # d margin / d final residual
    if tg["part"] == "attn" and not model.parallel:                      # this layer's MLP re-reads the attention output
        g = _vjp(lambda X: _rest_of_layer(model, L_of(tg), X), parts["x_in"] + parts["a_out"], g)
    if tg["post"]:                                                       # residual += res_scale · norm(raw)
        g = _vjp(lambda Y: model._post(np.asarray(Y, np.float32), tg["post"])[0], parts["raw"], g)
    return g / (np.linalg.norm(g) + 1e-12)


def _goal(model, tg, st, parts, pos, L, t, margin, log):
    """Smallest change of the MLP's raw output (along refined score directions) that makes t win by margin.
    Returns (dm, logits before, logits with dm, residual change)."""
    dm = np.zeros(model.D)
    run = lambda d: _finish(model, st, _resid(model, tg, parts, d, L)[None].astype(np.float32), pos, L)
    lg0, xf = run(dm)
    m, j = _margin(lg0, t)
    if m >= margin:
        return dm, lg0, lg0
    base = np.linalg.norm(parts["raw"] if tg["post"] else parts["x"][0]) + 1e-6
    for _round in range(4):
        g = _direction(model, tg, parts, t, j, xf)
        lo, hi, got = 0.0, None, None
        a = 0.02 * base
        best_a, best_m = 0.0, m
        for _ in range(18):
            lg = run(dm + a * g)[0]
            ma = _margin(lg, t)[0]
            if ma >= margin:
                hi, got = a, lg
                break
            if ma > best_m:
                best_a, best_m = a, ma
            lo, a = a, a * 2
        if hi is None:                               # not reached along this direction: keep the step that came closest
            dm = dm + best_a * g
            lg_, xf = run(dm)
            m, j = _margin(lg_, t)
            continue
        for _ in range(7):                           # bisect to the smallest step that still reaches the margin
            mid = (lo + hi) / 2
            lg = run(dm + mid * g)[0]
            if _margin(lg, t)[0] >= margin:
                hi, got = mid, lg
            else:
                lo = mid
        return dm + hi * g, lg0, got
    log(f"  could not make the token win by {margin} at position {pos}; using the closest change found")
    return dm, lg0, run(dm)[0]


_GOALS = {"model": None, "cache": {}}     # goals already found: settings that edit the same layer share them
_ACTIVE = []                               # patches currently added to the model (part of the goal's key)
_TOKEN = itertools.count()


class _Saved(list):
    pass


def goal(model, tg, st, parts, ids, pos, L, t, margin, log):
    """_goal, remembered: (dm, {"was", "was_p", "target_p_before", "goal_target_p"}). The same fix at the same layer
    with the same weights (no patch in place, or the same ones) always has the same goal, so trying several settings
    (rank, strength, mode) on one layer computes it once."""
    c = _GOALS
    if c["model"] is None or c["model"]() is not model:
        c.update(model=weakref.ref(model), cache={})
    key = (tg["part"], int(L), tuple(int(i) for i in ids[:pos + 1]), int(pos), int(t), float(margin), tuple(_ACTIVE))
    if key not in c["cache"]:
        dm, lg0, lg1 = _goal(model, tg, st, parts, pos, L, t, margin, log)
        p0, p1 = _probs(lg0), _probs(lg1)
        if len(c["cache"]) >= 20000:
            c["cache"].clear()
        c["cache"][key] = (dm, {"was": int(lg0.argmax()), "was_p": float(p0.max()), "target_p_before": float(p0[t]),
                                "goal_target_p": float(p1[t])})
    dm, info = c["cache"][key]
    return dm.copy(), dict(info)


def forget_goals():
    _GOALS.update(model=None, cache={})


def forced(model, ids):
    """Teacher-forced pass over ids: the trace arrays (as a saved trace has them) for positions 0..len-2."""
    st, rec = tc.State(model.n_layer), tc.Recorder()
    for s0 in range(0, len(ids) - 1, CHUNK):
        chunk = ids[s0:min(s0 + CHUNK, len(ids) - 1)]
        lg = model.logits(model.forward(chunk, st, rec))
        tc._record_preds(rec, lg, ids[s0 + 1:s0 + 1 + len(chunk)])
    return tc._pack(model, rec)


def _position_parts(model, tg, ids, pos, L, st=None):
    st = st if st is not None else _prefix_state(model, ids[:pos])
    parts = _upto(model, st, ids[pos], pos, L)
    if tg["post"] or tg["part"] == "attn":
        parts["raw"] = _raw(model, tg, parts, L)
    return st, parts


def _trace_keys(ws, model, which, tid, L, tg):
    """{group: keys [n, ff]} at every position of a saved trace, for the weights being edited."""
    info, A = tc._open_trace(ws, tid)
    mk = f"moe{L}_ids"
    if info.get("weights") == which and not tc._stale(ws, info) and "act" in A and (tg["kind"] in ("dense", "attn") or mk in A):
        act = A["act"][:, L].astype(np.float64)
        hv = A["heads"][:, L].astype(np.float64)
        moe = {k: A[f"moe{L}_{k}"] for k in ("ids", "w", "sg")} if tg["kind"] in ("shared", "routed") else None
    else:
        ids = info["ids"][:-1]
        st, rec, acts, hvs = tc.State(model.n_layer), tc.Recorder(), [], []
        for s0 in range(0, len(ids), CHUNK):
            x = model.embed(ids[s0:s0 + CHUNK])
            pos = np.arange(s0, s0 + len(x))
            for l in range(L + 1):
                r = model.block(l, x, pos, st, rec)
                x = r[0].astype(np.float32, copy=False)
            acts.append(r[1])
            hvs.append(np.asarray(r[2]).reshape(len(x), -1))
        act = np.concatenate(acts).astype(np.float64)
        hv = np.concatenate(hvs).astype(np.float64)
        moe = _moe_rows(rec, L)
    if tg["kind"] == "attn":
        return {None: hv[:, :tg["ff"]]}
    out = {}
    if tg["kind"] == "routed":
        ffe, k = tg["ff"], moe["ids"].shape[1]
        wf = np.ones_like(moe["w"]) if model.moe_weight_before else moe["w"]
        for s in range(k):
            keys = act[:, s * ffe:(s + 1) * ffe] * wf[:, s:s + 1]
            for e in np.unique(moe["ids"][:, s]):
                out.setdefault(int(e), []).append(keys[moe["ids"][:, s] == e])
        return {e: np.concatenate(v) for e, v in out.items()}
    if tg["kind"] == "shared":
        k = moe["ids"].shape[1]
        ffe = (act.shape[1] - tg["ff"]) // k if act.shape[1] > tg["ff"] else 0
        return {None: act[:, k * ffe:k * ffe + tg["ff"]] * moe["sg"][:, None]}
    return {None: act[:, :tg["ff"]]}


def _own_logprobs(model, ids):
    """Log-probability of each next token of ids and the top token, at every position."""
    st, rec, lps, tops = tc.State(model.n_layer), tc.NullRecorder(), [], []
    for s0 in range(0, len(ids) - 1, CHUNK):
        lg = model.logits(model.forward(ids[s0:min(s0 + CHUNK, len(ids) - 1)], st, rec)).astype(np.float64)
        m = lg.max(-1, keepdims=True)
        lse = (m + np.log(np.exp(lg - m).sum(-1, keepdims=True)))[:, 0]
        nxt = np.asarray(ids[s0 + 1:s0 + 1 + len(lg)])
        lps.append(lg[np.arange(len(lg)), nxt] - lse)
        tops.append(lg.argmax(-1))
    return np.concatenate(lps), np.concatenate(tops)


def _fix_ids(ws, f):
    return list(f["ids"]) if f.get("ids") is not None else tc._open_trace(ws, f["trace"])[0]["ids"]


def goal_costs(ws, fixes, layers, margin=1.0, which="edited", log=print, part="mlp"):
    """For each candidate layer: how far the residual must move (mean over the fixes) relative to that layer's
    typical MLP output. Lower = a smaller edit there. Returns [{"layer", "cost", ...}]."""
    model, tok = tc.get_model(ws, which, log)
    out = []
    bases = {}
    for L in layers:
        tg = edit_target(model, L, part)
        goals, typ = [], []
        for n, f in enumerate(fixes):
            tc.check_stop()
            ids, pos, t = _fix_ids(ws, f), int(f["pos"]), int(f["target"])
            if n not in bases:
                bases[n] = _prefix_state(model, ids[:pos])
            st, parts = _position_parts(model, tg, ids, pos, L, copy.deepcopy(bases[n]))
            dm, _info = goal(model, tg, st, parts, ids, pos, L, t, float(margin), lambda m: None)
            goals.append(float(np.linalg.norm(_resid(model, tg, parts, dm, L) - parts["x"][0])))
            typ.append(float(np.linalg.norm(parts["a_out"] if part == "attn" else parts["m_out"])) + 1e-9)
        out.append({"layer": L, "part": part, "cost": float(np.mean(goals) / np.mean(typ)), "goal_norms": [round(g, 3) for g in goals],
                    "typical_mlp_output": round(float(np.mean(typ)), 3), "kind": tg["kind"]})
        log(f"  layer {L} {'attention' if part == 'attn' else 'MLP'} ({tg['kind']}): residual push needed {np.mean(goals):.2f} "
            f"vs its output {np.mean(typ):.2f} -> cost {out[-1]['cost']:.2f}")
    return out


def _fit(K, R, C, nS, n_fix):
    """Columns S (neurons strongly active on the fixes, relative to how active they are elsewhere) and the
    smallest change dW on them, in the metric of the kept activations, with dW·k = r exactly for every key."""
    ff = K.shape[1]
    d = np.diag(C)
    eps = 1e-4 * float(np.mean(d)) + 1e-12
    if nS >= ff:
        S = np.arange(ff)
    else:
        score = (K[:n_fix] ** 2).sum(0) / (d + float(np.mean(d)))
        S = np.sort(np.argsort(-score)[:nS])
    Ks = K[:, S]
    A = C[np.ix_(S, S)] + eps * np.eye(len(S))
    AiKt = np.linalg.solve(A, Ks.T)                                   # [s, n]
    M = Ks @ AiKt
    M += 1e-9 * np.trace(M) / len(M) * np.eye(len(M))
    dW = (R.T @ np.linalg.solve(M, AiKt.T)).astype(np.float32)        # [D, s]
    fit = float(np.linalg.norm(dW @ Ks.T - R.T) / (np.linalg.norm(R) + 1e-12))
    return S, dW, fit


def _fit_general(K, R, C, nS, n_fix, rank, strength):
    """One shared correction of rank <= `rank` for all the fixes (instead of one exact answer per fix):
    minimise  sum_fix |dW k - r|^2 + 3 sum_protect |dW k|^2 + lam * (change of the output on the kept activations),
    lam = strength x (energy of the fix keys / energy of the kept keys). With M = K'K + lam C the best rank-r dW is the
    truncated SVD of R'K M^-1/2, times M^-1/2. Returns (columns, dW, relative miss on the fixes)."""
    ff = K.shape[1]
    d = np.diag(C)
    if nS >= ff:
        S = np.arange(ff)
    else:
        score = (K[:n_fix] ** 2).sum(0) / (d + float(np.mean(d)))
        S = np.sort(np.argsort(-score)[:nS])
    w = np.ones(len(K))
    w[n_fix:] = 3.0
    Kw = K[:, S] * np.sqrt(w)[:, None]
    Rw = R * np.sqrt(w)[:, None]
    G = Kw.T @ Kw
    Cs = C[np.ix_(S, S)]
    lam = float(strength) * float(np.trace(G)) / (float(np.trace(Cs)) + 1e-12)
    M = G + lam * Cs + 1e-6 * float(np.trace(G)) / len(G) * np.eye(len(G))
    ev, V = np.linalg.eigh(M)
    ev = np.maximum(ev, 1e-12 * ev.max())
    Mih = (V / np.sqrt(ev)) @ V.T
    B = Rw.T @ Kw @ Mih                                               # [D, s]
    U, sv, Vt = np.linalg.svd(B, full_matrices=False)
    r = max(1, min(int(rank), len(sv)))
    dW = (((U[:, :r] * sv[:r]) @ Vt[:r]) @ Mih).astype(np.float32)
    fit = float(np.linalg.norm(dW @ K[:n_fix, S].T - R[:n_fix].T) / (np.linalg.norm(R[:n_fix]) + 1e-12))
    return S, dW, fit


def _patch_matrix(model, p):
    W = model.T[p["tensor"]]
    return W[p["expert"]] if p["expert"] >= 0 else W


def _apply(model, patches):
    """Add the patches in place; returns what _restore needs to put the exact original values back."""
    _forget_prefix()                                   # states read with the old weights are no longer valid
    saved = _Saved()
    saved.token = next(_TOKEN)
    for p in patches:
        M = _patch_matrix(model, p)
        saved.append((M, p["cols"], M[:, p["cols"]].copy()))
        M[:, p["cols"]] += p["delta"]
    _ACTIVE.append(saved.token)
    return saved


def _restore(saved):
    _forget_prefix()
    for M, cols, orig in reversed(saved):
        M[:, cols] = orig
    tok = getattr(saved, "token", None)
    if tok in _ACTIVE:
        _ACTIVE.remove(tok)


def _logits_at(model, ids, pos):
    st = _prefix_state(model, ids[:pos])
    parts = _upto(model, st, ids[pos], pos, model.n_layer - 1)
    return model.logits(model.final(parts["x"]))[0].astype(np.float64)


def _solve_layer(ws, model, tok, fixes, L, neurons, margin, which, keep, keep_traces, protect, log, mode="exact",
                 rank=4, strength=0.5, part="mlp", goal_fn=None, exclude=()):
    """One layer's computed edit for these fixes (with any earlier edits already in place in the model).
    goal_fn(i, fix, tg, st, parts) -> (dm, info): the change each fix needs from this part (default: enough to make
    the right token win by the margin). Returns (patches, rows, info)."""
    tg = edit_target(model, L, part)
    D = model.D
    keys, rows = [], []                       # keys: (group, key, goal)
    for i, f in enumerate(fixes):
        tc.check_stop()
        ids, pos, t = _fix_ids(ws, f), int(f["pos"]), int(f["target"])
        st, parts = _position_parts(model, tg, ids, pos, L)
        dm, gi = goal(model, tg, st, parts, ids, pos, L, t, margin, log) if goal_fn is None else goal_fn(i, f, tg, st, parts)
        g, k = _keys(model, tg, parts)[0]                            # routed: the expert the token was sent to first
        keys.append((g, k, dm))
        rows.append({"trace": f.get("trace"), "pos": pos, "target": t, "target_text": tok.decode([t]),
                     "corrected": f.get("ids") is not None, "expert": g,
                     "was": gi["was"], "was_text": tok.decode([gi["was"]]),
                     "was_p": gi["was_p"], "target_p_before": gi["target_p_before"], "goal_target_p": gi["goal_target_p"],
                     "goal_norm": float(np.linalg.norm(dm))})
    groups = list(dict.fromkeys(g for g, _k, _r in keys))
    n_prot = 0
    for f in (protect or [])[:256]:                    # right answers: their MLP output stays exactly the same
        ids = _fix_ids(ws, f)
        _st, parts = _position_parts(model, tg, ids, int(f["pos"]), L)
        for g, k in _keys(model, tg, parts):
            if g in groups:
                keys.append((g, k, np.zeros(D)))
                n_prot += 1
    if n_prot:
        log(f"Protecting {n_prot} answer position(s) that must not change")
    fix_tr = [t for t in dict.fromkeys(f.get("trace") for f in fixes) if t]
    ex = set(exclude or ())                           # never fitted to: they measure the side effects afterwards
    others = [t["id"] for t in tc.list_traces(ws) if t["id"] not in fix_tr and t["id"] not in ex][:int(keep_traces)]
    if keep is not None:             # these first (e.g. a test's right answers), then other saved chats
        extra = [t for t in others if t not in set(keep)]
        others = [t for t in dict.fromkeys(keep) if t not in fix_tr][:200] + extra[:int(keep_traces)]
    kept = {}
    for tid in fix_tr + others:
        for g, Kg in _trace_keys(ws, model, which, tid, L, tg).items():
            if g in groups:
                kept.setdefault(g, []).append(Kg)
    patches, fits, nkept, last = [], [], 0, {}
    rng = np.random.default_rng(0)
    for g in groups:
        sel = [(k, r) for gg, k, r in keys if gg == g]
        n_fix_g = sum(1 for gg, _k, _r in keys[:len(rows)] if gg == g)
        K = np.array([k for k, _r in sel])
        R = np.array([r for _k, r in sel])
        Kk = np.concatenate(kept[g]) if g in kept else np.zeros((0, K.shape[1]))
        if len(Kk) > 20000:
            Kk = Kk[rng.choice(len(Kk), 20000, replace=False)]
        nkept += len(Kk)
        C = Kk.T @ Kk / len(Kk) if len(Kk) else np.eye(K.shape[1]) * float(np.mean(K ** 2) + 1e-12)
        ff = K.shape[1]
        nS = ff if int(neurons) <= 0 or int(neurons) >= ff else int(neurons)
        if mode == "general":
            S, dW, fit = _fit_general(K, R, C, nS, n_fix_g, rank, strength)
        else:
            S, dW, fit = _fit(K, R, C, nS, n_fix_g)
        last[g] = {"K": K, "R": R, "C": C, "n_fix": n_fix_g, "S": S, "dW": dW}
        patches.append({"tensor": tg["tensor"], "expert": -1 if g is None else int(g), "cols": S.astype(np.int64),
                        "delta": dW, "layer": L})
        fits.append(fit)
    global _LAST
    _LAST = last
    sizes = [float(np.linalg.norm(p["delta"]) / (np.linalg.norm(_patch_matrix(model, p)) + 1e-12) * 100) for p in patches]
    where = tg["tensor"] + (f", expert(s) {', '.join(str(p['expert']) for p in patches)}" if tg["kind"] == "routed" else "")
    log(f"Kept the outputs on {nkept:,} positions from {len(fix_tr) + len(others)} saved chat(s) as unchanged as possible")
    log(f"Solved{' (general, rank ' + str(rank) + ')' if mode == 'general' else ''}: {where}; "
        f"{sum(p['delta'].size for p in patches):,} numbers; size of the change = {max(sizes):.2f}% of the matrix")
    return patches, rows, {"layer": L, "part": part, "tensor": tg["tensor"], "kind": tg["kind"], "size": max(sizes),
                           "fit": max(fits), "kept": nkept, "n_fixes": len(fixes), "others": others}


def solve(ws, fixes, layer=None, neurons=0, margin=1.0, keep_traces=40, check_traces=8, which="edited", keep=None,
          check=None, protect=None, chain=2, mode="exact", rank=4, strength=0.5, part="mlp", exclude=(), log=print):
    """fixes: [{"trace": id, "pos": position (it predicts token pos+1), "target": token id, "ids": optional token
    list to use instead of the trace's (a corrected reply)}]. Nothing is saved.
    keep / check: trace ids whose outputs must stay the same / are re-run to measure side effects (default: the
    most recent other saved traces). protect: [{"trace" or "ids", "pos"}] positions whose MLP output must not
    change. chain: if some fixes aren't reached at `layer`, keep that edit and add one in up to `chain` earlier
    editable layers for the rest (a layer can only move the residual so far, e.g. behind an output norm).
    mode="exact": every fix reached exactly. mode="general": one shared correction of rank <= `rank` for all fixes,
    traded off against the kept outputs by `strength` (higher = change other answers less, fix fewer)."""
    if mode not in ("exact", "general"):
        raise ValueError("mode must be exact or general")
    if mode == "general":
        chain = 0                    # one shared correction in one layer: not chained
    if not fixes:
        raise ValueError("Add at least one fix: open a trace, click a token, pick the token it should have been")
    if len(fixes) > MAX_FIXES:
        raise ValueError(f"At most {MAX_FIXES} token fixes at once")
    t0 = time.time()
    model, tok = tc.get_model(ws, which, log)
    _forget_prefix()
    L0 = model.n_layer - 1 if layer in (None, "") else int(layer)
    edit_target(model, L0, part)
    for i, f in enumerate(fixes):
        ids, pos, t = _fix_ids(ws, f), int(f["pos"]), int(f["target"])
        if not 0 <= pos < len(ids) - (0 if f.get("ids") is not None else 1):
            raise ValueError(f"Fix {i + 1}: position {pos} isn't in the reply")
        if not 0 <= t < model.U.shape[0]:
            raise ValueError(f"Fix {i + 1}: token id must be 0–{model.U.shape[0] - 1}")
    order = [L0] + [l for l in reversed(editable_layers(model, part)) if l < L0][:max(0, int(chain))]
    margin = float(margin)
    rows = [None] * len(fixes)
    patches, steps, applied = [], [], []
    remaining = list(range(len(fixes)))
    try:
        for n, L in enumerate(order):
            log(f"Layer {L}: {len(remaining)} token fix(es)")
            ps, rws, info = _solve_layer(ws, model, tok, [fixes[i] for i in remaining], L, neurons, margin, which,
                                         keep, keep_traces, protect, log, mode, rank, strength, part, exclude=exclude)
            for i, r in zip(remaining, rws):
                rows[i] = rows[i] or r
            applied.append(_apply(model, ps))
            patches += ps
            steps.append(info)
            # re-check EVERY fix: an edit in an earlier layer changes what the later edited layers read
            remaining = [i for i in range(len(fixes)) if int(np.argmax(_logits_at(model, _fix_ids(ws, fixes[i]), rows[i]["pos"]))) != rows[i]["target"]]
            if not remaining:
                break
            if n + 1 < len(order):
                log(f"{len(remaining)} token fix(es) not reached at layer {L}; adding an edit in layer {order[n + 1]} for them")
        _final_rows(ws, model, tok, fixes, rows)
    finally:
        for s in reversed(applied):
            _restore(s)
    return _package(ws, model, fixes, rows, patches, steps, check, check_traces, which, margin, mode, rank, strength, part,
                    L0, t0, log)


def _final_rows(ws, model, tok, fixes, rows):
    """Every fix, with every layer's edit in place: what the model writes there now."""
    for r, f in zip(rows, fixes):
        lg = _logits_at(model, _fix_ids(ws, f), r["pos"])
        p = _probs(lg)
        r.update(now=int(lg.argmax()), now_text=tok.decode([int(lg.argmax())]), now_p=float(p.max()),
                 target_p_after=float(p[r["target"]]), fixed=int(lg.argmax()) == r["target"])


def _package(ws, model, fixes, rows, patches, steps, check, check_traces, which, margin, mode, rank, strength, part, L0,
             t0, log, extra=None):
    """Side-effect checks on other chats, then the solution record (kept in memory until saved)."""
    fix_tr = [t for t in dict.fromkeys(f.get("trace") for f in fixes) if t]
    others = steps[0]["others"]
    checks_on = [t for t in (check if check is not None else others[:int(check_traces)]) if t not in fix_tr]
    if max(s["size"] for s in steps) > BIG_EDIT_PCT and len(checks_on) > 3:
        checks_on = checks_on[:3]                     # far too big to apply: 3 chats are enough to show the damage
        log("The edit is too big to apply safely: checking its side effects on 3 chats only")
    checks = []
    for tid in checks_on:
        tc.check_stop()
        ids = tc._open_trace(ws, tid)[0]["ids"]
        lp0, top0 = _own_logprobs(model, ids)
        saved = _apply(model, patches)
        try:
            lp1, top1 = _own_logprobs(model, ids)
        finally:
            _restore(saved)
        checks.append({"trace": tid, "positions": len(lp0), "top_changed": int((top0 != top1).sum()),
                       "dlogp": float((lp1 - lp0).mean())})
        log(f"  check {tid}: top token changed at {checks[-1]['top_changed']}/{len(lp0)} positions")
    npos = sum(c["positions"] for c in checks)
    n_numbers = int(sum(p["delta"].size for p in patches))
    sid = uuid.uuid4().hex[:10]
    n_model = int(core.load_info(core.workspace_dir(ws)).get("n_parameters") or 0) or None
    kinds = {s["layer"]: s["kind"] for s in steps}
    res = {"sid": sid, "part": part, "tensor": steps[-1]["tensor"], "tensors": list(dict.fromkeys(p["tensor"] for p in patches)),
           "kind": steps[0]["kind"], "layer": L0, "layers": [s["layer"] for s in steps],
           "experts": [p["expert"] for p in patches if p["expert"] >= 0],
           "columns": sorted({int(c) for p in patches for c in p["cols"]}), "n_numbers": n_numbers,
           "pct_of_model": (100 * n_numbers / n_model) if n_model else None,
           "change_pct_of_tensor": max(s["size"] for s in steps), "fit_error": max(s["fit"] for s in steps),
           "steps": [{k: v for k, v in s.items() if k != "others"} for s in steps], "layer_kinds": kinds,
           "margin": margin, "mode": mode, "rank": int(rank) if mode == "general" else None,
           "strength": float(strength) if mode == "general" else None,
           "weights": which, "fixes": rows, "fixed": sum(r["fixed"] for r in rows),
           "checks": checks, "check_positions": npos,
           "check_top_changed_pct": (100 * sum(c["top_changed"] for c in checks) / npos) if npos else None,
           "check_dlogp": (sum(c["dlogp"] * c["positions"] for c in checks) / npos) if npos else None,
           "kept_positions": int(sum(s["kept"] for s in steps)), "seconds": round(time.time() - t0, 1), **(extra or {})}
    big = res["change_pct_of_tensor"] > BIG_EDIT_PCT
    res["too_big"] = big
    if big:
        tips = ["the general mode (one small shared correction)" if mode == "exact" else "a lower rank or a higher keep strength",
                "fewer answers at a time", "a smaller margin", "another layer"]
        res["warning"] = (f"This edit changes {res['tensor']} by {res['change_pct_of_tensor']:.0f}% of its size, which will very "
                          f"likely change many other answers. Try {', '.join(tips[:-1])} or {tips[-1]}.")
        log("Warning: " + res["warning"])
    _SOL[sid] = {"res": res, "patches": patches}
    while len(_SOL) > 8:
        _SOL.pop(next(iter(_SOL)))
    log(f"Fixed {res['fixed']}/{len(rows)} token(s) using layer(s) {', '.join(map(str, res['layers']))}"
        + (f"; other chats: top token changed at {res['check_top_changed_pct']:.1f}% of {npos} positions" if npos else ""))
    return res


# ------------------------------------------------------------------ one correction spread over several layers
def _resid_rows(model, tg, parts, DM, L):
    """_resid for many changes at once: residual after layer L for each row of DM [n, D]."""
    DM = np.asarray(DM, np.float64)
    x = parts["x"][0].astype(np.float64)
    rs = float(model.res_scale)
    post = lambda Y: model._post(np.asarray(Y, np.float32), tg["post"])[0].astype(np.float64)
    if tg["part"] == "attn":
        a_new = rs * post(parts["raw"][None] + DM) if tg["post"] else parts["a_out"][None] + rs * DM
        if model.parallel:
            return x[None] - parts["a_out"][None] + a_new
        return _rest_of_layer(model, L, parts["x_in"][None] + a_new)
    if tg["post"]:
        return (x - parts["m_out"])[None] + rs * post(parts["raw"][None] + DM)
    return x[None] + rs * DM


def _toward(model, tg, parts, L, delta, iters=8):
    """The change dm of this part's raw output that moves the residual after layer L by `delta` (as nearly as it
    can): exact for a plain residual add; through the output norm / this layer's MLP by Gauss–Newton steps with the
    finite-difference Jacobian (all D directions at once)."""
    rs = float(model.res_scale)
    if not tg["post"] and (tg["part"] == "mlp" or model.parallel):
        return np.asarray(delta, np.float64) / rs                     # the residual moves by exactly rs·dm
    x0 = _resid_rows(model, tg, parts, np.zeros((1, model.D)), L)[0]
    base = parts["raw"] if (tg["post"] or tg["part"] == "attn") else parts["x"][0]
    err = lambda d: float(np.linalg.norm(delta - (_resid_rows(model, tg, parts, d[None], L)[0] - x0)))
    dm, e = np.zeros(model.D), float(np.linalg.norm(delta))
    for _ in range(iters):
        if e <= 1e-4 * (np.linalg.norm(delta) + 1e-12):
            break
        miss = delta - (_resid_rows(model, tg, parts, dm[None], L)[0] - x0)
        h = 1e-2 * (float(np.sqrt(np.mean(np.asarray(base, np.float64) ** 2))) + 1e-6)
        J = np.zeros((model.D, model.D))
        for i0 in range(0, model.D, 1024):
            n = min(1024, model.D - i0)
            E = np.zeros((n, model.D))
            E[np.arange(n), i0 + np.arange(n)] = h
            J[:, i0:i0 + n] = ((_resid_rows(model, tg, parts, dm[None] + E, L) - _resid_rows(model, tg, parts, dm[None] - E, L)) / (2 * h)).T
        # an output norm ignores the length of its input: J has a direction that does (almost) nothing, which finite
        # differences blur into a tiny singular value; cut those directions instead of stepping hugely along them
        step = np.linalg.lstsq(J, miss, rcond=1e-3)[0]
        a = 1.0
        while a > 1e-3:                                # take the step only as far as it really lowers the miss
            e_new = err(dm + a * step)
            if e_new < e:
                dm, e = dm + a * step, e_new
                break
            a /= 2
        else:
            break
    return dm


def _top_residual(model, st, x, pos, L, L_top):
    """Residual after layer L_top at pos, from the residual after layer L (on a copy of the state)."""
    if L >= L_top:
        return np.asarray(x, np.float64).reshape(-1)
    st, rec, p = copy.deepcopy(st), tc.NullRecorder(), np.array([pos])
    x = np.asarray(x, np.float32).reshape(1, -1)
    for l in range(L + 1, L_top + 1):
        x = model.block(l, x, p, st, rec)[0].astype(np.float32, copy=False)
    return x[0].astype(np.float64)


def solve_spread(ws, fixes, spread, neurons=0, margin=1.0, keep_traces=40, check_traces=8, which="edited", keep=None,
                 check=None, protect=None, mode="exact", rank=4, strength=0.5, exclude=(), log=print):
    """One correction shared by several parts (MEMIT-style). spread: [(part "mlp" | "attn", layer, weight)], the
    weights from how much each part decided the mistakes.
      1. goal: with the real model, where the residual after the last part's layer must be for every fix
         (the change that makes the right token win by `margin` there)
      2. the parts in model order: each takes its share of what is still missing,
         share = weight / (its weight + the weights of the parts after it), moving the residual after its own layer
         by share × (target − where the edits so far got it); solved by least squares in that layer as usual
         (active-on-wrong neurons change most, neurons other answers use least, protected answers exactly 0)
      3. after each part the model is re-run, so the next part only covers what is still missing; the last part
         takes the rest, with the exact goal (the right token wins by the margin)
    Returns the same solution record as solve(), plus "spread": each part's weight, share and size."""
    if mode not in ("exact", "general"):
        raise ValueError("mode must be exact or general")
    if not fixes:
        raise ValueError("Add at least one fix")
    if len(fixes) > MAX_FIXES:
        raise ValueError(f"At most {MAX_FIXES} token fixes at once")
    t0 = time.time()
    model, tok = tc.get_model(ws, which, log)
    _forget_prefix()
    order = {"attn": 0, "mlp": 1}
    spread = sorted(((str(p), int(L), max(0.0, float(w))) for p, L, w in spread), key=lambda x: (x[1], order[x[0]]))
    spread = [x for x in spread if x[2] > 0] or [(p, L, 1.0) for p, L, _w in spread]
    if not spread:
        raise ValueError("No parts to spread the edit over")
    for p_, L, _w in spread:
        edit_target(model, L, p_)                      # refuses parts that can't be edited, with the reason
    if len(spread) == 1:
        p_, L, _w = spread[0]
        return solve(ws, fixes, layer=L, neurons=neurons, margin=margin, keep_traces=keep_traces, check_traces=check_traces,
                     which=which, keep=keep, check=check, protect=protect, chain=0, mode=mode, rank=rank, strength=strength,
                     part=p_, exclude=exclude, log=log)
    margin = float(margin)
    p_top, L_top, _w = spread[-1]
    tg_top = edit_target(model, L_top, p_top)
    names = lambda p_, L: f"{'attention' if p_ == 'attn' else 'MLP'} L{L}"
    log(f"Spreading one correction over {len(spread)} parts: " + ", ".join(f"{names(p_, L)} ({w:.2f})" for p_, L, w in spread))
    # 1. the target residual after the top layer, for every fix
    targets, infos = [], []
    for i, f in enumerate(fixes):
        tc.check_stop()
        ids, pos, t = _fix_ids(ws, f), int(f["pos"]), int(f["target"])
        st, parts = _position_parts(model, tg_top, ids, pos, L_top)
        dm, gi = goal(model, tg_top, st, parts, ids, pos, L_top, t, margin, log)
        targets.append(_resid(model, tg_top, parts, dm, L_top))
        infos.append(gi)
        if (i + 1) % 20 == 0 or i + 1 == len(fixes):
            log(f"  goals at {names(p_top, L_top)}: {i + 1}/{len(fixes)}")
    rows = [None] * len(fixes)
    patches, steps, applied, shares = [], [], [], []
    try:
        for j, (p_, L, w) in enumerate(spread):
            last = j == len(spread) - 1
            share = w / sum(x[2] for x in spread[j:])
            log(f"{names(p_, L)}: takes {share:.0%} of what is still missing" if not last else
                f"{names(p_, L)}: takes the rest (the right token must win by {margin})")
            moved = []

            def goal_fn(i, f, tg, st, parts, L=L, share=share, last=last, moved=moved):
                ids, pos, t = _fix_ids(ws, f), int(f["pos"]), int(f["target"])
                if last:
                    return goal(model, tg, st, parts, ids, pos, L, t, margin, log)
                now = _top_residual(model, st, parts["x"][0], pos, L, L_top)
                delta = share * (targets[i] - now)
                moved.append(float(np.linalg.norm(targets[i] - now)))
                return _toward(model, tg, parts, L, delta), infos[i]

            ps, rws, info = _solve_layer(ws, model, tok, fixes, L, neurons, margin, which, keep, keep_traces, protect, log,
                                         mode, rank, strength, p_, goal_fn=goal_fn, exclude=exclude)
            for i, r in enumerate(rws):
                rows[i] = rows[i] or r
            applied.append(_apply(model, ps))
            patches += ps
            steps.append(info)
            shares.append({"part": p_, "layer": L, "weight": w, "share": share, "size": info["size"], "tensor": info["tensor"],
                           "missing_before": float(np.mean(moved)) if moved else None})
        _final_rows(ws, model, tok, fixes, rows)
    finally:
        for s_ in reversed(applied):
            _restore(s_)
    wsum = sum(x[2] for x in spread)
    for x in shares:
        x["weight_pct"] = 100 * x["weight"] / wsum
    return _package(ws, model, fixes, rows, patches, steps, check, check_traces, which, margin, mode, rank, strength, p_top,
                    L_top, t0, log, extra={"spread": shares})


def save(ws, sid, name, force=False):
    if sid not in _SOL:
        raise ValueError("That result is gone (the app restarted?). Solve again.")
    key = _SOL[sid]["res"].get("weights_key")
    if key and key != [str(x) for x in tc._ws_stamp(core.workspace_dir(ws))]:
        raise ValueError("The workspace weights changed after this edit was computed (a weight was edited, or another "
                         "edit was applied or removed), so it no longer fits them. Compute it again.")
    if _SOL[sid]["res"].get("too_big") and not force:
        raise ValueError(_SOL[sid]["res"]["warning"] + " To apply it anyway, choose Apply anyway.")
    if not re.fullmatch(r"[\w-]{1,60}", name or ""):
        raise ValueError("Name: letters, digits, - and _ only")
    d = computed_dir(ws)
    d.mkdir(exist_ok=True)
    if (d / f"{name}.npz").exists():
        raise ValueError(f"A computed edit named {name!r} already exists")
    s = _SOL[sid]
    arrs = {"n": np.array(len(s["patches"]))}
    for i, p in enumerate(s["patches"]):
        arrs.update({f"tensor{i}": np.array(p["tensor"]), f"expert{i}": np.array(p["expert"]),
                     f"cols{i}": p["cols"], f"delta{i}": p["delta"]})
    np.savez(d / f"{name}.npz", **arrs)
    meta = {**s["res"], "name": name, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.pop("sid", None)
    (d / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return list_saved(ws)


def list_saved(ws):
    d = computed_dir(ws)
    out = []
    for p in sorted(d.glob("*.npz")) if d.is_dir() else []:
        j = p.with_suffix(".json")
        m = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
        out.append({"name": p.stem, "tensor": m.get("tensor"), "tensors": m.get("tensors") or [m.get("tensor")],
                    "layer": m.get("layer"), "n_numbers": m.get("n_numbers"),
                    "experts": m.get("experts", []), "fixed": m.get("fixed"), "n_fixes": len(m.get("fixes", [])),
                    "saved": m.get("saved"),
                    "fixes": [{k: f.get(k) for k in ("was_text", "target_text", "trace", "pos")} for f in m.get("fixes", [])]})
    return out


def delete(ws, name):
    if not re.fullmatch(r"[\w-]{1,60}", name or ""):
        raise ValueError("Bad name")
    d = computed_dir(ws)
    for ext in (".npz", ".json"):
        (d / f"{name}{ext}").unlink(missing_ok=True)
    return list_saved(ws)
