"""
Which layers really decided a token: total effects by switching parts off (causal), next to the direct push the
trace shows.

direct push        what a layer's attention / MLP adds straight to the token's score (Chat & trace's "push"). It
                   always favours the last layers: their output reaches the score directly, earlier layers work
                   through the later ones and get no credit for that.
total effect       switch the layer's attention or MLP off (its output replaced by its average over the text, at every
                   position) and run the rest of the model: how much the token's lead over a contrast token drops.
                   Counts everything that flows through the later layers.
contrastive        measured on (score of the token − score of the contrast token), e.g. the wrong answer vs the right
                   one, or the chosen token vs the runner-up. Layers every token depends on (the first layers, which
                   only read the text) hurt both scores alike and drop out, so what is left is what decided THIS choice.
flip layer         logit lens: the first layer after which the residual already prefers the token over the contrast,
                   and keeps preferring it.
"""
import time

import numpy as np

import trace_core as tc

CHUNK = 64


class _Hooks:
    """Wraps a model's mixer / ffn: records their raw outputs, or replaces one layer's output by a fixed vector."""
    def __init__(self, model):
        self.m, self.mixer, self.ffn = model, model.mixer, model.ffn
        self.rec = None
        self.off = None                          # (kind, layer, vector)

    def __enter__(self):
        m = self.m

        def mixer(L, h, pos, st, rec):
            a, hv = self.mixer(L, h, pos, st, rec)
            if self.rec is not None:
                self.rec[("attn", L)].append(np.asarray(a, np.float64))
            if self.off and self.off[0] == "attn" and self.off[1] == L:
                a = np.broadcast_to(self.off[2], a.shape).astype(np.float32).copy()
            return a, hv

        def ffn(L, h, rec=None):
            act, out = self.ffn(L, h, rec)
            if self.rec is not None:
                self.rec[("mlp", L)].append(np.asarray(out, np.float64))
            if self.off and self.off[0] == "mlp" and self.off[1] == L:
                out = np.broadcast_to(self.off[2], out.shape).astype(np.float32).copy()
            return act, out

        m.mixer, m.ffn = mixer, ffn
        return self

    def __exit__(self, *a):
        self.m.mixer, self.m.ffn = self.mixer, self.ffn


def _norm(text):
    """A token's text without spaces, case, punctuation and markup: ' Paris', 'Paris', '**Paris' are the same answer."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def best_contrast(lg, target, tok=None):
    """The strongest competitor that is a DIFFERENT answer: tokens that are only a spacing / case / markup variant of
    the target (or pure markup) are skipped, unless the target itself is markup."""
    order = np.argsort(-lg)
    tn = _norm(tok.decode([target])) if tok is not None else None
    for c in order[:200]:
        c = int(c)
        if c == target:
            continue
        if tok is None or not tn:
            return c
        cn = _norm(tok.decode([c]))
        if cn and cn != tn and not cn.startswith(tn) and not tn.startswith(cn):
            return c
    return int(order[0] if order[0] != target else order[1])


def _score(model, x_last, t, c):
    """(score of t − score of c, log-probability of t, top token) from the final residual at one position."""
    lg = model.logits(model.final(x_last[None]))[0].astype(np.float64)
    lse = lg.max() + np.log(np.exp(lg - lg.max()).sum())
    return float(lg[t] - lg[c]), float(lg[t] - lse), int(lg.argmax())


def layer_effects(model, ids, pos, target, contrast=None, log=print, baseline="mean", tok=None):
    """Effects of every layer's attention and MLP on the choice of `target` (over `contrast`, default: the strongest
    competitor that is a different answer, see best_contrast)
    at position `pos` (reading ids[pos], predicting the next token). Returns per-layer lists.
    baseline="mean": a switched-off part outputs its average over the text; "zero": it outputs nothing."""
    if not model.attribution_exact:
        raise ValueError("This architecture merges several residual streams (Gemma 3n); total effects aren't supported yet")
    ids = [int(i) for i in ids[:pos + 1]]
    nL = model.n_layer
    t0 = time.time()
    hk = _Hooks(model)
    with hk:
        # 1. normal pass: every layer's input (to restart from it) and raw outputs (for their average)
        hk.rec = {(k, L): [] for k in ("attn", "mlp") for L in range(nL)}
        xin = {L: [] for L in range(nL + 1)}
        st, rec = tc.State(nL), tc.NullRecorder()
        for s0 in range(0, len(ids), CHUNK):
            x = model.embed(ids[s0:s0 + CHUNK])
            p = np.arange(s0, s0 + len(x))
            for L in range(nL):
                xin[L].append(x)
                x = model.block(L, x, p, st, rec)[0].astype(np.float32, copy=False)
            xin[nL].append(x)
        xin = {L: np.concatenate(v) for L, v in xin.items()}
        recorded, hk.rec = hk.rec, None
        lg = model.logits(model.final(xin[nL][pos:pos + 1]))[0]
        if contrast is None:
            contrast = best_contrast(np.asarray(lg, np.float64), int(target), tok)
        base_diff, base_lp, base_top = _score(model, xin[nL][pos], target, contrast)
        means = {k: np.concatenate(v).mean(0) for k, v in recorded.items() if v}
        # 2. switch each part off, rerun from its layer
        eff = {"attn": [None] * nL, "mlp": [None] * nL}
        dlp = {"attn": [None] * nL, "mlp": [None] * nL}
        flips = {"attn": [False] * nL, "mlp": [False] * nL}
        total = sum(1 for k in ("attn", "mlp") for L in range(nL) if (k, L) in means)
        done = 0
        for kind in ("attn", "mlp"):
            for L in range(nL):
                if (kind, L) not in means:
                    continue
                tc.check_stop()
                done += 1
                if done == 1 or done % 6 == 0 or done == total:
                    log(f"Switching off {'attention' if kind == 'attn' else 'MLP'} of layer {L} ({done}/{total}, {time.time() - t0:.0f}s)")
                vec = means[(kind, L)] if baseline == "mean" else np.zeros_like(means[(kind, L)])
                hk.off = (kind, L, vec.astype(np.float32))
                st2, rec2 = tc.State(nL), tc.NullRecorder()
                last = None
                for s0 in range(0, len(ids), CHUNK):
                    x = xin[L][s0:s0 + CHUNK]
                    p = np.arange(s0, s0 + len(x))
                    for l in range(L, nL):
                        x = model.block(l, x, p, st2, rec2)[0].astype(np.float32, copy=False)
                    if s0 <= pos < s0 + len(x):
                        last = x[pos - s0]
                hk.off = None
                d, lp, top = _score(model, last, target, contrast)
                eff[kind][L] = base_diff - d
                dlp[kind][L] = base_lp - lp
                flips[kind][L] = top != base_top
    # 3. logit lens on the lead: what the residual after each layer prefers
    lens = []
    for L in range(nL + 1):
        lens.append(_score(model, xin[L][pos], target, contrast)[0])
    flip = None
    for L in range(nL + 1):
        if all(v > 0 for v in lens[L:]):
            flip = L - 1                                     # -1 = already from the embedding
            break
    log(f"Total effects: {sum(v is not None for v in eff['attn'] + eff['mlp'])} parts switched off in {time.time() - t0:.0f}s")
    return {"target": int(target), "contrast": int(contrast), "pos": int(pos), "base_lead": base_diff,
            "base_logp": base_lp, "effect": eff, "logp_drop": dlp, "top_changes": flips,
            "lens_lead": lens, "flip_after_layer": flip, "n_layer": nL, "seconds": round(time.time() - t0, 1)}


def top_parts(e, n=5):
    """The parts with the biggest contrastive total effect: [(effect, kind, layer)]."""
    parts = [(v, k, L) for k in ("attn", "mlp") for L, v in enumerate(e["effect"][k]) if v is not None]
    return sorted(parts, key=lambda x: -x[0])[:n]


def trace_effects(ws, tid, pos, target=None, contrast=None, log=print):
    """layer_effects for a saved trace position (weights the trace was recorded with)."""
    info, A = tc._open_trace(ws, tid)
    ids = info["ids"]
    if not 0 <= pos < len(ids) - 1:
        raise ValueError(f"Position must be between 0 and {len(ids) - 2}")
    model, tok = tc.get_model(ws, info["weights"], log)
    t = int(ids[pos + 1] if target is None else target)
    e = layer_effects(model, ids, pos, t, contrast, log, tok=tok)
    e["target_text"], e["contrast_text"] = tok.decode([e["target"]]), tok.decode([e["contrast"]])
    _i, A = tc._open_trace(ws, tid)
    att = tc.attribute(model, A, pos, e["target"])
    att_c = tc.attribute(model, A, pos, e["contrast"])
    e["direct"] = {"attn": [float(a - b) for a, b in zip(att["layer_attn"], att_c["layer_attn"])],
                   "mlp": [float(a - b) for a, b in zip(att["layer_mlp"], att_c["layer_mlp"])]}
    e["kinds"] = model.kinds() if hasattr(model, "kinds") else None
    return e


def _pair_scores(model, X, T, Cc):
    """score(T) − score(C) at many positions at once, from the final residuals X [n, D] (only two output rows each)."""
    xf = model.final(X.astype(np.float32)).astype(np.float64)
    U = model.U
    yt = np.einsum("nd,nd->n", xf, U[T].astype(np.float64))
    yc = np.einsum("nd,nd->n", xf, U[Cc].astype(np.float64))
    if model.out_bias is not None:
        yt, yc = yt + model.out_bias[T], yc + model.out_bias[Cc]
    if model.logit_scale != 1.0:
        yt, yc = yt / model.logit_scale, yc / model.logit_scale
    if model.softcap_final:
        c = float(model.softcap_final)
        yt, yc = np.tanh(yt / c) * c, np.tanh(yc / c) * c
    return yt - yc


def reply_effects(model, ids, positions, tok=None, log=print):
    """Total effects of every layer's attention and MLP on the tokens written at `positions` (each against its
    strongest different competitor), all from ONE switched-off re-run per part over the whole text: a token only
    depends on the text before it, so switching a part off everywhere and re-running once gives the drop at every
    position. Returns per-part mean effect, per-position effects and the top part of each position."""
    if not model.attribution_exact:
        raise ValueError("This architecture merges several residual streams (Gemma 3n); total effects aren't supported yet")
    ids = [int(i) for i in ids]
    positions = [int(p) for p in positions if 0 <= int(p) < len(ids) - 1]
    if not positions:
        raise ValueError("No reply positions to measure")
    last = max(positions)
    full = ids                                         # the tokens that were written (targets)
    ids = ids[:last + 1]                               # what has to be read to write them
    nL = model.n_layer
    t0 = time.time()
    hk = _Hooks(model)
    P = np.array(positions)
    with hk:
        hk.rec = {(k, L): [] for k in ("attn", "mlp") for L in range(nL)}
        xin = {L: [] for L in range(nL + 1)}
        st, rec = tc.State(nL), tc.NullRecorder()
        for s0 in range(0, len(ids), CHUNK):
            x = model.embed(ids[s0:s0 + CHUNK])
            p = np.arange(s0, s0 + len(x))
            for L in range(nL):
                xin[L].append(x)
                x = model.block(L, x, p, st, rec)[0].astype(np.float32, copy=False)
            xin[nL].append(x)
        xin = {L: np.concatenate(v) for L, v in xin.items()}
        recorded, hk.rec = hk.rec, None
        means = {k: np.concatenate(v).mean(0) for k, v in recorded.items() if v}
        T = np.array([full[p + 1] for p in positions])
        lg = model.logits(model.final(xin[nL][P])).astype(np.float64)            # once: to choose each competitor
        Cc = np.array([best_contrast(lg[n], int(T[n]), tok) for n in range(len(P))])
        base = _pair_scores(model, xin[nL][P], T, Cc)
        eff = {"attn": [None] * nL, "mlp": [None] * nL}
        total = sum(1 for k in ("attn", "mlp") for L in range(nL) if (k, L) in means)
        done = 0
        for kind in ("attn", "mlp"):
            for L in range(nL):
                if (kind, L) not in means:
                    continue
                tc.check_stop()
                done += 1
                if done == 1 or done % 6 == 0 or done == total:
                    log(f"Switching off {'attention' if kind == 'attn' else 'MLP'} of layer {L} ({done}/{total}, {time.time() - t0:.0f}s)")
                hk.off = (kind, L, means[(kind, L)].astype(np.float32))
                st2, rec2, outs = tc.State(nL), tc.NullRecorder(), []
                for s0 in range(0, len(ids), CHUNK):
                    x = xin[L][s0:s0 + CHUNK]
                    p = np.arange(s0, s0 + len(x))
                    for l in range(L, nL):
                        x = model.block(l, x, p, st2, rec2)[0].astype(np.float32, copy=False)
                    outs.append(x)
                hk.off = None
                eff[kind][L] = base - _pair_scores(model, np.concatenate(outs)[P], T, Cc)
    per_pos, mean = [], {"attn": [None] * nL, "mlp": [None] * nL}
    for k in ("attn", "mlp"):
        for L in range(nL):
            if eff[k][L] is not None:
                mean[k][L] = float(np.mean(eff[k][L]))
    for n, p in enumerate(positions):
        parts = [(float(eff[k][L][n]), k, L) for k in ("attn", "mlp") for L in range(nL) if eff[k][L] is not None]
        v, k, L = max(parts)
        per_pos.append({"pos": p, "token": int(T[n]), "contrast": int(Cc[n]), "lead": float(base[n]), "top": [k, L, v]})
    log(f"Total effects over {len(positions)} position(s): {done} parts switched off in {time.time() - t0:.0f}s")
    return {"positions": positions, "mean": mean, "per_position": per_pos, "n_layer": nL, "seconds": round(time.time() - t0, 1),
            "_eff": {k: [None if v is None else [float(x) for x in v] for v in eff[k]] for k in ("attn", "mlp")}}


def trace_reply_effects(ws, tid, log=print, only_content=True):
    """reply_effects for every reply token of a saved trace (skipping pure markup / spacing tokens)."""
    info, _A = tc._open_trace(ws, tid)
    model, tok = tc.get_model(ws, info["weights"], log)
    ids, npr = info["ids"], info["n_prompt"]
    pos = [p for p in range(npr - 1, len(ids) - 1) if not tok.is_stop(ids[p + 1])
           and (not only_content or _norm(tok.decode([ids[p + 1]])))]
    e = reply_effects(model, ids, pos, tok, log)
    for r in e["per_position"]:
        r["text"], r["contrast_text"] = tok.decode([r["token"]]), tok.decode([r["contrast"]])
    e["kinds"] = model.kinds() if hasattr(model, "kinds") else None
    cnt = {}
    for r in e["per_position"]:
        key = f"{r['top'][0]}:{r['top'][1]}"
        cnt[key] = cnt.get(key, 0) + 1
    e["top_counts"] = sorted(([k.split(":")[0], int(k.split(":")[1]), n] for k, n in cnt.items()), key=lambda x: -x[2])[:8]
    return e
