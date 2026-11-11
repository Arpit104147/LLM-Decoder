"""
Independent checks of the maths behind Fix answers and the held-out check:  python test_math.py
Each formula is checked against a different way of getting the same answer (optimality conditions, brute force,
finite differences, the real model, llama.cpp), not against itself.

A. least squares (no model)
   exact edit   ΔW = R (K A⁻¹ Kᵀ)⁻¹ K A⁻¹: every goal met exactly, protected keys exactly 0, and it is the smallest
                such change (Lagrange condition ΔW·A ∈ rows of K; random feasible alternatives are all bigger)
   general edit the best rank-r ΔW for  Σ|ΔW k − r|² + λ·tr(ΔW C ΔWᵀ):  equals the closed form at full rank,
                obeys Eckart–Young (cost = full-rank cost + dropped σ²), no random rank-r alternative is better
   McNemar      exact p-values against brute-force enumeration and known values
   LoRA form    B·A reproduces the edit to float precision, with the edit's true rank
B. on synthetic models (dense, output-norm, parallel, shared expert, routed experts, attention outputs)
   keys         changing the matrix by ΔW changes the layer's residual exactly as the solver predicts (_resid)
   direction    the goal direction is the true gradient of (right − wrong) score (finite differences, last layer)
   goal         the goal really makes the right token win by the margin
C. llama.cpp
   the edit as a LoRA adapter gives the same replies and probabilities as a model file with the edit written in,
   and switching it off gives the original model's
   check(): the held-out comparison asks the right questions, its "without" answers are the test's own, its
   "with" answers are the edited model's, and its counts add up
"""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core

tmp = Path(tempfile.mkdtemp(prefix="llmdec_math_"))
core.WORKSPACES = tmp / "workspaces"
import trace_core as tc
import fix_core as fc
import test_core as tst
import validate_core as vc
import test_trace_arch as arch

ok = fail = 0


def check(cond, msg):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


q = lambda m: None
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ A1. exact least squares
for trial, (D, ff, nfix, nprot, nkept) in enumerate([(6, 20, 3, 2, 200), (16, 64, 10, 6, 500), (8, 12, 4, 0, 50)]):
    K = rng.standard_normal((nfix + nprot, ff)) @ np.diag(rng.uniform(0.2, 2, ff))
    R = np.vstack([rng.standard_normal((nfix, D)), np.zeros((nprot, D))])
    Kk = rng.standard_normal((nkept, ff)) * rng.uniform(0.1, 3, ff)
    C = Kk.T @ Kk / nkept
    S, dW, fit = fc._fit(K, R, C, ff, nfix)
    dW = dW.astype(np.float64)
    eps = 1e-4 * float(np.mean(np.diag(C))) + 1e-12
    A = C + eps * np.eye(ff)
    meet = np.linalg.norm(dW @ K.T - R.T) / np.linalg.norm(R)
    prot = np.abs(dW @ K[nfix:].T).max() if nprot else 0.0
    # Lagrange: the gradient of tr(ΔW A ΔWᵀ) must lie in the span of the constraint keys: ΔW·A = Λ·K
    G = dW @ A
    P = K.T @ np.linalg.pinv(K.T)                                      # projector onto span of the keys
    kkt = np.linalg.norm(G - G @ P) / np.linalg.norm(G)
    cost = lambda X: float(np.trace(X @ A @ X.T))
    N = np.eye(ff) - P                                                 # moves that keep every constraint
    worse = all(cost(dW + rng.standard_normal((D, ff)) @ N * s) > cost(dW) for s in (1e-3, 1e-1, 1.0) for _ in range(20))
    check(meet < 1e-5 and prot < 1e-5 * np.abs(dW).max() and kkt < 1e-5 and worse,
          f"exact edit #{trial + 1}: goals met (error {meet:.1e}), protected keys unchanged ({prot:.1e}), "
          f"optimality condition holds ({kkt:.1e}), 60 other feasible edits all cost more")

# ------------------------------------------------------------------ A2. general (low-rank) least squares
for trial, (D, ff, nfix, nprot, rank, lam_s) in enumerate([(8, 24, 10, 4, 2, 1.0), (12, 30, 16, 0, 3, 0.3), (6, 10, 5, 3, 10, 2.0)]):
    K = rng.standard_normal((nfix + nprot, ff))
    R = np.vstack([rng.standard_normal((nfix, D)), np.zeros((nprot, D))])
    Kk = rng.standard_normal((300, ff)) * rng.uniform(0.2, 2, ff)
    C = Kk.T @ Kk / 300
    S, dW, fit = fc._fit_general(K, R, C, ff, nfix, rank, lam_s)
    dW = dW.astype(np.float64)
    w = np.ones(len(K))
    w[nfix:] = 3.0
    G = (K * w[:, None]).T @ K
    lam = lam_s * np.trace(G) / np.trace(C)
    M = G + lam * C + 1e-6 * np.trace(G) / ff * np.eye(ff)
    J = lambda X: float(np.sum(w * np.sum((X @ K.T - R.T) ** 2, 0)) + lam * np.trace(X @ C @ X.T)
                        + 1e-6 * np.trace(G) / ff * np.sum(X ** 2))
    full = (R.T * w) @ K @ np.linalg.inv(M)                           # unconstrained optimum (normal equations)
    B = (R.T * w) @ K @ np.real(np.linalg.inv(np.linalg.cholesky(M).T))
    sv = np.linalg.svd(B, compute_uv=False)
    eck = J(full) + float(np.sum(sv[rank:] ** 2))                      # Eckart–Young: what the best rank-r edit must cost
    r_ok = np.linalg.matrix_rank(dW, tol=1e-6 * np.abs(dW).max()) <= rank
    alts = []
    for _ in range(200):
        U_ = rng.standard_normal((D, rank))
        V_ = rng.standard_normal((rank, ff))
        X = dW + 0.05 * (U_ @ V_) / np.linalg.norm(U_ @ V_) * np.linalg.norm(dW)
        u, s_, vt = np.linalg.svd(X, full_matrices=False)
        alts.append(J((u[:, :rank] * s_[:rank]) @ vt[:rank]))          # a nearby rank-r edit
    check(abs(J(dW) - eck) <= 1e-6 * eck and r_ok and min(alts) >= J(dW) * (1 - 1e-9)
          and (rank < min(D, ff) or np.linalg.norm(dW - full) < 1e-6 * np.linalg.norm(full)),
          f"general edit #{trial + 1} (rank {rank}): cost {J(dW):.6f} = the Eckart–Young optimum {eck:.6f}, rank ≤ {rank}, "
          f"200 nearby rank-{rank} edits all cost more")

# ------------------------------------------------------------------ A3. McNemar and the verdict
from math import comb
brute = lambda b, c: min(1.0, sum(comb(b + c, k) for k in range(b + c + 1)
                                  if comb(b + c, k) <= comb(b + c, b) + 1e-9) / 2 ** (b + c)) if b + c else 1.0
pairs = [(10, 2), (2, 10), (0, 0), (5, 5), (12, 3), (30, 18), (1, 0), (9, 1)]
check(all(abs(vc.mcnemar_p(b, c) - brute(b, c)) < 1e-12 for b, c in pairs),
      "McNemar exact p = brute-force enumeration of all splits at least as uneven, for 8 cases")
check(abs(vc.mcnemar_p(10, 2) - 0.0385742) < 1e-6 and vc.mcnemar_p(5, 5) == 1.0 and vc.mcnemar_p(0, 0) == 1.0,
      "McNemar known values: 10 vs 2 -> p = 0.0386; 5 vs 5 -> 1; nothing changed -> 1")
v1, v2, v3 = vc.verdict(12, 2, 150), vc.verdict(12, 2, 150, tries=4), vc.verdict(2, 12, 150)
check(v1["verdict"] == "helps" and v2["verdict"] == "no clear effect" and v3["verdict"] == "hurts"
      and abs(v2["p_adjusted"] - 4 * v1["p"]) < 1e-12 and v1["ci_pct"][0] < v1["change_pct"] < v1["ci_pct"][1],
      f"verdicts: 12 fixed vs 2 broken helps (p {v1['p']:.4f}); the same after 4 tries doesn't (p×4 = {v2['p_adjusted']:.3f}); "
      "2 vs 12 hurts; the interval contains the change")

# ------------------------------------------------------------------ A4. LoRA factors


class _M:
    def __init__(self, W):
        self.T = {"w": W}


for true_r in (1, 3, 7):
    W = rng.standard_normal((20, 40)).astype(np.float32)
    cols = np.sort(rng.choice(40, 15, replace=False))
    delta = (rng.standard_normal((20, true_r)) @ rng.standard_normal((true_r, 15))).astype(np.float32)
    f = vc.lora_factors(_M(W), [{"tensor": "w", "expert": -1, "cols": cols, "delta": delta}])
    A_, B_, err = f["w"]
    full = np.zeros((20, 40))
    full[:, cols] = delta
    check(A_.shape == (true_r, 40) and B_.shape == (20, true_r) and err < 1e-6
          and np.abs(B_.astype(np.float64) @ A_ - full).max() < 1e-5 * np.abs(full).max(),
          f"LoRA form of a rank-{true_r} edit: rank {A_.shape[0]}, B·A = the edit (error {err:.1e})")
try:
    vc.lora_factors(_M(np.zeros((2, 4, 3), np.float32)), [{"tensor": "w", "expert": 1, "cols": np.arange(3), "delta": np.zeros((4, 3))}])
    check(False, "a routed-expert edit is refused as LoRA (checked with a temporary model instead)")
except vc.NotLoRA:
    check(True, "a routed-expert edit is refused as LoRA (checked with a temporary model instead)")


# ------------------------------------------------------------------ B. synthetic models
_WS = {}


def workspace(name, seed=4, calm=False):
    if (name, seed) in _WS:
        return _WS[(name, seed)]
    path = tmp / f"{name}_{seed}.gguf"
    arch.write_model(arch.FAMILIES[name], path, seed=seed)
    w = core.extract_workspace(path, log=q)
    if calm:            # random tied embeddings make a synthetic Gemma's scores 15-30 apart: scale them
        (core.WORKSPACES / w / "edit_weights.py").write_text('def edit(w):\n    w["token_embd.weight"] *= 0.1\n', encoding="utf-8")
    _WS[(name, seed)] = w
    return w


def model_checks(name, part, calm=False):
    w = workspace(name, calm=calm)
    m, tok = tc.get_model(w, "edited", q)
    layers = fc.editable_layers(m, part)
    ids = [int(x) for x in rng.integers(0, 150, 14)]
    pos = 11
    for L in sorted({layers[0], layers[-1]}):
        tg = fc.edit_target(m, L, part)
        st, parts = fc._position_parts(m, tg, ids, pos, L)
        g, k = fc._keys(m, tg, parts)[0]
        Wm = tg["W"][g] if g is not None else tg["W"]
        delta = rng.standard_normal(Wm.shape)
        delta *= 0.3 * np.linalg.norm(Wm) / np.linalg.norm(delta)
        dm = delta @ k
        predicted = fc._resid(m, tg, parts, dm, L)
        saved = fc._apply(m, [{"tensor": tg["tensor"], "expert": -1 if g is None else int(g), "cols": np.arange(Wm.shape[1]),
                               "delta": delta.astype(np.float32)}])
        try:
            st2 = fc._prefix_state(m, ids[:pos])
            actual = fc._upto(m, st2, ids[pos], pos, L)["x"][0].astype(np.float64)
        finally:
            fc._restore(saved)
        e = np.linalg.norm(predicted - actual) / (np.linalg.norm(actual - parts["x"][0]) + 1e-12)
        what = tg["kind"] + (" + output norm" if tg["post"] else "") + (" (parallel)" if m.parallel else "")
        check(e < 1e-3, f"{name} {part} L{L} [{what}]: a change ΔW moves the layer's residual exactly as predicted "
                        f"(error {e:.1e} of the move)")
    # direction = gradient (last editable layer of the model, where the direct path is the whole path)
    L = layers[-1]
    if L == m.n_layer - 1:
        tg = fc.edit_target(m, L, part)
        st, parts = fc._position_parts(m, tg, ids, pos, L)
        lg, xf = fc._finish(m, st, parts["x"], pos, L)
        order = np.argsort(-lg)
        j, t = int(order[0]), int(order[3])
        f_ = lambda d: (lambda l: l[t] - l[j])(fc._finish(m, st, fc._resid(m, tg, parts, d, L)[None].astype(np.float32), pos, L)[0])
        h = 3e-2 * (np.linalg.norm(parts["raw"] if tg["post"] or part == "attn" else parts["x"][0]) / np.sqrt(m.D))
        grad = np.array([(f_(h * e_) - f_(-h * e_)) / (2 * h) for e_ in np.eye(m.D)])
        dirn = fc._direction(m, tg, parts, t, j, xf)
        cos = float(grad @ dirn / (np.linalg.norm(grad) * np.linalg.norm(dirn)))
        # float32 rounding limits the finite-difference reference itself (Gemma's softcaps amplify it: two references
        # with different steps agree only to ~0.99 there), so softcapped models are held to 0.995
        if True:
            check(cos > (0.995 if m.softcap_final else 0.9999), f"{name} {part} L{L}: goal direction = the finite-difference gradient of (right − wrong) score (cos {cos:.6f})")
        dm, lg0, lg1 = fc._goal(m, tg, st, parts, pos, L, t, 1.0, q)
        real = fc._finish(m, st, fc._resid(m, tg, parts, dm, L)[None].astype(np.float32), pos, L)[0]
        mr, _j = fc._margin(real, t)
        check(mr >= 1.0 - 1e-4, f"{name} {part} L{L}: the goal makes the 4th-ranked token win by {mr:.2f} (≥ margin 1)")


for name, part, calm in (("llama", "mlp", False), ("gemma2", "mlp", True), ("command-r", "mlp", False), ("olmo2", "mlp", True),
                         ("qwen2moe", "mlp", False), ("qwen3moe", "mlp", False), ("llama", "attn", False),
                         ("gemma2", "attn", True), ("command-r", "attn", False)):
    try:
        model_checks(name, part, calm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, f"{name} {part}: {type(e).__name__}: {e}")


# ------------------------------------------------------------------ D. one correction spread over several layers
def spread_checks(name, calm=False):
    w = workspace(name, calm=calm)
    m, tok = tc.get_model(w, "edited", q)
    r2 = np.random.default_rng(11)
    fixes = []
    for _k in range(4):
        ids = [int(x) for x in r2.integers(0, 150, 12)]
        lg = fc._logits_at(m, ids, 10)
        t = int(np.argsort(-lg)[2])                                  # make the 3rd-ranked token win
        fixes.append({"ids": ids[:11] + [t], "pos": 10, "target": t})
    pids = [int(x) for x in r2.integers(0, 150, 12)]
    protect = [{"ids": pids, "pos": 9}]
    parts = [("mlp", 1, 1.0), ("attn", 2, 0.5), ("mlp", 3, 2.0)]
    res = fc.solve_spread(w, fixes, parts, mode="exact", protect=protect, margin=1.0, log=q)
    sp = res["spread"]
    want = [1.0 / 3.5, 0.5 / 2.5, 1.0]
    check(res["fixed"] == len(fixes) and [round(x["share"], 6) for x in sp] == [round(v, 6) for v in want]
          and [x["layer"] for x in sp] == [1, 2, 3] and len(res["tensors"]) == 3,
          f"{name} spread: one correction over MLP L1, attention L2, MLP L3 (weights 1 : 0.5 : 2 -> shares "
          f"{', '.join(f'{x['share']:.0%}' for x in sp)} of what is still missing); every fix reached ({res['fixed']}/{len(fixes)})")
    # protected answer: every part's own change is exactly 0 there (measured with the earlier parts' edits in place)
    pat = fc._SOL[res["sid"]]["patches"]
    worst, saved = 0.0, []
    try:
        for p_, L, _w in parts:
            tg = fc.edit_target(m, L, p_)
            pp = [x for x in pat if x["tensor"] == tg["tensor"]][0]
            _st, prt = fc._position_parts(m, tg, pids, 9, L)
            k = fc._keys(m, tg, prt)[0][1]
            out = pp["delta"].astype(np.float64) @ k[pp["cols"]]
            worst = max(worst, float(np.linalg.norm(out) / (np.linalg.norm(pp["delta"]) * np.linalg.norm(k[pp["cols"]]) + 1e-30)))
            saved.append(fc._apply(m, [pp]))
    finally:
        for x in reversed(saved):
            fc._restore(x)
    check(worst < 1e-5, f"{name} spread: the protected answer gets exactly no change from any part (worst {worst:.1e})")
    # each part's step: the residual after its layer moves by exactly the requested amount (when reachable)
    ids, pos = fixes[0]["ids"], 10
    errs = []
    for p_, L, _w in parts:
        tg = fc.edit_target(m, L, p_)
        _st, prt = fc._position_parts(m, tg, ids, pos, L)
        base = prt["raw"] if (tg["post"] or tg["part"] == "attn") else prt["x"][0]
        true = r2.standard_normal(m.D) * 0.1 * np.linalg.norm(base) / np.sqrt(m.D)
        x0 = fc._resid_rows(m, tg, prt, np.zeros((1, m.D)), L)[0]
        delta = fc._resid_rows(m, tg, prt, true[None], L)[0] - x0
        dm = fc._toward(m, tg, prt, L, delta)
        errs.append(float(np.linalg.norm(fc._resid_rows(m, tg, prt, dm[None], L)[0] - x0 - delta) / np.linalg.norm(delta)))
    # behind an output norm one direction (the input's length) does almost nothing: it is left out on purpose,
    # so those steps are exact to ~0.1 %, plain residual adds to float precision
    check(max(errs) < 5e-3, f"{name} spread: each part's step moves the residual by the share asked for "
                            f"(errors {', '.join(f'{e:.1e}' for e in errs)}; through output norms and the layer's MLP)")
    single = fc.solve(w, fixes, layer=3, part="mlp", chain=0, mode="exact", protect=protect, margin=1.0, log=q)
    check(res["change_pct_of_tensor"] < single["change_pct_of_tensor"],
          f"{name} spread: biggest change per matrix {res['change_pct_of_tensor']:.0f} % vs {single['change_pct_of_tensor']:.0f} % "
          f"when MLP L3 alone makes the same fixes")


for name, calm in (("llama", False), ("gemma2", True), ("command-r", False)):
    try:
        spread_checks(name, calm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, f"{name} spread: {type(e).__name__}: {e}")


# ------------------------------------------------------------------ C. llama.cpp: LoRA and the held-out check
def llama_checks():
    w = workspace("llama", seed=3)
    d = core.WORKSPACES / w
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    meta["tokenizer.chat_template"] = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                                       "<|im_start|>assistant\n")
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    m, tok = tc.get_model(w, "edited", q)
    L = m.n_layer - 2
    name = f"blk.{L}.ffn_down.weight"
    Wm = m.T[name]
    delta = (rng.standard_normal((Wm.shape[0], 2)) @ rng.standard_normal((2, Wm.shape[1]))).astype(np.float32)
    delta *= 2.0 * np.linalg.norm(Wm) / np.linalg.norm(delta)
    patch = [{"tensor": name, "expert": -1, "cols": np.arange(Wm.shape[1]), "delta": delta}]
    adapter = vc.write_lora(tmp / "a.gguf", meta["general.architecture"], vc.lora_factors(m, patch))
    npz = tmp / "p.npz"
    np.savez(npz, n=np.array(1), tensor0=np.array(name), expert0=np.array(-1), cols0=patch[0]["cols"], delta0=delta)
    edited_file = tmp / "edited.gguf"
    core.export_gguf(w, out_path=edited_file, log=q, exact=True, record=False, extra_computed=[npz])
    base_file = tst._model_file(w, "edited", q)
    prompts = [tc.prompt_ids(tok, tc.render_chat(meta, [{"role": "user", "content": f"say {x}"}])) for x in ("one", "two", "three", "four")]

    def ask(srv):
        out = []
        for p in prompts:
            r = srv.complete(p, 8) if not hasattr(srv, "probs") else None
            out.append(r["tokens"])
        return out

    def probs(srv, p):
        import urllib.request
        body = {"prompt": p, "n_predict": 1, "temperature": 0, "top_k": 1, "samplers": ["top_k"], "cache_prompt": False,
                "n_probs": 5}
        req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/completion", headers={"Content-Type": "application/json"},
                                     data=json.dumps(body).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return {int(x["id"]): x["logprob"] for x in r["completion_probabilities"][0]["top_logprobs"]}

    s1 = vc._server(base_file, 256, 1, q, [adapter])
    try:
        s1.set_adapters([0.0])
        off, p_off = ask(s1), probs(s1, prompts[0])
        s1.set_adapters([1.0])
        on, p_on = ask(s1), probs(s1, prompts[0])
        s1.set_adapters([0.0])
        off2 = ask(s1)
    finally:
        s1.close()
    s2 = vc._server(base_file, 256, 1, q)
    try:
        plain, p_plain = ask(s2), probs(s2, prompts[0])
    finally:
        s2.close()
    s3 = vc._server(edited_file, 256, 1, q)
    try:
        written, p_written = ask(s3), probs(s3, prompts[0])
    finally:
        s3.close()
    gap = lambda a, b: max(abs(a[k] - b[k]) for k in set(a) & set(b)) if set(a) & set(b) else 9.9
    check(off == plain and off2 == plain and gap(p_off, p_plain) < 1e-4,
          "LoRA switched off = the original model (same replies, probabilities within 1e-4), also after switching back")
    check(on == written and gap(p_on, p_written) < 2e-3 and on != plain,
          f"LoRA switched on = a model file with the edit written in (same replies, probabilities within "
          f"{gap(p_on, p_written):.1e}) and differs from the original")

    # check(): end to end on a small test
    text = "\n".join(f"say {x} | {a}" for x, a in zip("one two three four five six seven eight".split(),
                                                         ["toe", "hat", "42", "dot", "7", "sun", "ink", "9"]))
    r = tst.run(w, text, instruction="", max_tokens=6, which="edited", engine="llama", parallel=2, log=q)
    rr = tst.recommend(w, r["id"], split=0.5, mode="exact", measure=0, choose="cheapest", log=q)
    s = rr["solution"]
    out = vc.check(w, r["id"], [s["sid"]], parallel=2, train="all", log=q)
    c = out["candidates"][0]
    held, trn = s["split"]["held"], s["split"]["train"]
    items = {x["i"]: x for x in r["items"]}
    base_ok = [items[i]["correct"] for i in held]
    rows = {x["i"]: x for x in c["rows"]}
    h = c["held"]
    check(out["held"] == held == [4, 5, 6, 7] and sorted(out["asked"]) == sorted(held + trn),
          "check(): asks the 4 held-out questions (and, with train='all', the 4 it was computed from)")
    check([rows[i]["before"] for i in held] == base_ok,
          "check(): its answers without the edit are the test's own answers (same llama.cpp, same prompts)")
    check(h["after"] - h["before"] == h["gained"] - h["lost"] and h["before"] == sum(base_ok)
          and h["after"] == sum(rows[i]["after"] for i in held),
          f"check(): counts add up ({h['before']} -> {h['after']}: +{h['gained']} −{h['lost']})")
    # its "with" answers = the edited model's own replies
    def written_replies(sid):
        pts = fc._SOL[sid]["patches"]
        np.savez(npz, **{"n": np.array(len(pts)),
                         **{k: v for i, p in enumerate(pts)
                            for k, v in ((f"tensor{i}", np.array(p["tensor"])), (f"expert{i}", np.array(p["expert"])),
                                         (f"cols{i}", p["cols"]), (f"delta{i}", p["delta"]))}})
        core.export_gguf(w, out_path=edited_file, log=q, exact=True, record=False, extra_computed=[npz])
        s4 = vc._server(edited_file, 256, 1, q)
        try:
            out_ = []
            for i in held:
                p = tc.prompt_ids(tok, tc.render_chat(meta, [{"role": "user", "content": items[i]["question"]}]))
                gen = s4.complete(p, 6)["tokens"]
                out_.append(tok.decode([x for x in gen if not tok.is_stop(x)]))
            return out_
        finally:
            s4.close()

    replies = written_replies(s["sid"])
    check(all(rows[i]["reply_after"] == rp for i, rp in zip(held, replies)),
          "check(): its answers with the edit are exactly the replies of a model file with the edit written in")
    # the same for an edit spread over several layers (several tensors in one LoRA adapter)
    rs_ = tst.recommend(w, r["id"], split=0.5, spread="0-3", mode="exact", measure=2, log=q)["solution"]
    o3 = vc.check(w, r["id"], [rs_["sid"]], parallel=2, train="none", log=q)
    rows3 = {x["i"]: x for x in o3["candidates"][0]["rows"]}
    rep3 = written_replies(rs_["sid"])
    check(len(rs_["tensors"]) >= 2 and all(rows3[i]["reply_after"] == rp for i, rp in zip(held, rep3)),
          f"check() of a spread edit ({len(rs_['tensors'])} tensors in one LoRA adapter): its answers with the edit are "
          "exactly a model file's with the edit written in")
    hist = vc.list_checks(w, r["id"])
    check("validation" in fc._SOL[s["sid"]]["res"] and len(hist) >= 1 and hist[0]["candidates"][0]["sid"] == s["sid"],
          "check(): the result is kept with the edit (shown before applying, saved with it) and in the test's history")
    s0 = vc.search(w, r["id"], split=0.5, measure=2, parallel=2, train="wrong", log=q)
    check(s0.get("none_applicable") and not s0["candidates"] and s0["best"] is None
          and all("too big" in x["reason"] for x in s0["skipped"]),
          "search(): when every edit is too big to apply safely (random toy model), it checks none and says so")
    sr = vc.search(w, r["id"], split=0.5, measure=2, parallel=2, train="wrong", allow_big=True, log=q)
    nc = len(sr["candidates"])
    check(nc >= 1 and sr["tries"] == nc and all(c["held"]["tries"] == nc for c in sr["candidates"])
          and len(sr["tried"]) + len(sr["skipped"]) == len(sr["settings"])
          and all(abs(c["held"]["p_adjusted"] - min(1.0, c["held"]["p"] * nc)) < 1e-12 for c in sr["candidates"])
          and (sr["best"] is None or any(c["sid"] == sr["best"] and c["held"]["verdict"] == "helps" for c in sr["candidates"])),
          f"search(): {len(sr['settings'])} settings -> {nc} edit(s) checked on the held-out questions, p-values × {nc}, "
          f"'best' only when it really helps ({sr['best'] or 'none here'})")
    try:
        rr2 = tst.recommend(w, r["id"], split=None, mode="exact", measure=0, choose="cheapest", log=q)
        vc.check(w, r["id"], [rr2["solution"]["sid"]], log=q)
        check(False, "check() refuses an edit computed from all the questions (nothing held out)")
    except ValueError:
        check(True, "check() refuses an edit computed from all the questions (nothing held out)")


try:
    llama_checks()
except Exception as e:
    import traceback
    traceback.print_exc()
    check(False, f"llama.cpp checks: {type(e).__name__}: {e}")

tc.unload() if hasattr(tc, "unload") else None
core.release_sources()
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
