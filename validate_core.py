"""
Check a computed edit the way it will really be used: ask the questions it was NOT computed from (held out), once
without and once with the edit, grade both, and test whether the difference is real or could be chance.

  llama.cpp writes all the answers (greedy, the test's own prompt, instruction and token limit). The edit is
  loaded as a LoRA adapter: ΔW = B·A, the exact SVD of the edit, so llama.cpp computes W·x + B·(A·x), the same
  numbers as the edited matrix. One server holds the original and every candidate edit; they are switched on and
  off between batches. Edits to routed experts (which LoRA can't hold) are checked with a temporary exported model.

  Paired comparison on the same questions: gained = wrong -> right, lost = right -> wrong. McNemar's exact test:
  if the edit did nothing, gained and lost would be like coin flips, p = P(a split at least this uneven).
  When several candidate edits are tried, p is multiplied by their number (Bonferroni), so picking the best of
  several can't make chance look like an improvement.

  Verdict: "helps" (more gained than lost, p < 0.05), "hurts" (more lost, p < 0.05), else "no clear effect".
"""
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

import gguf_core as core
import trace_core as tc
import fix_core as fc
import test_core as tst

ALPHA = 0.05


# ------------------------------------------------------------------ statistics
def mcnemar_p(gained, lost):
    """Two-sided exact McNemar p-value: P(a split of the n = gained + lost changed answers at least this uneven, if
    each change were equally likely to go either way)."""
    n = int(gained) + int(lost)
    if n == 0:
        return 1.0
    k = min(int(gained), int(lost))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_ci(gained, lost, n, z=1.96):
    """95% interval for the change in accuracy (after − before), paired (Wald on the difference of proportions)."""
    if n == 0:
        return 0.0, 0.0
    d = (gained - lost) / n
    var = (gained + lost) / n - d * d
    se = math.sqrt(max(var, 0.0) / n)
    return d - z * se, d + z * se


def verdict(gained, lost, n, tries=1):
    p = mcnemar_p(gained, lost)
    p_adj = min(1.0, p * max(1, int(tries)))
    lo, hi = paired_ci(gained, lost, n)
    if gained > lost and p_adj < ALPHA:
        v = "helps"
    elif lost > gained and p_adj < ALPHA:
        v = "hurts"
    else:
        v = "no clear effect"
    return {"verdict": v, "p": p, "p_adjusted": p_adj, "tries": int(tries), "change_pct": 100 * (gained - lost) / n if n else 0.0,
            "ci_pct": [100 * lo, 100 * hi]}


def verdict_text(h):
    n, b, a = h["n"], h["before"], h["after"]
    s = (f"{a}/{n} right with the edit vs {b}/{n} without ({h['gained']} fixed, {h['lost']} broken; "
         f"p = {h['p_adjusted']:.3g}{' after correcting for ' + str(h['tries']) + ' tries' if h['tries'] > 1 else ''})")
    if h["verdict"] == "helps":
        return "Helps on questions it never saw: " + s
    if h["verdict"] == "hurts":
        return "Hurts on questions it never saw: " + s
    return ("No clear effect on questions it never saw: " + s + ". A difference this small could be chance"
            + (f"; with {n} held-out questions only a change of about {needed_change(n)} questions would show." if n else "."))


def needed_change(n, discordant_share=0.15):
    """Roughly how many more questions must be gained than lost to reach p < 0.05, when ~15% of answers change."""
    m = max(2, int(round(n * discordant_share)))
    for k in range(m + 1):
        if mcnemar_p((m + k) // 2 + (m + k) % 2, (m - k) // 2) < ALPHA:
            return k
    return m


# ------------------------------------------------------------------ the edit as a LoRA adapter
class NotLoRA(Exception):
    pass


def lora_factors(model, patches, tol=1e-7):
    """{tensor: (A [r, n_in], B [n_out, r], relative error)} with B·A = the summed edit of that tensor (exact SVD,
    singular values below tol × the largest dropped). Routed-expert patches raise NotLoRA."""
    full = {}
    for p in patches:
        if int(p["expert"]) >= 0:
            raise NotLoRA(p["tensor"])
        W = fc._patch_matrix(model, p)
        f = full.setdefault(p["tensor"], np.zeros(W.shape, np.float64))
        f[:, np.asarray(p["cols"])] += np.asarray(p["delta"], np.float64)
    out = {}
    for name, f in full.items():
        cols = np.flatnonzero(np.any(f != 0, axis=0))
        if not len(cols):
            continue
        U, s, Vt = np.linalg.svd(f[:, cols], full_matrices=False)
        r = int((s > tol * s[0]).sum())
        A = np.zeros((r, f.shape[1]), np.float32)
        A[:, cols] = Vt[:r]
        B = (U[:, :r] * s[:r]).astype(np.float32)
        err = float(np.linalg.norm(B.astype(np.float64) @ A.astype(np.float64) - f) / (np.linalg.norm(f) + 1e-30))
        out[name] = (A, B, err)
    return out


def write_lora(path, arch, factors):
    """A llama.cpp LoRA adapter file (alpha 0: the adapter's scale is used as is)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), arch)
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 0.0)
    for name, (A, B, _e) in factors.items():
        w.add_tensor(name + ".lora_a", np.ascontiguousarray(A))
        w.add_tensor(name + ".lora_b", np.ascontiguousarray(B))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


# ------------------------------------------------------------------ asking the questions
def _ask(srv, tok, stop, prompts, max_tokens, parallel, log, label):
    """Greedy replies for every prompt (in order): [(reply text, generated ids, cut off?)]."""
    from concurrent.futures import ThreadPoolExecutor
    out = [None] * len(prompts)
    t0 = time.time()
    done = 0

    ev = getattr(tc._JOB, "cancel", None)                 # the job's stop flag (the workers are other threads)

    def one(i):
        if ev is not None and ev.is_set():
            raise tc.Stopped("Stopped")
        r = srv.complete(prompts[i], int(max_tokens))
        gen = [int(x) for x in r.get("tokens", [])]
        if r.get("stop_type") == "eos" and stop is not None and (not gen or not tok.is_stop(gen[-1])):
            gen.append(stop)
        cut = r.get("stop_type") == "limit" or (len(gen) >= int(max_tokens) and not (gen and tok.is_stop(gen[-1])))
        return i, (tok.decode([x for x in gen if not tok.is_stop(x)]), gen, cut)

    with ThreadPoolExecutor(int(parallel)) as ex:
        for i, res in ex.map(one, range(len(prompts))):
            out[i] = res
            done += 1
            if done % 10 == 0 or done == len(prompts):
                log(f"  {label}: {done}/{len(prompts)} answered ({time.time() - t0:.0f}s)")
    return out


def _server(model_file, ctx, par, log, adapters=()):
    extra = []
    for a in adapters:
        extra += ["--lora", str(a)]
    if adapters:
        extra.append("--lora-init-without-apply")
    return tst._Llama(model_file, ctx, par, log, extra=extra)


def _summary(rows_before, rows_after, idx):
    b = sum(rows_before[i]["correct"] for i in idx)
    a = sum(rows_after[i]["correct"] for i in idx)
    g = sum((not rows_before[i]["correct"]) and rows_after[i]["correct"] for i in idx)
    l_ = sum(rows_before[i]["correct"] and not rows_after[i]["correct"] for i in idx)
    return {"n": len(idx), "before": b, "after": a, "gained": g, "lost": l_}


def check(ws, tid, sids, parallel=3, train="wrong", tries=None, log=print):
    """Ask the held-out questions of test `tid` without and with each unsaved edit in `sids` (all computed from the
    same split of that test), grade, and compare. train: also ask the questions the edit was computed from:
    "wrong" (did it fix them in real replies?), "all", or "none". Returns {"baseline", "candidates": [...]} and
    stores each result in its solution (shown before applying, and saved with the edit)."""
    r = tst.get_test(ws, tid)
    if not sids:
        raise ValueError("No edit to check")
    sols = []
    for sid in sids:
        if sid not in fc._SOL:
            raise ValueError("That computed edit isn't in memory any more (the app keeps the last 8 and forgets them on "
                             "restart): compute it again")
        sols.append(fc._SOL[sid])
    split = sols[0]["res"].get("split") or {}
    held = list(split.get("held") or [])
    trn = list(split.get("train") or [])
    for s in sols:
        if s["res"].get("test") != tid or (s["res"].get("split") or {}).get("held") != held:
            raise ValueError("All edits must be computed from the same test and split")
        if s["res"].get("weights_key") and s["res"]["weights_key"] != tst._stamp(ws):
            raise ValueError("The workspace weights changed after this edit was computed (an edit was applied or "
                             "removed): compute it again")
    if not held:
        raise ValueError("This edit was computed from all the questions, so none are held out to check it on. "
                         "Set 'Compute from' to 50–80 % and compute it again.")
    items = {x["i"]: x for x in r["items"]}
    wrong_t = [i for i in trn if not items[i]["correct"]]
    ask = held + (wrong_t if train == "wrong" else trn if train == "all" else [])
    meta = tst._meta(ws, r["weights"])
    tok = tc.make_tokenizer(meta)
    stop = tst._stop_id(tok)
    prompts = [tc.prompt_ids(tok, tc.render_chat(meta, [{"role": "user", "content": tst._prompt(items[i], r.get("instruction", ""))}]))
               for i in ask]
    mt = int(r.get("max_tokens", 160))
    ctx = max(map(len, prompts)) + mt + 16
    model, _t = tc.get_model(ws, r["weights"], log)
    cache = core.workspace_dir(ws) / "cache" / "checks"
    cache.mkdir(parents=True, exist_ok=True)
    lora, own = [], []                         # (sol index, adapter path) / (sol index, model file)
    for n, s in enumerate(sols):
        try:
            f = lora_factors(model, s["patches"])
            worst = max(e for _a, _b, e in f.values())
            if worst > 1e-5:
                raise NotLoRA(f"factorisation error {worst:.1e}")
            lora.append((n, write_lora(cache / f"{s['res']['sid']}.gguf", meta["general.architecture"], f)))
            log(f"Edit {n + 1}: as a LoRA adapter (rank {', '.join(str(a.shape[0]) for a, _b, _e in f.values())})")
        except NotLoRA:
            npz = cache / f"{s['res']['sid']}.npz"
            arrs = {"n": np.array(len(s["patches"]))}
            for i, p in enumerate(s["patches"]):
                arrs.update({f"tensor{i}": np.array(p["tensor"]), f"expert{i}": np.array(p["expert"]),
                             f"cols{i}": p["cols"], f"delta{i}": p["delta"]})
            np.savez(npz, **arrs)
            mf = cache / f"{s['res']['sid']}_model.gguf"
            log(f"Edit {n + 1}: edits routed experts (no LoRA form): writing a temporary model with it…")
            core.export_gguf(core.workspace_dir(ws).name, out_path=mf, log=lambda m: None, exact=True, record=False,
                             extra_computed=[npz])
            own.append((n, mf))
    base_file = tst._model_file(ws, r["weights"], log)
    t0 = time.time()
    log(f"Asking {len(ask)} question(s) ({len(held)} held out{f', {len(ask) - len(held)} it was computed from' if len(ask) > len(held) else ''}) "
        f"without the edit and with each of {len(sols)} edit(s)…")
    answers = {}

    def grade_all(out):
        rows = {}
        for i, (reply, gen, cut) in zip(ask, out):
            ok, got, off = tst._grade_row(items[i], reply, cut)
            rows[i] = {"correct": bool(ok), "got": got, "reply": reply, "unfinished": cut, "n_reply": len(gen)}
        return rows

    srv = _server(base_file, ctx, parallel, log, [p for _n, p in lora])
    try:
        if lora:
            srv.set_adapters([0.0] * len(lora))
        answers["base"] = grade_all(_ask(srv, tok, stop, prompts, mt, parallel, log, "without the edit"))
        for k, (n, _p) in enumerate(lora):
            srv.set_adapters([1.0 if j == k else 0.0 for j in range(len(lora))])
            answers[n] = grade_all(_ask(srv, tok, stop, prompts, mt, parallel, log, f"with edit {n + 1}"))
    finally:
        srv.close()
        for _n, f in lora:                               # the adapter files are only needed while the server runs
            Path(f).unlink(missing_ok=True)
    for n, mf in own:
        srv = _server(mf, ctx, parallel, log)
        try:
            answers[n] = grade_all(_ask(srv, tok, stop, prompts, mt, parallel, log, f"with edit {n + 1}"))
        finally:
            srv.close()
            for f in (mf, cache / f"{sols[n]['res']['sid']}.npz"):
                f.unlink(missing_ok=True)
    base = answers["base"]
    tries = int(tries or len(sols))
    cands = []
    for n, s in enumerate(sols):
        a = answers[n]
        h = _summary(base, a, held)
        h.update(verdict(h["gained"], h["lost"], h["n"], tries))
        h["text"] = verdict_text(h)
        res = {"sid": s["res"]["sid"], "label": _label(s["res"]), "held": h,
               "rows": [{"i": i, "before": base[i]["correct"], "after": a[i]["correct"], "got_before": base[i]["got"],
                         "got_after": a[i]["got"], "reply_after": a[i]["reply"][-400:], "unfinished_after": a[i]["unfinished"]}
                        for i in ask]}
        if wrong_t and train in ("wrong", "all"):
            res["train_wrong"] = {"n": len(wrong_t), "fixed": sum(a[i]["correct"] for i in wrong_t),
                                  "fixed_without": sum(base[i]["correct"] for i in wrong_t)}
        if train == "all":
            right_t = [i for i in trn if items[i]["correct"]]
            res["train_right"] = {"n": len(right_t), "kept": sum(a[i]["correct"] for i in right_t)}
        s["res"]["validation"] = {k: v for k, v in res.items() if k != "rows"}
        cands.append(res)
        log(f"Edit {n + 1} ({res['label']}): {h['text']}")
    out = {"test": tid, "held": held, "asked": ask, "tries": tries, "seconds": round(time.time() - t0, 1),
           "baseline": {"held_right": sum(base[i]["correct"] for i in held), "n": len(held),
                        "unfinished": sum(base[i]["unfinished"] for i in held)},
           "candidates": cands, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    best = max(cands, key=lambda c: (c["held"]["verdict"] == "helps", c["held"]["after"] - c["held"]["before"]))
    out["best"] = best["sid"] if best["held"]["verdict"] == "helps" else None
    _store(ws, tid, out)
    return out


def _label(res):
    how_ = f"general, rank {res.get('rank')}, keep {res.get('strength')}" if res.get("mode") == "general" else "exact"
    if res.get("spread"):
        sp = res["spread"]
        return (f"spread over {len(sp)} parts (" + ", ".join(f"{'attn' if x['part'] == 'attn' else 'MLP'} L{x['layer']}"
                                                            for x in sp) + f") · {how_}")
    part = "attention" if res.get("part") == "attn" else "MLP"
    layers = res.get("layers") or [res.get("layer")]
    how = f"general, rank {res.get('rank')}, keep {res.get('strength')}" if res.get("mode") == "general" else "exact"
    return f"{part} L{', L'.join(map(str, layers))} · {how}"


def _store(ws, tid, out):
    d = tst.tests_dir(ws)
    p = d / f"{tid}.checks.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    old.append({k: v for k, v in out.items()})
    p.write_text(json.dumps(old[-20:], ensure_ascii=False, indent=1), encoding="utf-8")


def list_checks(ws, tid):
    p = tst.tests_dir(ws) / f"{tid}.checks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


# ------------------------------------------------------------------ try several settings, keep what really helps
DEFAULT_SETTINGS = [
    # the part that decided the mistakes (total effect): where a fix "should" go, but often far from the output, so
    # its goals can need very large changes; a high keep strength keeps the edit small
    {"choose": "responsible", "mode": "general", "rank": 1, "strength": 1.0},
    {"choose": "responsible", "mode": "general", "rank": 4, "strength": 4.0},
    # the layer needing the smallest change (usually the last): a late correction, small enough to apply
    {"choose": "cheapest", "mode": "general", "rank": 1, "strength": 1.0},
    {"choose": "cheapest", "mode": "general", "rank": 4, "strength": 1.0},
    {"choose": "cheapest", "mode": "exact", "per_answer": 1},
    # one correction spread over the parts that decided the mistakes (MEMIT-style), each by its measured share
    {"spread": "auto", "mode": "general", "rank": 4, "strength": 1.0},
    {"spread": "auto", "mode": "exact", "per_answer": 1},
]


def search(ws, tid, split=0.5, settings=None, measure=8, parallel=3, train="wrong", allow_big=False, shuffle=False,
           log=print):
    """Compute an edit for each setting (from the first `split` of the questions), then check them all on the held-out
    rest (check()). Edits too big to apply safely are not checked (unless allow_big, for research). The p-values are corrected for the number of
    edits checked. Returns check()'s result plus the settings and what was skipped."""
    if not split or float(split) >= 1:
        raise ValueError("Finding an edit that helps needs held-out questions: compute from 50–80 % of them")
    settings = settings or DEFAULT_SETTINGS
    a = tst.analyze(ws, tid, log, with_ids=True, split=split, measure=measure, shuffle=shuffle)
    if not a["rows"]:
        raise ValueError("No wrong answers to fix in the questions the edit is computed from"
                         + (" (all were skipped)" if a["skipped"] else ""))
    sids, skipped, tried = [], [], []
    for n, st in enumerate(settings, 1):
        tc.check_stop()
        log(f"Setting {n}/{len(settings)}: {st}")
        try:
            rr = tst.recommend(ws, tid, split=split, measure=measure, analysis=a, log=log, **st)
            s = rr["solution"]
        except tc.Stopped:
            raise
        except Exception as e:
            skipped.append({"setting": st, "reason": f"{type(e).__name__}: {e}"})
            continue
        tried.append({"setting": st, "sid": s["sid"], "label": _label(s), "size": s["change_pct_of_tensor"],
                      "fixed": s["fixed"], "n_fixes": len(s["fixes"])})
        if s.get("too_big") and not allow_big:
            skipped.append({"setting": st, "reason": f"too big to apply safely ({s['change_pct_of_tensor']:.0f}% of the matrix)"})
            continue
        sids.append(s["sid"])
    if not sids:                                       # a result too: no edit of these kinds can be applied safely
        log("No setting gave an edit small enough to apply safely; nothing to check on the held-out questions")
        out = {"test": tid, "held": a["held"], "asked": [], "tries": 0, "baseline": None, "candidates": [], "best": None,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": settings, "tried": tried, "skipped": skipped,
               "none_applicable": True}
        _store(ws, tid, out)
        return out
    out = check(ws, tid, sids, parallel=parallel, train=train, log=log)
    out.update(settings=settings, tried=tried, skipped=skipped)
    _store_last(ws, tid, out)
    return out


def _store_last(ws, tid, out):
    p = tst.tests_dir(ws) / f"{tid}.checks.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    if old:
        old[-1] = out
        p.write_text(json.dumps(old[-20:], ensure_ascii=False, indent=1), encoding="utf-8")
