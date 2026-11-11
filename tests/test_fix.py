"""
Tests for the Test tab and Fix answers (computed edits), on a synthetic Llama model in a temporary workspace.
  python test_fix.py

1. question parsing (pipe lines, JSON, JSONL / GSM8K, CSV) and grading (numbers, number words, \\boxed{}, keywords)
2. run a test -> analyse -> computed edit: every fix is reached exactly, protected positions don't move
3. saving the edit: the workspace weights, measure, Chat & trace reload and the exported GGUF all include it;
   deleting it restores the original weights
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core

tmp = Path(tempfile.mkdtemp(prefix="llmdec_fix_"))
core.WORKSPACES = tmp / "workspaces"                  # keep the real workspaces folder untouched
import trace_core as tc
import fix_core as fc
import test_core as tst
import effects_core as ec
import test_trace_arch as arch

ok = fail = 0


def check(cond, msg):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


q = lambda m: None

# ------------------------------------------------------------------ 1. parsing and grading
it = tst.parse_questions("What is 2+2? | 4\nCapital of France? | Paris / paris city\n# comment\n")
check(len(it) == 2 and it[0]["kind"] == "number" and it[0]["value"] == 4 and it[1]["keywords"] == ["Paris", "paris city"],
      "pipe lines: numbers and several right answers")
it = tst.parse_questions(json.dumps([{"q": "a?", "kw": ["x", "y"]}, {"q": "b?", "num": 12.5}]))
check(it[0]["kind"] == "text" and it[0]["keywords"] == ["x", "y"] and it[1]["value"] == 12.5, "JSON list in the study format")
it = tst.parse_questions('{"row": {"question": "How many?", "answer": "work... #### 1,234"}}\n{"question": "c?", "answer": "z"}')
check(it[0]["kind"] == "number" and it[0]["value"] == 1234 and it[1]["answer"] == "z", "JSONL with GSM8K answers (#### n)")
it = tst.parse_questions("question,answer\n\"Hi, there?\",hello\nNum?,7\n")
check(len(it) == 2 and it[0]["question"] == "Hi, there?" and it[1]["value"] == 7, "CSV with question,answer columns")
try:
    tst.parse_questions("no separator here")
    check(False, "a line without | is refused")
except ValueError:
    check(True, "a line without | is refused")
N = lambda a: tst._item({"question": "q", "answer": a}, 0)
check(tst.grade(N("391"), "17*23 = 391")[0] and not tst.grade(N("391"), "391? no, 442")[0], "numbers: the last number counts")
check(tst.grade(N("391"), "so \\boxed{391}. Check: 17*23=391, 1")[0], "numbers: \\boxed{} wins over later numbers")
check(tst.grade(N("1234"), "It is 1,234.")[0] and tst.grade(N("8"), "A spider has **eight** legs.")[0],
      "numbers: thousands separators and number words")
check(tst.grade(N("cheetah"), "Cheetahs.")[0] and not tst.grade(N("cat"), "concatenate")[0], "text: plurals, whole words only")
check(not tst.grade(tst._item({"question": "q", "answer": ["same"], "not": ["heavier"]}, 0), "the same, iron is heavier")[0],
      "text: a 'not' word makes it wrong")

# ------------------------------------------------------------------ 2. a synthetic model: test, analyse, fix
src = tmp / "tiny_llama.gguf"
arch.write_model(arch.FAMILIES["llama"], src, seed=3)
ws = core.extract_workspace(src, log=q)
d = core.WORKSPACES / ws
meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
meta["tokenizer.chat_template"] = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                                   "<|im_start|>assistant\n")
(d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
text = "say one | toe\nsay two | hat\nsay three | 42\nsay four | dot\nsay five | 7"
r = tst.run(ws, text, instruction="", max_tokens=6, which="edited", log=q)
check(len(r["items"]) == 5 and all(x["trace"] for x in r["items"]), f"test run: 5 traced answers ({r['right']} right)")
a = tst.analyze(ws, r["id"], log=q, with_ids=True)
check(len(a["rows"]) + len(a["skipped"]) == 5 - r["right"] and a["recommended_layer"] in a["editable_layers"],
      f"analysis: every wrong answer located or skipped with a reason ({len(a['rows'])} located, {a['n_token_fixes']} token fixes)")
model, tok = tc.get_model(ws, "edited", q)
if a["rows"]:
    x0 = a["rows"][0]
    ok_all = True
    for f in x0["fixes"]:
        A2 = fc.forced(model, f["ids"])
        ok_all &= f["ids"][f["pos"] + 1] == f["target"] and int(A2["top_ids"][f["pos"], 0]) == f["wrong"] != f["target"]
    check(ok_all and 1 <= x0["n_tokens"] <= tst.MAX_TOKENS_PER_ANSWER,
          f"every token fix is a token of the right text the model wouldn't write there ({x0['n_tokens']} for {x0['should']!r})")
    f0 = x0["fixes"][0]
    gap_sum = abs(sum(f0["mlp_gap"]) + sum(f0["attn_gap"]) - f0["gap"])
    check(gap_sum < 0.2, f"per-layer pushes add up to the score gap between wrong and right token (off by {gap_sum:.3f})")
    if len(x0["fixes"]) > 1:
        f1 = x0["fixes"][1]
        check(f1["ids"][:len(f0["ids"])] == f0["ids"], "later token fixes read the corrected text before them")

# calculation steps: the first wrong one is fixed, not the final answer
reply = "Working: 12 * 4 = 46, so 46 + 2 = 48. Answer: 48"
rids = tok.encode(reply, special=False)
row = {"kind": "number", "offset": reply.rindex("48"), "got": "48", "answer": "50", "question": "q"}
w = tst._what_to_fix(tok, row, rids)
check(w[0] == "calculation step" and w[2].strip() == "48" and "12 * 4 = 46" in w[4],
      f"a wrong calculation step is found first ({w[4]!r} -> fix to {w[2].strip()!r})")
r2txt = "12 * 4 = 48, so 48 + 2 = 50? No: 49"
w = tst._what_to_fix(tok, {**row, "answer": "50", "offset": r2txt.rindex("49"), "got": "49"}, tok.encode(r2txt, special=False))
check(w[0] == "final answer" and w[2].strip() == "50" and w[5], "with every step right, the final answer is fixed (and ends the reply)")
w = tst._what_to_fix(tok, {**row, "solution": "12 * 4 = 48 and 48 + 2 = 50"}, rids)
check(w[0] == "reference working" and w[1] == 0, "a solution field is used as the whole right reply")

L = model.n_layer - 1
traces = [x["trace"] for x in r["items"]]
fixes = [{"trace": f["trace"], "ids": f["ids"], "pos": f["pos"], "target": f["target"]} for x in a["rows"] for f in x["fixes"]][:4]
info_p = tc._open_trace(ws, traces[-1])[0]
protect = [{"trace": traces[-1], "pos": info_p["n_prompt"] - 1}]
if fixes:
    s = fc.solve(ws, fixes, layer=L, neurons=0, margin=1.0, which="edited", keep=traces, check=[], protect=protect, log=q)
    check(s["fixed"] == len(fixes), f"computed edit: {s['fixed']}/{len(fixes)} token fixes now give the wanted token")
    check(s["fit_error"] < 1e-4, f"least squares reaches every goal exactly (relative error {s['fit_error']:.1e})")
    sol = fc._SOL[s["sid"]]
    last = fc._LAST[None]
    pt = sol["patches"][0]
    moved = np.abs(pt["delta"].astype(np.float64) @ last["K"][last["n_fix"]:, pt["cols"]].T).max()
    check(moved < 1e-4, f"a protected position's layer output doesn't move (max change {moved:.1e})")
    s64 = fc.solve(ws, fixes, layer=L, neurons=8, margin=1.0, which="edited", keep=traces, check=[], log=q)
    check(len(s64["columns"]) == 8 and s64["n_numbers"] == 8 * model.D, "neurons=8 changes only 8 columns of ffn_down")

    # ------------------------------------------------------------------ 3. save, reload, export, delete
    W0 = np.array(model.T[f"blk.{L}.ffn_down.weight"], copy=True)
    if s["too_big"]:
        try:
            fc.save(ws, s["sid"], "t0")
            check(False, "an edit bigger than half its matrix is refused without 'apply anyway'")
        except ValueError:
            check(not fc.list_saved(ws), f"an edit bigger than half its matrix ({s['change_pct_of_tensor']:.0f}%) is refused without 'apply anyway'")
    fc.save(ws, s["sid"], "t1", force=True)
    check([e["name"] for e in fc.list_saved(ws)] == ["t1"] and (d / "computed" / "t1.npz").exists(), "saved as computed/t1.npz")
    try:
        fc.save(ws, s["sid"], "t1", force=True)
        check(False, "a second edit with the same name is refused")
    except ValueError:
        check(True, "a second edit with the same name is refused")
    m2, _ = tc.get_model(ws, "edited", q)
    exp = W0.copy()
    exp[:, pt["cols"]] += pt["delta"]
    check(m2 is not model and np.allclose(m2.T[f"blk.{L}.ffn_down.weight"], exp, atol=1e-6),
          "Chat & trace reloads the workspace weights with the edit added")
    st = core.measure_changes(ws, log=q)
    names = {t["name"] for t in st["tensors"] if t["edited"]}
    check(f"blk.{L}.ffn_down.weight" in names, "measure counts the computed edit")
    check("t1" in core.pending_changes(ws)["computed"], "Changes & Export lists it")
    out = tmp / "exported.gguf"
    core.export_gguf(ws, out_path=out, log=q, exact=True)
    from gguf import GGUFReader
    t_out = {t.name: t for t in GGUFReader(str(out)).tensors}[f"blk.{L}.ffn_down.weight"]
    check(np.allclose(core.full_tensor(t_out), exp, atol=1e-6), "the exported GGUF holds original + computed values")
    r2 = tst.run(ws, text, instruction="", max_tokens=6, which="edited", log=q)
    check(r2["computed_edits"] == ["t1"], "a test run records which computed edits it ran with")
    fc.delete(ws, "t1")
    m3, _ = tc.get_model(ws, "edited", q)
    check(np.array_equal(m3.T[f"blk.{L}.ffn_down.weight"], W0) and not fc.list_saved(ws),
          "deleting the edit gives back the exact original weights")
else:
    check(False, "the synthetic model got every answer right; nothing to fix")

# ------------------------------------------------------------------ 5. fast mode, held-out split, general mode
rt = tst.run(ws, text, instruction="", max_tokens=6, which="edited", engine="tracer", log=q)
rf = tst.run(ws, text, instruction="", max_tokens=6, which="edited", engine="llama", parallel=2, log=q)
same = [x["reply"] == y["reply"] for x, y in zip(rt["items"], rf["items"])]
check(sum(same) >= 4 and all(x["trace"] is None for x in rf["items"]),
      f"fast mode: llama.cpp gives the tracer's greedy replies ({sum(same)}/5 identical) and traces nothing yet")
r2 = tst.ensure_traces(ws, rf["id"], [0, 1], log=q)
tr0 = r2["items"][0]["trace"]
info0, A0 = tc._open_trace(ws, tr0)
npr = info0["n_prompt"]
agree = float(np.mean(A0["top_ids"][npr - 1:, 0] == np.array(info0["ids"][npr:])))
check(tr0 and r2["items"][2]["trace"] is None and agree >= 0.8,
      f"traces on demand: only the asked replies are replayed, and the tracer predicts {agree:.0%} of the reply")
tr_, held_ = tst.split_of(rf, 0.6)
check(tr_ == [0, 1, 2] and held_ == [3, 4], "split: the first 60 % compute the edit, the rest is held out")
a2 = tst.analyze(ws, rf["id"], log=q, split=0.6)
check(all(x["i"] in tr_ for x in a2["rows"]) and a2["held"] == [3, 4] and all(x.get("trace") for x in tst.get_test(ws, rf["id"])["items"][:3]),
      "analysis with a split looks only at the first part (and traces just those replies)")
if fixes:
    sg = fc.solve(ws, fixes, layer=L, which="edited", keep=traces, check=[], mode="general", rank=1, strength=0.3, log=q)
    pt = fc._SOL[sg["sid"]]["patches"][0]
    rk = np.linalg.matrix_rank(pt["delta"].astype(np.float64), tol=1e-4 * np.abs(pt["delta"]).max())
    check(rk == 1 and sg["mode"] == "general" and sg["layers"] == [L],
          f"general mode: one shared rank-1 correction ({sg['fixed']}/{len(fixes)} token fixes reached, change "
          f"{sg['change_pct_of_tensor']:.1f}% vs exact {s['change_pct_of_tensor']:.1f}%)")
    fc.save(ws, sg["sid"], "g1", force=True)
    try:
        tst.ensure_traces(ws, rf["id"], [4], log=q)
        check(False, "after an edit, old fast-mode replies are not traced with the new weights")
    except ValueError:
        check(True, "after an edit, old fast-mode replies are not traced with the new weights")
    fc.delete(ws, "g1")

# ------------------------------------------------------------------ 6. total effects (switching parts off)
a3 = tst.analyze(ws, rf["id"], log=q, measure=2)
mx = [x for x in a3["rows"] if x.get("total_top")]
check(a3["measured"] == min(2, len(a3["rows"])) and len(mx) == a3["measured"] and a3["top_layers_total"],
      f"analysis measures total effects on {a3['measured']} answer(s) and ranks layers by them")
if fixes:
    out = tst.recommend(ws, rf["id"], measure=2, log=q)
    top = out["analysis"]["top_parts_total"][0]
    check(out["solution"]["layer"] == top["layer"] and out["solution"]["part"] == top["part"],
          f"the edit goes to the most responsible part by default ({top['part']} L{out['solution']['layer']})")
    out2 = tst.recommend(ws, rf["id"], choose="cheapest", log=q)
    check(out2["analysis"]["measured"] == 0 and out2["solution"]["layer_costs"], "'cheapest' skips the measuring and picks by cost")


def effects_exact(name):
    path = tmp / f"eff_{name}.gguf"
    arch.write_model(arch.FAMILIES[name], path, seed=4)
    m, _t = tc.get_model(core.extract_workspace(path, log=q), "edited", q)
    ids = [int(t_) for t_ in np.random.default_rng(2).integers(0, 150, 20)]
    A = fc.forced(m, ids)
    pos, tgt, con = 15, int(A["top_ids"][15, 0]), int(A["top_ids"][15, 1])
    lg0 = fc._logits_at(m, ids, pos)
    e = ec.layer_effects(m, ids, pos, tgt, con, log=q, baseline="zero")
    worst, n = 0.0, 0
    for kind, wname in (("mlp", "ffn_down.weight"), ("attn", "attn_output.weight")):
        for Lx in range(m.n_layer):
            W = m.w(Lx, wname)
            if e["effect"][kind][Lx] is None or W is None or m.w(Lx, wname.replace("weight", "bias")) is not None:
                continue
            if kind == "mlp" and m.is_moe(Lx):
                continue
            keep = W.copy()
            W[:] = 0
            fc._forget_prefix()                       # weights changed by hand: no saved prefix state is valid
            try:
                lg = fc._logits_at(m, ids, pos)
            finally:
                W[:] = keep
                fc._forget_prefix()
            worst = max(worst, abs((float(lg0[tgt] - lg0[con]) - float(lg[tgt] - lg[con])) - e["effect"][kind][Lx]))
            n += 1
    em = ec.layer_effects(m, ids, pos, tgt, con, log=q)
    ok_lens = abs(em["lens_lead"][-1] - em["base_lead"]) < 1e-4
    check(n > 0 and worst < 1e-4 and np.array_equal(lg0, fc._logits_at(m, ids, pos)) and ok_lens,
          f"{name}: switching a part off through the hooks = zeroing its weights ({n} parts, worst gap {worst:.1e}); "
          f"the model is unchanged afterwards; the lens ends at the real lead")


def reply_matches(name):
    path = tmp / f"reff_{name}.gguf"
    arch.write_model(arch.FAMILIES[name], path, seed=4)
    m, _t = tc.get_model(core.extract_workspace(path, log=q), "edited", q)
    ids = [int(t_) for t_ in np.random.default_rng(3).integers(0, 150, 18)]
    last = len(ids) - 2
    r = ec.reply_effects(m, ids, [8, 12, last], log=q)
    c = r["per_position"][-1]["contrast"]
    e = ec.layer_effects(m, ids, last, ids[last + 1], c, log=q)
    worst = 0.0
    for k in ("attn", "mlp"):
        for Lx in range(m.n_layer):
            if e["effect"][k][Lx] is None:
                continue
            per = r["_eff"][k][Lx][-1]                # the last position's effect from the whole-reply run
            worst = max(worst, abs(per - e["effect"][k][Lx]))
    check(len(r["per_position"]) == 3 and worst < 1e-4,
          f"{name}: one whole-reply re-run per part gives the same effect at the last token as measuring that token alone "
          f"(worst gap {worst:.1e}); 3 positions at once")


for name in ("llama", "gemma2", "command-r", "qwen3moe"):
    try:
        reply_matches(name)
    except Exception as e:
        check(False, f"{name} whole-reply effects: {type(e).__name__}: {e}")
for name in ("llama", "gemma2", "command-r", "qwen3moe"):
    try:
        effects_exact(name)
    except Exception as e:
        check(False, f"{name} total effects: {type(e).__name__}: {e}")

# ------------------------------------------------------------------ 4. every kind of layer
def family(name, export_check=False, rank=5, calm=False):
    path = tmp / f"{name}.gguf"
    arch.write_model(arch.FAMILIES[name], path, seed=5)
    wsf = core.extract_workspace(path, log=q)
    if calm:     # random tied embeddings make a synthetic Gemma's scores 15-30 apart (or tied at its cap): scale them
        (core.WORKSPACES / wsf / "edit_weights.py").write_text(
            'def edit(w):\n    w["token_embd.weight"] *= 0.1\n', encoding="utf-8")
    m, _t = tc.get_model(wsf, "edited", q)
    layers = fc.editable_layers(m)
    if not layers:
        return wsf, None, None
    L = layers[-1]
    tg = fc.edit_target(m, L)
    rng = np.random.default_rng(1)
    V = m.U.shape[0]
    seq = [int(t) for t in rng.integers(0, min(V, 150), 14)]
    seq2 = [int(t) for t in rng.integers(0, min(V, 150), 10)]
    A0 = fc.forced(m, seq)
    fx = []
    for pos in (5, 8, 11):
        top = int(A0["top_ids"][pos, 0])
        fx.append({"ids": seq, "pos": pos, "target": int(A0["top_ids"][pos, rank])})
    prot = [{"ids": seq2, "pos": 6}]
    _st, before = fc._position_parts(m, tg, seq2, 6, L)
    s = fc.solve(wsf, fx, layer=L, which="edited", keep=[], check=[], protect=prot, log=q)
    sol = fc._SOL[s["sid"]]
    saved = fc._apply(m, sol["patches"])
    try:
        _st, after = fc._position_parts(m, tg, seq2, 6, L)
    finally:
        fc._restore(saved)
    moved = float(np.abs(after["m_out"] - before["m_out"]).max() / (np.abs(before["m_out"]).max() + 1e-9))
    what = tg["kind"] + (" + output norm" if tg["post"] else "") + (" (parallel)" if m.parallel else "")
    one = len(s["layers"]) == 1                # chained edits change earlier positions too, so only check one layer
    check(s["fixed"] == 3 and (moved < 1e-3 or not one),
          f"{name}: layer {L} {what} -> {s['fixed']}/3 fixed (layers {s['layers']})"
          + (f", protected output moved {moved:.1e}" if one else "") + (f", experts {s['experts']}" if s["experts"] else ""))
    if export_check:
        exp = {}
        for pt in sol["patches"]:
            if pt["tensor"] not in exp:
                exp[pt["tensor"]] = np.array(m.T[pt["tensor"]], copy=True)
            M = exp[pt["tensor"]][pt["expert"]] if pt["expert"] >= 0 else exp[pt["tensor"]]
            M[:, pt["cols"]] += pt["delta"]
        fc.save(wsf, s["sid"], "x", force=True)
        m2, _t = tc.get_model(wsf, "edited", q)
        same = all(np.allclose(m2.T[n], e, atol=1e-6) for n, e in exp.items())
        out = tmp / f"{name}_x.gguf"
        core.export_gguf(wsf, out_path=out, log=q, exact=True)
        from gguf import GGUFReader
        tens = {t_.name: t_ for t_ in GGUFReader(str(out)).tensors}
        same_out = all(np.allclose(core.full_tensor(tens[n]), e, atol=1e-6) for n, e in exp.items())
        check(same and same_out, f"{name}: the saved edit is in Chat & trace and in the exported GGUF ({', '.join(exp)})")
    return wsf, s, tg


for name, exp_ in (("llama", False), ("olmo2", False), ("command-r", False),
                   ("qwen2moe", True), ("deepseek3", False), ("qwen3moe", True), ("mixtral", False), ("olmoe", False),
                   ("gpt-oss", False), ("granitemoe", False), ("glm4moe", False), ("jamba", False), ("lfm2", False)):
    try:
        family(name, exp_)
    except Exception as e:
        check(False, f"{name}: {type(e).__name__}: {e}")
# behind an output norm a layer can only move the residual so far: chained layers reach further
for name in ("gemma2", "gemma3"):
    try:
        _w, s_, _t = family(name, name == "gemma2", rank=1, calm=True)
    except Exception as e:
        check(False, f"{name} (close call): {type(e).__name__}: {e}")
# attention output edits (attn_output / ssm_out), solved through the rest of the layer
def attn_family(name, calm=False):
    path = tmp / f"att_{name}.gguf"
    arch.write_model(arch.FAMILIES[name], path, seed=5)
    wsf = core.extract_workspace(path, log=q)
    if calm:
        (core.WORKSPACES / wsf / "edit_weights.py").write_text('def edit(w):\n    w["token_embd.weight"] *= 0.1\n', encoding="utf-8")
    m, _t = tc.get_model(wsf, "edited", q)
    layers = fc.editable_layers(m, "attn")
    L = layers[len(layers) // 2]                                 # a middle layer: the rest of the model reacts
    tg = fc.edit_target(m, L, "attn")
    rng = np.random.default_rng(1)
    seq = [int(t) for t in rng.integers(0, min(m.U.shape[0], 150), 14)]
    seq2 = [int(t) for t in rng.integers(0, min(m.U.shape[0], 150), 10)]
    A0 = fc.forced(m, seq)
    fx = [{"ids": seq, "pos": p_, "target": int(A0["top_ids"][p_, 1])} for p_ in (5, 8, 11)]
    _st, before = fc._position_parts(m, tg, seq2, 6, L)
    s_ = fc.solve(wsf, fx, layer=L, which="edited", keep=[], check=[], protect=[{"ids": seq2, "pos": 6}], part="attn", log=q)
    sol = fc._SOL[s_["sid"]]
    saved = fc._apply(m, sol["patches"])
    try:
        _st, after = fc._position_parts(m, tg, seq2, 6, L)
    finally:
        fc._restore(saved)
    moved = float(np.abs(after["a_out"] - before["a_out"]).max() / (np.abs(before["a_out"]).max() + 1e-9))
    one = len(s_["layers"]) == 1
    check(s_["fixed"] == 3 and s_["part"] == "attn" and sol["patches"][0]["tensor"] == tg["tensor"] and (moved < 1e-3 or not one),
          f"{name}: attention output of layer {L} ({tg['tensor']}{', parallel' if m.parallel else ''}{', output norm' if tg['post'] else ''}) "
          f"-> {s_['fixed']}/3 fixed (layers {s_['layers']})" + (f", protected attention output moved {moved:.1e}" if one else ""))


for name, calm in (("llama", False), ("gemma2", True), ("command-r", False), ("qwen3moe", False), ("olmo2", True)):
    try:
        attn_family(name, calm)
    except Exception as e:
        check(False, f"{name} attention edit: {type(e).__name__}: {e}")
for name in ("mamba", "rwkv6"):
    path = tmp / f"{name}.gguf"
    arch.write_model(arch.FAMILIES[name], path, seed=5) if name != "rwkv6" else arch.write_rwkv(arch.FAMILIES[name], path, seed=5)
    m, _t = tc.get_model(core.extract_workspace(path, log=q), "edited", q)
    try:
        fc.edit_target(m, m.n_layer - 1)
        check(False, f"{name}: layers without an MLP output matrix are refused")
    except ValueError as e:
        check(True, f"{name}: refused with a clear reason ({e})")

tc.unload()
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(arch.TMP, ignore_errors=True)
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
