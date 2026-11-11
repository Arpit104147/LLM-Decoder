"""
Pattern analysis of the 100 traced chats (results_<name>.jsonl + their traces).
  python analyze.py [name]        -> report_<name>.json and report_<name>.md

All numbers come from the recorded traces and the model's own weights:
  - accuracy / confidence per category; right vs wrong
  - layer activity (size of what each layer's attention and MLP added) per category
  - category-specialist MLP neurons: mean activation over the reply, category vs the rest (effect size d)
  - neurons that separate right from wrong answers (within the same categories, permutation p-value)
  - where answers form: the first layer after which the logit lens ranks the chosen token first and keeps it there
  - what pushed the chosen tokens: direct logit attribution summed per layer, top heads and neurons
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import inspect_core as ic
import trace_core as tc

WS = "Qwen3.5-0.8B-Q8_0"
NAME = sys.argv[1] if len(sys.argv) > 1 else "original"
CATS = ["math", "facts", "reasoning", "language", "code", "commonsense"]
rng = np.random.default_rng(0)


def effect(a, b):
    """Cohen's d of a vs b along axis 0."""
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    return (a.mean(0) - b.mean(0)) / np.sqrt((va + vb) / 2 + 1e-8)


def main():
    t0 = time.time()
    rows = [json.loads(l) for l in (HERE / f"results_{NAME}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    model, tok = tc.get_model(WS, "original" if NAME == "original" else "edited", log=lambda m: None)
    v = ic.vocab(WS)
    L, F = model.n_layer, model.ff
    out = {"n": len(rows)}

    # ---- per chat summaries from the traces
    mean_act, mean_abs, layer_a, layer_m, conf, push_a, push_m, form = [], [], [], [], [], [], [], []
    neuron_push_sum = {c: np.zeros((L, F), np.float64) for c in CATS}
    head_push_sum = {c: {} for c in CATS}
    npos = {c: 0 for c in CATS}
    for r in rows:
        info, A = tc._open_trace(WS, r["trace"])
        n = A["act"].shape[0]
        rp = np.arange(max(0, info["n_prompt"] - 1), n)                 # positions that produced reply tokens
        act = A["act"][rp].astype(np.float32)                          # [p, L, F]
        mean_act.append(act.mean(0).ravel())
        mean_abs.append(np.abs(act).mean(0).ravel())
        layer_a.append(A["attn_norm"][rp].mean(0))
        layer_m.append(A["mlp_norm"][rp].mean(0))
        conf.append(float(np.exp(A["next_lp"][rp]).mean()))
        # attribution + logit lens on up to 12 reply positions per chat
        ids = info["ids"]
        sample = rp if len(rp) <= 12 else rp[np.linspace(0, len(rp) - 1, 12).astype(int)]
        pa, pm, fl = np.zeros(L), np.zeros(L), []
        for pos in sample:
            tgt = ids[pos + 1]
            att = tc.attribute(model, A, int(pos), int(tgt))
            pa += att["layer_attn"]
            pm += att["layer_mlp"]
            neuron_push_sum[r["cat"]] += att["neuron_push"]
            for Lh, hrow in enumerate(att["head_push"]):
                for h, val in enumerate(hrow):
                    head_push_sum[r["cat"]][(Lh, h)] = head_push_sum[r["cat"]].get((Lh, h), 0.0) + float(val)
            lg = model.logits(model.final(A["resid"][pos].astype(np.float32)))      # [L+1, V]
            top1 = lg.argmax(-1) == tgt
            later = [i for i in range(L + 1) if top1[i:].all()]
            fl.append((later[0] if later else L) - 1)                  # lens row i = after layer i - 1 (row 0 = embedding, -1)
        npos[r["cat"]] += len(sample)
        push_a.append(pa / len(sample))
        push_m.append(pm / len(sample))
        form.append(float(np.mean(fl)))
        tc._TRACES.clear()
    mean_act, mean_abs = np.array(mean_act), np.array(mean_abs)
    layer_a, layer_m, push_a, push_m = map(np.array, (layer_a, layer_m, push_a, push_m))
    cats = np.array([r["cat"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    conf, form = np.array(conf), np.array(form)

    # ---- accuracy and confidence
    out["accuracy"] = {c: {"n": int((cats == c).sum()), "correct": int(correct[cats == c].sum()),
                           "mean_confidence": round(float(conf[cats == c].mean()), 3),
                           "answer_forms_after_layer": round(float(form[cats == c].mean()), 1),
                           "reply_tokens": round(float(np.mean([r["n_reply"] for r in rows if r["cat"] == c])), 1)}
                       for c in CATS}
    out["right_vs_wrong"] = {k: {"n": int(m.sum()), "mean_confidence": round(float(conf[m].mean()), 3) if m.any() else None,
                                 "answer_forms_after_layer": round(float(form[m].mean()), 1) if m.any() else None,
                                 "reply_tokens": round(float(np.mean([r["n_reply"] for r, ok in zip(rows, m) if ok])), 1) if m.any() else None}
                             for k, m in (("right", correct), ("wrong", ~correct))}
    out["wrong"] = [{"id": r["id"], "cat": r["cat"], "got": r["got"], "reply": r["reply"][-160:]} for r in rows if not r["correct"]]

    # ---- layer activity per category (relative to all chats)
    base_a, base_m = layer_a.mean(0), layer_m.mean(0)
    out["layer_activity"] = {c: {"attn_rel": (layer_a[cats == c].mean(0) / base_a).round(3).tolist(),
                                 "mlp_rel": (layer_m[cats == c].mean(0) / base_m).round(3).tolist()} for c in CATS}
    out["layer_push"] = {c: {"attn": push_a[cats == c].mean(0).round(3).tolist(), "mlp": push_m[cats == c].mean(0).round(3).tolist()}
                         for c in CATS}

    # ---- specialist neurons per category
    def promotes(Lr, j, k=6):
        col = model.w(Lr, "ffn_down.weight")[:, j] * model.out_norm
        lg = model.U @ col
        top = np.argsort(-lg)[:k]
        return [v["texts"][i] for i in top]

    spec = {}
    for c in CATS:
        d = effect(mean_act[cats == c], mean_act[cats != c])
        size = np.abs(mean_act[cats == c].mean(0))
        score = np.where(size > np.percentile(size, 90), d, 0)        # only neurons that are clearly active
        top = np.argsort(-np.abs(score))[:12]
        spec[c] = [{"layer": int(i // F), "neuron": int(i % F), "d": round(float(d[i]), 2),
                    "mean_act_in_cat": round(float(mean_act[cats == c][:, i].mean()), 3),
                    "mean_act_elsewhere": round(float(mean_act[cats != c][:, i].mean()), 3),
                    "promotes": promotes(int(i // F), int(i % F))} for i in top]
    out["specialists"] = spec
    # where the specialists live
    out["specialist_layers"] = {c: np.bincount([s["layer"] for s in spec[c]], minlength=L).tolist() for c in CATS}

    # ---- right vs wrong neurons (only categories with both)
    mix = [c for c in CATS if 0 < correct[cats == c].sum() < (cats == c).sum()]
    sel = np.isin(cats, mix)
    if sel.sum() >= 6 and (~correct[sel]).sum() >= 3:
        X = mean_act[sel]
        # remove each category's own mean so the difference isn't just "math is harder"
        for c in mix:
            m = cats[sel] == c
            X[m] -= X[m].mean(0)
        y = correct[sel]
        d = effect(X[y], X[~y])
        top = np.argsort(-np.abs(d))[:12]
        perm = []
        for _ in range(200):
            yp = rng.permutation(y)
            perm.append(np.abs(effect(X[yp], X[~yp])).max())
        out["right_vs_wrong_neurons"] = {
            "categories": mix, "n_right": int(y.sum()), "n_wrong": int((~y).sum()),
            "max_abs_d_by_chance_95pct": round(float(np.percentile(perm, 95)), 2),
            "top": [{"layer": int(i // F), "neuron": int(i % F), "d": round(float(d[i]), 2),
                     "higher_when": "right" if d[i] > 0 else "wrong", "promotes": promotes(int(i // F), int(i % F))} for i in top]}

    # ---- heads and neurons that push answers most (summed over sampled reply tokens)
    out["top_heads"] = {}
    out["top_push_neurons"] = {}
    for c in CATS:
        hs = sorted(head_push_sum[c].items(), key=lambda kv: -abs(kv[1]))[:8]
        out["top_heads"][c] = [{"layer": k[0], "head": k[1], "kind": model.kinds()[k[0]], "mean_push": round(val / npos[c], 3)} for k, val in hs]
        ps = neuron_push_sum[c] / npos[c]
        top = np.argsort(-ps.ravel())[:8]
        out["top_push_neurons"][c] = [{"layer": int(i // F), "neuron": int(i % F), "mean_push": round(float(ps.ravel()[i]), 3),
                                       "promotes": promotes(int(i // F), int(i % F), 4)} for i in top]
    out["seconds"] = round(time.time() - t0)
    (HERE / f"report_{NAME}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(out)
    print(f"analysis done in {out['seconds']} s -> report_{NAME}.md")


def write_md(o):
    L = []
    a = o["accuracy"]
    L.append(f"# Qwen3.5-0.8B — 100 traced chats ({NAME} weights)\n")
    L.append("| Category | Correct | Confidence | Top guess from layer on | Reply tokens |")
    L.append("|---|---|---|---|---|")
    tot = sum(x["correct"] for x in a.values())
    for c, x in a.items():
        L.append(f"| {c} | {x['correct']}/{x['n']} | {x['mean_confidence']:.2f} | {x['answer_forms_after_layer']} | {x['reply_tokens']} |")
    L.append(f"| **all** | **{tot}/{o['n']}** | | | |\n")
    rw = o["right_vs_wrong"]
    L.append("| | Answers | Confidence | Top guess from layer on | Reply tokens |\n|---|---|---|---|---|")
    for k in ("right", "wrong"):
        x = rw[k]
        L.append(f"| {k} | {x['n']} | {x['mean_confidence']} | {x['answer_forms_after_layer']} | {x['reply_tokens']} |")
    L.append("\n## Specialist neurons (top 5 per category)\n")
    L.append("| Category | Neuron | Effect size d | Active in category / elsewhere | Pushes words like |\n|---|---|---|---|---|")
    for c, lst in o["specialists"].items():
        for s in lst[:5]:
            L.append(f"| {c} | L{s['layer']} #{s['neuron']} | {s['d']} | {s['mean_act_in_cat']} / {s['mean_act_elsewhere']} | {' '.join(repr(t) for t in s['promotes'][:4])} |")
    if "right_vs_wrong_neurons" in o:
        r = o["right_vs_wrong_neurons"]
        L.append(f"\n## Right vs wrong ({r['n_right']} right, {r['n_wrong']} wrong in {', '.join(r['categories'])}; chance level |d| ≈ {r['max_abs_d_by_chance_95pct']})\n")
        L.append("| Neuron | d | Higher when | Pushes words like |\n|---|---|---|---|")
        for s in r["top"][:8]:
            L.append(f"| L{s['layer']} #{s['neuron']} | {s['d']} | {s['higher_when']} | {' '.join(repr(t) for t in s['promotes'][:4])} |")
    (HERE / f"report_{NAME}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
