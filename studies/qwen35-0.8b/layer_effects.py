"""
Is the layer Chat & trace highlights the layer that decided the answer?

For the ANSWER token of study chats (the answer number for math, the first answer word otherwise; original weights),
against the runner-up token at that position:
  direct push    what each layer's attention / MLP adds straight to (answer − runner-up)  (the app's "push")
  total effect   how much (answer − runner-up) drops when that part is switched off (output -> its average over the
                 text) and everything after it is re-run: includes what it does through later layers
  flip layer     logit lens: the first layer after which the residual prefers the answer and keeps preferring it
  python studies/qwen35-0.8b/layer_effects.py [n_per_category]        -> layer_effects.json / .md
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import trace_core as tc
import effects_core as ec
import test_core as tst

WS = "Qwen3.5-0.8B-Q8_0"
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rows = [json.loads(l) for l in (HERE / "results_original.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
qs = {q["id"]: q for q in json.loads((HERE / "questions.json").read_text(encoding="utf-8"))}
model, tok = tc.get_model(WS, "original", lambda m: None)
nL = model.n_layer


def answer_pos(r, info):
    """Position that writes the answer token: the graded number for math, the first answer word otherwise."""
    q = qs[r["id"]]
    item = tst._item({"question": q["q"], **({"num": q["num"]} if "num" in q else {"answer": q["kw"]})}, 0)
    ids, npr = info["ids"], info["n_prompt"]
    reply_ids = [i for i in ids[npr:] if not tok.is_stop(i)]
    reply = tok.decode(reply_ids)
    if item["kind"] == "number":
        _ok, _got, off = tst.grade(item, reply)
        j = tst._char_to_token(tok, reply_ids, off) if off is not None else None
    else:
        j = next((k for k, t in enumerate(reply_ids) if not tst.MARKUP.match(tok.decode([t]))), 0)
    return None if j is None else npr + j - 1


cats = {}
for r in rows:
    cats.setdefault(r["cat"], []).append(r)
pick = [r for c in cats.values() for r in c[:PER]]
out = []
for k, r in enumerate(pick, 1):
    info, _A = tc._open_trace(WS, r["trace"])
    pos = answer_pos(r, info)
    if pos is None:
        print(f"[{k}/{len(pick)}] {r['id']}: no answer token found, skipped", flush=True)
        continue
    e = ec.trace_effects(WS, r["trace"], pos, log=lambda m: None)
    parts = lambda v: [(v["attn"][L], "attn", L) for L in range(nL) if v["attn"][L] is not None] + \
                      [(v["mlp"][L], "mlp", L) for L in range(nL) if v["mlp"][L] is not None]
    bd, bt = max(parts(e["direct"])), max(parts(e["effect"]))
    o = {"id": r["id"], "cat": r["cat"], "correct": r["correct"], "pos": pos, "token": e["target_text"],
         "runner_up": e["contrast_text"], "lead": e["base_lead"], "direct": e["direct"], "total": e["effect"],
         "lens": e["lens_lead"], "flip": e["flip_after_layer"], "top_direct": [bd[1], bd[2], bd[0]], "top_total": [bt[1], bt[2], bt[0]],
         "top3_total": [[kk, L, v] for v, kk, L in ec.top_parts(e, 3)]}
    out.append(o)
    print(f"[{k}/{len(pick)}] {r['id']} {o['token']!r} over {o['runner_up']!r} (lead {o['lead']:.2f}): "
          f"app shows {bd[1]} L{bd[2]} ({bd[0]:+.2f}) · switching off hurts most: {bt[1]} L{bt[2]} ({bt[0]:+.2f}) · "
          f"lens: leads from L{o['flip']}", flush=True)
    (HERE / "layer_effects.json").write_text(json.dumps(out), encoding="utf-8")

D = {kk: np.array([[v if v is not None else 0 for v in o["direct"][kk]] for o in out]) for kk in ("attn", "mlp")}
T = {kk: np.array([[v if v is not None else 0 for v in o["total"][kk]] for o in out]) for kk in ("attn", "mlp")}
same = sum(o["top_direct"][:2] == o["top_total"][:2] for o in out)
cd = Counter(f"{o['top_direct'][0]} L{o['top_direct'][1]}" for o in out)
ct = Counter(f"{o['top_total'][0]} L{o['top_total'][1]}" for o in out)
fl = Counter(o["flip"] for o in out)
lines = [f"# Which layer decided the answer: direct push vs total effect ({len(out)} study chats, answer token vs runner-up)", "",
         "Direct push = what Chat & trace highlights. Total effect = how much the answer's lead over the runner-up drops "
         "when the part is switched off (output -> its average over the text) and everything after it is re-run.", "",
         f"Same top part by both: **{same}/{len(out)}** chats", "",
         f"Top part by direct push: {', '.join(f'{k} ×{v}' for k, v in cd.most_common(6))}",
         f"Top part by total effect: {', '.join(f'{k} ×{v}' for k, v in ct.most_common(8))}",
         f"Answer leads (logit lens) from layer: {', '.join(f'L{k} ×{v}' for k, v in sorted(fl.items(), key=lambda x: (x[0] is None, x[0] or 0)))}", "",
         "| Layer | Attention direct | Attention total | MLP direct | MLP total |", "|---|---|---|---|---|"]
for L in range(nL):
    lines.append(f"| {L} | {D['attn'][:, L].mean():+.2f} | {T['attn'][:, L].mean():+.2f} | {D['mlp'][:, L].mean():+.2f} | {T['mlp'][:, L].mean():+.2f} |")
lines += ["", "| Chat | Answer over runner-up | App highlights (direct) | Biggest total effects | Leads from |", "|---|---|---|---|---|"]
for o in out:
    lines.append(f"| {o['id']} | {o['token']!r} over {o['runner_up']!r} ({o['lead']:.1f}) | {o['top_direct'][0]} L{o['top_direct'][1]} | "
                 + ", ".join(f"{k} L{L} ({v:+.1f})" for k, L, v in o["top3_total"]) + f" | L{o['flip']} |")
(HERE / "layer_effects.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
