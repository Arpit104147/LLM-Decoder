"""
Causal weight-edit tests on the 100 study chats (Qwen3.5-0.8B, original weights).

Each variant edits weights the study pointed at, then replays every saved chat (prompt + the model's own reply,
teacher-forced) and measures how the reply tokens' scores move:
  - a head ×k   = its columns of attn_output (full attention) or ssm_out (linear attention) multiplied by k
  - a neuron ×k = its column of ffn_down multiplied by k   (k = 0 removes it, k = 2 doubles what it writes)
These are real weight edits: the same change exported as a GGUF gives the same model in llama.cpp.

  python studies/qwen35-0.8b/edit_tests.py [--shard=k/N] [--merge]
Writes edit_tests.json (+ .md). Layers below the earliest edited layer are computed once per chat and reused.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import trace_core as tc

WS = "Qwen3.5-0.8B-Q8_0"
CHUNK = 64

# name -> list of (kind, layer, index, factor)
VARIANTS = {
    "L23 head 6 ×0": [("head", 23, 6, 0.0)],
    "L23 head 6 ×2": [("head", 23, 6, 2.0)],
    "L23 heads 2,4,5,6 ×0": [("head", 23, h, 0.0) for h in (2, 4, 5, 6)],
    "L21 head 15 ×0": [("head", 21, 15, 0.0)],
    "L21 head 15 ×2": [("head", 21, 15, 2.0)],
    "L23 #9 ×0": [("neuron", 23, 9, 0.0)],
    "L23 #9 ×2": [("neuron", 23, 9, 2.0)],
    "L19 #0 ×0": [("neuron", 19, 0, 0.0)],
    "L19 #0 ×2": [("neuron", 19, 0, 2.0)],
    "L21 #0 ×0": [("neuron", 21, 0, 0.0)],
    "L23 #9 + L19 #0 + L21 #0 ×0": [("neuron", 23, 9, 0.0), ("neuron", 19, 0, 0.0), ("neuron", 21, 0, 0.0)],
    "math: L23 #223 ×0": [("neuron", 23, 223, 0.0)],
    "math: L23 #223 ×2": [("neuron", 23, 223, 2.0)],
    "code: L17 #603 ×0": [("neuron", 17, 603, 0.0)],
    "code: L23 #165 ×0": [("neuron", 23, 165, 0.0)],
    "control: L23 head 0 ×0": [("head", 23, 0, 0.0)],
    "control: L23 head 1 ×0": [("head", 23, 1, 0.0)],
    "control: L23 #1000 ×0": [("neuron", 23, 1000, 0.0)],
    "control: L23 #2000 ×0": [("neuron", 23, 2000, 0.0)],
    "control: L19 #1500 ×0": [("neuron", 19, 1500, 0.0)],
}


def tensor_cols(model, kind, L, i):
    """(tensor name, column slice) that a head / neuron writes through."""
    if kind == "neuron":
        return f"blk.{L}.ffn_down.weight", slice(i, i + 1)
    d = model.head_dim(L)
    name = f"blk.{L}.attn_output.weight" if model.full[L] else f"blk.{L}.ssm_out.weight"
    return name, slice(i * d, (i + 1) * d)


class Edited:
    """Context manager: scale the columns in place, put the exact originals back afterwards."""
    def __init__(self, model, edits):
        self.model, self.edits, self.saved = model, edits, []

    def __enter__(self):
        for kind, L, i, k in self.edits:
            name, cols = tensor_cols(self.model, kind, L, i)
            W = self.model.T[name]
            self.saved.append((W, cols, W[:, cols].copy()))
            W[:, cols] *= np.float32(k)

    def __exit__(self, *a):
        for W, cols, orig in reversed(self.saved):
            W[:, cols] = orig


def run_from(model, xin, start, n, reply_pos):
    """Layers start..end over the chat (chunked, fresh state) -> log-probs at reply positions [r, V]."""
    st, rec = tc.State(model.n_layer), tc.Recorder()
    outs = []
    for s0 in range(0, n, CHUNK):
        x = xin[start][s0:s0 + CHUNK]
        pos = np.arange(s0, s0 + len(x))
        for L in range(start, model.n_layer):
            x = model.block(L, x, pos, st, rec)[0].astype(np.float32, copy=False)
        outs.append(x)
    x = np.concatenate(outs)[reply_pos]
    lg = model.logits(model.final(x)).astype(np.float32)
    m = lg.max(-1, keepdims=True)
    return lg - (m + np.log(np.exp(lg - m).sum(-1, keepdims=True)))


def base_pass(model, ids, keep_from):
    """Original model over the chat; returns the input of every layer >= keep_from ([n, D] each)."""
    n = len(ids)
    st, rec = tc.State(model.n_layer), tc.Recorder()
    xin = {L: [] for L in range(keep_from, model.n_layer)}
    for s0 in range(0, n, CHUNK):
        x = model.embed(ids[s0:s0 + CHUNK])
        pos = np.arange(s0, s0 + len(x))
        for L in range(model.n_layer):
            if L >= keep_from:
                xin[L].append(x)
            x = model.block(L, x, pos, st, rec)[0].astype(np.float32, copy=False)
    return {L: np.concatenate(v) for L, v in xin.items()}


def main():
    shard = next((a.split("=")[1] for a in sys.argv if a.startswith("--shard=")), None)
    if "--merge" in sys.argv:
        return merge()
    rows = [json.loads(l) for l in (HERE / "results_original.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tag = ""
    if shard:
        k, N = map(int, shard.split("/"))
        rows, tag = rows[k::N], f".shard{k}"
    out = HERE / f"edit_tests{tag}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    model, _tok = tc.get_model(WS, "original", log=lambda m: None)
    assert isinstance(model, tc.Qwen35)
    for kind_L_i_k in [e for v in VARIANTS.values() for e in v]:        # every column actually exists
        name, cols = tensor_cols(model, *kind_L_i_k[:3])
        assert cols.stop <= model.T[name].shape[1], (name, cols)
    keep_from = min(e[1] for v in VARIANTS.values() for e in v)
    t0 = time.time()
    for n_done, row in enumerate(rows, 1):
        if row["id"] in res:
            continue
        info, _A = tc._open_trace(WS, row["trace"])
        ids, n_prompt = info["ids"], info["n_prompt"]
        n = len(ids) - 1                                   # position p predicts token p+1
        reply_pos = np.arange(n_prompt - 1, n)
        target = np.asarray(ids[n_prompt:])
        xin = base_pass(model, ids[:-1], keep_from)
        base = run_from(model, xin, keep_from, n, reply_pos)
        b_lp = base[np.arange(len(target)), target]
        b_top = base.argmax(-1)
        pb = np.exp(base)
        r = {"cat": row["cat"], "correct": row["correct"], "n_reply": int(len(target)),
             "base_top1_is_reply": float((b_top == target).mean()), "base_first_p": float(np.exp(b_lp[0])), "v": {}}
        for vname, edits in VARIANTS.items():
            start = min(e[1] for e in edits)
            with Edited(model, edits):
                lp = run_from(model, xin, start, n, reply_pos)
            t_lp = lp[np.arange(len(target)), target]
            top = lp.argmax(-1)
            kl = (pb * (base - lp)).sum(-1)
            r["v"][vname] = {"dlp": float((t_lp - b_lp).mean()), "dlp_first": float(t_lp[0] - b_lp[0]),
                             "first_p": float(np.exp(t_lp[0])), "flip": float((top != b_top).mean()),
                             "first_flip": bool(top[0] != b_top[0]), "kl": float(kl.mean()),
                             "first_new": int(top[0]) if top[0] != b_top[0] else None}
        res[row["id"]] = r
        out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        el = time.time() - t0
        print(f"[{n_done:3d}/{len(rows)}] {row['id']} n={n} reply={len(target)} {el / 60:.1f} min", flush=True)
    if not shard:
        merge()


def merge():
    res = {}
    for p in sorted(HERE.glob("edit_tests.shard*.json")):
        res.update(json.loads(p.read_text(encoding="utf-8")))
    single = HERE / "edit_tests.json"
    if not res and single.exists():
        res = json.loads(single.read_text(encoding="utf-8"))
    if "summary" in res:
        res = res["chats"]
    model, tok = tc.get_model(WS, "original", log=lambda m: None)
    cats = ["math", "facts", "reasoning", "language", "code", "commonsense"]
    summ = {}
    for vname in VARIANTS:
        s = {}
        for c in cats + ["all"]:
            rs = [r for r in res.values() if c == "all" or r["cat"] == c]
            if not rs:
                continue
            v = [r["v"][vname] for r in rs]
            w = np.array([r["n_reply"] for r in rs], float)
            s[c] = {"chats": len(rs),
                    "dlp": float(np.average([x["dlp"] for x in v], weights=w)),
                    "first_p_change": float(np.mean([x["first_p"] - r["base_first_p"] for x, r in zip(v, rs)])),
                    "flip_pct": 100 * float(np.average([x["flip"] for x in v], weights=w)),
                    "first_flip": int(sum(x["first_flip"] for x in v)),
                    "kl": float(np.average([x["kl"] for x in v], weights=w))}
        s["first_new_tokens"] = [[k, tok.decode([x["first_new"]]) if hasattr(tok, "decode") else x["first_new"]]
                                 for k, r in res.items() for x in [r["v"][vname]] if x["first_new"] is not None][:12]
        summ[vname] = s
    (HERE / "edit_tests.json").write_text(json.dumps({"summary": summ, "chats": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Qwen3.5-0.8B — weight-edit tests on the {len(res)} study chats (teacher-forced replay)", "",
             "Δ log-prob = mean change in log-probability of the model's own original reply tokens (negative = the edit "
             "makes the original answer less likely). Changed top token = % of reply positions where the top guess "
             "changes. KL = mean divergence from the original next-token distribution.", "",
             "| Edit | Δ log-prob (all) | Changed top token | First reply token changed | KL | Δ log-prob math | facts | reasoning | language | code | commonsense |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for vname, s in summ.items():
        a = s["all"]
        lines.append(f"| {vname} | {a['dlp']:+.3f} | {a['flip_pct']:.1f}% | {a['first_flip']}/{a['chats']} | {a['kl']:.3f} | "
                     + " | ".join(f"{s[c]['dlp']:+.3f}" if c in s else "" for c in cats) + " |")
    (HERE / "edit_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
