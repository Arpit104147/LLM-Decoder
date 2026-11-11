"""
Checks every saved chat trace of the 100-question study: is what the Chat & trace view highlights correct?
  python verify_traces.py [results_name] [--no-llama]

For every chat and every position it checks:
  recompute   run the same tokens again in one fresh pass (prefill, not the token-by-token path the chat used) and
              compare neuron activations, head outputs, layer outputs and token probabilities with the saved trace
  layers      layer activity (attn_norm / mlp_norm, the "Layer activity" map) = size of the saved layer outputs
  residual    hidden state after layer L = before + mixer output + MLP output (nothing lost between layers)
  neurons     MLP output of each layer = sum over neurons of activation × that neuron's ffn_down column,
              so the highlighted neuron really is the column the grid buttons open
  heads       mixer output = sum over heads of head output × its block of attn_output / ssm_out columns
  pushes      per-neuron pushes add up to each layer's MLP push, per-head pushes to its attention push, and
              embedding + all layers = the token's real score (the "Where the push came from" table)
  lens        the last logit-lens row = the model's real top guess
  tokens      each token's colour/probability in the conversation belongs to that token (no off-by-one)
  attention   highlighted attention only points at the current or earlier tokens, weights between 0 and 1
  links       every weight-grid button points inside its tensor
  llama.cpp   llama.cpp's own probabilities for the same chat agree with the trace (independent program)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import inspect_core as ic
import trace_core as tc

WS = "Qwen3.5-0.8B-Q8_0"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NAME = args[0] if args else "original"
LLAMA = "--no-llama" not in sys.argv


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))


def check_links(model):
    shapes = {t["name"]: t["matrix"] for t in json.loads((tc.core.workspace_dir(WS) / "tensors.json").read_text(encoding="utf-8"))}
    bad = []

    def dims(name):        # (rows, cols) as the weight grid shows the tensor
        s = shapes.get(name)
        return None if s is None else (int(s[0]), int(s[1]))

    F = model.ff
    for L in range(model.n_layer):
        ref = tc.neuron_ref({}, None, 0, L, F - 1)
        for k, x in ref["links"].items():
            d = dims(x[0])
            if d is None:
                bad.append(f"L{L} {k}: no tensor {x[0]}")
            elif (x[1] == "col" and x[2] >= d[1]) or (x[1] == "row" and x[2] >= d[0]):
                bad.append(f"L{L} {k}: {x[1]} {x[2]} outside {x[0]} {d}")
        hl = model.head_layout(L)
        if hl:
            H = model.n_heads(L)
            d = dims(hl["out"])
            if d is None or H * hl["out_cols"] > d[1]:
                bad.append(f"L{L} head out {hl['out']} {d} < {H}×{hl['out_cols']}")
            for r in hl["reads"]:
                d = dims(r[0])
                last = (r[5] if len(r) > 5 else 0) + ((H - 1) // r[3]) * r[2] + r[2]
                if d is None or last > d[0]:
                    bad.append(f"L{L} head read {r[4]} rows up to {last} > {d}")
    return bad


def recompute(model, ids):
    st, rec = tc.State(model.n_layer), tc.Recorder()
    lgs = []
    for s0 in range(0, len(ids), 64):
        lgs.append(model.logits(model.forward(ids[s0:s0 + 64], st, rec)))
    return {k: np.concatenate(v) for k, v in rec.cols.items()}, np.concatenate(lgs)


def check_chat(model, row, v):
    info, A = tc._open_trace(WS, row["trace"])
    ids = info["ids"]
    n = A["act"].shape[0]
    r = {"id": row["id"], "positions": n}
    assert n == len(ids) - 1, "positions != tokens - 1"
    B, lg = recompute(model, ids[:-1])
    lse = lg.max(-1) + np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1))
    lp = lg[np.arange(n), ids[1:]] - lse
    # 1. the saved trace vs a fresh pass
    r["re_act"] = rel(A["act"], B["act"])
    r["re_heads"] = rel(A["heads"], B["heads"])
    r["re_layer_norms"] = max(rel(A["attn_norm"], B["attn_norm"]), rel(A["mlp_norm"], B["mlp_norm"]))
    r["re_prob"] = float(np.abs(np.exp(A["next_lp"]) - np.exp(lp)).max())
    r["re_top1_same"] = float((A["top_ids"][:, 0] == lg.argmax(-1)).mean())
    # 2. layer activity map = size of the saved layer outputs
    av, mv = A["a_vec"].astype(np.float32), A["m_vec"].astype(np.float32)
    r["layers"] = max(rel(np.linalg.norm(av, axis=-1), A["attn_norm"]), rel(np.linalg.norm(mv, axis=-1), A["mlp_norm"]))
    # 3. residual stream adds up
    res = A["resid"].astype(np.float32)
    r["residual"] = rel(res[:, 1:] - res[:, :-1], av + mv)
    # 4. neurons x ffn_down = MLP output ; 5. heads x out_proj = mixer output
    nm, hm = 0.0, 0.0
    for L in range(model.n_layer):
        down = model.mlp_down(L)
        nm = max(nm, rel(A["act"][:, L, :down.shape[1]].astype(np.float32) @ down.T, mv[:, L]))
        op, H, dh = model.out_proj(L), model.n_heads(L), model.head_dim(L)
        hm = max(hm, rel(A["heads"][:, L, :H * dh].astype(np.float32) @ op.T, av[:, L]))
    r["neurons"], r["heads"] = nm, hm
    # 6. attribution adds up, at every position, for the real next token (what the table shows by default)
    pe = te = 0.0
    for pos in range(n):
        t = ids[pos + 1]
        att = tc.attribute(model, A, pos, t)
        sn = att["neuron_push"].sum(1)
        sh = np.array([h.sum() for h in att["head_push"]])
        sc = np.abs(att["layer_mlp"]).max() + np.abs(att["layer_attn"]).max() + 1e-6
        pe = max(pe, float(max(np.abs(sn - att["layer_mlp"]).max(), np.abs(sh - att["layer_attn"]).max()) / sc))
        te = max(te, abs(att["logit"] - float(lg[pos, t])) / (abs(float(lg[pos, t])) + 1))
    r["pushes"], r["push_total"] = pe, te
    # 7. logit lens: last row = real top guess (at a few positions)
    bad = 0
    for pos in np.linspace(0, n - 1, min(n, 8)).astype(int):
        ll = model.logits(model.final(A["resid"][pos].astype(np.float32)))
        bad += int(ll[-1].argmax() != A["top_ids"][pos][0] and lg[pos].max() - np.sort(lg[pos])[-2] > 0.05)
    r["lens_bad"] = bad
    # 8. tokens in the conversation: text, probability and top guess belong to the right token
    s = tc.summary(WS, row["trace"])
    tb = 0
    for i, k in enumerate(s["tokens"]):
        tb += k["id"] != ids[i] or k["text"] != v["texts"][ids[i]]
        if i >= 1:
            tb += abs(k["p"] - float(np.exp(lp[i - 1]))) > 2e-3 or k["top1"] != int(A["top_ids"][i - 1][0])
    tb += len(s["tokens"]) != len(ids) or s["reply"] != info["reply"]
    rng = np.random.default_rng(len(ids))
    for _ in range(3):                              # neuron highlight = that neuron's activation at each token
        L, j = int(rng.integers(model.n_layer)), int(rng.integers(model.ff))
        nv = tc.neuron_over_trace(WS, row["trace"], L, j)["act"]
        tb += len(nv) != n or np.abs(np.array(nv) - A["act"][:, L, j].astype(np.float32)).max() > 1e-3
    r["tokens_bad"] = int(tb)
    # 9. attention highlights: never a later token, weights in [0, 1], match the fresh pass
    ab, am = 0, 0.0
    for L in range(model.n_layer):
        if f"attn{L}_i" not in A:
            continue
        I, W = A[f"attn{L}_i"], A[f"attn{L}_w"].astype(np.float32)
        P = np.arange(n)[:, None, None]
        ab += int(((I > P) & (W > 0)).sum() + (W < 0).sum() + (W > 1.001).sum() + (W.sum(-1) > 1.01).sum())
        am = max(am, float(np.abs(W[..., 0] - B[f"attn{L}_w"].astype(np.float32)[..., 0]).max()))
    r["attention_bad"], r["re_attention"] = ab, am
    tc._TRACES.clear()
    return r


def llama_check(rows, path):
    """llama.cpp follows each chat with the trace's own tokens (teacher-forced at each divergence)."""
    import test_trace_arch as T
    out = {}
    with T.Server(path, ctx=4096):
        for row in rows:
            info, A = tc._open_trace(WS, row["trace"])
            ids, npr = info["ids"], info["n_prompt"]
            diffs, same, steps, k = [], 0, 0, npr
            while k < len(ids):
                r = T.post("/completion", {"prompt": ids[:k], "n_predict": len(ids) - k, "temperature": 0, "top_k": 1,
                                           "samplers": ["top_k"], "n_probs": 5, "cache_prompt": True, "return_tokens": True,
                                           "ignore_eos": True})
                cps = r["completion_probabilities"]
                moved = 0
                for c in cps:
                    pos = k - 1                                  # the trace position that predicted ids[k]
                    ours = {int(i): float(np.exp(l)) for i, l in zip(A["top_ids"][pos], A["top_lp"][pos])}
                    for t in c["top_logprobs"]:
                        if t["id"] in ours:
                            diffs.append(abs(ours[t["id"]] - float(np.exp(t["logprob"]))))
                    steps += 1
                    same += c["id"] == ids[k]
                    moved += 1
                    k += 1
                    if c["id"] != ids[k - 1] or k >= len(ids):
                        break
                if not moved:
                    break
            out[row["id"]] = {"steps": steps, "same_token": same, "prob_diff_mean": float(np.mean(diffs)) if diffs else None,
                              "prob_diff_max": float(np.max(diffs)) if diffs else None}
            tc._TRACES.clear()
            print(f"  llama.cpp {row['id']}: same token {same}/{steps}, prob diff mean {out[row['id']]['prob_diff_mean'] * 100:.2f} pts", flush=True)
    return out


def main():
    """--shard k/N checks every N-th chat (run N at once); --merge joins the shards and runs the llama.cpp check."""
    t0 = time.time()
    rows = [json.loads(l) for l in (HERE / f"results_{NAME}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), None)
    if "--merge" in sys.argv:
        parts = sorted(HERE.glob(f"verify_{NAME}.shard*.json"))
        reps = [json.loads(p.read_text(encoding="utf-8")) for p in parts]
        order = {r["id"]: i for i, r in enumerate(rows)}
        rep = {"links_bad": reps[0]["links_bad"], "chats": sorted([c for r in reps for c in r["chats"]], key=lambda c: order[c["id"]])}
        if LLAMA:
            rep["llama"] = llama_check(rows, json.loads((tc.core.workspace_dir(WS) / "source.json").read_text(encoding="utf-8"))["gguf_path"])
        (HERE / f"verify_{NAME}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"merged {len(rep['chats'])} chats from {len(parts)} shards in {time.time() - t0:.0f} s")
        return
    model, tok = tc.get_model(WS, "original" if NAME == "original" else "edited", log=lambda m: None)
    v = ic.vocab(WS)
    k, N = (int(x) for x in shard.split("/")) if shard else (0, 1)
    rep = {"links_bad": check_links(model), "chats": []}
    print("links:", rep["links_bad"] or "all inside their tensors", flush=True)
    mine = rows[k::N]
    for i, row in enumerate(mine, 1):
        r = check_chat(model, row, v)
        rep["chats"].append(r)
        print(f"[{i:3d}/{len(mine)}] {r['id']} n={r['positions']:3d} act {r['re_act']:.1e} prob {r['re_prob']:.1e} "
              f"resid {r['residual']:.1e} neurons {r['neurons']:.1e} heads {r['heads']:.1e} pushes {r['pushes']:.1e} "
              f"total {r['push_total']:.1e} tokens_bad {r['tokens_bad']} attn_bad {r['attention_bad']}", flush=True)
    rep["seconds"] = round(time.time() - t0)
    out = f"verify_{NAME}.shard{k}.json" if shard else f"verify_{NAME}.json"
    (HERE / out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if not shard and LLAMA:
        rep["llama"] = llama_check(rows, json.loads((tc.core.workspace_dir(WS) / "source.json").read_text(encoding="utf-8"))["gguf_path"])
        (HERE / out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"done in {rep['seconds']} s")


if __name__ == "__main__":
    main()
