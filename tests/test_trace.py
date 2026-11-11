"""
Checks the chat tracer against llama.cpp (LM Studio's llama-server), on a real workspace.
  python test_trace.py [workspace]          (default: Qwen3.5-0.8B-Q8_0)

1. tokenizer: same token ids as llama.cpp for many kinds of text
2. chat template: same prompt text as llama.cpp's /apply-template
3. greedy replies: same tokens as llama.cpp, and close next-token probabilities
4. attribution: the pushes of all parts add up to the target token's score
5. edited weights: tracing the workspace's edits behaves like the exported edited model (if one exists)
"""
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core
import inspect_core as ic
import trace_core as tc

WS = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-0.8B-Q8_0"
PORT = 8140
ok = fail = 0


def check(cond, msg):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode())
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def start_server(model):
    log = tempfile.TemporaryFile()
    p = subprocess.Popen([str(ic.find_llama_server()), "-m", str(model), "--port", str(PORT), "-c", "4096", "-ngl", "0",
                          "--no-warmup", "-np", "1", "-t", "6"], stdout=subprocess.DEVNULL, stderr=log,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return p, log
        except Exception:
            if p.poll() is not None:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace")[-2000:])
            time.sleep(0.4)
    raise RuntimeError("llama-server didn't start")


def compare(model, tok, ids, N):
    """Greedy reply of both engines. Returns (tokens agreeing, compared, divergence is a near-tie, prob diffs, mine).
    llama.cpp rounds activations to 8 bits inside Q8_0 matmuls; this tracer computes in float32. So when two tokens
    are almost tied, the order can flip; a real bug would show as a divergence where one engine is confident."""
    theirs, tprobs = server_greedy(ids, N)
    all_ids, npr, rec, timing = tc.run(model, tok, ids, max_new=N, temperature=0, log=lambda m: None)
    mine = all_ids[npr:]
    top_ids, top_lp = np.concatenate(rec.cols["top_ids"]), np.concatenate(rec.cols["top_lp"])
    same = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
    diffs, tie = [], True
    for k in range(min(same + 1, len(tprobs))):
        row = npr - 1 + k
        mp = {int(i): float(np.exp(l)) for i, l in zip(top_ids[row], top_lp[row])}
        diffs += [abs(mp[t] - p) for t, p in tprobs[k].items() if t in mp]
        if k == same and same < min(len(mine), len(theirs)):     # the divergence point
            a, b = mine[k], theirs[k]
            tie = abs(mp.get(a, 0) - mp.get(b, 0)) < 0.1 and abs(tprobs[k].get(a, 0) - tprobs[k].get(b, 0)) < 0.1
    return same, min(len(mine), len(theirs)), tie, diffs, mine, timing


def server_greedy(ids, n):
    r = post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "n_probs": 5,
                             "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"]})
    toks = r.get("tokens") or [t["id"] for t in r["completion_probabilities"]]
    probs = []
    for c in r["completion_probabilities"]:
        tops = c.get("top_logprobs") or c.get("probs") or []
        probs.append({t["id"]: float(np.exp(t["logprob"])) if "logprob" in t else t["prob"] for t in tops})
    return toks, probs


d = core.workspace_dir(WS)
info = core.load_info(d)
t0 = time.time()
model, tok = tc.get_model(WS, "original", log=lambda m: None)
print(f"loaded original weights in {time.time() - t0:.0f} s")
srv, srvlog = start_server(info["gguf_path"])
try:
    # 1. tokenizer
    texts = ["Hello world", "  two  spaces and\ttab\n\nnew lines\r\n", "What is 17*23? Answer: 391.", "1234567890 3.14159",
             "don't I'll they're WE'VE", "def f(x):\n    return x**2  # square", "नमस्ते दुनिया, आप कैसे हैं?",
             "日本語のテキスト、中文文本。", "emoji 😀👍🏽 and ümlauts café", "<|im_start|>user\nhi<|im_end|>",
             "naïve façade — “quotes” …", "x" * 50 + " " + "abc" * 20, "   leading and trailing   ", "\n\n\n", "á è"]
    bad = []
    for s in texts:
        mine = tok.encode(s)
        theirs = post("/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"]
        theirs = [t["id"] if isinstance(t, dict) else t for t in theirs]
        if mine != theirs:
            bad.append((s, mine[:20], theirs[:20]))
    check(not bad, f"tokenizer matches llama.cpp on {len(texts) - len(bad)}/{len(texts)} texts" + (f": {bad[:2]}" if bad else ""))
    check(all(tok.decode(tok.encode(s)) == s for s in texts), "decode(encode(text)) gives the text back")

    # 2. chat template
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    mine = tc.render_chat(tc._MODEL["meta"], msgs, thinking=False)
    try:
        theirs = post("/apply-template", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        check(mine == theirs, "chat template renders the same prompt as llama.cpp" + ("" if mine == theirs else f": {mine!r} vs {theirs!r}"))
    except Exception as e:
        print("skip: /apply-template not available:", e)

    # 3. greedy replies
    prompts = ["What is 17 * 23? Answer with just the number.", "Write one sentence about the sea.",
               "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
               "How many clips did Natalia sell altogether in April and May?"]
    agree_all, diffs, speeds = [], [], []
    for q in prompts:
        ids = tok.encode(tc.render_chat(tc._MODEL["meta"], [{"role": "user", "content": q}], False))
        same, n, tie, dd, mine, timing = compare(model, tok, ids, 48)
        agree_all.append((same, n, tie))
        diffs += dd
        speeds.append(timing["tok_per_s"])
        print(f"   {q[:40]!r}: same first {same} of {n} tokens{'' if same == n else ' (then a near-tie)' if tie else ' (then a CONFIDENT difference)'}; "
              f"mine: {tok.decode(mine)[:70]!r}")
    check(all(s == n or t for s, n, t in agree_all) and sum(s for s, _, _ in agree_all) >= 60,
          f"greedy replies match llama.cpp token for token until a near-tie (agreement: {', '.join(f'{s}/{n}' for s, n, _ in agree_all)})")
    check(np.mean(diffs) < 0.01 and max(diffs) < 0.07, f"next-token probabilities match llama.cpp (mean difference "
          f"{np.mean(diffs) * 100:.2f} points, max {max(diffs) * 100:.2f}, over {len(diffs)} values)")
    print(f"   speed: {', '.join(map(str, speeds))} tokens/s")

    # 4. attribution adds up, on a saved trace
    res = tc.chat(WS, [{"role": "user", "content": "What is 2 + 2?"}], max_new=12, temperature=0, which="original",
                  log=lambda m: None)
    tid = res["id"]
    _, A = tc._open_trace(WS, tid)
    worst = 0
    for pos in range(res["n_prompt"] - 1, res["n_positions"]):
        det = tc.position_detail(WS, tid, pos, log=lambda m: None)
        xf = tc._rms(A["resid"][pos][-1].astype(np.float32), model.out_norm, model.eps)
        true_logit = float(xf @ model.U[det["target"]["id"]])
        worst = max(worst, abs(det["target_logit"] - true_logit) / max(1.0, abs(true_logit)))
    check(worst < 0.01, f"pushes of all parts add up to the token's score (worst relative error {worst:.4f})")
    det = tc.position_detail(WS, tid, res["n_prompt"] - 1, log=lambda m: None)
    check(det["lens"][-1]["top"][0]["id"] == det["top"][0]["id"], "logit lens after the last layer = the model's own prediction")
    check(len(det["top_up"]) == 20 and det["top_up"][0]["links"]["down"][0].endswith("ffn_down.weight"),
          "top neurons point to their ffn_down / ffn_up / ffn_gate weights")
    print("   reply:", repr(res["reply"]), "| strongest neuron:", det["top_up"][0]["layer"], det["top_up"][0]["neuron"])
    tc.delete_trace(WS, tid)
finally:
    srv.kill()
    srv.wait()
    srvlog.close()

# 5. edited weights vs the exported edited model
exp = sorted((d / "exports").glob("big_laser23_k25.gguf"))
if exp and (d / "edit_weights.py").read_text(encoding="utf-8").count("low_rank(w[n], 0.25)"):
    model, tok = tc.get_model(WS, "edited", log=lambda m: None)
    srv, srvlog = start_server(exp[0])
    try:
        ids = tok.encode(tc.render_chat(tc._MODEL["meta"], [{"role": "user", "content": prompts[2]}], False))
        same, n, tie, dd, _, _ = compare(model, tok, ids, 32)
        check((same == n or tie) and np.mean(dd) < 0.01, f"workspace edits (edit_weights.py) traced = exported edited model "
              f"({same}/{n} tokens agree{', then a near-tie' if same < n else ''}; mean probability difference {np.mean(dd) * 100:.2f} points)")
    finally:
        srv.kill()
        srv.wait()
        srvlog.close()
else:
    print("skip: no matching exported edited model")

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
