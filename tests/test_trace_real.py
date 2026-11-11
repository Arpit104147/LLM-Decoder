"""
Checks the chat tracer against llama.cpp on any real .gguf model file (no workspace needed):
  python test_trace_real.py <model.gguf> [more.gguf ...]

For a few prompts it compares, step by step along llama.cpp's own greedy reply: the tokenizer, the top-1 token
and the probabilities of llama.cpp's top-5 candidates. Real files are quantized (Q8_0, Q4_K…): llama.cpp then
rounds activations to 8 bits inside its matrix multiplications while the tracer computes in float32, so
expect probabilities within about a percentage point and the same tokens until two candidates are nearly tied.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import test_trace_arch as T
import trace_core as tc

PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Write a Python function that returns the square of a number.",
           "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
           "How many clips did Natalia sell altogether in April and May?"]


def compare_file(path, n_gen=24, patch=None, quiet=False):
    """Returns (ok, summary). patch(model) may change the model first (used to try variants)."""
    t0 = time.time()
    model, tok, meta = tc.load_file(path)
    if patch:
        patch(model)
    load_s = time.time() - t0
    rows, tok_bad = [], 0
    with T.Server(path):
        for q in PROMPTS:
            prompt = tc.render_chat(meta, [{"role": "user", "content": q}], thinking=False)
            ids = tc.prompt_ids(tok, prompt)
            theirs_tok = T.post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
            theirs_tok = [t["id"] if isinstance(t, dict) else t for t in theirs_tok]
            if tok.bos is not None and (not theirs_tok or theirs_tok[0] != tok.bos):
                theirs_tok = [tok.bos] + theirs_tok
            tok_bad += ids != theirs_tok
            r = T.post("/completion", {"prompt": ids, "n_predict": n_gen, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                                       "n_probs": 5, "cache_prompt": False, "return_tokens": True})
            theirs = r.get("tokens") or []
            cps = r["completion_probabilities"]
            st, rec = tc.State(model.n_layer), tc.Recorder()
            lg = model.logits(model.forward(ids, st, rec))[-1]
            same, diffs = None, []
            for k, y in enumerate(theirs):
                lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
                if k < len(cps) and cps[k].get("id") == y:
                    diffs += [abs(float(np.exp(lp[t["id"]])) - float(np.exp(t["logprob"]))) for t in cps[k]["top_logprobs"]]
                if same is None and int(np.argmax(lg)) != y:
                    same = k
                lg = model.logits(model.forward([y], st, rec))[-1]          # follow llama.cpp's path
            same = len(theirs) if same is None else same
            rows.append((q[:38], same, len(theirs), max(diffs) if diffs else 0, float(np.mean(diffs)) if diffs else 0,
                         tok.decode(theirs)[:50]))
    mean = float(np.mean([r[4] for r in rows]))
    ok = tok_bad == 0 and mean < 0.01 and all(r[1] >= min(8, r[2]) for r in rows)
    if not quiet:
        print(f"\n{Path(path).name}  ({meta.get('general.architecture')}, loaded in {load_s:.0f} s)")
        print(f"  tokenizer identical on {len(PROMPTS) - tok_bad}/{len(PROMPTS)} prompts")
        for q, s, n, mx, mn, txt in rows:
            print(f"  {q!r:42s} top-1 same for {s:2d}/{n} steps · probability diff mean {mn * 100:.2f} max {mx * 100:.2f} points · {txt!r}")
    return ok, mean


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        ok, _ = compare_file(p)
        print("  PASS" if ok else "  FAIL")
        bad += not ok
    sys.exit(1 if bad else 0)
