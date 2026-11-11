"""
Benchmarks for weight-edited Qwen3.5-0.8B models, run in llama.cpp (CPU).

  python bench.py export "<variant>" ...        build exports/edit_<slug>.gguf for variants of edit_tests.py
  python bench.py run <name|original> [--port=8170] [--gsm=50]
        1. the 100 study questions, same prompt tokens as the traced chats, greedy, graded like run_chats.py
        2. GSM8K test problems 0..N-1, thinking off, greedy, 1024-token cap, answer in \\boxed{}
  python bench.py gsm <name> <a> <b> [--port=]  GSM8K problems a..b-1 only (ranges can run side by side)
  python bench.py compare <name1> <name2>       paired GSM8K comparison over every problem both have (McNemar)
  python bench.py report                        bench_results.md from every bench_<name>.json

Each run uses one llama.cpp slot and 4 threads, so it is deterministic and several runs can go side by side.
The workspace's own edit_weights.py is put back after each export.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import gguf_core as core
import inspect_core as ic

WS = "Qwen3.5-0.8B-Q8_0"
D = core.workspace_dir(WS)
GSM = HERE / "gsm8k_test_0_299_solutions.jsonl"      # the first 300 GSM8K test problems, with the worked solutions
HEAD_DIM = {"full": 256, "linear": 128}          # attention.value_length ; ssm.inner_size / ssm.time_step_rank
FULL = {3, 7, 11, 15, 19, 23}


def gsm_rows():
    return [json.loads(l) for l in GSM.read_text(encoding="utf-8").splitlines() if l.strip()]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("×", "x")).strip("_")


def edit_code(name, edits):
    lines = [f'"""Bench edit "{name}" (from studies/qwen35-0.8b/edit_tests.py)."""', "", "", "def edit(w):"]
    for kind, L, i, k in edits:
        if kind == "neuron":
            lines.append(f'    w["blk.{L}.ffn_down.weight"][:, {i}] *= {k}      # neuron {i} of layer {L}')
        else:
            d = HEAD_DIM["full" if L in FULL else "linear"]
            t = "attn_output" if L in FULL else "ssm_out"
            lines.append(f'    w["blk.{L}.{t}.weight"][:, {i * d}:{(i + 1) * d}] *= {k}      # head {i} of layer {L}')
    return "\n".join(lines) + "\n"


def export(names):
    from edit_tests import VARIANTS
    p = D / "edit_weights.py"
    backup = p.read_text(encoding="utf-8")
    try:
        for name in names:
            out = D / "exports" / f"edit_{slug(name)}.gguf"
            p.write_text(edit_code(name, VARIANTS[name]), encoding="utf-8")
            t = time.time()
            print(f"{name} -> {out.name}", flush=True)
            keep = ("Metadata", "Script edit", "Grid", "grid", "Weights you changed", "Values that differ", "Encoding")
            core.export_gguf(WS, out_path=out, log=lambda m: print("   ", m, flush=True) if m.startswith(keep) else None)
            print(f"    {time.time() - t:.0f}s", flush=True)
    finally:
        p.write_text(backup, encoding="utf-8")


class Server:
    def __init__(self, model, port):
        self.port = port
        self.logf = open(HERE / f"bench_server_{port}.log", "wb")
        self.p = subprocess.Popen([str(ic.find_llama_server()), "-m", str(model), "--port", str(port), "-c", "4096", "-ngl", "0",
                                   "--device", "none", "--no-warmup", "-np", "1", "-t", "4"],
                                  stdout=self.logf, stderr=self.logf, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                return
            except Exception:
                if self.p.poll() is not None:
                    raise SystemExit("llama-server didn't start; see " + self.logf.name)
                time.sleep(0.5)

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers={"Content-Type": "application/json"},
                                     data=json.dumps(body).encode())
        return json.loads(urllib.request.urlopen(req, timeout=1800).read())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.p.kill()
        self.p.wait()
        self.logf.close()


def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", "").replace("$", ""))
    return float(m[-1]) if m else None


def boxed(text):
    hits = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text or "")
    return num(hits[-1]) if hits else None


def run(name, port, n_gsm):
    sys.argv = [sys.argv[0], "original"]                    # run_chats reads argv at import
    import run_chats as rc
    import trace_core as tc
    model = model_path(name)
    out = HERE / f"bench_{name}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"study": {}, "gsm8k": {}}
    items = {q["id"]: q for q in json.loads((HERE / "questions.json").read_text(encoding="utf-8"))}
    rows = [json.loads(l) for l in (HERE / "results_original.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cap = HERE / "bench_gsm_cap.txt"                         # lowers --gsm for runs not started yet
    if cap.exists():
        n_gsm = min(n_gsm, int(cap.read_text().strip()))
    gsm = gsm_rows()[:n_gsm]
    t0 = time.time()
    with Server(model, port) as s:
        for row in rows:
            if row["id"] in res["study"]:
                continue
            item = items[row["id"]]
            info = json.loads((tc.traces_dir(WS) / row["trace"] / "trace.json").read_text(encoding="utf-8"))
            r = s.post("/completion", {"prompt": info["ids"][:info["n_prompt"]], "n_predict": 300 if item["cat"] == "math" else 80,
                                       "temperature": 0, "top_k": 1, "samplers": ["top_k"], "cache_prompt": False})
            ok, got = rc.grade(item, r["content"])
            res["study"][row["id"]] = {"cat": item["cat"], "correct": ok, "got": got, "reply": r["content"],
                                       "same_as_trace": r["content"].strip() == row["reply"].strip()}
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        st = res["study"]
        print(f"{name}: study {sum(x['correct'] for x in st.values())}/{len(st)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        gsm_loop(s, name, gsm, 0, res, out, t0)
    print(f"{name}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


def gsm_one(s, g):
    prompt = g["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    r = s.post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1024,
                                        "temperature": 0, "top_k": 1, "cache_prompt": False,
                                        "chat_template_kwargs": {"enable_thinking": False}})
    ch = r["choices"][0]
    text = ch["message"].get("content") or ""
    gold = num(g["answer"].split("####")[-1])
    pred = boxed(text)
    if pred is None and ch.get("finish_reason") != "length":
        pred = num(text)                                  # a finished answer without \boxed{}: its last number
    return {"gold": gold, "pred": pred, "correct": pred is not None and abs(pred - gold) < 1e-6,
            "capped": ch.get("finish_reason") == "length", "tokens": r["usage"]["completion_tokens"], "tail": text[-300:]}


def gsm_loop(s, name, rows, start, res, out, t0):
    for j, g in enumerate(rows):
        i = start + j
        if str(i) in res["gsm8k"]:
            continue
        res["gsm8k"][str(i)] = gsm_one(s, g)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        if j % 10 == 9:
            gs = res["gsm8k"]
            print(f"{name}: gsm8k {sum(x['correct'] for x in gs.values())}/{len(gs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def model_path(name):
    return core.load_info(D)["gguf_path"] if name == "original" else D / "exports" / f"edit_{name}.gguf"


def gsm_range(name, a, b, port):
    """GSM8K problems a..b-1 only, into bench_<name>.gsm<a>-<b>.json (several ranges can run side by side)."""
    out = HERE / f"bench_{name}.gsm{a}-{b}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"gsm8k": {}}
    rows = gsm_rows()[a:b]
    t0 = time.time()
    with Server(model_path(name), port) as s:
        gsm_loop(s, f"{name} [{a}-{b})", rows, a, res, out, t0)
    print(f"{name} [{a}-{b}): done in {(time.time() - t0) / 60:.1f} min", flush=True)


def gsm_all(name):
    """Every GSM8K problem solved by a model: its main run plus all range runs."""
    g = {}
    for p in [HERE / f"bench_{name}.json"] + sorted(HERE.glob(f"bench_{name}.gsm*.json")):
        if p.exists():
            g.update(json.loads(p.read_text(encoding="utf-8"))["gsm8k"])
    return g


def compare(a, b):
    """Paired GSM8K comparison on the problems both models solved (exact McNemar test)."""
    from math import comb
    ga, gb = gsm_all(a), gsm_all(b)
    ks = sorted(set(ga) & set(gb), key=int)
    only_a = sum(ga[k]["correct"] and not gb[k]["correct"] for k in ks)
    only_b = sum(gb[k]["correct"] and not ga[k]["correct"] for k in ks)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    rows = [f"GSM8K problems: {len(ks)} (both models)",
            "| Model | Correct | Hit 1024-token cap | Mean tokens |", "|---|---|---|---|"]
    for m, g in ((a, ga), (b, gb)):
        rows.append(f"| {m} | {sum(g[k]['correct'] for k in ks)}/{len(ks)} ({100 * sum(g[k]['correct'] for k in ks) / len(ks):.1f} %) | "
                    f"{sum(g[k]['capped'] for k in ks)} | {round(sum(g[k]['tokens'] for k in ks) / len(ks))} |")
    rows.append(f"\nOnly {a} right: {only_a} · only {b} right: {only_b} · exact McNemar p = {p:.3f}")
    print("\n".join(rows))
    return "\n".join(rows)


def report():
    cats = ["math", "facts", "reasoning", "language", "code", "commonsense"]
    lines = ["| Model | Study 100 | math | facts | reasoning | language | code | commonsense | Same reply as original | GSM8K 0-29 | GSM8K capped | GSM8K tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = None
    p0 = HERE / "bench_original.json"
    if p0.exists():
        base = json.loads(p0.read_text(encoding="utf-8"))
    for p in sorted(HERE.glob("bench_*.json"), key=lambda p: (p.stem != "bench_original", p.stem)):
        r = json.loads(p.read_text(encoding="utf-8"))
        st = r["study"]
        gs = {k: v for k, v in r["gsm8k"].items() if int(k) < 30}     # problems 0-29: every model ran these
        same = sum(base["study"].get(k, {}).get("reply") == v["reply"] for k, v in st.items()) if base else 0
        cell = lambda c: f"{sum(x['correct'] for x in st.values() if x['cat'] == c)}/{sum(x['cat'] == c for x in st.values())}"
        lines.append(f"| {p.stem[6:]} | **{sum(x['correct'] for x in st.values())}/{len(st)}** | " + " | ".join(cell(c) for c in cats)
                     + f" | {same}/{len(st)} | **{sum(x['correct'] for x in gs.values())}/{len(gs)}** | "
                     f"{sum(x['capped'] for x in gs.values())} | {round(sum(x['tokens'] for x in gs.values()) / max(1, len(gs)))} |")
    (HERE / "bench_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        sys.path.insert(0, str(HERE))
        export(sys.argv[2:])
    elif cmd == "run":
        arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
        run(sys.argv[2], int(arg("port", 8170)), int(arg("gsm", 50)))
    elif cmd == "gsm":
        arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
        gsm_range(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(arg("port", 8170)))
    elif cmd == "compare":
        txt = compare(sys.argv[2], sys.argv[3])
        (HERE / f"gsm8k_{sys.argv[2]}_vs_{sys.argv[3]}.md").write_text(txt + "\n", encoding="utf-8")
    elif cmd == "report":
        report()
