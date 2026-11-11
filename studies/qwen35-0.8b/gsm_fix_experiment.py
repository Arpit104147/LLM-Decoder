"""
Does a computed edit from the Test tab help math problems it never saw?

  python studies/qwen35-0.8b/gsm_fix_experiment.py

1. GSM8K test problems 0-299 on the workspace weights (fast mode: llama.cpp writes the replies, greedy, thinking off)
2. an edit computed only from the wrong answers in problems 0-149 (the Test tab's Analyse + Compute edit with
   "compute from: first 50 %"), in two modes: exact (fix these answers) and general (one shared rank-4 correction)
3. each edit applied in turn, problems 150-299 asked again: held-out accuracy before vs after (and the edit removed)
Every step is the code the Test tab's buttons call. Results: gsm_fix_experiment.json / .md (resumes where it stopped).
"""
import json
import sys
import time
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fix_core as fc
import test_core as tst

WS = "Qwen3.5-0.8B-Q8_0"
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
OUT = HERE / "gsm_fix_experiment.json"
Q = (HERE / "gsm8k_test_0_299.jsonl").read_text(encoding="utf-8")
PAR = 4
st = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
save = lambda: OUT.write_text(json.dumps(st, indent=1), encoding="utf-8")
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)


def mcnemar(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


for e in fc.list_saved(WS):
    if e["name"].startswith("gsm-"):
        fc.delete(WS, e["name"])
if "base" not in st:
    log("1. baseline: 300 problems")
    r = tst.run(WS, Q, instruction=INSTR, max_tokens=1024, which="edited", engine="llama", parallel=PAR,
                name="GSM8K 0-299 (before any fix)", log=lambda m: log("   " + m) if not m.startswith("[") else None)
    st["base"] = r["id"]
    save()
base = tst.get_test(WS, st["base"])
held = [x for x in base["items"] if x["i"] >= 150]
train = [x for x in base["items"] if x["i"] < 150]
log(f"baseline: {base['right']}/300 · problems 0-149: {sum(x['correct'] for x in train)}/150 · "
    f"150-299: {sum(x['correct'] for x in held)}/150")
held_text = "\n".join(json.dumps({"question": x["question"], "answer": x["answer"]}) for x in held)

for mode, kw in (("exact", {}), ("general", {"rank": 4, "strength": 0.5})):
    key = f"edit_{mode}"
    if key in st and "held_after" in st[key]:
        continue
    log(f"2. {mode} edit from the wrong answers in problems 0-149")
    t0 = time.time()
    out = tst.recommend(WS, st["base"], split=0.5, mode=mode, log=lambda m: log("   " + m)
                        if not m.startswith(("Fix ", "  check", "  traced")) else None, **kw)
    s = out["solution"]
    a = out["analysis"]
    info = {"layer": s["layer"], "layers": s["layers"], "mode": mode, "token_fixes": len(s["fixes"]), "reached": s["fixed"],
            "answers": len(a["rows"]), "skipped": len(a["skipped"]), "calc_steps": sum(x["kind"] == "calculation step" for x in a["rows"]),
            "change_pct": s["change_pct_of_tensor"], "n_numbers": s["n_numbers"],
            "right_answers_top_changed_pct": s["check_top_changed_pct"], "seconds": round(time.time() - t0)}
    log(f"   {info}")
    fc.save(WS, s["sid"], f"gsm-{mode}", force=True)   # measured even if big
    try:
        log(f"3. held-out problems 150-299 with the {mode} edit")
        r = tst.run(WS, held_text, instruction=INSTR, max_tokens=1024, which="edited", engine="llama", parallel=PAR,
                    name=f"GSM8K 150-299 with the {mode} edit", log=lambda m: log("   " + m) if not m.startswith("[") else None)
    finally:
        fc.delete(WS, f"gsm-{mode}")
    after = {x["question"]: x for x in r["items"]}
    gained = sum(1 for x in held if not x["correct"] and after[x["question"]]["correct"])
    lost = sum(1 for x in held if x["correct"] and not after[x["question"]]["correct"])
    info.update(held_test=r["id"], held_before=sum(x["correct"] for x in held), held_after=r["right"], gained=gained, lost=lost,
                p=mcnemar(gained, lost))
    st[key] = info
    save()
    log(f"   held-out: {info['held_before']}/150 -> {info['held_after']}/150 (now right {gained}, now wrong {lost}, p = {info['p']:.3f})")

lines = ["# Computed edits on GSM8K: compute from problems 0-149, test on 150-299", "",
         f"Workspace weights, llama.cpp greedy, thinking off, 1024-token cap. Baseline: {base['right']}/300.", "",
         "| Edit | Layer(s) | Token fixes (reached) | Change | Right answers changed | Held-out before | Held-out after | Now right / now wrong | p |",
         "|---|---|---|---|---|---|---|---|---|"]
for mode in ("exact", "general"):
    i = st.get(f"edit_{mode}")
    if i and "held_after" in i:
        lines.append(f"| {mode}{' (rank 4)' if mode == 'general' else ''} | {', '.join(map(str, i['layers']))} | {i['token_fixes']} ({i['reached']}) | "
                     f"{i['change_pct']:.1f}% | {i['right_answers_top_changed_pct']:.1f}% | {i['held_before']}/150 | **{i['held_after']}/150** | "
                     f"{i['gained']} / {i['lost']} | {i['p']:.2f} |")
(HERE / "gsm_fix_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
