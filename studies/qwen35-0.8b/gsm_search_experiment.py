"""
Does a computed edit really improve GSM8K questions it never saw?  (Qwen3.5-0.8B, workspace weights)

  1. copy the saved 300-problem baseline test, re-graded with the current grader (cut-off replies flagged)
  2. validate_core.search: compute edits from the wrong answers in problems 1-150 with 4 settings
     (general rank 1 / rank 4 at the most responsible part, rank 4 at the second, exact with 1 token per answer),
     then ask problems 151-300 (and the wrong ones of 1-150) without and with each edit in llama.cpp
  3. write gsm_search_experiment.md / .json

  python studies/qwen35-0.8b/gsm_search_experiment.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import test_core as tst
import validate_core as vc
import grade_core as g

WS = "Qwen3.5-0.8B-Q8_0"
BASE = json.loads((HERE / "gsm_fix_experiment.json").read_text(encoding="utf-8"))["base"]
LOG = HERE / "gsm_search_experiment.log"


def log(m):
    line = time.strftime("%H:%M:%S ") + str(m)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def regraded_copy():
    r = tst.get_test(WS, BASE)
    tid = BASE + "-rg"
    p = tst.tests_dir(WS) / f"{tid}.json"
    if p.exists():
        return tid, json.loads(p.read_text(encoding="utf-8"))
    changed = 0
    for x in r["items"]:
        it = g.item({"question": x["question"], "answer": x["answer"]}, 0)
        cut = x.get("n_reply", 0) >= int(r["max_tokens"])               # llama.cpp stopped at the limit
        ok, got, off = tst._grade_row(it, x["reply"], cut)
        changed += ok != x["correct"]
        x.update(correct=bool(ok), got=got, offset=off, unfinished=cut)
    r["id"], r["name"] = tid, r["name"] + " (re-graded)"
    r["right"] = sum(x["correct"] for x in r["items"])
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"re-graded copy {tid}: {r['right']}/300 right ({changed} grade(s) changed, "
        f"{sum(x['unfinished'] for x in r['items'])} cut off at {r['max_tokens']} tokens)")
    return tid, r


def main():
    t0 = time.time()
    log("=== GSM8K: search for an edit that helps held-out questions ===")
    tid, r = regraded_copy()
    # run 1 (gsm_search_experiment_run1.log) already showed general rank 1 / rank 4 at the most responsible part
    # (attention L0) need changes of ~78,800 % of the matrix: not repeated here
    settings = [x for x in vc.DEFAULT_SETTINGS if x != {"choose": "responsible", "mode": "general", "rank": 1, "strength": 1.0}]
    out = vc.search(WS, tid, split=0.5, settings=settings, measure=8, parallel=3, train="wrong", log=lambda m: log("   " + str(m)))
    (HERE / "gsm_search_experiment.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if not out["candidates"]:
        lines = ["# Computed edits on GSM8K, checked on questions they never saw", "",
                 f"Edits computed from the wrong answers in problems 1-150 · {(time.time() - t0) / 3600:.1f} h", "",
                 "| Setting | Edit | Size of change | Token fixes reached |", "|---|---|---:|---|"]
        lines += [f"| `{json.dumps(t['setting'])}` | {t['label']} | {t['size']:.0f} % | {t['fixed']}/{t['n_fixes']} |" for t in out["tried"]]
        lines += ["", "**Result: every edit was too big to apply safely, so none was checked on problems 151-300.**"]
        (HERE / "gsm_search_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log("\n".join(lines))
        return
    b = out["baseline"]
    lines = ["# Computed edits on GSM8K, checked on questions they never saw", "",
             f"Qwen3.5-0.8B Q8_0 workspace weights · llama.cpp greedy · {r['max_tokens']}-token limit · "
             f"edits computed from the wrong answers in problems 1-150, checked on problems 151-300 · "
             f"{(time.time() - t0) / 3600:.1f} h", "",
             f"Without an edit: **{b['held_right']}/{b['n']}** held-out problems right ({b['unfinished']} cut off).", "",
             "| Edit | Size of change | Held-out right | Fixed | Broken | p (×{}) | Verdict | Its own wrong answers now right |".format(out["tries"]),
             "|---|---:|---|---:|---:|---:|---|---|"]
    for c in out["candidates"]:
        t = next((x for x in out["tried"] if x["sid"] == c["sid"]), {})
        h = c["held"]
        tw = c.get("train_wrong")
        own = f"{tw['fixed']}/{tw['n']}" if tw else ""
        lines.append(f"| {c['label']} | {t.get('size', 0):.1f} % | {h['before']} → **{h['after']}** / {h['n']} | {h['gained']} | {h['lost']} | "
                     f"{h['p_adjusted']:.3g} | {h['verdict']} | {own} |")
    if out["skipped"]:
        lines += ["", "Not checked: " + "; ".join(f"{json.dumps(x['setting'])}: {x['reason']}" for x in out["skipped"])]
    lines += ["", ("**Result: an edit that really helps held-out problems was found.**" if out["best"] else
                   "**Result: none of the edits clearly helps problems it never saw.**")]
    (HERE / "gsm_search_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
