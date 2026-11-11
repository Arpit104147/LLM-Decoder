"""
Ask the 100 questions in questions.json through the chat tracer and grade the answers.
  python run_chats.py [original|edited] [out_name]
Every chat is saved as a trace in the workspace (open them in the app: Chat & trace -> Saved traces) and one line
per question goes to results_<out_name>.jsonl (the run resumes where it stopped).
Greedy decoding (temperature 0), thinking off, so a run is repeatable.
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import trace_core as tc

WS = "Qwen3.5-0.8B-Q8_0"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "original"
OUT = HERE / f"results_{sys.argv[2] if len(sys.argv) > 2 else WHICH}.jsonl"


def prompt_for(item):
    if item["cat"] == "math":
        return item["q"] + " Show brief working, then give the final answer as a number on the last line."
    return item["q"] + " Answer briefly."


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("−", "-"))
    nums = [n.rstrip(".").replace(",", "") for n in nums if any(ch.isdigit() for ch in n)]
    return float(nums[-1]) if nums else None


def grade(item, reply):
    if "num" in item:
        got = last_number(reply)
        return got is not None and abs(got - item["num"]) < 1e-6, got
    text = reply if item.get("case") else reply.lower()
    for bad in item.get("not", []):
        if bad.lower() in reply.lower():
            return False, bad
    for kw in item["kw"]:
        k = kw if item.get("case") else kw.lower()
        if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", text) if re.match(r"^[\w ]+$", k) else k in text:
            return True, kw
    return False, None


def main():
    items = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    t0 = time.time()
    for n, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        t = time.time()
        res = tc.chat(WS, [{"role": "user", "content": prompt_for(item)}], max_new=300 if item["cat"] == "math" else 80,
                      temperature=0, which=WHICH, log=lambda m: None)
        ok, got = grade(item, res["reply"])
        row = {"id": item["id"], "cat": item["cat"], "correct": ok, "got": got, "reply": res["reply"], "trace": res["id"],
               "n_tokens": len(res["tokens"]), "n_reply": res["timing"]["n_gen"], "seconds": round(time.time() - t, 1)}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{n:3d}/100] {item['id']} {'RIGHT' if ok else 'wrong'}  {row['seconds']:5.1f}s  {res['reply'][:70]!r}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
