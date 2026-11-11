"""
The web app's own rules, on a tiny synthetic model in a temporary workspaces folder (no download, no llama.cpp):
  python tests/test_app.py

1. who may call the API (this page only), and what is refused while a background task runs
2. the model file: downloaded again (new date), replaced by another model, moved (locate it), extracted again
3. the Test tab's bookkeeping: tokens kept with every reply, deleted traces, replies ending at the token limit,
   calculation checks, random split, an edit computed for other weights
4. bad input: infinite weights, a layer count that doesn't match the tensors
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core

tmp = Path(tempfile.mkdtemp(prefix="llmdec_app_"))
core.WORKSPACES = tmp / "workspaces"                  # keep the real workspaces folder untouched
import trace_core as tc
import fix_core as fc
import test_core as tst
import test_trace_arch as arch
import app as A

ok = fail = 0


def check(cond, msg):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


q = lambda m: None
src = tmp / "tiny_llama.gguf"
arch.write_model(arch.FAMILIES["llama"], src, seed=3)
other = tmp / "other.gguf"
arch.write_model(arch.FAMILIES["llama"], other, seed=4)
ws = core.extract_workspace(src, log=q)
d = core.WORKSPACES / ws
meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
meta["tokenizer.chat_template"] = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                                   "<|im_start|>assistant\n")
(d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
c = A.app.test_client()
U = lambda path: f"/api/ws/{ws}{path}"


def wait(resp):
    j = resp.get_json()
    while True:
        s = c.get(f"/api/jobs/{j['job']}").get_json()
        if s["status"] != "running":
            return s
        time.sleep(0.03)


# ------------------------------------------------------------------ 1. who may call, and when
r = c.post(U("/traces/delete"), data=json.dumps({"ids": ["x"]}), content_type="text/plain", headers={"Origin": "http://evil.example"})
check(r.status_code == 403, "a request from another web site's page is refused")
r = c.post(U("/metadata/reset"), json={"key": "general.name"}, headers={"Host": "evil.example"})
check(r.status_code == 403, "a request for another host name (DNS rebinding) is refused")
r = c.get("/api/workspaces", headers={"Sec-Fetch-Site": "cross-site"})
check(r.status_code == 403, "a cross-site read is refused")
r = c.get("/api/workspaces", headers={"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"})
check(r.status_code == 200 and c.get("/api/workspaces").status_code == 200, "the app's own page (and plain local requests) are served")
check(c.post(U("/traces/delete"), json={"ids": []}, headers={"Origin": "http://localhost:9999"}).status_code == 403,
      "a page on another local port is refused")

name = next(e["name"] for e in core.load_manifest(d) if e["name"].endswith("ffn_down.weight"))
A.BUSY.acquire()
A.RUNNING.update(job="x", title="exporting a GGUF", started=time.time(), stoppable=False)
codes = [c.post(U("/tensor"), json={"name": name, "cells": [[0, 0, 1.5]]}).status_code,
         c.put(U("/metadata"), json={"key": "general.name", "value": "x"}).status_code,
         c.post(U("/chat/unload")).status_code, c.delete(U("/fix/nothing")).status_code, c.delete(f"/api/ws/{ws}").status_code]
A.RUNNING.clear()
A.BUSY.release()
check(codes == [409] * 5 and d.is_dir(), "while a task runs, weights / metadata / computed edits / the workspace can't be changed")
check(c.post(U("/tensor"), json={"name": name, "cells": [[0, 0, 1.5]]}).status_code == 200
      and c.post(U("/tensor/revert"), json={"name": name}).status_code == 200, "...and can be again when it has finished")

j = c.post(U("/measure")).get_json()
check(j.get("stoppable") is False, "a task that can't be stopped says so (no Stop button)")
wait(c.post(U("/measure")) if False else type("R", (), {"get_json": lambda self: j})())
j = c.post(U("/chat"), json={"messages": [{"role": "user", "content": "say one"}], "max_tokens": 2}).get_json()
check(j.get("stoppable") is True, "a chat reply can be stopped")
check(wait(type("R", (), {"get_json": lambda self: j})())["status"] == "done", "chat job finishes")
for _ in range(A.KEEP_JOBS + 6):
    wait(c.post(U("/measure")))
check(len(A.JOBS) <= A.KEEP_JOBS + 1 and len(A.CANCEL) == len(A.JOBS), f"finished jobs are forgotten (kept {len(A.JOBS)} of {A.KEEP_JOBS + 8})")
s = wait(c.post(U("/chat"), json={"messages": [{"role": "user", "content": "hi"}], "temperature": 1, "top_k": 10 ** 9, "max_tokens": 2}))
check(s["status"] == "done", "top-k larger than the vocabulary works")

# unloading the model while a reply is written doesn't lose the reply
n = [0]


def emit(x):
    n[0] += 1
    if n[0] == 2:
        tc.unload()


try:
    r = tc.chat(ws, [{"role": "user", "content": "say one"}], max_new=5, emit=emit, log=q)
    check(bool(r["id"]), "a reply is saved even if the model is unloaded while it is written")
except Exception as e:
    check(False, f"a reply is saved even if the model is unloaded while it is written ({type(e).__name__}: {e})")

# ------------------------------------------------------------------ 4. bad input
r1 = c.post(U("/tensor"), json={"name": name, "cells": [[0, 0, "inf"]]})
r2 = c.post(U("/tensor"), json={"name": name, "cells": [[0, 1, 1e40]]})
r3 = c.post(U("/tensor"), json={"name": name, "cells": [[1]]})
check(r1.status_code == r2.status_code == r3.status_code == 400 and "float32" in r2.get_json()["error"],
      "infinite / too large weights and malformed cells are refused")
check(c.get(U(f"/tile?name={name}&r0=-3&nr=4")).status_code == 400, "a negative tile position is refused")
c.put(U("/metadata"), json={"key": "llama.block_count", "value": 999})
s = wait(c.post(U("/export")))
check(s["status"] == "error" and "block_count" in s["error"], "export refuses a layer count that doesn't match the tensors")
c.post(U("/metadata/reset"), json={"key": "llama.block_count"})
check(wait(c.post(U("/export")))["status"] == "done", "...and works again with the right one")

# ------------------------------------------------------------------ 3. the Test tab's bookkeeping
text = "say one | toe\nsay two | hat\nsay three | 42\nsay four | dot\nsay five | 7"
t = wait(c.post(U("/test/run"), json={"text": text, "instruction": "", "max_tokens": 6}))["result"]
tid = t["id"]
check(all(x.get("ids") and x.get("n_prompt") for x in t["items"]), "a test run with the tracer keeps every reply's tokens")
a1 = wait(c.post(U(f"/test/{tid}/analyze"), json={"split": 0.6, "measure": 2}))
check(a1["status"] == "done" and a1["result"]["rows"], "analyse: wrong answers located")
c.post(U("/traces/delete"), json={"ids": [x["trace"] for x in t["items"]]})
s = wait(c.post(U(f"/test/{tid}/recommend"), json={"split": 0.6, "measure": 2, "layer": 1, "mode": "exact"}))
check(s["status"] == "done", f"compute right after its traces were deleted: they are made again ({s['error']})")
sid = s["result"]["solution"]["sid"]
check(s["result"]["solution"].get("check_unseen") is False, "few right answers: the side-effect check says it used the fitted ones")

# a test saved by an earlier version: no tokens with the replies, and its traces deleted
p = d / "tests" / f"{tid}.json"
old = json.loads(p.read_text(encoding="utf-8"))
c.post(U("/traces/delete"), json={"ids": [x["trace"] for x in old["items"] if x.get("trace")]})
old = json.loads(p.read_text(encoding="utf-8"))
for x in old["items"]:
    x.pop("ids", None), x.pop("n_prompt", None)
p.write_text(json.dumps(old), encoding="utf-8")
s = wait(c.post(U(f"/test/{tid}/analyze"), json={"split": 0.8, "measure": 0}))
check(s["status"] == "done", f"analyse a test saved without tokens after its traces were deleted ({s['error']})")

# applying an edit after the weights changed is refused
c.post(U("/tensor"), json={"name": name, "cells": [[0, 0, 0.25]]})
r = c.post(U("/fix/save"), json={"sid": sid, "name": "late"})
check(r.status_code == 400 and "changed" in r.get_json().get("error", ""), f"an edit computed for other weights can't be applied ({r.status_code} {str(r.get_json())[:160]})")
c.post(U("/tensor/revert"), json={"name": name})

# a reply that ends exactly at the token limit is not "cut off"
model, tok = tc.get_model(ws, "edited", q)
stop, orig, k = tst._stop_id(tok), tc._sample, [0]


def fake(logits, *a):
    k[0] += 1
    return stop if k[0] == 3 else orig(logits, *a)


tc._sample = fake
x = tst.run(ws, "say one | toe", instruction="", max_tokens=3, log=q)["items"][0]
tc._sample = orig
check(x["n_reply"] == 3 and x["unfinished"] is False, "a reply ended by its stop token at exactly Max tokens isn't marked cut off")

# calculation checks: whole calculations only
AE = lambda s: [(e[1], e[2]) for e in tst.arithmetic_errors(s)]
check(all(AE(s) == [] for s in ("(2 + 3) * 4 + 5 = 25", "1/2 + 1/4 = 3/4", "2^3 * 2 = 16", "at 3:00 + 2 = 5:00",
                                "10 / 3 = 3 remainder 1", "16 - 3 - 4 = 9 and 9 * 2 = 18")),
      "correct working isn't flagged (part of a longer calculation, fractions, powers, clock times, remainders)")
check(AE("3 + 4 = 8") == [("8", "7")] and AE("Step 1: 16 - 3 = 12") == [("12", "13")] and AE("- 3 + 4 = 8") == [("8", "7")]
      and AE("(a) 3 + 4 = 8") == [("8", "7")] and AE("$9 * $2 = $19") == [("19", "18")],
      "wrong results are still found (after a step label, a bullet, a list letter, with $ signs)")

# random split: the same pick every time, sizes as asked
R = {"items": [{"i": i} for i in range(20)]}
a, b = tst.split_of(R, 0.7, shuffle=True)
check((a, b) == tst.split_of(R, 0.7, shuffle=True) and len(a) == 14 and len(b) == 6 and sorted(a + b) == list(range(20))
      and a != list(range(14)) and tst.split_of(R, 0.7) == (list(range(14)), list(range(14, 20))),
      "split: first 70 % by default; picked at random (same every time) when asked")

# ------------------------------------------------------------------ 2. the model file
tc.unload()
core.release_sources()
stamp0 = tc._ws_stamp(d)
data = src.read_bytes()
src.unlink()
r = c.get(f"/api/ws/{ws}").get_json()
check(r["source_missing"] == [str(src)], "a missing model file is reported with the workspace")
time.sleep(1.1)
src.write_bytes(data)                                                   # "downloaded again": same contents, new date
core.release_sources()
s = wait(c.post(U("/export")))
check(s["status"] == "done" and tc._ws_stamp(d) == stamp0 and core.load_info(d).get("accepted"),
      f"the same model with a new date is accepted, and the weights don't count as changed ({s['error']})")
core.release_sources()
time.sleep(1.1)
src.write_bytes(other.read_bytes())                                     # another model at the same place, same size
s = wait(c.post(U("/export")))
check(s["status"] == "error" and "changed since extraction" in s["error"], "another model in its place is refused")
core.release_sources()
src.unlink()
moved = tmp / "moved" / "renamed.gguf"
moved.parent.mkdir()
moved.write_bytes(data)
check(c.post(U("/source"), json={"path": str(other)}).status_code == 400, "locate file: another model is refused")
r = c.post(U("/source"), json={"path": str(moved)})
check(r.status_code == 200 and c.get(f"/api/ws/{ws}").get_json()["source_missing"] == [] and tc._ws_stamp(d) == stamp0
      and wait(c.post(U("/export")))["status"] == "done", "locate file: the same model in a new place is used from there")

# a workspace made before fingerprints existed: the tensor list and the CSV copies decide
core.release_sources()
info = core.load_info(d)
info.pop("fingerprint"), info.pop("accepted")
(d / "source.json").write_text(json.dumps(info), encoding="utf-8")
time.sleep(1.1)
moved.write_bytes(other.read_bytes())
s1 = wait(c.post(U("/export")))
core.release_sources()
time.sleep(1.1)
moved.write_bytes(data)
s2 = wait(c.post(U("/export")))
check(s1["status"] == "error" and s2["status"] == "done", "older workspaces: same model accepted, another one refused")

# extract again: edits go, saved tests and traces stay
tc.unload()
core.release_sources()
n_tests = len(list((d / "tests").glob("*.json")))
(d / "added" / "marker.txt").write_text("x")
shutil.copy(moved, src)
log = []
core.extract_workspace(src, overwrite=True, log=log.append)
check(n_tests > 0 and len(list((d / "tests").glob("*.json"))) == n_tests and not (d / "added" / "marker.txt").exists()
      and [x.name for x in core.WORKSPACES.iterdir()] == [ws] and any("Kept" in m for m in log),
      f"extract again keeps the {n_tests} saved test file(s) and removes the edits")

tc.unload()
core.release_sources()
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
