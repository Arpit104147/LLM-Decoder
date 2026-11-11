"""End-to-end test of the running app's HTTP API on a real model, the way the UI uses it:
edit weights (inside + outside the CSV block), edit metadata, add a pass-through layer and a token,
export, run the result in LM Studio's llama.cpp, then undo everything.

Run (app must be running):  python test_e2e_api.py [workspace]      default: LFM3-VL-2B-Q8_0
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

BASE = "http://127.0.0.1:8765"
WS = sys.argv[1] if len(sys.argv) > 1 else "LFM3-VL-2B-Q8_0"
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok  " if cond else "  BAD ") + msg, flush=True)


def call(method, path, body=None, expect=200):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            code, data = r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        code, data = e.code, json.loads(e.read() or b"{}")
    if expect is not None and code != expect:
        raise AssertionError(f"{method} {path} -> {code}: {data}")
    return data


def job(path, body=None):
    jid = call("POST", path, body)["job"]
    while True:
        j = call("GET", f"/api/jobs/{jid}")
        if j["status"] != "running":
            if j["status"] == "error":
                raise AssertionError(f"{path} failed: {j['error']}")
            return j["result"]
        time.sleep(0.5)


W = f"/api/ws/{urllib.parse.quote(WS)}"
q = urllib.parse.quote

print(f"0) starting state of {WS}")
info = call("GET", W)
c0 = call("GET", W + "/changes")
check(c0["stats"]["edited"] == 0 and not c0["structure"] and not c0["metadata"]["changed"], "workspace starts clean")
L0, V0 = info["highlights"]["Layers"], info["highlights"]["Vocabulary"]
total0 = c0["stats"]["original_total"]
arch = info["source"]["architecture"]
print(f"     {arch}: {L0} layers, {V0:,} tokens, {total0:,} weights")

print("1) bad input is rejected with clear messages")
e = call("POST", "/api/extract", {"path": "C:/nope/missing.gguf"}, expect=400)
check("not found" in e["error"].lower(), f"missing file: {e['error']}")
e = call("PUT", W + "/metadata/raw", {"text": "{ bad json"}, expect=400)
check("line 1" in e["error"], f"broken metadata JSON: {e['error']}")
e = call("PUT", W + "/metadata", {"key": f"{arch}.block_count", "value": "many"}, expect=400)
check("whole number" in e["error"], f"wrong metadata type: {e['error']}")
e = call("POST", W + "/tensor", {"name": "token_embd.weight", "cells": [[10 ** 9, 0, 1.0]]}, expect=400)
check("outside" in e["error"], f"cell outside the tensor: {e['error']}")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core
import inspect_core as ic
from gguf import GGUFReader
server = ic.find_llama_server()
orig = info["source"]["gguf_path"]


def run(model, prompt, port=8141):
    import tempfile
    errlog = tempfile.TemporaryFile()        # a file, not a pipe: an unread pipe fills up and freezes llama.cpp
    p = subprocess.Popen([str(server), "-m", model, "--port", str(port), "-c", "1024", "-ngl", "0", "--no-warmup"],
                         stdout=subprocess.DEVNULL, stderr=errlog)
    try:
        for _ in range(600):
            if p.poll() is not None:
                errlog.seek(0)
                return {"error": errlog.read().decode(errors="replace")[-600:]}
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except Exception:
                time.sleep(0.4)
        post = lambda path, b: json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
        # reasoning models think before answering, so leave room for that
        chat = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": 0})
        tok = post("/tokenize", {"content": "<e2e_test_token>"})["tokens"]
        msg = chat["choices"][0]["message"]
        return {"reply": (msg.get("content") or "").strip(), "thinking": (msg.get("reasoning_content") or "").strip(), "tok": tok}
    finally:
        p.kill(); p.wait()


question = "What is the capital of France? Answer in one short sentence."
a = run(orig, question)
print(f"     original reply: {a.get('reply', a.get('error'))!r}")
check("Paris" in (a.get("reply") or ""), "original model answers correctly")

print("2) grow only: +1 pass-through layer, +1 token -> must behave EXACTLY like the original")
r = job(W + "/grow/layers", {"count": 1, "mode": "identity"})
check(f"layer {L0 - 1}" in r["label"], r["label"])
r = job(W + "/grow/tokens", {"tokens": ["<e2e_test_token>"], "token_type": "user_defined", "init": "mean"})
check(str(V0) in r["label"], r["label"])
c2 = call("GET", W + "/changes")
check(c2["stats"]["edited"] == 0 and c2["stats"]["added"] > 0 and len(c2["structure"]) == 2,
      f"0 original weights changed, {c2['stats']['added']:,} added = {c2['stats']['added_pct']} %")
t0 = time.time()
res = job(W + "/export", {"exact": False})
print(f"     exported {res['size'] / 1e9:.2f} GB in {time.time() - t0:.0f}s")
g = run(res["path"], question)
print(f"     grown reply:    {g.get('reply', g.get('error'))!r}")
check("error" not in g, f"grown model ({L0 + 1} layers, {V0 + 1:,} tokens) loads and chats in llama.cpp")
check(g.get("reply") == a.get("reply") and g.get("thinking") == a.get("thinking") and g.get("thinking"),
      "identical answer AND identical reasoning text to the original")
check(g.get("tok") == [V0], f"new token is recognised as one token, id {V0}: {g.get('tok')}")

print("3) now edit metadata + weights (inside and far outside the CSV block)")
call("PUT", W + "/metadata", {"key": "general.name", "value": "LFM3 e2e test"})
call("POST", W + "/tensor", {"name": "token_embd.weight", "cells": [[5, 5, 0.1], [120000, 2000, -0.05]]})
ffn = next(t["name"] for t in info["tensors"] if t["name"].endswith("ffn_down.weight") and ".10." in t["name"])
call("POST", W + "/tensor", {"name": ffn, "cells": [[100, 7000, 0.02]]})
tile = call("GET", W + f"/tile?name=token_embd.weight&r0=119990&nr=20&c0=1990&nc=20")
check(abs(tile["values"][10][10] + 0.05) < 1e-12 and [10, 10] in tile["changed"], "far-away edit visible in the grid tile")
c1 = call("GET", W + "/changes")
check(c1["stats"]["edited"] == 3 and c1["stats"]["edited_pct"] == f"{3 * 100 / total0:.10f}", f"3 weights changed = {c1['stats']['edited_pct']} %")
res = job(W + "/export", {"exact": False})
types = {t["name"]: (t["from"], t["to"], t["note"]) for t in res["tensors"]}
check(types[ffn][1] == "Q8_0" and "1 of" in types[ffn][2], f"{ffn}: {types[ffn]}")
check(types["token_embd.weight"][1] == "Q8_0", f"token_embd (resized for the new token) stays Q8_0: {types['token_embd.weight']}")
check(res["stats"]["edited"] == 3, f"export report: {res['stats']['edited']} changed = {res['stats']['edited_pct']} %")
out = GGUFReader(res["path"])
meta = core.read_metadata(out)
emb = next(t for t in out.tensors if t.name == "token_embd.weight")
rows = int(emb.shape[1])
row120k = core._decode(emb.data.reshape(rows, -1)[120000:120001], emb.tensor_type).reshape(-1)
check(meta["general.name"] == "LFM3 e2e test" and meta[f"{arch}.block_count"] == L0 + 1 and rows == V0 + 1,
      f"file has the new name, {L0 + 1} layers, {V0 + 1:,} token rows")
check(abs(row120k[2000] + 0.05) < 0.002, f"edit at [120000, 2000] is in the file: {row120k[2000]:.6f} (Q8_0 of -0.05)")
del out, emb
core.release_sources()
b = run(res["path"], question)
print(f"     edited reply:   {b.get('reply', b.get('error'))!r}")
check("Paris" in (b.get("reply") or ""), "edited model still answers correctly (a changed active weight may change wording)")

print("4) exact mode keeps typed digits")
res2 = job(W + "/export", {"exact": True})
types2 = {t["name"]: t["to"] for t in res2["tensors"]}
out = GGUFReader(res2["path"])
f = next(t for t in out.tensors if t.name == ffn)
val = core.full_tensor(f).reshape(-1, f.shape[0])[100, 7000]
check(types2[ffn] == "F32" and val == np.float32(0.02), f"{ffn} stored as {types2[ffn]}, value {val!r}")
del out, f
core.release_sources()

print("5) undo everything")
job(W + "/undo"); job(W + "/undo")
call("POST", W + "/tensor/revert", {"name": "token_embd.weight"})
call("POST", W + "/tensor/revert", {"name": ffn})
call("POST", W + "/metadata/reset", {"key": "general.name"})
c3 = call("GET", W + "/changes")
check(c3["stats"]["edited"] == 0 and not c3["structure"] and not any(c3["metadata"].values()) and not c3["weights"],
      "workspace is exactly back to the original")
info3 = call("GET", W)
check(info3["highlights"]["Layers"] == L0 and info3["highlights"]["Vocabulary"] == V0, f"{L0} layers, {V0:,} tokens again")
for p in Path(res["path"]).parent.glob("*"):
    p.unlink()
print("ALL END-TO-END CHECKS PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
