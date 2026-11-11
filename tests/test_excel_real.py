"""Excel round-trip on a real model through the running app, using real Microsoft Excel to edit:
export the workbook, import it untouched (must change nothing), edit cells in Excel and save,
import again (exactly those cells must change), then undo everything.

Needs: the app running and Excel installed.
Run:  python test_excel_real.py [workspace]      default: LFM3-VL-2B-Q8_0
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
W = f"/api/ws/{urllib.parse.quote(WS)}"
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


def job(path, body=None, fail_ok=False):
    jid = call("POST", path, body)["job"]
    while True:
        j = call("GET", f"/api/jobs/{jid}")
        if j["status"] != "running":
            if j["status"] == "error" and not fail_ok:
                raise AssertionError(f"{path} failed: {j['error']}")
            return j
        time.sleep(1)


def excel_edit(path, edits):
    """Open the workbook in real Excel (hidden), set cells, save. edits: [(sheet, row1, col1, value)]."""
    lines = ["$ErrorActionPreference = 'Stop'",
             "$x = New-Object -ComObject Excel.Application",
             "$x.Visible = $false; $x.DisplayAlerts = $false",
             "try {",
             f"  $wb = $x.Workbooks.Open('{path}')"]
    for sheet, r, c, v in edits:
        val = f"'{v}'" if isinstance(v, str) else repr(float(v))
        lines.append(f"  $wb.Worksheets.Item('{sheet}').Cells.Item({r}, {c}).Value2 = {val}")
    lines += ["  $wb.Save(); $wb.Close($false)", "} finally { $x.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($x) }"]
    t = time.time()
    p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "\n".join(lines)],
                       capture_output=True, text=True, timeout=3600)
    if p.returncode:
        raise AssertionError("Excel failed: " + (p.stderr or p.stdout)[-800:])
    return time.time() - t


def cell(name, r, c):
    t = call("GET", W + f"/tile?name={urllib.parse.quote(name)}&r0={r}&nr=1&c0={c}&nc=1")
    return t["values"][0][0], [0, 0] in t["changed"]


c0 = call("GET", W + "/changes")
check(c0["stats"]["edited"] == 0 and not c0["weights"], "workspace starts clean")
info = call("GET", W)
ffn = next(t["name"] for t in info["tensors"] if t["name"].endswith("ffn_down.weight") and ".10." in t["name"])
EMB = "token_embd.weight"

print("1) export the workbook")
t = time.time()
r = job(W + "/excel/export")["result"]
xlsx = Path(r["path"])
print(f"     {xlsx.name}: {xlsx.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f}s")
import openpyxl
wb = openpyxl.load_workbook(xlsx, read_only=True)
index = {row[0]: row[1] for row in wb["Index"].iter_rows(min_row=2, max_col=2, values_only=True) if row[0] and row[1]}
n_sheets = len(wb.sheetnames)
wb.close()
check(n_sheets == len(info["tensors"]) + 1, f"{n_sheets - 1} tensor sheets + Index")
before_emb, before_ffn = cell(EMB, 5, 5)[0], cell(ffn, 100, 200)[0]

print("2) import the untouched workbook -> must change nothing")
t = time.time()
r = job(W + "/excel/import")["result"]
print(f"     import took {time.time() - t:.0f}s")
c = call("GET", W + "/changes")
check(r["cells"] == 0 and c["stats"]["edited"] == 0, f"0 numbers changed after an untouched round trip ({r['cells']})")

print("3) edit in Excel with one bad cell -> import must refuse and change nothing")
secs = excel_edit(xlsx, [(index[EMB], 6, 6, 0.1), (index[ffn], 101, 201, "abc")])
print(f"     Excel open + edit + save took {secs:.0f}s")
j = job(W + "/excel/import", fail_ok=True)
check(j["status"] == "error" and "not a number" in j["error"], f"import refused: {j.get('error')}")
c = call("GET", W + "/changes")
check(c["stats"]["edited"] == 0 and cell(EMB, 5, 5)[0] == before_emb,
      f"nothing imported from the refused file (edited = {c['stats']['edited']})")

print("4) fix the bad cell in Excel -> import must bring in exactly the edits")
secs = excel_edit(xlsx, [(index[ffn], 101, 201, 0.02), (index[ffn], 256, 256, -0.03)])
print(f"     Excel open + edit + save took {secs:.0f}s")
t = time.time()
r = job(W + "/excel/import")["result"]
print(f"     import took {time.time() - t:.0f}s")
check(r["cells"] == 3 and sorted(r["tensors"]) == sorted([EMB, ffn]), f"imported {r['cells']} numbers in {r['tensors']}")
v1, ch1 = cell(EMB, 5, 5)
v2, ch2 = cell(ffn, 100, 200)
v3, ch3 = cell(ffn, 255, 255)
check(np.float32(v1) == np.float32(0.1) and ch1, f"{EMB}[5,5] = {v1} (was {before_emb})")
check(np.float32(v2) == np.float32(0.02) and ch2, f"{ffn}[100,200] = {v2} (was {before_ffn})")
check(np.float32(v3) == np.float32(-0.03) and ch3, f"{ffn}[255,255] = {v3} (last cell of the block)")
c = call("GET", W + "/changes")
total = c["stats"]["original_total"]
check(c["stats"]["edited"] == 3 and c["stats"]["edited_pct"] == f"{3 * 100 / total:.10f}",
      f"Changes page: 3 weights = {c['stats']['edited_pct']} %")

print("5) undo")
call("POST", W + "/tensor/revert", {"name": EMB})
call("POST", W + "/tensor/revert", {"name": ffn})
c = call("GET", W + "/changes")
check(c["stats"]["edited"] == 0 and not c["weights"], "workspace back to the original")
xlsx.unlink()
print("ALL EXCEL CHECKS PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
