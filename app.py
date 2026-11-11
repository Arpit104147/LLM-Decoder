"""
LLM Decoder - local web app.
Run:  python app.py      (opens http://127.0.0.1:8765 in your browser)
"""
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

import gguf_core as core
import inspect_core as ic
import trace_core as tc
import fix_core as fc
import effects_core as ec
import test_core as tst
import validate_core as vc

PORT = 8765
app = Flask(__name__, static_folder="static", static_url_path="/static")
JOBS = {}
CANCEL = {}
BUSY = threading.Lock()  # one long job (extract/export/grow) at a time


def err(msg, code=400, **extra):
    return jsonify({"error": msg, **extra}), code


def ws_dir(ws):
    try:
        return core.workspace_dir(ws)
    except FileNotFoundError:
        abort(404, f"No workspace named {ws!r}")


def entry(d, name):
    for e in core.load_manifest(d):
        if e["name"] == name:
            return e
    abort(404, f"No tensor named {name!r}")


def body():
    return request.get_json(force=True, silent=True) or {}


# ------------------------------------------------------------------ background jobs

TITLES = {"trace_effects": "measuring which layers decided a token", "recommend": "computing an edit",
          "analyze": "analysing a test", "run": "running a test", "chat": "writing a chat reply",
          "export_gguf": "exporting a GGUF", "solve": "computing an edit", "extract_workspace": "extracting a model",
          "reply_effects": "measuring layers over a reply", "trace_reply_effects": "measuring layers over a reply",
          "check": "checking an edit on held-out questions", "search": "finding an edit that helps held-out questions",
          "measure_changes": "measuring the changes", "export_xlsx": "writing the Excel workbook",
          "import_xlsx": "reading the Excel workbook", "add_layers": "adding layers", "add_tokens": "adding tokens",
          "insert_slices": "inserting rows / columns", "undo_last": "undoing a change",
          "install_to_lmstudio": "adding the model to LM Studio", "logit_lens": "running the logit lens",
          "relink_source": "checking the model file"}
KEEP_JOBS = 20          # finished jobs remembered (the page reads a job's result right after it ends)
RUNNING = {}            # the job holding BUSY: {"job", "title", "started"}
STOPPABLE = {"trace_effects", "reply_effects", "trace_reply_effects", "recommend", "analyze", "solve", "run", "check", "search"}   # loops that call check_stop()


def start_job(fn, *args, stream=False, title=None, stoppable=None, **kwargs):
    """Run fn in the background. stream=True also passes emit(item) (items the page reads as they come, e.g.
    generated tokens). fn gets cancel (a threading.Event set by POST /api/jobs/<id>/cancel) when it takes one.
    stoppable: say so for a function (a lambda) whose loop calls check_stop()."""
    import inspect
    if not BUSY.acquire(blocking=False):
        return None
    for old in [j for j, v in JOBS.items() if v["status"] != "running"][:-KEEP_JOBS]:
        JOBS.pop(old, None)                             # results can be large (a whole test, a chat's trace summary)
        CANCEL.pop(old, None)
    jid = uuid.uuid4().hex[:10]
    job = JOBS[jid] = {"status": "running", "log": [], "result": None, "error": None, "stream": []}
    CANCEL[jid] = threading.Event()
    try:
        takes_cancel = "cancel" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes_cancel = False
    if stream:
        kwargs.update(emit=job["stream"].append)
    if stream or takes_cancel:
        kwargs.update(cancel=CANCEL[jid])
    job["stoppable"] = bool(stream or takes_cancel or getattr(fn, "__name__", "") in STOPPABLE) if stoppable is None else bool(stoppable)
    RUNNING.clear()
    RUNNING.update(job=jid, title=title or TITLES.get(getattr(fn, "__name__", ""), "a task"), started=time.time(),
                   stoppable=job["stoppable"])

    def run():
        tc.set_stop_flag(CANCEL[jid])
        try:
            job["result"] = fn(*args, log=lambda m: job["log"].append(str(m)), **kwargs)
            job["status"] = "done"
        except tc.Stopped:
            job["error"] = "Stopped"
            job["status"] = "error"
        except Exception as e:
            job["error"] = str(e)
            job["status"] = "error"
        finally:
            tc.set_stop_flag(None)
            RUNNING.clear()
            BUSY.release()

    threading.Thread(target=run, daemon=True).start()
    return jid


def job_response(jid):
    if jid:
        return jsonify({"job": jid, "stoppable": bool(JOBS[jid].get("stoppable"))})
    return busy_error()


def busy_error():
    r = dict(RUNNING)
    what = f" ({r['title']}, {int(time.time() - r['started'])} s so far)" if r else ""
    return err(f"Another task is still running{what}. Wait for it to finish" + (", or stop it." if r.get("stoppable") else "."),
               409, running=r.get("job"), stoppable=bool(r.get("stoppable")))


def when_idle(view):
    """For requests that change the workspace (or the loaded model) directly: refused while a background task is
    running, which reads the same files and weights (an export, a test, a reply being written)."""
    @functools.wraps(view)
    def wrapped(*a, **k):
        if BUSY.locked():
            return busy_error()
        return view(*a, **k)
    return wrapped


@app.get("/api/jobs/current")
def job_current():
    r = dict(RUNNING)
    return jsonify({**r, "seconds": int(time.time() - r["started"])} if r else {})


@app.get("/api/jobs/<jid>")
def job_status(jid):
    job = JOBS.get(jid) or abort(404)
    since, ssince = int(request.args.get("since", 0)), int(request.args.get("ssince", 0))
    return jsonify({**job, "log": job["log"][since:], "log_len": len(job["log"]),
                    "stream": job["stream"][ssince:], "stream_len": len(job["stream"])})


@app.post("/api/jobs/<jid>/cancel")
def job_cancel(jid):
    (CANCEL.get(jid) or abort(404)).set()
    return jsonify({"ok": True})


# ------------------------------------------------------------------ pages & errors

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _host_port(value):
    """(host name, port or None) of a Host header ("localhost:8765") or an Origin ("http://localhost:8765")."""
    from urllib.parse import urlsplit
    try:
        u = urlsplit(value if "://" in (value or "") else "//" + (value or ""))
        return (u.hostname or "").lower(), u.port
    except ValueError:
        return "", None


@app.before_request
def only_this_page():
    """The app has no login: it is meant for the person at this computer. A web page from somewhere else, open in the
    same browser, could otherwise send requests here (delete traces, change metadata, start jobs). Refuse requests
    that weren't made for this address (DNS rebinding) or that come from another site's page."""
    if not request.path.startswith("/api/"):
        return None
    host, port = _host_port(request.host)
    if host not in LOCAL_HOSTS:
        return err("This app only answers on 127.0.0.1 / localhost", 403)
    origin = request.headers.get("Origin")
    if origin is not None:
        o_host, o_port = _host_port(origin)
        if o_host not in LOCAL_HOSTS or (o_port or 80) != (port or 80):
            return err("Requests from other web sites are not accepted", 403)
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return err("Requests from other web sites are not accepted", 403)
    return None


@app.after_request
def no_stale_page(resp):
    """The page and its files are checked with the server on every load (cheap: unchanged files answer 304), so
    the browser never keeps running an old version of the app after an update."""
    if request.path == "/" or request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.errorhandler(404)
def not_found(e):
    return err(getattr(e, "description", "Not found"), 404)


@app.errorhandler(Exception)
def crashed(e):
    if hasattr(e, "code") and hasattr(e, "description"):
        return err(e.description, e.code)
    return err(str(e) if isinstance(e, (ValueError, RuntimeError, FileNotFoundError)) else f"{type(e).__name__}: {e}", 400)


# ------------------------------------------------------------------ model loading

@app.post("/api/browse")
def browse():
    """Native file picker (the app only runs on this computer)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(parent=root, title="Choose a GGUF model",
                                      filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
    root.destroy()
    return jsonify({"path": str(Path(path)) if path else ""})


@app.post("/api/extract")
def extract():
    b = body()
    path = (b.get("path") or "").strip().strip('"')
    if not path:
        return err("Enter the path of a .gguf file")
    p = Path(path).expanduser()
    if not p.is_file():
        return err(f"File not found: {p}")
    if p.suffix.lower() != ".gguf":
        return err("That file isn't a .gguf model")
    try:
        rows, cols = int(b.get("rows", 256)), int(b.get("cols", 256))
    except (TypeError, ValueError):
        return err("Rows and columns must be whole numbers")
    if not (1 <= rows <= 100_000 and 1 <= cols <= 100_000):
        return err("Rows and columns must be between 1 and 100000")
    name = core.workspace_name_for(p)
    if (core.WORKSPACES / name).exists() and not b.get("overwrite"):
        return err(f"A workspace for '{name}' already exists.", 409, exists=True, name=name)
    return job_response(start_job(core.extract_workspace, str(p), rows, cols, overwrite=bool(b.get("overwrite"))))


@app.get("/api/workspaces")
def workspaces():
    return jsonify(core.list_workspaces())


@app.get("/api/ws/<ws>")
def workspace(ws):
    d = ws_dir(ws)
    info = core.load_info(d)
    meta_ok, meta = True, {}
    try:
        meta = core.load_user_metadata(d)
    except ValueError as e:
        meta_ok = str(e)
    arch = meta.get("general.architecture") or info.get("architecture") or ""
    pick = lambda k: meta.get(f"{arch}.{k}")
    man = core.load_manifest(d)
    return jsonify({
        "name": ws, "source": info, "metadata_ok": meta_ok, "path": str(d), "source_missing": core.source_missing(d),
        "n_parameters": sum(e["n_elements"] for e in man), "n_tensors": len(man),
        "highlights": {
            "Layers": pick("block_count"), "Context length": pick("context_length"),
            "Embedding size": pick("embedding_length"), "FFN size": pick("feed_forward_length"),
            "Attention heads": pick("attention.head_count"), "KV heads": pick("attention.head_count_kv"),
            "Vocabulary": len(meta["tokenizer.ggml.tokens"]) if isinstance(meta.get("tokenizer.ggml.tokens"), list) else None,
            "Tokenizer": meta.get("tokenizer.ggml.model"),
        },
        "tensors": [{k: e.get(k) for k in ("name", "type", "numpy_shape", "window", "complete", "layout",
                                           "n_elements", "source", "added")} for e in man],
        "exports": [{"name": f.name, "size": f.stat().st_size} for f in sorted((d / "exports").glob("*.gguf"))],
        "history": core.list_history(ws),
        "companions": [{"path": p, "name": Path(p).name, "size": Path(p).stat().st_size if Path(p).is_file() else 0,
                        "workspace": core.safe_name(Path(p).stem) if (core.WORKSPACES / core.safe_name(Path(p).stem)).is_dir() else None}
                       for p in (info.get("companions") or core.find_companions(info["gguf_path"]))],
    })


@app.delete("/api/ws/<ws>")
@when_idle
def delete_workspace(ws):
    d = ws_dir(ws)
    tc.unload()
    core.release_sources()
    shutil.rmtree(d)
    return jsonify({"ok": True})


@app.post("/api/ws/<ws>/source")
@when_idle
def relink(ws):
    """The model file was moved or downloaded again somewhere else: use it from its new place (same model only)."""
    ws_dir(ws)
    path = (body().get("path") or "").strip().strip('"')
    if not path:
        return err("Choose the model's .gguf file")
    return jsonify(core.relink_source(ws, path, log=lambda m: None))


# ------------------------------------------------------------------ metadata

LARGE_ITEMS, LARGE_CHARS = 64, 400


def _value_summary(v):
    if isinstance(v, list):
        s = json.dumps(v[:20], ensure_ascii=False)
        return {"preview": (s[:-1] + ", …]") if len(v) > 20 else s, "large": len(v) > LARGE_ITEMS, "count": len(v)}
    s = json.dumps(v, ensure_ascii=False)
    return {"preview": s if len(s) <= LARGE_CHARS else s[:LARGE_CHARS] + "…", "large": len(s) > LARGE_CHARS, "count": None}


@app.get("/api/ws/<ws>/metadata")
def metadata(ws):
    d = ws_dir(ws)
    try:
        meta = core.load_user_metadata(d)
    except ValueError as e:
        return err(f"{e}. Fix it in the Code view (or re-extract).")
    try:
        fields = core.open_source(d, check=False)[1].fields
    except Exception:
        fields = {}
    rows = [{"key": k, "type": core.type_label(fields[k]) if k in fields else "new", **_value_summary(v)}
            for k, v in meta.items()]
    return jsonify(rows)


@app.get("/api/ws/<ws>/metadata/value")
def metadata_value(ws):
    meta = core.load_user_metadata(ws_dir(ws))
    key = request.args.get("key", "")
    if key not in meta:
        abort(404, f"No key {key!r}")
    return jsonify({"key": key, "value": meta[key]})


@app.get("/api/ws/<ws>/metadata/raw")
def metadata_raw(ws):
    return jsonify({"text": (ws_dir(ws) / "metadata.json").read_text(encoding="utf-8-sig")})


@app.put("/api/ws/<ws>/metadata/raw")
@when_idle
def metadata_raw_save(ws):
    """Save metadata.json from the code editor, after checking it is valid and exportable."""
    d = ws_dir(ws)
    text = body().get("text", "")
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as e:
        return err(f"Not valid JSON: line {e.lineno}, column {e.colno}: {e.msg}", line=e.lineno, col=e.colno)
    if not isinstance(meta, dict):
        return err("metadata.json must be one JSON object { ... }")
    core.validate_metadata(meta, core.open_source(d, check=False)[1].fields)
    (d / "metadata.json").write_text(text, encoding="utf-8")
    return jsonify({"ok": True})


@app.put("/api/ws/<ws>/metadata")
@when_idle
def metadata_set(ws):
    d = ws_dir(ws)
    b = body()
    key = (b.get("key") or "").strip()
    if not key:
        return err("Key can't be empty")
    meta = core.load_user_metadata(d)
    if b.get("new") and key in meta:
        return err(f"'{key}' already exists; edit it instead")
    meta[key] = b.get("value")
    core.validate_metadata(meta, core.open_source(d, check=False)[1].fields)
    core.write_metadata(d, meta)
    return jsonify({"ok": True})


@app.delete("/api/ws/<ws>/metadata")
@when_idle
def metadata_delete(ws):
    d = ws_dir(ws)
    key = request.args.get("key", "")
    if key == "general.architecture":
        return err("general.architecture is required")
    meta = core.load_user_metadata(d)
    meta.pop(key, None)
    core.write_metadata(d, meta)
    return jsonify({"ok": True})


@app.post("/api/ws/<ws>/metadata/reset")
@when_idle
def metadata_reset(ws):
    d = ws_dir(ws)
    key = body().get("key")
    meta = core.load_user_metadata(d)
    orig = core.read_metadata(core.open_source(d, check=False)[1])
    if key:
        if key in orig:
            meta[key] = core.json_safe(orig[key])
        else:
            meta.pop(key, None)
    else:
        meta = {k: core.json_safe(v) for k, v in orig.items() if k not in core.PSEUDO_KEYS}
    core.write_metadata(d, meta)
    return jsonify({"ok": True})


# ------------------------------------------------------------------ weights grid

def _n_edits(d, ot, e):
    """Cells of this tensor that differ from its base (CSV block + edits outside it)."""
    n = 0
    try:
        cur = core.read_csv(d / e["csv"])
        base = core.entry_window(d, ot, e)
        if cur.shape == base.shape:
            n = int((~np.isclose(cur, base, rtol=core.CHANGE_RTOL, atol=core.CHANGE_ATOL, equal_nan=True)).sum())
    except Exception:
        pass
    return n + len(core.load_sparse(d, e))


@app.get("/api/ws/<ws>/tensor")
def tensor(ws):
    """Tensor info for the grid (values come in tiles)."""
    d = ws_dir(ws)
    e = entry(d, request.args.get("name", ""))
    _, _, ot = core.open_source(d, check=False)
    R, C = core.ui_dims(e)
    return jsonify({**e, "rows": R, "cols": C, "n_changed": _n_edits(d, ot, e), "role": ic.describe_tensor(e["name"])})


@app.get("/api/ws/<ws>/tile")
def tile(ws):
    """Values of grid rows [r0, r0+nr) x cols [c0, c0+nc), with edits, plus which cells are edited."""
    d = ws_dir(ws)
    e = entry(d, request.args.get("name", ""))
    try:
        r0, nr = int(request.args.get("r0", 0)), int(request.args.get("nr", 64))
        c0, nc = int(request.args.get("c0", 0)), int(request.args.get("nc", 64))
    except ValueError:
        return err("Rows and columns must be whole numbers")
    if min(r0, c0) < 0 or min(nr, nc) < 1:
        return err("Rows and columns can't be negative")
    if nr * nc > 400_000:
        return err("Tile too large")
    _, _, ot = core.open_source(d, check=False)
    cur, base = core.tile(d, ot, e, r0, nr, c0, nc)
    mask = ~((cur == base) | (np.isnan(cur) & np.isnan(base)))
    vals = np.where(np.isfinite(cur), cur, np.nan)
    return jsonify({"r0": r0, "c0": c0, "values": [[None if x != x else float(x) for x in row] for row in vals],
                    "changed": np.argwhere(mask).tolist()})


_STATS = {}


@app.get("/api/ws/<ws>/tensor/stats")
def tensor_stats(ws):
    d = ws_dir(ws)
    e = entry(d, request.args.get("name", ""))
    info = core.load_info(d)
    key = (ws, e["name"], e["source"], e.get("npy"), info["file_mtime_ns"])
    if key not in _STATS:
        _, _, ot = core.open_source(d, check=False)
        _STATS[key] = core.tensor_stats(d, ot, e)
    return jsonify(_STATS[key])


@app.post("/api/ws/<ws>/tensor")
@when_idle
def tensor_save(ws):
    ws_dir(ws)
    b = body()
    cells = b.get("cells", [])
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            return err("Each cell must be [row, column, value]")
        r, c, v = cell
        if v is not None and not (isinstance(v, str) and v.lower() == "nan"):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return err(f"Cell [{r}, {c}]: {v!r} is not a number")
            if not math.isfinite(x) or abs(x) > 3.4028234663852886e38:
                return err(f"Cell [{r}, {c}]: {v} is too large for a weight (float32 holds up to ±3.4e38); "
                           "an infinite weight makes the model's output meaningless")
    return jsonify(core.save_cells(ws, b.get("name", ""), cells))


@app.get("/api/ws/<ws>/raw")
def raw(ws):
    """How grid cell [r, c] is really stored in the model file (quantized integers, scales, bytes)."""
    d = ws_dir(ws)
    e = entry(d, request.args.get("name", ""))
    r, c = int(request.args.get("r", 0)), int(request.args.get("c", 0))
    row, col = (0, r) if e["layout"] == "column" else (r, c)
    if e["source"] != "original":
        what = "a new tensor (all zeros until you edit it)" if e["source"] == "zeros" else "a resized tensor"
        return jsonify({"kind": "workspace", "type": e["type"],
                        "explain": f"This is {what}. It isn't in the original file yet; on export it is "
                                   f"encoded as {e['type']}."})
    _, _, ot = core.open_source(d, check=False)
    res = core.raw_storage(ot[e["orig"]], row, col)
    res["matrix_index"] = [row, col]
    return jsonify(res)


@app.post("/api/ws/<ws>/tensor/revert")
@when_idle
def tensor_revert(ws):
    ws_dir(ws)
    core.revert_tensor(ws, body().get("name", ""))
    return jsonify({"ok": True})


@app.post("/api/ws/<ws>/install")
def install(ws):
    ws_dir(ws)
    return job_response(start_job(core.install_to_lmstudio, ws))


# ------------------------------------------------------------------ grow model (structure edits)

@app.post("/api/ws/<ws>/grow/layers")
def grow_layers(ws):
    ws_dir(ws)
    b = body()
    return job_response(start_job(core.add_layers, ws, b.get("count", 1), b.get("mode", "identity"),
                                  b.get("copy_from")))


@app.post("/api/ws/<ws>/grow/tokens")
def grow_tokens(ws):
    ws_dir(ws)
    b = body()
    toks = b.get("tokens") or []
    if isinstance(toks, str):
        toks = [t for t in toks.splitlines() if t.strip()]
    return job_response(start_job(core.add_tokens, ws, toks, b.get("token_type", "user_defined"),
                                  b.get("init", "mean"), b.get("copy_from")))


@app.post("/api/ws/<ws>/grow/insert")
def grow_insert(ws):
    ws_dir(ws)
    b = body()
    return job_response(start_job(core.insert_slices, ws, b.get("name"), b.get("axis", "rows"), b.get("at"),
                                  b.get("count", 1), b.get("init", "zeros"), b.get("copy_from")))


@app.post("/api/ws/<ws>/undo")
def undo(ws):
    ws_dir(ws)
    return job_response(start_job(core.undo_last, ws))


# ------------------------------------------------------------------ changes, export, files

@app.get("/api/ws/<ws>/changes")
def changes(ws):
    ws_dir(ws)
    return jsonify(core.pending_changes(ws))


@app.post("/api/ws/<ws>/measure")
def measure(ws):
    ws_dir(ws)
    return job_response(start_job(core.measure_changes, ws))


@app.post("/api/ws/<ws>/export")
def export(ws):
    ws_dir(ws)
    return job_response(start_job(core.export_gguf, ws, exact=bool(body().get("exact"))))


@app.post("/api/ws/<ws>/excel/export")
def excel_export(ws):
    ws_dir(ws)
    return job_response(start_job(core.export_xlsx, ws))


@app.post("/api/ws/<ws>/excel/import")
def excel_import(ws):
    ws_dir(ws)
    return job_response(start_job(core.import_xlsx, ws))


def _open(path):
    """Open a file or folder with the system's default program."""
    if hasattr(os, "startfile"):
        os.startfile(path)                                   # Windows
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def _vscode(*args):
    code = shutil.which("code")
    if not code:
        return False
    subprocess.Popen([code, *map(str, args)])
    return True


@app.post("/api/ws/<ws>/open")
def open_target(ws):
    d = ws_dir(ws)
    b = body()
    what = b.get("what")
    files = {"metadata": d / "metadata.json", "script": d / "edit_weights.py", "readme": d / "README_FOR_AI.md"}
    if what in ("folder", "exports", "excel"):
        target = d if what == "folder" else d / what
        target.mkdir(exist_ok=True)
        _open(target)
    elif what == "excelfile":
        f = d / "excel" / f"{ws}_weights.xlsx"
        if not f.is_file():
            return err("Export to Excel first")
        _open(f)
    elif what == "vscode":
        _vscode(d) or _open(d)
    elif what in files:
        _vscode(d, "-g", files[what]) or _open(files[what])
    elif what == "csv":
        f = d / entry(d, b.get("name", ""))["csv"]
        _vscode(d, "-g", f) or _open(f)
    else:
        return err("Unknown target")
    return jsonify({"ok": True})


# ------------------------------------------------------------------ inspect (layer tools)

@app.get("/api/ws/<ws>/inspect/anatomy")
def inspect_anatomy(ws):
    ws_dir(ws)
    return jsonify(ic.anatomy(ws))


@app.get("/api/ws/<ws>/inspect/tokens")
def inspect_tokens(ws):
    ws_dir(ws)
    q = request.args.get("q", "")
    if request.args.get("id") not in (None, ""):
        v = ic.vocab(ws)
        i = int(request.args["id"])
        return jsonify([ic._tok(v, i)] if 0 <= i < len(v["tokens"]) else [])
    return jsonify(ic.search_tokens(ws, q) if q else [])


@app.post("/api/ws/<ws>/inspect/nearest")
def inspect_nearest(ws):
    ws_dir(ws)
    tid = int(body().get("id", 0))          # read the request now, not inside the background thread
    return job_response(start_job(lambda log: ic.nearest_tokens(ws, tid, log=log), title="finding the nearest tokens"))


@app.post("/api/ws/<ws>/inspect/neuron")
def inspect_neuron(ws):
    ws_dir(ws)
    b = body()
    name, index, axis = b.get("name"), int(b.get("index", 0)), b.get("axis") or None
    return job_response(start_job(lambda log: ic.neuron(ws, name, index, axis, log=log), title="analysing a neuron"))


@app.get("/api/ws/<ws>/inspect/lens")
def inspect_lens_info(ws):
    ws_dir(ws)
    layers = [int(x) for x in request.args.get("layers", "").split(",") if x.strip().isdigit()]
    return jsonify({"n_layers": ic.lens_layers(ws), **ic.lens_disk(ws, layers)})


@app.post("/api/ws/<ws>/inspect/lens")
def inspect_lens(ws):
    ws_dir(ws)
    b = body()
    return job_response(start_job(ic.logit_lens, ws, b.get("prompt", ""), b.get("layers", []), int(b.get("k", 8))))


@app.post("/api/ws/<ws>/inspect/lens/clear")
@when_idle
def inspect_lens_clear(ws):
    ws_dir(ws)
    return jsonify(ic.clear_lens(ws))


# ------------------------------------------------------------------ chat + trace (the model run in numpy)

@app.get("/api/ws/<ws>/chat/status")
def chat_status(ws):
    ws_dir(ws)
    return jsonify(tc.status(ws))


@app.post("/api/ws/<ws>/chat/load")
def chat_load(ws):
    ws_dir(ws)
    which = body().get("weights", "edited")
    return job_response(start_job(lambda log: (tc.get_model(ws, which, log), tc.status(ws))[1], title="loading the model"))


@app.post("/api/ws/<ws>/chat/unload")
@when_idle
def chat_unload(ws):
    ws_dir(ws)
    tc.unload()
    return jsonify(tc.status(ws))


@app.post("/api/ws/<ws>/chat")
def chat(ws):
    ws_dir(ws)
    b = body()
    try:
        opts = dict(max_new=max(1, min(2048, int(b.get("max_tokens", 256)))), temperature=float(b.get("temperature", 0)),
                    top_p=float(b.get("top_p", 0.8)), top_k=int(b.get("top_k", 20)), seed=int(b.get("seed", 0)),
                    thinking=bool(b.get("thinking")), which="original" if b.get("weights") == "original" else "edited")
    except (TypeError, ValueError):
        return err("Settings must be numbers")
    msgs = [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in b.get("messages", [])
            if m.get("role") in ("system", "user", "assistant")]
    return job_response(start_job(tc.chat, ws, msgs, stream=True, **opts))


@app.get("/api/ws/<ws>/traces")
def traces(ws):
    ws_dir(ws)
    return jsonify(tc.list_traces(ws))


@app.get("/api/ws/<ws>/trace/<tid>")
def trace(ws, tid):
    ws_dir(ws)
    return jsonify(tc.summary(ws, tid))


@app.post("/api/ws/<ws>/traces/delete")
@when_idle
def traces_delete(ws):
    ws_dir(ws)
    ids = [str(i) for i in body().get("ids", [])][:5000]
    n = 0
    for tid in ids:
        try:
            tc.delete_trace(ws, tid)
            n += 1
        except (FileNotFoundError, ValueError):
            pass
    tst.forget_traces(ws, ids)
    return jsonify({"deleted": n, "traces": tc.list_traces(ws)})


@app.delete("/api/ws/<ws>/trace/<tid>")
@when_idle
def trace_delete(ws, tid):
    ws_dir(ws)
    return jsonify(tc.delete_trace(ws, tid))


@app.get("/api/ws/<ws>/trace/<tid>/neuron")
def trace_neuron(ws, tid):
    ws_dir(ws)
    ex = request.args.get("expert")
    return jsonify(tc.neuron_over_trace(ws, tid, int(request.args.get("layer", 0)), int(request.args.get("neuron", 0)),
                                        int(ex) if ex not in (None, "", "null") else None))


@app.get("/api/ws/<ws>/trace/<tid>/pos/<int:pos>")
def trace_pos(ws, tid, pos):
    ws_dir(ws)
    t = request.args.get("target")
    if not tc.ready_for(ws, tid):
        return err("The model isn't loaded for this trace yet", 409, need_load=tc.trace_weights(ws, tid))
    return jsonify(tc.position_detail(ws, tid, pos, int(t) if t not in (None, "") else None))


# ------------------------------------------------------------------ Fix answers (computed edits) and Test

def _num_arg(b, k, dflt, kind=float):
    v = b.get(k, dflt)
    return dflt if v in (None, "") else kind(v)


@app.post("/api/ws/<ws>/fix/solve")
def fix_solve(ws):
    ws_dir(ws)
    b = body()
    try:
        opts = dict(layer=_num_arg(b, "layer", None, int), neurons=_num_arg(b, "neurons", 64, int),
                    margin=_num_arg(b, "margin", 1.0), which="original" if b.get("weights") == "original" else "edited",
                    part="attn" if b.get("part") == "attn" else "mlp")
        fixes = [{"trace": str(f["trace"]), "pos": int(f["pos"]), "target": int(f["target"])} for f in b.get("fixes", [])]
    except (TypeError, ValueError, KeyError):
        return err("Settings must be numbers")
    return job_response(start_job(fc.solve, ws, fixes, **opts))


@app.post("/api/ws/<ws>/fix/save")
@when_idle
def fix_save(ws):
    ws_dir(ws)
    b = body()
    return jsonify(fc.save(ws, str(b.get("sid", "")), str(b.get("name", "")).strip(), force=bool(b.get("force"))))


@app.get("/api/ws/<ws>/fix")
def fix_list(ws):
    ws_dir(ws)
    return jsonify(fc.list_saved(ws))


@app.delete("/api/ws/<ws>/fix/<name>")
@when_idle
def fix_delete(ws, name):
    ws_dir(ws)
    return jsonify(fc.delete(ws, name))


@app.post("/api/ws/<ws>/test/parse")
def test_parse(ws):
    ws_dir(ws)
    items = tst.parse_questions(body().get("text", ""))
    return jsonify({"n": len(items), "items": items[:5]})


@app.post("/api/ws/<ws>/test/run")
def test_run(ws):
    ws_dir(ws)
    b = body()
    try:
        opts = dict(instruction=str(b.get("instruction", "")), max_tokens=max(1, min(2048, _num_arg(b, "max_tokens", 160, int))),
                    which="original" if b.get("weights") == "original" else "edited", name=str(b.get("name", ""))[:80],
                    engine="llama" if b.get("engine") == "llama" else "tracer",
                    parallel=max(1, min(8, _num_arg(b, "parallel", 3, int))))
    except (TypeError, ValueError):
        return err("Max tokens must be a number")
    tst.parse_questions(b.get("text", ""))                       # bad input fails now, not inside the job
    return job_response(start_job(tst.run, ws, b.get("text", ""), stream=True, **opts))


@app.get("/api/ws/<ws>/test/cache")
def test_cache(ws):
    ws_dir(ws)
    return jsonify(tst.cache_info(ws))


@app.delete("/api/ws/<ws>/test/cache")
@when_idle
def test_cache_clear(ws):
    ws_dir(ws)
    return jsonify(tst.clear_cache(ws))


@app.get("/api/ws/<ws>/tests")
def tests_list(ws):
    ws_dir(ws)
    return jsonify(tst.list_tests(ws))


@app.get("/api/ws/<ws>/test/<tid>")
def test_get(ws, tid):
    ws_dir(ws)
    return jsonify(tst.get_test(ws, tid))


@app.delete("/api/ws/<ws>/test/<tid>")
@when_idle
def test_delete(ws, tid):
    ws_dir(ws)
    return jsonify(tst.delete_test(ws, tid))


@app.post("/api/ws/<ws>/test/<tid>/analyze")
def test_analyze(ws, tid):
    ws_dir(ws)
    try:
        split = _num_arg(body(), "split", None)
        measure = max(0, min(200, _num_arg(body(), "measure", 8, int)))
    except (TypeError, ValueError):
        return err("Split must be a number")
    return job_response(start_job(tst.analyze, ws, tid, split=split, measure=measure, shuffle=bool(body().get("shuffle"))))


@app.post("/api/ws/<ws>/test/<tid>/trace/<int:i>")
def test_trace(ws, tid, i):
    ws_dir(ws)
    return job_response(start_job(lambda log: next(x["trace"] for x in tst.ensure_traces(ws, tid, [i], log)["items"] if x["i"] == i),
                                  title="tracing a reply", stoppable=True))


@app.post("/api/ws/<ws>/test/<tid>/recommend")
def test_recommend(ws, tid):
    ws_dir(ws)
    b = body()
    try:
        opts = dict(layer=_num_arg(b, "layer", None, int), neurons=_num_arg(b, "neurons", 0, int),
                    margin=_num_arg(b, "margin", 1.0), use=[int(i) for i in b["use"]] if b.get("use") is not None else None,
                    split=_num_arg(b, "split", None), mode="general" if b.get("mode") == "general" else "exact",
                    rank=max(1, min(64, _num_arg(b, "rank", 4, int))), strength=max(0.0, _num_arg(b, "strength", 0.5)),
                    measure=max(0, min(200, _num_arg(b, "measure", 8, int))),
                    choose="cheapest" if b.get("choose") == "cheapest" else "responsible",
                    part="attn" if b.get("part") == "attn" else "mlp",
                    per_answer=_num_arg(b, "per_answer", None, int) or None,
                    spread=str(b["spread"]).strip() if b.get("spread") else None, shuffle=bool(b.get("shuffle")))
    except (TypeError, ValueError):
        return err("Settings must be numbers")
    return job_response(start_job(tst.recommend, ws, tid, **opts))


def _train_arg(b):
    return b.get("train") if b.get("train") in ("wrong", "all", "none") else "wrong"


@app.post("/api/ws/<ws>/test/<tid>/check")
def test_check(ws, tid):
    """Ask the held-out questions without and with the computed edit(s) and compare (validate_core.check)."""
    ws_dir(ws)
    b = body()
    sids = [str(x) for x in (b.get("sids") or ([b["sid"]] if b.get("sid") else []))]
    try:
        par = max(1, min(8, _num_arg(b, "parallel", 3, int)))
    except (TypeError, ValueError):
        return err("Settings must be numbers")
    return job_response(start_job(vc.check, ws, tid, sids, parallel=par, train=_train_arg(b)))


@app.post("/api/ws/<ws>/test/<tid>/search")
def test_search(ws, tid):
    """Compute edits with several settings and keep the one that really helps held-out questions (validate_core.search)."""
    ws_dir(ws)
    b = body()
    try:
        opts = dict(split=_num_arg(b, "split", 0.5), measure=max(0, min(200, _num_arg(b, "measure", 8, int))),
                    parallel=max(1, min(8, _num_arg(b, "parallel", 3, int))), train=_train_arg(b), shuffle=bool(b.get("shuffle")))
    except (TypeError, ValueError):
        return err("Settings must be numbers")
    return job_response(start_job(vc.search, ws, tid, **opts))


@app.get("/api/ws/<ws>/test/<tid>/checks")
def test_checks(ws, tid):
    ws_dir(ws)
    tst.get_test(ws, tid)
    return jsonify(vc.list_checks(ws, tid))


@app.post("/api/ws/<ws>/trace/<tid>/effects")
def trace_reply_effects(ws, tid):
    ws_dir(ws)
    return job_response(start_job(ec.trace_reply_effects, ws, tid))


@app.post("/api/ws/<ws>/trace/<tid>/pos/<int:pos>/effects")
def trace_effects(ws, tid, pos):
    ws_dir(ws)
    b = body()
    try:
        t = _num_arg(b, "target", None, int)
        c = _num_arg(b, "contrast", None, int)
    except (TypeError, ValueError):
        return err("Token ids must be numbers")
    return job_response(start_job(ec.trace_effects, ws, tid, pos, t, c))


@app.get("/api/ws/<ws>/readme")
def readme(ws):
    d = ws_dir(ws)
    return jsonify({"text": (d / "README_FOR_AI.md").read_text(encoding="utf-8"), "path": str(d)})


if __name__ == "__main__":
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"LLM Decoder running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
