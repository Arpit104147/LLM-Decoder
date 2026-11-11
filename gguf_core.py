"""
Core engine for LLM Decoder.

  extract_workspace(gguf)  -> workspaces/<model>/  (editable metadata.json, weights/*.csv, edit_weights.py)
  add_layers / add_tokens / insert_slices / undo_last   -> structure edits (grow the model)
  export_xlsx / import_xlsx                            -> edit the weights in Excel and read them back
  pending_changes(ws)      -> what an export would change
  export_gguf(ws)          -> workspaces/<model>/exports/<model>_edited.gguf

Design: the original .gguf stays the source of truth. tensors.json lists every tensor of the
OUTPUT model; each entry says where its numbers come from:
  source "original" -> a tensor of the original file (maybe under a new name, e.g. a copied layer)
  source "zeros"    -> all zeros (pass-through parts of new layers)
  source "npy"      -> a full array saved in added/ (resized tensors, e.g. after adding tokens)
On export, untouched original tensors are copied byte-for-byte; everything else is re-encoded.
"""
import datetime
import gc
import hashlib
import importlib.util
import json
import os
import re
import shutil
import traceback
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType as VT, GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize

ROOT = Path(__file__).resolve().parent
WORKSPACES = ROOT / "workspaces"
PSEUDO_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",   # reader-only, not real metadata
               "split.no", "split.count", "split.tensors.count"}      # split bookkeeping; exports are one file
SPLIT_RX = re.compile(r"^(?P<base>.+)-(?P<no>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)
CHUNK_VALUES = 4_000_000     # numbers decoded at once when streaming a tensor (~16 MB as float32)
# types the gguf Python library can re-quantize; others (Q4_K, Q6_K, IQ*) fall back to F16 when edited
REQUANTIZABLE = {QT.BF16, QT.Q4_0, QT.Q4_1, QT.Q5_0, QT.Q5_1, QT.Q8_0, QT.TQ1_0, QT.TQ2_0, QT.MXFP4}
INT_TYPES = {QT.I8: np.int8, QT.I16: np.int16, QT.I32: np.int32, QT.I64: np.int64}
PLAIN_TYPES = {QT.F32, QT.F16, QT.F64, *INT_TYPES}
CHANGE_RTOL, CHANGE_ATOL = 1e-6, 1e-9  # ignore rounding noise when a CSV was re-saved by Excel
# tensors whose output is added to the residual stream; zeroing them makes a layer pass-through
RESIDUAL_OUT = ("attn_output.", "ffn_down.", "ffn_down_exps.", "ffn_down_shexp.", "ssm_out.",
                "shortconv.out_proj.", "time_mix_output.", "channel_mix_value.")  # + LFM conv, RWKV
TOKEN_TYPES = {"normal": 1, "control": 3, "user_defined": 4}


# ============================================================== metadata reading

def _to_str(part):
    # the gguf library decodes strictly; a single byte-level token would break the whole vocab
    return bytes(part).decode("utf-8", errors="replace")


def _f32_short(x):
    """Shortest decimal that is exactly this float32 (1e-12, not 9.999999960041972e-13)."""
    return float(str(np.float32(x)))


def field_value(field):
    """GGUF key/value field -> plain Python (str, int, float, bool, list)."""
    if not field.types:
        return None
    main = field.types[0]
    if main == VT.ARRAY:
        if len(field.types) > 2:
            return f"<nested array, {len(field.data)} items, not decoded>"
        sub = field.types[-1]
        if sub == VT.STRING:
            return [_to_str(field.parts[i]) for i in field.data]
        vals = [v for i in field.data for v in field.parts[i].tolist()]
        if sub == VT.FLOAT32:
            return [_f32_short(v) for v in vals]
        return [bool(v) for v in vals] if sub == VT.BOOL else vals
    if main == VT.STRING:
        return _to_str(field.parts[-1])
    val = field.parts[-1].tolist()[0]
    if main == VT.FLOAT32:
        return _f32_short(val)
    return bool(val) if main == VT.BOOL else val


def read_metadata(reader):
    meta = {}
    for name, field in reader.fields.items():
        try:
            meta[name] = field_value(field)
        except Exception as e:
            meta[name] = f"<unreadable: {e}>"
    return meta


def json_safe(v):
    """NaN/Infinity aren't valid JSON."""
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    return v


def type_label(field):
    t = [x.name for x in field.types]
    return f"ARRAY[{t[-1]}]" if t and t[0] == "ARRAY" else (t[0] if t else "?")


# ============================================================== tensor reading

def tensor_dims(t):
    """(rows, cols) of a GGUF tensor viewed as a 2D matrix; ggml ne[0] is the row length."""
    ne = [int(x) for x in t.shape]
    return (int(np.prod(ne[1:])) if len(ne) > 1 else 1), ne[0]


def natural_shape(t):
    """numpy-order shape (outermost first), e.g. token_embd.weight -> (n_vocab, n_embd)."""
    return tuple(int(x) for x in reversed(t.shape))


def _decode(chunk, qtype):
    if chunk.dtype == np.uint8:                      # quantized blocks / BF16 as raw bytes
        return dequantize(np.ascontiguousarray(chunk), qtype).astype(np.float32, copy=False)
    if chunk.dtype == np.float64 or chunk.dtype.kind == "i":
        return np.array(chunk, dtype=np.float64)     # keep precision of F64 / integers
    return np.array(chunk, dtype=np.float32)


def load_rows(t, take):
    """First `take` rows of the tensor (as a 2D matrix). Only those rows are dequantized."""
    rows, cols = tensor_dims(t)
    chunk = t.data.reshape(rows, -1)[:take]
    return _decode(chunk, t.tensor_type).reshape(take, cols)


def full_tensor(t):
    """Whole tensor as floats, numpy-order shape."""
    return _decode(t.data, t.tensor_type).reshape(natural_shape(t))


def window_array(t, layout, wr, wc):
    """The part of a tensor shown in its CSV: 'matrix' = top-left wr x wc, 'column' = 1D as one column."""
    if layout == "column":
        return load_rows(t, 1)[0, :wr].reshape(-1, 1)
    return load_rows(t, wr)[:, :wc]


# ============================================================== raw storage (how a number is really stored)

def _f16(b, o):
    return float(np.frombuffer(bytes(b[o:o + 2].astype(np.uint8)), dtype=np.float16)[0])


def _k_scale_min(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _decode_block(qtype, b):
    """Decode one quantized block into its stored parts. Returns (fields, q, per-value scale, per-value offset,
    formula). value[i] = scale[i] * q[i] - offset[i]."""
    b = np.frombuffer(bytes(b), dtype=np.uint8).astype(np.int64)
    if qtype == QT.Q8_0:
        d = _f16(b, 0)
        q = b[2:34].astype(np.int8).astype(np.int64)
        q = np.where(q > 127, q - 256, q)
        return {"d (scale, f16)": d}, q, np.full(32, d), np.zeros(32), "value = d × q"
    if qtype == QT.Q4_0:
        d = _f16(b, 0)
        qs = b[2:18]
        q = np.concatenate([qs & 0xF, qs >> 4]) - 8
        return {"d (scale, f16)": d}, q, np.full(32, d), np.zeros(32), "value = d × (nibble − 8)"
    if qtype == QT.Q4_1:
        d, m = _f16(b, 0), _f16(b, 2)
        qs = b[4:20]
        q = np.concatenate([qs & 0xF, qs >> 4])
        return {"d (scale, f16)": d, "m (min, f16)": m}, q, np.full(32, d), np.full(32, -m), "value = d × q + m"
    if qtype in (QT.Q4_K, QT.Q5_K):
        d, dmin = _f16(b, 0), _f16(b, 2)
        sc = b[4:16]
        if qtype == QT.Q4_K:
            qh, qs = None, b[16:144]
        else:
            qh, qs = b[16:48], b[48:176]
        q, scale, off, subs = np.zeros(256, np.int64), np.zeros(256), np.zeros(256), []
        for chunk in range(4):
            s1, m1 = _k_scale_min(2 * chunk, sc)
            s2, m2 = _k_scale_min(2 * chunk + 1, sc)
            subs += [[int(s1), int(m1)], [int(s2), int(m2)]]
            lo, hi = qs[chunk * 32:(chunk + 1) * 32] & 0xF, qs[chunk * 32:(chunk + 1) * 32] >> 4
            if qh is not None:
                lo = lo + (((qh >> (2 * chunk)) & 1) << 4)
                hi = hi + (((qh >> (2 * chunk + 1)) & 1) << 4)
            base = chunk * 64
            q[base:base + 32], q[base + 32:base + 64] = lo, hi
            scale[base:base + 32], off[base:base + 32] = d * s1, dmin * m1
            scale[base + 32:base + 64], off[base + 32:base + 64] = d * s2, dmin * m2
        bits = "4-bit" if qtype == QT.Q4_K else "5-bit"
        return ({"d (super-block scale, f16)": d, "dmin (super-block min, f16)": dmin,
                 "sub-block [scale, min] (6-bit each, 8 sub-blocks of 32)": subs},
                q, scale, off, f"value = d × sub_scale × q − dmin × sub_min   (q is {bits})")
    if qtype == QT.Q6_K:
        ql, qh = b[0:128], b[128:192]
        scs = b[192:208].astype(np.int8).astype(np.int64)
        scs = np.where(scs > 127, scs - 256, scs)
        d = _f16(b, 208)
        q, scale = np.zeros(256, np.int64), np.zeros(256)
        for half in range(2):
            l = np.arange(32)
            L, H, S, o = ql[half * 64:], qh[half * 32:], scs[half * 8:], half * 128
            is_ = l // 16
            q[o + l] = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32
            q[o + l + 32] = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
            q[o + l + 64] = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
            q[o + l + 96] = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
            for k in range(4):
                scale[o + l + 32 * k] = d * S[is_ + 2 * k]
        return ({"d (super-block scale, f16)": d, "sub-block scales (int8, 16 sub-blocks of 16)": scs.tolist()},
                q, scale, np.zeros(256), "value = d × sub_scale × q   (q is 6-bit, −32..31)")
    return None


def raw_storage(t, row, col):
    """Exactly how element [row, col] of tensor `t` is stored in the file."""
    from gguf.constants import GGML_QUANT_SIZES
    rows, cols = tensor_dims(t)
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"[{row}, {col}] is outside {rows} x {cols}")
    qt = t.tensor_type
    bs, ts = GGML_QUANT_SIZES[qt]
    raw = t.data.reshape(rows, -1)
    bi, pos = col // bs, col % bs
    if raw.dtype != np.uint8:                                   # plain F32/F16/F64/int storage
        v = raw[row, col]
        return {"type": qt.name, "kind": "plain", "bytes": np.array(v).tobytes().hex(" "),
                "value": float(v), "value_repr": str(v), "dtype": str(raw.dtype),
                "explain": f"Stored directly as one {raw.dtype} number ({np.dtype(raw.dtype).itemsize} bytes)."}
    blk = raw[row, bi * ts:(bi + 1) * ts]
    values = dequantize(np.ascontiguousarray(blk).reshape(1, -1), qt).reshape(-1).astype(np.float64)
    res = {"type": qt.name, "kind": "block", "block_size": bs, "block_bytes": ts, "block_index": bi,
           "position": pos, "block_first_col": bi * bs, "bytes": bytes(blk).hex(" "),
           "values": values.tolist(), "value": float(values[pos]), "value_repr": str(np.float32(values[pos])),
           "bits_per_weight": round(ts * 8 / bs, 2)}
    dec = _decode_block(qt, blk) if qt != QT.BF16 else None
    if dec:
        fields, q, scale, off, formula = dec
        recon = scale * q - off
        res.update({"fields": fields, "q": q.tolist(), "formula": formula, "q_here": int(q[pos]),
                    "scale_here": float(scale[pos]), "offset_here": float(off[pos]),
                    "decode_matches": bool(np.allclose(recon, values, rtol=1e-5, atol=1e-7))})
    res["explain"] = (f"{qt.name} stores {bs} weights in {ts} bytes ({res['bits_per_weight']} bits each). "
                      f"This weight is #{pos} of block {bi} in row {row}.")
    return res


# ============================================================== workspace basics

def safe_name(name):
    return re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip(". ") or "model"


def workspace_dir(ws):
    d = (WORKSPACES / ws).resolve()
    if d.parent != WORKSPACES.resolve() or not d.is_dir():
        raise FileNotFoundError(f"No workspace named {ws!r}")
    return d


def list_workspaces():
    out = []
    if WORKSPACES.is_dir():
        for d in sorted(WORKSPACES.iterdir()):
            src = d / "source.json"
            if src.is_file():
                info = json.loads(src.read_text(encoding="utf-8"))
                out.append({"name": d.name, **{k: info.get(k) for k in
                            ("model_name", "architecture", "n_parameters", "gguf_path", "extracted_at")}})
    return out


def load_manifest(d):
    return json.loads((d / "tensors.json").read_text(encoding="utf-8"))


def save_manifest(d, man):
    (d / "tensors.json").write_text(json.dumps(man, indent=1), encoding="utf-8")


def load_info(d):
    return json.loads((d / "source.json").read_text(encoding="utf-8"))


def dumps_metadata(meta):
    """Readable JSON: one setting per line, long lists wrapped into rows (not one item per line)."""
    out, items = ["{"], list(meta.items())
    for i, (k, v) in enumerate(items):
        if isinstance(v, list) and len(v) > 8:
            per = 12 if all(isinstance(x, (int, float)) for x in v[:64]) else 8
            rows = [", ".join(json.dumps(x, ensure_ascii=False) for x in v[j:j + per]) for j in range(0, len(v), per)]
            val = "[\n    " + ",\n    ".join(rows) + "\n  ]"
        else:
            val = json.dumps(v, ensure_ascii=False)
        out.append(f"  {json.dumps(k, ensure_ascii=False)}: {val}{',' if i < len(items) - 1 else ''}")
    out.append("}")
    return "\n".join(out) + "\n"


def write_metadata(d, meta):
    (d / "metadata.json").write_text(dumps_metadata(meta), encoding="utf-8")


def _layout(numpy_shape, max_rows, max_cols):
    """(layout, window, complete, matrix) for a tensor of this shape."""
    if len(numpy_shape) == 1:
        n = int(numpy_shape[0])
        return "column", [n, 1], True, [1, n]
    cols = int(numpy_shape[-1])
    rows = int(np.prod(numpy_shape[:-1]))
    wr, wc = min(rows, max_rows), min(cols, max_cols)
    return "matrix", [wr, wc], (wr == rows and wc == cols), [rows, cols]


def _shape_fields(numpy_shape, info):
    layout, window, complete, matrix = _layout(numpy_shape, info["max_rows"], info["max_cols"])
    return {"numpy_shape": [int(x) for x in numpy_shape], "shape_ggml": [int(x) for x in reversed(numpy_shape)],
            "matrix": matrix, "layout": layout, "window": window, "complete": complete,
            "n_elements": int(np.prod(numpy_shape))}


def _csv_fmt(e):
    return "%.17g" if e["type"] in ("F64", "I8", "I16", "I32", "I64") else "%.9g"


def write_csv(d, e, arr):
    p = d / e["csv"]
    np.savetxt(p, arr, delimiter=",", fmt=_csv_fmt(e))
    e["csv_mtime_ns"] = p.stat().st_mtime_ns


def read_csv(path):
    return np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64, encoding="utf-8-sig")


def _unique_csv(man, name, extra=()):
    used = {Path(e["csv"]).stem.lower() for e in man} | {x.lower() for x in extra}
    f = safe_name(name)
    while f.lower() in used:
        f += "_"
    return f


EDIT_TEMPLATE = '''"""
Full-tensor weight edits. This file runs when you press Export (after the CSV/grid edits are applied).
Leave edit() empty if you only edit in the grid or the CSV files.

  w.names()            -> list of all tensor names (including layers/tokens you added)
  w.info(name)         -> {"shape": numpy shape, "type": "Q4_K", ...}
  w[name]              -> the WHOLE tensor as a numpy float array (numpy-order shape,
                          e.g. token_embd.weight is (n_vocab, n_embd): row = token id)
  w[name] = array      -> replace it (same shape)
  in-place edits work too:  w[name][0, :] *= 2

Only tensors you actually change are re-encoded; everything else is copied unchanged.
"""
import numpy as np


def edit(w):
    # Examples (uncomment to use):
    # w["blk.0.attn_q.weight"][:, 0] = 0.0                 # zero one column
    # w["output_norm.weight"] *= 1.05                      # scale a norm vector
    # emb = w["token_embd.weight"]; emb[100] = emb[200]    # copy token 200's embedding onto token 100
    pass
'''

AI_README = """# Editable LLM workspace: {model_name}

Source model: `{gguf_path}`
Architecture: `{architecture}` · {n_parameters:,} parameters · {n_tensors} tensors

This folder is an editable copy of a GGUF model. Edit the files below, then press
**Export GGUF** in the LLM Decoder app (or run `python gguf_core.py export {ws}`) to build
`exports/{ws}_edited.gguf`.

## Files you may edit

| File | What it is | Rules |
|---|---|---|
| `metadata.json` | All model settings: architecture, context length, RoPE, chat template, tokenizer | Keep valid JSON. Keep each value's type (number stays number, list stays list). Add a key = new setting, delete a key = remove it. |
| `weights/<tensor>.csv` | The top-left part of each weight matrix as plain numbers | Change numbers only. Do NOT add/remove rows or columns. Row r, column c = element [r, c] of the matrix. 1D tensors are one column. |
| `edits/<tensor>.json` | Edits to cells OUTSIDE the CSV block, as `{{"row,col": value}}` (null = NaN) | Grid coordinates (1D tensors: `"i,0"`). Add, change or remove entries; delete the file to drop them all. |
| `edit_weights.py` | Python `edit(w)` function for whole-tensor edits with numpy | Use this for bulk math on whole tensors. |

Read-only reference: `tensors.json` (every tensor of the output model: name, shape, type, which CSV
holds it, whether the CSV is the complete tensor or a crop), `source.json` (where the model came from).
Don't edit `tensors.json` or the `added/` and `history/` folders by hand; to add layers, tokens or
rows/columns use the app's "Grow model" tab (or `python gguf_core.py add-layers ...`).

## Tips for AI assistants
- Tensor names follow llama.cpp: `token_embd.weight` (rows = token ids), `blk.N.attn_q/k/v/output.weight`,
  `blk.N.ffn_up/gate/down.weight`, `*_norm.weight`, `output.weight` (rows = token ids).
- A CSV covers only `window` rows x cols of the tensor (see `tensors.json` -> `complete`).
  To change values beyond it, write code in `edit_weights.py`.
- Edited quantized tensors are re-quantized (Q4_0/Q5_0/Q8_0/BF16...) or stored as F16 when the
  original type (Q4_K, Q6_K, IQ*) can't be re-quantized in Python. The file can grow a little.
- Large edits to weights change the model's behavior unpredictably; small, targeted changes are safer.
"""


class SplitReader:
    """The parts of a split model (model-00001-of-00004.gguf, ...) read as ONE model:
    metadata from the first part, tensors from all parts (each stays memory-mapped from its own file)."""

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        self.parts = [GGUFReader(p, "r") for p in self.paths]
        self.fields = self.parts[0].fields
        self.tensors = [t for r in self.parts for t in r.tensors]
        names = [t.name for t in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError("The parts of this split model contain the same tensor twice")


def split_parts(src):
    """All files of a split model, given any one of them (first part first). [src] for a normal file."""
    src = Path(src)
    m = SPLIT_RX.match(src.name)
    if not m:
        return [src]
    n = int(m["n"])
    width = len(m["no"])
    parts = [src.with_name(f"{m['base']}-{i:0{width}d}-of-{m['n']}.gguf") for i in range(1, n + 1)]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"This is part of a {n}-file split model, but these parts are missing next to it: "
                                f"{', '.join(missing)}. Download all parts into the same folder.")
    return parts


def open_gguf(src):
    """GGUFReader for a normal file, SplitReader for a split model."""
    parts = split_parts(src)
    if len(parts) == 1:
        r = GGUFReader(str(parts[0]), "r")
        sc = r.fields.get("split.count")
        if sc is not None and field_value(sc) > 1:
            raise ValueError(f"{Path(src).name} says it is one of {field_value(sc)} parts of a split model, but its name "
                             f"doesn't follow the usual '-00001-of-0000N.gguf' pattern, so the other parts can't be found.")
        return r, parts
    return SplitReader(parts), parts


def workspace_name_for(path):
    """Workspace folder name for a model file (split parts share one name: the part before -0000N-of-)."""
    p = Path(path)
    m = SPLIT_RX.match(p.name)
    return safe_name(m["base"] if m and int(m["n"]) > 1 else p.stem)


def _parts_stamp(parts):
    return [[str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in parts]


KEPT_ON_REEXTRACT = ("tests", "traces")       # your results: extracting a model again replaces everything else


def extract_workspace(gguf_path, max_rows=256, max_cols=256, overwrite=False, log=print):
    """Make the workspace of a model file. overwrite: extract an existing workspace again from scratch; its edits
    are deleted, its saved tests and traces are kept."""
    hold = None
    try:
        first = split_parts(Path(gguf_path).expanduser().resolve())[0]
        old = WORKSPACES / workspace_name_for(first)
    except Exception:
        old = None
    if overwrite and old is not None and old.is_dir():
        hold = WORKSPACES / f".{old.name}.kept"
        for sub in KEPT_ON_REEXTRACT:
            if (old / sub).is_dir() and not (hold / sub).exists():
                hold.mkdir(parents=True, exist_ok=True)
                (old / sub).rename(hold / sub)
    try:
        return _extract_workspace(gguf_path, max_rows, max_cols, overwrite, log)
    finally:
        if hold is not None and hold.is_dir():                     # put them back, whether or not it worked
            old.mkdir(parents=True, exist_ok=True)
            n = 0
            for sub in list(hold.iterdir()):
                if not (old / sub.name).exists():
                    n += sum(1 for _ in sub.iterdir())
                    sub.rename(old / sub.name)
            try:
                hold.rmdir()
            except OSError:
                pass
            if n:
                log(f"Kept the saved tests and traces ({n} item(s))")


def _extract_workspace(gguf_path, max_rows, max_cols, overwrite, log):
    src = Path(gguf_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"File not found: {src}")
    if max_rows < 1 or max_cols < 1:
        raise ValueError("rows and columns must be at least 1")
    log(f"Reading {src}")
    reader, parts = open_gguf(src)
    src = Path(parts[0])
    if len(parts) > 1:
        log(f"Split model: reading all {len(parts)} parts as one model")
    meta = read_metadata(reader)
    ws = workspace_name_for(src)
    d = WORKSPACES / ws
    if d.exists():
        if not overwrite:
            raise FileExistsError(f"Workspace '{ws}' already exists (re-extracting deletes its edits)")
        gc.collect()
        try:
            shutil.rmtree(d)
        except PermissionError as ex:
            raise RuntimeError(f"Can't replace workspace '{ws}': a file in it is open in another program "
                               f"({Path(ex.filename).name}). Close it (e.g. eject the model in LM Studio) and retry.")
    for sub in ("weights", "exports", "added", "history"):
        (d / sub).mkdir(parents=True)

    split = len(parts) if len(parts) > 1 else None
    write_metadata(d, {k: json_safe(v) for k, v in meta.items() if k not in PSEUDO_KEYS})
    log("Wrote metadata.json")

    info = {"max_rows": max_rows, "max_cols": max_cols}
    man, n = [], len(reader.tensors)
    for i, t in enumerate(reader.tensors, 1):
        e = {"name": t.name, "csv": f"weights/{_unique_csv(man, t.name)}.csv", "type": t.tensor_type.name,
             "source": "original", "orig": t.name, "added": False, **_shape_fields(natural_shape(t), info),
             "n_bytes": int(t.n_bytes)}
        write_csv(d, e, window_array(t, e["layout"], *e["window"]))
        man.append(e)
        if i % 20 == 0 or i == n:
            log(f"Extracted {i}/{n} tensors")
    save_manifest(d, man)

    stamp = _parts_stamp(parts)
    info.update({
        "gguf_path": str(src), "file_size": sum(p[1] for p in stamp), "file_mtime_ns": max(p[2] for p in stamp),
        "parts": stamp if len(parts) > 1 else None,
        "extracted_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "gguf_version": field_value(reader.fields["GGUF.version"]) if "GGUF.version" in reader.fields else None,
        "architecture": meta.get("general.architecture"), "model_name": meta.get("general.name") or src.stem,
        "n_tensors": n, "n_parameters": sum(x["n_elements"] for x in man),
        "split_count": split,
        "tensor_types": sorted({x["type"] for x in man}),
        "companions": find_companions(src),
        "fingerprint": source_fingerprint({t.name: t for t in reader.tensors}),
    })
    (d / "source.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (d / "edit_weights.py").write_text(EDIT_TEMPLATE, encoding="utf-8")
    (d / "README_FOR_AI.md").write_text(AI_README.format(ws=ws, n_tensors=n, **{
        k: info[k] for k in ("model_name", "gguf_path", "architecture", "n_parameters")}), encoding="utf-8")
    log(f"Done: workspace '{ws}'")
    return ws


# ============================================================== tensor sources

_SOURCES = {}  # parsed original files, so the app doesn't re-parse a model (and its tokenizer) on every click


def _cached_reader(src, stamp):
    key = str(src)
    hit = _SOURCES.get(key)
    if hit and hit["stamp"] == stamp:
        return hit
    reader, _ = open_gguf(src)
    hit = _SOURCES[key] = {"stamp": stamp, "reader": reader,
                           "ot": {t.name: t for t in reader.tensors}, "meta": None}
    while len(_SOURCES) > 4:
        _SOURCES.pop(next(iter(_SOURCES)))
    return hit


def original_metadata(reader):
    """json-safe metadata of an original file (cached when it comes from open_source)."""
    for hit in _SOURCES.values():
        if hit["reader"] is reader:
            if hit["meta"] is None:
                hit["meta"] = {k: json_safe(v) for k, v in read_metadata(reader).items()}
            return hit["meta"]
    return {k: json_safe(v) for k, v in read_metadata(reader).items()}


def release_sources():
    """Close cached files (needed on Windows before deleting/replacing them)."""
    _SOURCES.clear()
    gc.collect()


def source_fingerprint(ot):
    """What identifies a model file's contents: every tensor's name, type, shape and size, and the first bytes of
    ~48 tensors spread over the file. The same after downloading the file again; different for another model."""
    h = hashlib.sha256()
    names = sorted(ot)
    for n in names:
        t = ot[n]
        h.update(f"{n}|{t.tensor_type.name}|{[int(x) for x in t.shape]}|{int(t.n_bytes)}\n".encode())
    for n in names[::max(1, len(names) // 48)]:
        h.update(np.ascontiguousarray(np.asarray(ot[n].data).reshape(-1)[:2048]).tobytes())
    return h.hexdigest()[:32]


def _same_model(d, info, ot):
    """Is this file (its tensors `ot`) the model the workspace was extracted from?"""
    if info.get("fingerprint"):
        return source_fingerprint(ot) == info["fingerprint"]
    # workspaces made before fingerprints: the same tensors, types and sizes as when it was extracted
    if len(ot) != info.get("n_tensors"):
        return False
    orig = [e for e in load_manifest(d) if e.get("source") == "original"]
    for e in orig:
        t = ot.get(e["orig"])
        if t is None or t.tensor_type.name != e["type"] or int(t.n_elements) != int(e["n_elements"]):
            return False
    # ... and the same numbers: the CSV copies in weights/ hold each tensor's top-left block as it was extracted. A
    # few of them may have been edited since, so most (not all) of a sample must still match the file.
    sample = orig[::max(1, len(orig) // 8)][:8]
    same = 0
    for e in sample:
        try:
            cur, base = read_csv(d / e["csv"]), entry_window(d, ot, e)
            same += cur.shape == base.shape and bool(np.isclose(cur, base, rtol=CHANGE_RTOL, atol=CHANGE_ATOL, equal_nan=True).mean() > 0.9)
        except Exception:
            pass
    return same * 2 >= len(sample)


def _write_info(d, info):
    """Rewrite source.json without changing its date (the date is part of the weights' stamp: tests and traces
    would otherwise look out of date although no weight changed)."""
    p = d / "source.json"
    st = p.stat()
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def _source_parts(info):
    return [Path(p[0]) for p in info["parts"]] if info.get("parts") else [Path(info["gguf_path"])]


def source_missing(d):
    """The source file(s) of a workspace that are not on disk any more."""
    return [str(p) for p in _source_parts(load_info(d)) if not p.is_file()]


def open_source(d, check=True):
    info = load_info(d)
    src = Path(info["gguf_path"])
    parts = _source_parts(info)
    for p in parts:
        if not p.is_file():
            raise FileNotFoundError(f"The original model file is gone: {p}")
    stamp = _parts_stamp(parts)
    hit = _cached_reader(src, stamp)
    if check:
        want = info.get("parts") or [[str(src), info["file_size"], info["file_mtime_ns"]]]
        same = len(stamp) == len(want) and all(size == size0 and mt == mt0 for (_p, size, mt), (_q, size0, mt0) in zip(stamp, want))
        if not same and [list(x) for x in stamp] != info.get("accepted"):
            # a new date with the same size: the file was copied or downloaded again. Accept it when it holds the
            # same model (and remember that), instead of asking for a new extraction that would throw the edits away.
            sizes = len(stamp) == len(want) and all(x[1] == y[1] for x, y in zip(stamp, want))
            if not (sizes and _same_model(d, info, hit["ot"])):
                raise RuntimeError(f"The original model file changed since extraction: {stamp[0][0]}. Re-extract it.")
            info["accepted"] = [list(x) for x in stamp]
            info.setdefault("fingerprint", source_fingerprint(hit["ot"]))
            _write_info(d, info)
    return info, hit["reader"], hit["ot"]


def relink_source(ws, path, log=print):
    """Point a workspace at its model file in a new place (moved, or downloaded again somewhere else). Refused
    unless the file holds the same model."""
    d = workspace_dir(ws)
    info = load_info(d)
    src = Path(str(path).strip().strip('"')).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"File not found: {src}")
    reader, parts = open_gguf(src)
    ot = {t.name: t for t in reader.tensors}
    stamp = _parts_stamp(parts)
    want = info.get("parts") or [[info["gguf_path"], info["file_size"], info["file_mtime_ns"]]]
    if len(stamp) != len(want) or any(x[1] != y[1] for x, y in zip(stamp, want)) or not _same_model(d, info, ot):
        raise ValueError(f"{src.name} isn't the model this workspace was extracted from (its size or its tensors are "
                         "different). Pick the same file, or extract this one as a new workspace.")
    info["gguf_path"] = str(parts[0])
    if info.get("parts"):
        info["parts"] = [[x[0], y[1], y[2]] for x, y in zip(stamp, want)]
    info["accepted"] = [list(x) for x in stamp]
    info.setdefault("fingerprint", source_fingerprint(ot))
    info["companions"] = find_companions(parts[0])
    del reader, ot
    release_sources()
    _write_info(d, info)
    log(f"Workspace '{ws}' now reads {parts[0]}")
    return {"path": info["gguf_path"]}


def entry_full(d, ot, e):
    """The tensor's base numbers (before CSV/script edits), numpy-order shape."""
    if e["source"] == "original":
        return full_tensor(ot[e["orig"]])
    if e["source"] == "zeros":
        return np.zeros(e["numpy_shape"], dtype=np.float32)
    return np.array(np.load(d / e["npy"]), dtype=np.float64 if _csv_fmt(e) == "%.17g" else np.float32)


def entry_window(d, ot, e):
    """The base numbers shown in the entry's CSV (before CSV edits)."""
    wr, wc = e["window"]
    if e["source"] == "original":
        return window_array(ot[e["orig"]], e["layout"], wr, wc).astype(np.float64)
    if e["source"] == "zeros":
        return np.zeros((wr, 1) if e["layout"] == "column" else (wr, wc))
    a = np.load(d / e["npy"], mmap_mode="r")
    if e["layout"] == "column":
        return np.array(a.reshape(-1)[:wr], dtype=np.float64).reshape(-1, 1)
    return np.array(a.reshape(e["matrix"])[:wr, :wc], dtype=np.float64)


def _apply_csv(full, new, mask, e):
    if e["layout"] == "column":
        flat = full.reshape(-1)
        flat[: new.shape[0]][mask[:, 0]] = new[:, 0][mask[:, 0]]
    else:
        r, c = e["window"]
        block = full.reshape(e["matrix"])[:r, :c]
        block[mask] = new[mask]
    return full


def csv_changes(d, ot, man=None, only_touched=True):
    """{tensor name: (new window, changed mask, entry)} for every CSV whose numbers differ from its base."""
    out = {}
    for e in man if man is not None else load_manifest(d):
        p = d / e["csv"]
        if not p.is_file():
            raise FileNotFoundError(f"{e['csv']} is missing (don't delete weight files; restore it or re-extract)")
        if only_touched and p.stat().st_mtime_ns == e["csv_mtime_ns"]:
            continue
        try:
            new = read_csv(p)
        except Exception as ex:
            raise ValueError(f"{e['csv']}: can't read numbers ({ex})")
        base = entry_window(d, ot, e)
        if new.shape != base.shape:
            raise ValueError(f"{e['csv']}: has {new.shape[0]} x {new.shape[1]} numbers but must have "
                             f"{base.shape[0]} x {base.shape[1]} (don't add or remove rows/columns; "
                             f"use Grow model -> insert rows/columns instead)")
        mask = ~np.isclose(new, base, rtol=CHANGE_RTOL, atol=CHANGE_ATOL, equal_nan=True)
        if mask.any():
            out[e["name"]] = (new, mask, e)
    return out


def current_full(d, ot, e):
    """Base numbers with the entry's CSV and sparse edits applied."""
    full = entry_full(d, ot, e)
    ch = csv_changes(d, ot, [e], only_touched=False)
    if e["name"] in ch:
        new, mask, _ = ch[e["name"]]
        _apply_csv(full, new, mask, e)
    _apply_sparse(full, e, load_sparse(d, e))
    return full


# ---------------------------------------------------------------- whole-tensor view & sparse edits
# The grid can show and edit ANY cell of a tensor. Cells inside the CSV block are saved in the CSV
# (so VS Code / AI tools see them); cells outside it go to edits/<tensor>.json as {"row,col": value}.
# Coordinates are grid coordinates: (row, col) of the 2D matrix; 1D tensors are one column (i, 0).

def ui_dims(e):
    return (int(e["numpy_shape"][0]), 1) if e["layout"] == "column" else (int(e["matrix"][0]), int(e["matrix"][1]))


def edits_rel(e):
    return f"edits/{Path(e['csv']).stem}.json"


def load_sparse(d, e):
    p = d / edits_rel(e)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as ex:
        raise ValueError(f"{edits_rel(e)} is not valid JSON: line {ex.lineno}: {ex.msg}")
    out = {}
    R, C = ui_dims(e)
    for k, v in raw.items():
        try:
            r, c = (int(x) for x in k.split(","))
        except ValueError:
            raise ValueError(f"{edits_rel(e)}: key {k!r} must look like \"row,col\"")
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"{edits_rel(e)}: cell {k} is outside the {R} x {C} tensor")
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise ValueError(f"{edits_rel(e)}: cell {k} must be a number (or null for NaN)")
        out[(r, c)] = float("nan") if v is None else float(v)
    return out


def save_sparse(d, e, sp):
    p = d / edits_rel(e)
    if not sp:
        p.unlink(missing_ok=True)
        return
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({f"{r},{c}": (None if v != v else v) for (r, c), v in sorted(sp.items())}, indent=0),
                 encoding="utf-8")


def _apply_sparse(full, e, sp):
    if not sp:
        return full
    view = full.reshape(-1) if e["layout"] == "column" else full.reshape(e["matrix"])
    for (r, c), v in sp.items():
        if e["layout"] == "column":
            view[r] = v
        else:
            view[r, c] = v
    return full


def _coords_new_part(e, coords):
    if not e.get("inserts") or not coords:
        return 0
    a = np.array(list(coords), dtype=np.int64).reshape(-1, 2)
    return int(_inserted_mask(e, a[:, 0], a[:, 1] if e["layout"] != "column" else np.zeros(len(a), np.int64)).sum())


def base_tile(d, ot, e, r0, nr, c0, nc):
    """Base values (before any edits) of grid rows [r0, r0+nr) x cols [c0, c0+nc)."""
    R, C = ui_dims(e)
    r1, c1 = min(R, r0 + nr), min(C, c0 + nc)
    if e["layout"] == "column":
        if e["source"] == "zeros":
            return np.zeros((r1 - r0, 1))
        if e["source"] == "npy":
            return np.array(np.load(d / e["npy"], mmap_mode="r").reshape(-1)[r0:r1], dtype=np.float64).reshape(-1, 1)
        return load_rows(ot[e["orig"]], 1)[0, r0:r1].astype(np.float64).reshape(-1, 1)
    if e["source"] == "zeros":
        return np.zeros((r1 - r0, c1 - c0))
    if e["source"] == "npy":
        return np.array(np.load(d / e["npy"], mmap_mode="r").reshape(e["matrix"])[r0:r1, c0:c1], dtype=np.float64)
    t = ot[e["orig"]]
    rows, cols = tensor_dims(t)
    chunk = t.data.reshape(rows, -1)[r0:r1]
    return _decode(chunk, t.tensor_type).reshape(r1 - r0, cols)[:, c0:c1].astype(np.float64)


def tile(d, ot, e, r0, nr, c0, nc):
    """(current values, base values) of a grid rectangle: base + CSV edits + sparse edits."""
    base = base_tile(d, ot, e, r0, nr, c0, nc)
    cur = base.copy()
    r1, c1 = r0 + base.shape[0], c0 + base.shape[1]
    wr, wc = e["window"]
    if r0 < wr and c0 < wc:
        csv = read_csv(d / e["csv"])
        rr, cc = min(r1, wr, csv.shape[0]), min(c1, wc, csv.shape[1])
        part, b = csv[r0:rr, c0:cc], cur[: rr - r0, : cc - c0]
        # only real edits count: CSV text of an unchanged float32 is not bit-identical to it
        edited = ~np.isclose(part, b, rtol=CHANGE_RTOL, atol=CHANGE_ATOL, equal_nan=True)
        b[edited] = part[edited]
    for (r, c), v in load_sparse(d, e).items():
        if r0 <= r < r1 and c0 <= c < c1:
            cur[r - r0, c - c0] = v
    return cur, base


def _same_value(e, a, b):
    if a != a and b != b:
        return True
    if e["type"] in ("F64", "I8", "I16", "I32", "I64"):
        return a == b
    return np.float32(a) == np.float32(b)


def save_cells(ws, name, cells):
    """Save grid edits [(row, col, value)] anywhere in the tensor."""
    d = workspace_dir(ws)
    man = load_manifest(d)
    e = next((x for x in man if x["name"] == name), None)
    if e is None:
        raise ValueError(f"No tensor named {name!r}")
    _, _, ot = open_source(d, check=False)
    R, C = ui_dims(e)
    wr, wc = e["window"]
    csv, csv_dirty = read_csv(d / e["csv"]), False
    sp = load_sparse(d, e)
    base_rows = {}
    for r, c, v in cells:
        r, c = int(r), int(c)
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"Cell [{r}, {c}] is outside the {R} x {C} tensor")
        v = float("nan") if (v is None or (isinstance(v, str) and v.lower() == "nan")) else float(v)
        if r < wr and c < wc:
            csv[r, c] = v
            csv_dirty = True
        else:
            if r not in base_rows:
                base_rows[r] = base_tile(d, ot, e, r, 1, 0, C)[0]
            if _same_value(e, v, float(base_rows[r][c])):
                sp.pop((r, c), None)          # back to the original value: no edit to remember
            else:
                sp[(r, c)] = v
    if csv_dirty:
        np.savetxt(d / e["csv"], csv, delimiter=",", fmt=_csv_fmt(e))
    save_sparse(d, e, sp)
    return {"csv": csv_dirty, "sparse": len(sp)}


def revert_tensor(ws, name):
    d = workspace_dir(ws)
    man = load_manifest(d)
    e = next((x for x in man if x["name"] == name), None)
    if e is None:
        raise ValueError(f"No tensor named {name!r}")
    _, _, ot = open_source(d, check=False)
    write_csv(d, e, entry_window(d, ot, e))
    save_sparse(d, e, {})
    save_manifest(d, man)


def tensor_stats(d, ot, e, chunk_rows=4096):
    """min / max / mean / std of the WHOLE tensor (original values), read in chunks to save memory."""
    R, C = ui_dims(e)
    n = s = s2 = 0.0
    lo, hi, bad = np.inf, -np.inf, 0
    for r0 in range(0, R, chunk_rows if e["layout"] != "column" else R):
        b = base_tile(d, ot, e, r0, chunk_rows if e["layout"] != "column" else R, 0, C)
        f = b[np.isfinite(b)]
        bad += b.size - f.size
        if f.size:
            n += f.size; s += f.sum(); s2 += (f * f).sum()
            lo, hi = min(lo, f.min()), max(hi, f.max())
    mean = s / n if n else 0.0
    return {"min": float(lo) if n else None, "max": float(hi) if n else None, "mean": mean,
            "std": float(np.sqrt(max(s2 / n - mean * mean, 0))) if n else None, "nonfinite": int(bad),
            "count": int(R * C)}


# ============================================================== metadata edits

def load_user_metadata(d):
    try:
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise ValueError(f"metadata.json is not valid JSON: line {e.lineno}, column {e.colno}: {e.msg}")
    if not isinstance(meta, dict):
        raise ValueError("metadata.json must be a JSON object { ... }")
    return meta


def metadata_diff(reader, user_meta):
    orig = {k: v for k, v in original_metadata(reader).items() if k not in PSEUDO_KEYS}
    return {
        "changed": [k for k in user_meta if k in orig and user_meta[k] != orig[k]],
        "added": [k for k in user_meta if k not in orig],
        "removed": [k for k in orig if k not in user_meta],
    }


def _coerce(key, x, vt):
    if vt == VT.BOOL:
        if not isinstance(x, bool):
            raise ValueError(f"metadata '{key}': must be true or false")
        return x
    if vt in (VT.UINT8, VT.INT8, VT.UINT16, VT.INT16, VT.UINT32, VT.INT32, VT.UINT64, VT.INT64):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or float(x) != int(x):
            raise ValueError(f"metadata '{key}': must be a whole number, got {x!r}")
        bits = {VT.UINT8: 8, VT.INT8: 8, VT.UINT16: 16, VT.INT16: 16, VT.UINT32: 32, VT.INT32: 32,
                VT.UINT64: 64, VT.INT64: 64}[vt]
        lo, hi = (0, 2 ** bits - 1) if vt.name.startswith("U") else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        if not lo <= int(x) <= hi:
            raise ValueError(f"metadata '{key}': {x} is out of range for {vt.name}")
        return int(x)
    if vt in (VT.FLOAT32, VT.FLOAT64):
        if isinstance(x, str) and x.lower() in ("nan", "inf", "-inf"):
            return float(x)
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"metadata '{key}': must be a number, got {x!r}")
        return float(x)
    if vt == VT.STRING:
        if not isinstance(x, str):
            raise ValueError(f"metadata '{key}': must be text, got {x!r}")
        return x
    raise ValueError(f"metadata '{key}': unsupported type {vt.name}")


def _infer_type(key, x):
    if isinstance(x, bool):
        return VT.BOOL
    if isinstance(x, int):
        return VT.INT32 if -2 ** 31 <= x < 2 ** 31 else VT.INT64
    if isinstance(x, float):
        return VT.FLOAT32
    if isinstance(x, str):
        return VT.STRING
    raise ValueError(f"metadata '{key}': can't store {type(x).__name__} (use number, text, true/false or a list)")


def validate_metadata(meta, fields):
    """Raise ValueError with a clear message if any value can't be written with its type."""
    if not isinstance(meta.get("general.architecture"), str) or not meta["general.architecture"]:
        raise ValueError("metadata 'general.architecture' is required and must be text")
    for key, value in meta.items():
        if key in PSEUDO_KEYS:
            raise ValueError(f"'{key}' is computed automatically; remove it from metadata.json")
        f = fields.get(key)
        if f is not None and f.types[0] == VT.ARRAY:
            if not isinstance(value, list):
                raise ValueError(f"metadata '{key}': must stay a list [ ... ]")
            for x in value:
                _coerce(key, x, f.types[-1])
        elif f is not None:
            _coerce(key, value, f.types[0])
        elif isinstance(value, list):
            sub = _infer_type(key, value[0]) if value else VT.STRING
            for x in value:
                _coerce(key, x, sub)
        else:
            _infer_type(key, value)


def _write_kv(writer, key, value, field):
    if field is not None:                                  # existing key: keep its original type
        main = field.types[0]
        if main == VT.ARRAY:
            if not isinstance(value, list):
                raise ValueError(f"metadata '{key}': must stay a list")
            sub = field.types[-1]
            writer.add_key_value(key, [_coerce(key, x, sub) for x in value], VT.ARRAY, sub_type=sub)
        else:
            writer.add_key_value(key, _coerce(key, value, main), main)
        return
    if isinstance(value, list):                            # new key: infer the type
        sub = _infer_type(key, value[0]) if value else VT.STRING
        writer.add_key_value(key, [_coerce(key, x, sub) for x in value], VT.ARRAY, sub_type=sub)
    else:
        vt = _infer_type(key, value)
        writer.add_key_value(key, _coerce(key, value, vt), vt)


def _raw_array(field):
    sub = field.types[-1]
    if sub == VT.STRING:
        return [bytes(field.parts[i]) for i in field.data]
    vals = [v for i in field.data for v in field.parts[i].tolist()]
    return [bool(v) for v in vals] if sub == VT.BOOL else vals


def _copy_kv(writer, key, field):
    """Byte-exact copy of an unchanged key (keeps invalid-UTF-8 tokens and exact number types)."""
    main = field.types[0]
    if main == VT.ARRAY:
        if len(field.types) > 2:
            raise ValueError(f"metadata '{key}': nested arrays can't be exported")
        writer.add_key_value(key, _raw_array(field), VT.ARRAY, sub_type=field.types[-1])
    elif main == VT.STRING:
        writer.add_key_value(key, bytes(field.parts[-1]), VT.STRING)
    else:
        v = field.parts[-1].tolist()[0]
        writer.add_key_value(key, bool(v) if main == VT.BOOL else v, main)


def _write_extended_array(writer, key, value, field, orig_value):
    """A list that only grew at the end (e.g. tokens added): keep the original items byte-exact."""
    sub = field.types[-1]
    raw = _raw_array(field)
    extra = [_coerce(key, x, sub) for x in value[len(orig_value):]]
    writer.add_key_value(key, raw + extra, VT.ARRAY, sub_type=sub)


# ============================================================== structure edits (with undo)

class _Change:
    """Snapshot of every file an operation touches, so it can be undone."""

    def __init__(self, d, label):
        self.d, self.label = d, label
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.dir = d / "history" / self.id
        self.saved, self.created = [], []

    def touch(self, rel):
        rel = str(rel).replace("\\", "/")
        if rel in self.saved or rel in self.created:
            return
        p = self.d / rel
        if p.exists():
            dst = self.dir / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            self.saved.append(rel)
        else:
            self.created.append(rel)

    def commit(self, details):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "change.json").write_text(json.dumps({
            "label": self.label, "details": details, "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "saved": self.saved, "created": self.created}, indent=1), encoding="utf-8")

    def rollback(self):
        _restore(self.d, self.dir, self.saved, self.created)


def _restore(d, hdir, saved, created):
    for rel in created:
        try:
            (d / rel).unlink()
        except FileNotFoundError:
            pass
    for rel in saved:
        shutil.copy2(hdir / "files" / rel, d / rel)
    shutil.rmtree(hdir, ignore_errors=True)


def list_history(ws):
    d = workspace_dir(ws)
    out = []
    for h in sorted((d / "history").glob("*/change.json")):
        c = json.loads(h.read_text(encoding="utf-8"))
        out.append({"id": h.parent.name, "label": c["label"], "details": c["details"], "time": c["time"]})
    return out


def undo_last(ws, log=print):
    d = workspace_dir(ws)
    hs = sorted((d / "history").glob("*/change.json"))
    if not hs:
        raise ValueError("Nothing to undo")
    c = json.loads(hs[-1].read_text(encoding="utf-8"))
    gc.collect()
    _restore(d, hs[-1].parent, c["saved"], c["created"])
    log(f"Undid: {c['label']}")
    return c["label"]


def _structural(fn):
    """Wrap a structure edit: snapshot, run, commit (or roll back on error)."""
    def run(ws, *args, log=print, **kw):
        d = workspace_dir(ws)
        info, reader, ot = open_source(d)
        man, meta = load_manifest(d), load_user_metadata(d)
        ch = _Change(d, fn.__name__)
        ch.touch("tensors.json")
        ch.touch("metadata.json")
        try:
            label, details = fn(d, info, ot, man, meta, ch, *args, log=log, **kw)
            save_manifest(d, man)
            write_metadata(d, meta)
            ch.label = label
            ch.commit(details)
        except Exception:
            gc.collect()
            ch.rollback()
            raise
        log(f"Done: {label}")
        return {"label": label, "details": details}
    run.__name__ = fn.__name__
    return run


def _editable_type(type_name, ndim):
    """Output type for a re-encoded tensor: keep it if we can write it, else F16 (2D+) / F32 (1D)."""
    import kquants
    qt = QT[type_name]
    if qt in PLAIN_TYPES or qt in REQUANTIZABLE or qt.name in kquants.ENCODERS:
        return type_name
    return "F16" if ndim >= 2 else "F32"


def _new_entry(man, name, source, numpy_shape, type_name, info, **extra):
    e = {"name": name, "csv": f"weights/{_unique_csv(man, name)}.csv", "type": type_name, "source": source,
         "added": True, **_shape_fields(numpy_shape, info), **extra}
    return e


def _arch_key(meta, key):
    arch = meta.get("general.architecture")
    return f"{arch}.{key}"


@_structural
def add_layers(d, info, ot, man, meta, ch, count=1, mode="identity", copy_from=None, log=print):
    """Append `count` transformer blocks after the last one.
    mode 'identity': copy of a layer with its residual outputs zeroed (the model's output is unchanged
                     for pre-norm models like Llama/Qwen/Mistral/Gemma); 'copy': exact copy of a layer."""
    count = int(count)
    if not 1 <= count <= 64:
        raise ValueError("Add between 1 and 64 layers at a time")
    if mode not in ("identity", "copy"):
        raise ValueError("mode must be 'identity' or 'copy'")
    bkey = _arch_key(meta, "block_count")
    L = meta.get(bkey)
    if not isinstance(L, int):
        raise ValueError(f"metadata has no '{bkey}', so this model's layers can't be extended")
    s = L - 1 if copy_from in (None, "") else int(copy_from)
    if not 0 <= s < L:
        raise ValueError(f"copy_from must be a layer between 0 and {L - 1}")
    prefix = f"blk.{s}."
    src = [e for e in man if e["name"].startswith(prefix)]
    if not src:
        raise ValueError(f"No tensors named {prefix}* found")
    insert_at = max(i for i, e in enumerate(man) if e["name"].startswith(f"blk.{L - 1}.")) + 1 \
        if any(e["name"].startswith(f"blk.{L - 1}.") for e in man) else len(man)
    new = []
    for j in range(L, L + count):
        for e in src:
            rest = e["name"][len(prefix):]
            name = f"blk.{j}.{rest}"
            if any(x["name"] == name for x in man + new):
                raise ValueError(f"Tensor {name} already exists")
            zero = mode == "identity" and rest.startswith(RESIDUAL_OUT)
            ne = {k: v for k, v in e.items() if k not in ("csv", "csv_mtime_ns")}
            ne.update(name=name, added=True, csv=f"weights/{_unique_csv(man + new, name)}.csv")
            ch.touch(ne["csv"])
            if zero:
                ne.update(source="zeros", type=_editable_type(e["type"], len(e["numpy_shape"])))
                ne.pop("orig", None); ne.pop("npy", None)
                write_csv(d, ne, entry_window(d, ot, ne))
            else:
                shutil.copy2(d / e["csv"], d / ne["csv"])   # carries over edits made to the source layer
                ne["csv_mtime_ns"] = 0                        # force a content check on export
                if (d / edits_rel(e)).is_file():
                    ch.touch(edits_rel(ne))
                    (d / "edits").mkdir(exist_ok=True)
                    shutil.copy2(d / edits_rel(e), d / edits_rel(ne))
            new.append(ne)
    man[insert_at:insert_at] = new
    # per-layer lists in metadata (some architectures have them) grow with the layer count
    arch = meta["general.architecture"]
    grown = []
    for k, v in list(meta.items()):
        if k.startswith(arch + ".") and isinstance(v, list) and len(v) == L:
            meta[k] = v + [v[s]] * count
            grown.append(k)
    meta[bkey] = L + count
    log(f"Added {count} layer(s): blk.{L}..blk.{L + count - 1} ({len(new)} tensors, {mode} of layer {s})")
    return (f"Added {count} layer{'s' if count > 1 else ''} on top ({'pass-through' if mode == 'identity' else 'copy'} "
            f"of layer {s})", {"layers": [L, L + count - 1], "tensors": len(new), "metadata_lists_grown": grown})


def _materialize_insert(d, info, ot, man, ch, e, axis, at, count, init, copy_from=None):
    """Turn entry `e` into a resized array saved in added/, inserting `count` rows/cols at `at`."""
    ndim = len(e["numpy_shape"])
    if ndim > 2:
        raise ValueError(f"{e['name']}: inserting is only supported for 1D and 2D tensors")
    if ndim == 1 and axis != "rows":
        raise ValueError(f"{e['name']} is a 1D vector; insert values (rows) instead")
    full = entry_full(d, ot, e)
    baked = csv_changes(d, ot, [e], only_touched=False)
    if e["name"] in baked:                        # grid edits get baked into the resized array
        new_, mask_, _ = baked[e["name"]]
        _apply_csv(full, new_, mask_, e)
        n_new = int(new_part(e, mask_))
        e["baked_edits"] = e.get("baked_edits", 0) + int(mask_.sum()) - n_new
        e["baked_new"] = e.get("baked_new", 0) + n_new
    sp = load_sparse(d, e)
    if sp:                                        # edits outside the CSV block get baked in too
        _apply_sparse(full, e, sp)
        n_new = _coords_new_part(e, sp.keys())
        e["baked_edits"] = e.get("baked_edits", 0) + len(sp) - n_new
        e["baked_new"] = e.get("baked_new", 0) + n_new
        ch.touch(edits_rel(e))
        save_sparse(d, e, {})
    ax = 0 if axis == "rows" else 1
    n = full.shape[ax]
    if at in (None, "", "end"):
        at = n
    at, count = int(at), int(count)
    if not 0 <= at <= n:
        raise ValueError(f"Position must be between 0 and {n}")
    if not 1 <= count <= 1_000_000:
        raise ValueError("Count must be at least 1")
    if init == "zeros":
        piece = np.zeros_like(np.take(full, [0] * count, axis=ax))
    elif init == "mean":
        piece = np.repeat(full.mean(axis=ax, keepdims=True), count, axis=ax)
    elif init == "copy":
        src = int(copy_from) if copy_from not in (None, "") else max(at - 1, 0)
        if not 0 <= src < n:
            raise ValueError(f"Copy source must be between 0 and {n - 1}")
        piece = np.take(full, [src] * count, axis=ax)
    else:
        raise ValueError("init must be 'zeros', 'mean' or 'copy'")
    new = np.concatenate([np.take(full, range(at), axis=ax), piece, np.take(full, range(at, n), axis=ax)], axis=ax)
    npy = f"added/{safe_name(e['name'])}.{ch.id}.npy"
    ch.touch(npy)
    ch.touch(e["csv"])
    (d / "added").mkdir(exist_ok=True)
    np.save(d / npy, new.astype(np.float64 if _csv_fmt(e) == "%.17g" else np.float32))
    e["inserted"] = e.get("inserted", 0) + int(new.size - full.size)
    e.setdefault("inserts", []).append([axis, at, count])  # to tell new cells from original ones later
    e.update(source="npy", npy=npy, type=_editable_type(e["type"], ndim), **_shape_fields(new.shape, info))
    e.pop("orig", None)
    write_csv(d, e, entry_window(d, ot, e))
    return new.shape


@_structural
def add_tokens(d, info, ot, man, meta, ch, tokens, token_type="user_defined", init="mean", copy_from=None, log=print):
    """Add new tokens to the vocabulary, with new rows in token_embd.weight (and output.weight)."""
    tokens = [t for t in (tokens or []) if t != ""]
    if not tokens:
        raise ValueError("Enter at least one token")
    if token_type not in TOKEN_TYPES:
        raise ValueError(f"token type must be one of {', '.join(TOKEN_TYPES)}")
    vocab = meta.get("tokenizer.ggml.tokens")
    if not isinstance(vocab, list):
        raise ValueError("This model has no tokenizer.ggml.tokens list")
    have = set(vocab)
    dup = [t for t in tokens if t in have] + [t for i, t in enumerate(tokens) if t in tokens[:i]]
    if dup:
        raise ValueError(f"Already in the vocabulary (or listed twice): {', '.join(map(repr, dup[:5]))}")
    n0, k = len(vocab), len(tokens)
    targets = [e for e in man if e["name"] in ("token_embd.weight", "output.weight")]
    if not any(e["name"] == "token_embd.weight" for e in targets):
        raise ValueError("No token_embd.weight tensor found")
    for e in targets:
        if len(e["numpy_shape"]) != 2 or e["numpy_shape"][0] != n0:
            raise ValueError(f"{e['name']} has {e['numpy_shape'][0]} rows but the vocabulary has {n0} tokens; "
                             f"can't add tokens safely")
    if init == "copy" and copy_from not in (None, ""):
        if not 0 <= int(copy_from) < n0:
            raise ValueError(f"Token id to copy must be between 0 and {n0 - 1}")
    meta["tokenizer.ggml.tokens"] = vocab + tokens
    extended = ["tokenizer.ggml.tokens"]
    if isinstance(meta.get("tokenizer.ggml.scores"), list) and len(meta["tokenizer.ggml.scores"]) == n0:
        meta["tokenizer.ggml.scores"] = meta["tokenizer.ggml.scores"] + [0.0] * k
        extended.append("tokenizer.ggml.scores")
    if isinstance(meta.get("tokenizer.ggml.token_type"), list) and len(meta["tokenizer.ggml.token_type"]) == n0:
        meta["tokenizer.ggml.token_type"] = meta["tokenizer.ggml.token_type"] + [TOKEN_TYPES[token_type]] * k
        extended.append("tokenizer.ggml.token_type")
    vkey = _arch_key(meta, "vocab_size")
    if isinstance(meta.get(vkey), int):
        meta[vkey] += k
        extended.append(vkey)
    for e in targets:
        _materialize_insert(d, info, ot, man, ch, e, "rows", n0, k, init,
                            copy_from if init == "copy" else None)
        log(f"{e['name']}: {n0} -> {n0 + k} rows ({init})")
    ids = list(range(n0, n0 + k))
    return (f"Added {k} token{'s' if k > 1 else ''} (ids {ids[0]}–{ids[-1]})",
            {"tokens": tokens, "ids": ids, "tensors": [e["name"] for e in targets], "metadata": extended})


@_structural
def insert_slices(d, info, ot, man, meta, ch, name, axis="rows", at=None, count=1, init="zeros", copy_from=None,
                  log=print):
    """Insert rows or columns into one tensor. Changes its shape: the model only loads if every
    tensor and setting that depends on this size is changed to match."""
    e = next((x for x in man if x["name"] == name), None)
    if e is None:
        raise ValueError(f"No tensor named {name!r}")
    if axis not in ("rows", "cols"):
        raise ValueError("axis must be 'rows' or 'cols'")
    old = list(e["numpy_shape"])
    shape = _materialize_insert(d, info, ot, man, ch, e, axis, at, count, init, copy_from)
    log(f"{name}: {old} -> {list(shape)}")
    what = "row" if axis == "rows" else "column"
    return (f"Inserted {count} {what}{'s' if int(count) > 1 else ''} into {name}",
            {"tensor": name, "from": old, "to": list(shape)})


# ============================================================== companion files & LM Studio

def find_companions(gguf_path):
    """Vision/audio encoder files (mmproj*.gguf) that belong to a model: same folder, not the model itself."""
    p = Path(gguf_path)
    if p.name.lower().startswith("mmproj"):
        return []
    return [str(x) for x in sorted(p.parent.glob("*.gguf")) if x.name.lower().startswith("mmproj") and x != p]


def lmstudio_models_dir():
    home = Path.home()
    for f in (home / ".lmstudio" / "settings.json", home / ".cache" / "lm-studio" / "settings.json"):
        try:
            folder = json.loads(f.read_text(encoding="utf-8")).get("downloadsFolder")
            if folder and Path(folder).is_dir():
                return Path(folder)
        except Exception:
            pass
    for cand in (home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError("LM Studio's models folder wasn't found (is LM Studio installed?)")


def _link_or_copy(src, dst, log):
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)                  # same drive: instant, no extra disk space
        log(f"Linked {dst.name}")
    except OSError:
        log(f"Copying {dst.name} ({Path(src).stat().st_size / 1e6:.0f} MB)...")
        shutil.copy2(src, dst)


def install_to_lmstudio(ws, log=print):
    """Put the exported model (plus its vision encoder, edited or original) into LM Studio's models folder."""
    d = workspace_dir(ws)
    info = load_info(d)
    model = d / "exports" / f"{ws}_edited.gguf"
    if not model.is_file():
        raise FileNotFoundError("Build the edited GGUF first (Export)")
    target = lmstudio_models_dir() / "llm-decoder" / f"{ws}-edited"
    target.mkdir(parents=True, exist_ok=True)
    _link_or_copy(model, target / f"{ws}-edited.gguf", log)
    used = []
    for comp in info.get("companions") or find_companions(info["gguf_path"]):
        cws = safe_name(Path(comp).stem)
        edited = WORKSPACES / cws / "exports" / f"{cws}_edited.gguf"
        src = edited if edited.is_file() else Path(comp)
        _link_or_copy(src, target / Path(comp).name, log)
        used.append({"file": Path(comp).name, "edited": edited.is_file()})
    log(f"Done: {target}  (restart or refresh LM Studio's model list to see it)")
    return {"path": str(target), "companions": used}


# ============================================================== Excel round trip

def _sheet_name(name, used):
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[-31:].strip("'") or "tensor"
    if base.lower() == "history":
        base = "history_"
    cand, i = base, 1
    while cand.lower() in used:
        suf = f"~{i}"
        cand, i = base[: 31 - len(suf)] + suf, i + 1
    used.add(cand.lower())
    return cand


def export_xlsx(ws, log=print):
    """Write the workspace's current weight grids (with your edits) to an editable Excel file."""
    import xlsxwriter
    d = workspace_dir(ws)
    man = load_manifest(d)
    out = d / "excel" / f"{ws}_weights.xlsx"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    wb = xlsxwriter.Workbook(str(tmp), {"constant_memory": True, "nan_inf_to_errors": True})
    index = wb.add_worksheet("Index")
    used, rows = {"index"}, []
    for i, e in enumerate(man, 1):
        sheet = _sheet_name(e["name"], used)
        ws_ = wb.add_worksheet(sheet)
        arr = read_csv(d / e["csv"])
        for r in range(arr.shape[0]):
            ws_.write_row(r, 0, arr[r].tolist())
        rows.append([e["name"], sheet, f"{arr.shape[0]} x {arr.shape[1]}", e["type"],
                     "x".join(map(str, e["numpy_shape"])), "complete" if e["complete"] else "top-left block"])
        if i % 20 == 0 or i == len(man):
            log(f"Wrote {i}/{len(man)} sheets")
    hdr = wb.add_format({"bold": True, "bg_color": "#DDDDDD"})
    index.write_row(0, 0, ["tensor", "sheet", "grid size", "type", "full shape", "grid shows"], hdr)
    for i, r in enumerate(rows, 1):
        index.write_row(i, 0, r)
    index.write(len(rows) + 2, 0, "Edit numbers on the tensor sheets, save, then use 'Import from Excel' in "
                                  "LLM Decoder. Don't rename sheets or add/remove rows or columns.")
    index.set_column(0, 0, 42)
    index.set_column(1, 5, 20)
    wb.close()
    try:
        os.replace(tmp, out)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{out.name} is open in Excel. Close it and export again.")
    log(f"Done: {out}")
    return {"path": str(out)}


def import_xlsx(ws, path=None, log=print):
    """Read edited numbers back from the Excel file into the weight CSVs."""
    import openpyxl
    d = workspace_dir(ws)
    man = {e["name"]: e for e in load_manifest(d)}
    p = Path(path) if path else d / "excel" / f"{ws}_weights.xlsx"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; export to Excel first")
    try:
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    except PermissionError:
        raise RuntimeError(f"{p.name} is locked; save and close it in Excel first")
    try:
        if "Index" not in wb.sheetnames:
            raise ValueError("This Excel file has no Index sheet (was it made by LLM Decoder?)")
        pairs = [(r[0], r[1]) for r in wb["Index"].iter_rows(min_row=2, max_col=2, values_only=True) if r[0] and r[1]]
        changed_tensors, cells, updates = [], 0, []
        for i, (tname, sheet) in enumerate(pairs, 1):
            e = man.get(tname)
            if e is None or sheet not in wb.sheetnames:
                continue
            cur = read_csv(d / e["csv"])
            vals = np.full(cur.shape, np.nan)
            for r, row in enumerate(wb[sheet].iter_rows(max_row=cur.shape[0], max_col=cur.shape[1], values_only=True)):
                for c, v in enumerate(row):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals[r, c] = v
                    elif isinstance(v, str) and v.strip():
                        try:
                            vals[r, c] = float(v)
                        except ValueError:
                            if v.startswith("#"):    # Excel error such as #NUM! = NaN/Inf weight
                                vals[r, c] = np.nan
                            else:
                                raise ValueError(f"Sheet '{sheet}' cell R{r + 1}C{c + 1}: {v!r} is not a number")
                    elif v is None:
                        raise ValueError(f"Sheet '{sheet}' cell R{r + 1}C{c + 1} is empty (don't delete numbers)")
            mask = ~np.isclose(vals, cur, rtol=CHANGE_RTOL, atol=CHANGE_ATOL, equal_nan=True)
            if mask.any():
                cur[mask] = vals[mask]
                updates.append((e, cur))
                changed_tensors.append(tname)
                cells += int(mask.sum())
            if i % 20 == 0 or i == len(pairs):
                log(f"Checked {i}/{len(pairs)} sheets")
    finally:
        wb.close()          # also on errors, so Excel can save the file again
    # write only after every sheet was read without errors, so a refused file changes nothing
    for e, cur in updates:
        np.savetxt(d / e["csv"], cur, delimiter=",", fmt=_csv_fmt(e))
    log(f"Imported {cells} changed number(s) in {len(changed_tensors)} tensor(s)")
    return {"tensors": changed_tensors, "cells": cells}


# ============================================================== pending changes & export

def computed_patches(d):
    """Saved Fix-answers edits (computed/<name>.npz: tensor, columns, values to add), in name order."""
    q = Path(d) / "computed"
    return sorted(q.glob("*.npz")) if q.is_dir() else []


def read_computed(path):
    """[(tensor, expert or -1, columns, values to add)] of one computed edit file (old single-patch files too)."""
    with np.load(path) as z:
        if "n" not in z.files:
            return [(str(z["tensor"]), -1, z["cols"], z["delta"])]
        return [(str(z[f"tensor{i}"]), int(z[f"expert{i}"]), z[f"cols{i}"], z[f"delta{i}"]) for i in range(int(z["n"]))]


def script_has_code(d):
    p = d / "edit_weights.py"
    return p.is_file() and p.read_text(encoding="utf-8") != EDIT_TEMPLATE


def pct(n, total, places=10):
    """Exact percentage n/total*100 as a decimal string (integer math, no float rounding error)."""
    from fractions import Fraction
    if not total:
        return "0." + "0" * places
    f = Fraction(int(n) * 100, int(total))
    q = int(round(f * 10 ** places))
    return f"{q // 10 ** places}.{q % 10 ** places:0{places}d}"


def _inserted_mask(e, ri, ci):
    """For cells at current (row, col) indices, True where the cell was inserted (not an original weight)."""
    ri, ci = np.array(ri, dtype=np.int64), np.array(ci, dtype=np.int64)
    new = np.zeros(ri.shape, dtype=bool)
    for axis, at, count in reversed(e.get("inserts", [])):
        x = ri if axis == "rows" else ci
        hit = (x >= at) & (x < at + count) & ~new
        new |= hit
        x[(x >= at + count)] -= count
    return new


def new_part(e, mask):
    """How many changed cells of a grid mask lie in inserted rows/columns."""
    if not e.get("inserts") or not mask.any():
        return 0
    idx = np.argwhere(mask)
    ri, ci = (idx[:, 0], np.zeros(len(idx), np.int64)) if e["layout"] == "column" else (idx[:, 0], idx[:, 1])
    return int(_inserted_mask(e, ri, ci).sum())


def _new_part_full(e, diff):
    """Same as new_part for a full-tensor difference mask (numpy-order shape)."""
    if not e.get("inserts") or not diff.any():
        return 0
    idx = np.argwhere(diff.reshape(-1) if len(e["numpy_shape"]) == 1 else diff.reshape(e["matrix"]))
    ri, ci = (idx[:, 0], np.zeros(len(idx), np.int64)) if len(e["numpy_shape"]) == 1 else (idx[:, 0], idx[:, 1])
    return int(_inserted_mask(e, ri, ci).sum())


def change_stats(man, ot, edited_by_tensor, new_by_tensor=None):
    """Exact counts of how much of the model's weights changed.
    edited_by_tensor: {tensor: values that differ from its base}; new_by_tensor: {tensor: how many of those
    lie in inserted rows/columns (they are new weights, not original ones)}."""
    new_by_tensor = new_by_tensor or {}
    total = sum(int(t.n_elements) for t in ot.values())          # parameters of the ORIGINAL model
    edited_orig, edited_new, rows = 0, 0, []
    added = 0
    for e in man:
        n = int(edited_by_tensor.get(e["name"], 0))
        if e.get("added"):                                         # whole tensor is new (added layer)
            added += e["n_elements"]
            o, nw = 0, n
        else:
            added += int(e.get("inserted", 0))
            nw = int(new_by_tensor.get(e["name"], 0)) + int(e.get("baked_new", 0))
            o = n - int(new_by_tensor.get(e["name"], 0)) + int(e.get("baked_edits", 0))
        edited_orig += o
        edited_new += nw
        if o + nw:
            rows.append({"name": e["name"], "edited": o, "edited_new": nw, "n_elements": e["n_elements"],
                         "new": bool(e.get("added")), "pct_of_tensor": pct(o + nw, e["n_elements"], 6)})
    return {"original_total": total, "edited": edited_orig, "edited_pct": pct(edited_orig, total),
            "added": added, "added_pct": pct(added, total), "edited_in_new": edited_new,
            "final_total": sum(e["n_elements"] for e in man), "tensors": rows}


def pending_changes(ws):
    d = workspace_dir(ws)
    _, reader, ot = open_source(d)
    man = load_manifest(d)
    res = {"metadata": None, "metadata_error": None, "weights": [], "weights_error": None,
           "script": script_has_code(d), "structure": list_history(ws)}
    try:
        res["metadata"] = metadata_diff(reader, load_user_metadata(d))
    except Exception as e:
        res["metadata_error"] = str(e)
    counts, newp = {}, {}
    try:
        ch = csv_changes(d, ot, man)
        counts = {k: int(m.sum()) for k, (_, m, _) in ch.items()}
        newp = {k: new_part(e, m) for k, (_, m, e) in ch.items()}
        for e in man:
            sp = load_sparse(d, e)
            if sp:
                counts[e["name"]] = counts.get(e["name"], 0) + len(sp)
                newp[e["name"]] = newp.get(e["name"], 0) + _coords_new_part(e, sp.keys())
        res["weights"] = [{"name": k, "cells": v} for k, v in counts.items()]
    except Exception as e:
        res["weights_error"] = str(e)
    res["stats"] = change_stats(man, ot, counts, newp)
    res["stats"]["includes_script"] = False
    res["computed"] = [p.stem for p in computed_patches(d)]
    return res


def _grid_cols(e):
    return int(e["numpy_shape"][0]) if e["layout"] == "column" else int(e["matrix"][1])


def _grid_edits(d, ot, man, log=print):
    """Grid / CSV / Excel edits of every tensor as {name: {flat index: value}}, without loading whole tensors.
    Flat index = row * cols + col of the 2D matrix (1D tensors: the index). Also returns exact counts."""
    edits, counts, newp = {}, {}, {}
    for name, (new, mask, e) in csv_changes(d, ot, man).items():
        ed, cols = edits.setdefault(name, {}), _grid_cols(e)
        for r, c in zip(*np.nonzero(mask)):
            ed[int(r) if e["layout"] == "column" else int(r) * cols + int(c)] = float(new[r, c])
        counts[name] = int(mask.sum())
        newp[name] = new_part(e, mask)
        log(f"Grid edit: {name} ({int(mask.sum())} numbers)")
    for e in man:                                        # edits outside the CSV block
        sp = load_sparse(d, e)
        if sp:
            ed, cols = edits.setdefault(e["name"], {}), _grid_cols(e)
            for (r, c), v in sp.items():
                ed[r if e["layout"] == "column" else r * cols + c] = v
            counts[e["name"]] = counts.get(e["name"], 0) + len(sp)
            newp[e["name"]] = newp.get(e["name"], 0) + _coords_new_part(e, sp.keys())
            log(f"Grid edit (outside the CSV block): {e['name']} ({len(sp)} numbers)")
    return edits, counts, newp


def _apply_flat(full, ed):
    if ed:
        flat = full.reshape(-1)
        idx = np.fromiter(ed.keys(), dtype=np.int64, count=len(ed))
        flat[idx] = np.fromiter(ed.values(), dtype=np.float64, count=len(ed))
    return full


def _apply_all_edits(d, ot, man, log=print, extra_computed=()):
    """Grid/CSV edits + edit_weights.py. Returns (Weights, grid edits, modified names, exact counts, new-part counts).
    Without a script nothing is loaded into memory: export streams each edited tensor. With a script, the
    tensors it touches are loaded whole (with the grid edits applied) so it sees the model you are building."""
    edits, counts, newp = _grid_edits(d, ot, man, log)
    w = Weights(d, ot, man)
    modified = set(edits) | {e["name"] for e in man if e["source"] != "original"}
    patches = computed_patches(d) + [Path(p) for p in extra_computed]   # extra: an unsaved edit being checked
    if not script_has_code(d) and not patches:
        return w, edits, modified, counts, newp        # grid masks are already exact counts
    w._pending = edits                                 # applied when the script (or export) loads a tensor
    if script_has_code(d):
        log("Running edit_weights.py")
        spec = importlib.util.spec_from_file_location("edit_weights_user", d / "edit_weights.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            if hasattr(mod, "edit"):
                mod.edit(w)
        except Exception:
            raise RuntimeError("edit_weights.py failed:\n" + traceback.format_exc(limit=-3))
    for p in patches:                                  # Fix answers (computed edits): added last
        for name, expert, cols, delta in read_computed(p):
            if name not in w:
                raise RuntimeError(f"Computed edit {p.stem}: no tensor named {name!r}")
            arr = w[name]
            M = arr[expert] if expert >= 0 and arr.ndim == 3 and expert < arr.shape[0] else (arr if expert < 0 else None)
            if M is None or M.ndim != 2 or delta.shape != (M.shape[0], len(cols)) or (len(cols) and cols.max() >= M.shape[1]):
                raise RuntimeError(f"Computed edit {p.stem} doesn't fit {name} {arr.shape} (the model changed?)")
            M[:, cols] += delta.astype(M.dtype)
            log(f"Computed edit {p.stem}: {name}{f' expert {expert}' if expert >= 0 else ''}, {len(cols)} column(s)")
    for name, arr in w._cache.items():
        base = entry_full(d, ot, w._e[name])
        dmask = ~((arr == base) | (np.isnan(arr) & np.isnan(base)))
        diff = int(np.count_nonzero(dmask))
        if diff:
            if name not in edits:
                log(f"Script edit: {name} ({diff} numbers)")
            counts[name] = diff
            newp[name] = _new_part_full(w._e[name], dmask)
            modified.add(name)
        else:
            counts.pop(name, None)
            newp.pop(name, None)
        del base, dmask
    return w, edits, modified, counts, newp


def measure_changes(ws, log=print):
    """Exact change statistics including what edit_weights.py does (dry run, nothing is written)."""
    d = workspace_dir(ws)
    _, _, ot = open_source(d)
    man = load_manifest(d)
    _, _, _, counts, newp = _apply_all_edits(d, ot, man, log=log)
    st = change_stats(man, ot, counts, newp)
    st["includes_script"] = True
    log(f"Changed {st['edited']:,} of {st['original_total']:,} original weights = {st['edited_pct']} %")
    return st


class Weights:
    """What edit_weights.py receives as `w`."""

    def __init__(self, d, ot, man):
        self._d, self._ot = d, ot
        self._e = {e["name"]: e for e in man}
        self._cache = {}
        self._pending = {}           # grid edits applied when a tensor is first loaded

    def names(self):
        return list(self._e)

    def info(self, name):
        e = self._e[name]
        return {"shape": tuple(e["numpy_shape"]), "type": e["type"], "n_elements": e["n_elements"]}

    def __contains__(self, name):
        return name in self._e

    def __getitem__(self, name):
        if name not in self._e:
            raise KeyError(f"No tensor named {name!r}")
        if name not in self._cache:
            self._cache[name] = _apply_flat(entry_full(self._d, self._ot, self._e[name]), self._pending.get(name))
        return self._cache[name]

    def __setitem__(self, name, value):
        if name not in self._e:
            raise KeyError(f"No tensor named {name!r} (add layers/tokens in the Grow model tab)")
        arr = np.array(value, dtype=np.float64 if np.asarray(value).dtype == np.float64 else np.float32)
        shape = tuple(self._e[name]["numpy_shape"])
        if arr.shape != shape:
            raise ValueError(f"{name}: new array has shape {arr.shape}, must be {shape}")
        self._cache[name] = arr


def _encode(arr, numpy_shape, type_name, orig_type=None, orig_tensor=None):
    """Encode floats as `type_name`. Returns (data, raw_dtype or None, written type, note)."""
    arr = np.asarray(arr).reshape(numpy_shape)
    tt = QT[type_name]
    if tt == QT.F32:
        return arr.astype(np.float32), None, "F32", ""
    if tt == QT.F16:
        return arr.astype(np.float16), None, "F16", ""
    if tt == QT.F64:
        return arr.astype(np.float64), None, "F64", ""
    if tt in INT_TYPES:
        info = np.iinfo(INT_TYPES[tt])
        return np.clip(np.rint(arr), info.min, info.max).astype(INT_TYPES[tt]), None, tt.name, ""
    from gguf.constants import GGML_QUANT_SIZES
    bs, ts = GGML_QUANT_SIZES[tt]
    shape = tuple(int(x) for x in numpy_shape)
    rows, cols = (int(np.prod(shape[:-1])) if len(shape) > 1 else 1), shape[-1]
    if cols % bs == 0:
        nb = cols // bs
        a2 = arr.reshape(rows, cols).astype(np.float32)
        if (orig_tensor is not None and orig_tensor.tensor_type == tt and orig_tensor.data.dtype == np.uint8
                and natural_shape(orig_tensor) == shape):
            # re-encode ONLY the blocks that contain an edit; every other block keeps its original bytes
            base = full_tensor(orig_tensor).reshape(rows, cols)
            blk = (~((a2 == base) | (np.isnan(a2) & np.isnan(base)))).reshape(rows, nb, bs).any(-1)
            del base
            k = int(blk.sum())
            if k == 0:
                return orig_tensor.data, tt, tt.name, "unchanged"
            enc = _quantize_blocks(a2.reshape(rows, nb, bs)[blk], tt)
            if enc is not None:
                raw = np.array(orig_tensor.data).reshape(rows, nb, ts)
                raw[blk] = enc
                return (raw.reshape(orig_tensor.data.shape), tt, tt.name,
                        f"re-encoded {k:,} of {rows * nb:,} blocks; all other blocks kept byte-for-byte")
        else:
            enc = _quantize_blocks(a2.reshape(-1, bs), tt)
            if enc is not None:
                return enc.reshape(*shape[:-1], nb * ts), tt, tt.name, "encoded"
    fb = "F16" if arr.ndim >= 2 else "F32"
    return (arr.astype(np.float16 if fb == "F16" else np.float32), None, fb,
            f"{orig_type or type_name} has no encoder in Python; stored as {fb}")


def _quantize_blocks(vals, qt):
    """vals: (k, block_size) floats -> (k, block_bytes) uint8, or None if no encoder exists for qt."""
    import kquants
    vals = np.ascontiguousarray(vals, dtype=np.float32).reshape(len(vals), -1)
    if qt.name in kquants.ENCODERS:
        return kquants.quantize_blocks(vals, qt.name)
    if qt in REQUANTIZABLE:
        try:
            return quantize(vals, qt).reshape(len(vals), -1)
        except Exception:
            return None
    return None


_ENCODER_OK = {}


def _has_encoder(tt):
    if tt not in _ENCODER_OK:
        from gguf.constants import GGML_QUANT_SIZES
        _ENCODER_OK[tt] = _quantize_blocks(np.zeros((1, GGML_QUANT_SIZES[tt][0]), np.float32), tt) is not None
    return _ENCODER_OK[tt]


PLAIN_NP = {QT.F32: np.float32, QT.F16: np.float16, QT.F64: np.float64, **INT_TYPES}


def _written_type(e, target):
    """The type a modified tensor is really written as (types without a Python encoder fall back to F16/F32)."""
    from gguf.constants import GGML_QUANT_SIZES
    tt = QT[target]
    if tt in PLAIN_NP:
        return tt
    if int(e["numpy_shape"][-1]) % GGML_QUANT_SIZES[tt][0] == 0 and _has_encoder(tt):
        return tt
    return QT.F16 if len(e["numpy_shape"]) >= 2 else QT.F32


def _type_bytes(tt, n_elements):
    from gguf.constants import GGML_QUANT_SIZES
    bs, ts = GGML_QUANT_SIZES[tt]
    return int(n_elements) // bs * ts


def total_ram():
    """Physical memory in bytes (0 if unknown)."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = MS()
            m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.total)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 0


def check_disk_space(folder, need_bytes, what, hint=""):
    """Raise a clear error instead of failing halfway with 'No space left on device'."""
    folder = Path(folder)
    probe = folder
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    margin = 512 * 1024 ** 2
    if free < need_bytes + margin:
        drive = probe.anchor or str(probe)
        raise RuntimeError(f"Not enough disk space for {what}: it needs about {need_bytes / 1e9:.1f} GB on {drive} "
                           f"but only {free / 1e9:.1f} GB is free.{(' ' + hint) if hint else ''}")
    return free


def _row_source(d, ot, e):
    """(rows, cols, base(r0, r1) -> float rows before edits, original tensor or None) for streaming."""
    rows, cols = (1, int(e["numpy_shape"][0])) if e["layout"] == "column" else (int(e["matrix"][0]), int(e["matrix"][1]))
    fdt = np.float64 if _csv_fmt(e) == "%.17g" else np.float32
    if e["source"] == "original":
        t = ot[e["orig"]]
        data = t.data.reshape(rows, -1)
        return rows, cols, (lambda a, b: _decode(data[a:b], t.tensor_type).reshape(b - a, cols)), t
    if e["source"] == "zeros":
        return rows, cols, (lambda a, b: np.zeros((b - a, cols), fdt)), None
    arr = np.load(d / e["npy"], mmap_mode="r").reshape(rows, cols)
    return rows, cols, (lambda a, b: np.array(arr[a:b], dtype=fdt)), None


def _diff_count(a, b):
    return int(np.count_nonzero(~((a == b) | (np.isnan(a) & np.isnan(b)))))


def _stream_tensor(d, ot, e, ed, target, tmp_files, orig_type=None):
    """Write one modified tensor (grid edits `ed` = {flat index: value}) WITHOUT loading it whole: it is
    processed a few million numbers at a time into a temporary file next to the export. For a tensor that
    keeps its quantized type, only the rows holding edits are decoded and only the edited blocks re-encoded.
    Returns (array for the writer, raw dtype or None, written type name, note, values that differ in the file)."""
    from gguf.constants import GGML_QUANT_SIZES
    rows, cols, base, t = _row_source(d, ot, e)
    shape = tuple(int(x) for x in e["numpy_shape"])
    tt = _written_type(e, target)
    note = "" if tt == QT[target] else f"{orig_type or target} has no encoder in Python; stored as {tt.name}"
    count = e["source"] == "original" and not e.get("added")
    by_row = {}
    for i, v in ed.items():
        by_row.setdefault(i // cols, []).append((i % cols, v))
    by_row = {r: (np.array([c for c, _ in cv], np.int64), np.array([v for _, v in cv], np.float64)) for r, cv in by_row.items()}
    edit_rows = sorted(by_row)
    step = max(1, CHUNK_VALUES // max(cols, 1))

    def with_edits(chunk, r0):
        import bisect
        lo, hi = bisect.bisect_left(edit_rows, r0), bisect.bisect_left(edit_rows, r0 + len(chunk))
        if lo < hi:
            chunk = np.array(chunk)                       # base rows may be read-only views
            for r in edit_rows[lo:hi]:
                c, v = by_row[r]
                chunk[r - r0, c] = v
        return chunk

    def temp(dtype, tshape):
        p = d / "exports" / f".stream_{len(tmp_files)}_{Path(e['csv']).stem}.tmp"
        mm = np.memmap(p, dtype=dtype, mode="w+", shape=tshape)
        tmp_files.append((p, mm))
        return mm

    stored = 0
    if tt in PLAIN_NP:                                    # F32 / F16 / F64 / integers: convert row chunks
        dt = PLAIN_NP[tt]
        out = temp(dt, shape)
        flat = out.reshape(rows, cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            b = base(r0, r1)
            cur = with_edits(b, r0)
            if tt in INT_TYPES:
                lim = np.iinfo(dt)
                enc = np.clip(np.rint(cur), lim.min, lim.max).astype(dt)
            else:
                enc = cur.astype(dt, copy=False)
            flat[r0:r1] = enc
            if count and (enc is not b or cur is not b):
                stored += _diff_count(enc.astype(b.dtype, copy=False), b)
            del b, cur, enc
        out.flush()
        return out, None, tt.name, note, (stored if count else None)

    bs, ts = GGML_QUANT_SIZES[tt]
    nb = cols // bs
    keep = (t is not None and t.tensor_type == tt and t.data.dtype == np.uint8 and natural_shape(t) == shape)
    if keep:
        # re-encode ONLY the blocks that contain an edit; every other block keeps its original bytes
        patches, k = [], 0
        for r in edit_rows:
            b = base(r, r + 1)[0]
            cur = with_edits(b[None, :], r)[0].astype(np.float32)
            blk = ~((cur == b) | (np.isnan(cur) & np.isnan(b)))
            blk = blk.reshape(nb, bs).any(-1)
            if blk.any():
                patches.append((r, blk, _quantize_blocks(cur.reshape(nb, bs)[blk], tt), b))
                k += int(blk.sum())
        if k == 0:
            return t.data, tt, tt.name, "unchanged", 0
        out = temp(np.uint8, t.data.shape)
        flat, raw = out.reshape(rows, nb * ts), t.data.reshape(rows, nb * ts)
        bstep = max(1, (64 * 1024 ** 2) // (nb * ts))     # copy the original bytes 64 MB at a time
        for r0 in range(0, rows, bstep):
            flat[r0:r0 + bstep] = raw[r0:r0 + bstep]
        for r, blk, enc, b in patches:
            row = flat[r].reshape(nb, ts)
            row[blk] = enc
            if count:
                stored += _diff_count(_decode(flat[r:r + 1], tt).reshape(cols), b)
        out.flush()
        return (out, tt, tt.name, f"re-encoded {k:,} of {rows * nb:,} blocks; all other blocks kept byte-for-byte",
                stored if count else None)

    # a new or resized tensor (or a type change): encode every block, chunk by chunk
    out = temp(np.uint8, (*shape[:-1], nb * ts))
    flat = out.reshape(rows, nb * ts)
    zero_row = None
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        import bisect
        has_edit = bisect.bisect_left(edit_rows, r0) < bisect.bisect_left(edit_rows, r1)
        if e["source"] == "zeros" and not has_edit:      # pass-through layers: every block is the same
            if zero_row is None:
                zero_row = np.tile(_quantize_blocks(np.zeros((1, bs), np.float32), tt)[0], nb)
            flat[r0:r1] = zero_row
            continue
        b = base(r0, r1)
        cur = with_edits(b, r0)
        enc = _quantize_blocks(np.asarray(cur, np.float32).reshape(-1, bs), tt).reshape(r1 - r0, nb * ts)
        flat[r0:r1] = enc
        if count:
            stored += _diff_count(_decode(enc, tt).reshape(r1 - r0, cols), b)
    out.flush()
    return out, tt, tt.name, "encoded", (stored if count else None)


def _release_temp(tmp_files):
    for _, mm in tmp_files:
        try:
            mm._mmap.close()
        except Exception:
            pass
    arrays = [mm for _, mm in tmp_files]
    paths = [p for p, _ in tmp_files]
    tmp_files.clear()
    del arrays
    gc.collect()
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def check_layer_count(meta, man, orig_meta, orig_names):
    """<architecture>.block_count must stay in step with the layers the tensors really have (blk.0 … blk.N-1), as it
    was in the original file: llama.cpp can't load a file that says another number."""
    def gap(m, names):
        n = m.get(f"{m.get('general.architecture')}.block_count")
        layers = {int(x.group(1)) for x in (re.match(r"blk\.(\d+)\.", nm) for nm in names) if x}
        return (n, max(layers) + 1) if isinstance(n, int) and not isinstance(n, bool) and layers else None
    now, was = gap(meta, [e["name"] for e in man]), gap(orig_meta, orig_names)
    if now and was and was[0] == was[1] and now[0] != now[1]:     # (a file that never matched is left alone)
        want = now[1]
        raise ValueError(f"metadata '{meta.get('general.architecture')}.block_count' is {now[0]}, but the model has {now[1]} "
                         f"layer(s) of tensors (blk.0 to blk.{now[1] - 1}). llama.cpp can't load a file like that: set it to "
                         f"{want}, or add / remove layers in Grow model (which keeps the two in step).")


def export_gguf(ws, out_path=None, log=print, exact=False, record=True, extra_computed=()):
    """exact=True: store edited quantized / F16 tensors as F32 so typed values are kept exactly
    (instead of snapping them to the quantized format's grid). Untouched tensors are always copied as-is."""
    d = workspace_dir(ws)
    info, reader, ot = open_source(d)
    out = Path(out_path) if out_path else d / "exports" / f"{ws}_edited.gguf"
    sources = [Path(p[0]).resolve() for p in info["parts"]] if info.get("parts") else [Path(info["gguf_path"]).resolve()]
    if out.resolve() in sources:
        raise ValueError("Refusing to overwrite the original model")
    out.parent.mkdir(parents=True, exist_ok=True)
    (d / "exports").mkdir(exist_ok=True)
    man = load_manifest(d)
    if info.get("split_count"):
        log(f"Split model: the {info['split_count']} parts are merged into one file")

    # 1) metadata
    user_meta = load_user_metadata(d)
    validate_metadata(user_meta, reader.fields)
    mdiff = metadata_diff(reader, user_meta)
    orig_vals = original_metadata(reader)
    check_layer_count(user_meta, man, orig_vals, list(ot))
    log(f"Metadata: {len(mdiff['changed'])} changed, {len(mdiff['added'])} added, {len(mdiff['removed'])} removed")

    # 2) grid/CSV edits + 3) edit_weights.py
    w, edits, modified, counts, newp = _apply_all_edits(d, ot, man, log=log, extra_computed=extra_computed)

    def target_of(e):
        if exact and QT[e["type"]] not in (QT.F32, QT.F64, *INT_TYPES):
            return "F32"                  # F16 keeps only ~3-4 digits; F32 keeps every digit shown in the grid
        return e["type"]

    # disk space: the new file, plus temporary files for the edited tensors (deleted afterwards)
    out_bytes = temp_bytes = 0
    for e in man:
        if e["name"] in modified:
            b = _type_bytes(_written_type(e, target_of(e)), e["n_elements"])
            out_bytes += b
            if e["name"] not in w._cache:
                temp_bytes += b
        else:
            out_bytes += int(ot[e["orig"]].n_bytes)
    out_bytes += (d / "metadata.json").stat().st_size + 1024 ** 2
    check_disk_space(out.parent, out_bytes + temp_bytes, "the export",
                     f"(new model {out_bytes / 1e9:.1f} GB + {temp_bytes / 1e9:.1f} GB of temporary files). "
                     f"Free up space or delete old files in {out.parent}.")

    # 4) write
    writer = GGUFWriter(str(out), user_meta["general.architecture"])
    for key, value in user_meta.items():
        if key == "general.architecture":
            continue
        field = reader.fields.get(key)
        if key == "general.alignment":
            writer.add_custom_alignment(int(value))
        elif field is not None and value == orig_vals.get(key):
            _copy_kv(writer, key, field)
        elif (field is not None and field.types[0] == VT.ARRAY and isinstance(value, list)
              and isinstance(orig_vals.get(key), list) and len(value) > len(orig_vals[key])
              and value[:len(orig_vals[key])] == orig_vals[key]):
            _write_extended_array(writer, key, value, field, orig_vals[key])
        else:
            _write_kv(writer, key, value, field)

    notes, stored_changed, tmp_files = [], 0, []
    tmp = out.with_name(out.name + ".partial")       # a failed export never destroys the previous one
    try:
        for e in man:
            if e["name"] in modified:
                orig_type = ot[e["orig"]].tensor_type.name if e["source"] == "original" else None
                target = target_of(e)
                if e["name"] in w._cache:                 # touched by edit_weights.py: already in memory
                    data, raw, tname, note = _encode(w[e["name"]], e["numpy_shape"], target, orig_type,
                                                     ot[e["orig"]] if e["source"] == "original" else None)
                    n = None
                    if e["source"] == "original" and not e.get("added"):
                        written = _decode(data, QT[tname] if raw is None else raw).reshape(e["numpy_shape"])
                        n = _diff_count(written, full_tensor(ot[e["orig"]]))
                        del written
                else:
                    log(f"Encoding {e['name']} …")
                    data, raw, tname, note, n = _stream_tensor(d, ot, e, edits.get(e["name"], {}), target,
                                                               tmp_files, orig_type)
                if target != e["type"]:
                    note = f"kept your exact values: stored as {tname} instead of {e['type']}"
                writer.add_tensor(e["name"], data, raw_dtype=raw)
                src_label = {"original": orig_type, "zeros": "new (zeros)", "npy": "resized"}[e["source"]]
                item = {"name": e["name"], "from": src_label, "to": tname, "note": note,
                        "edited": counts.get(e["name"], 0)}
                if n is not None:
                    # how many values really differ in the written file (re-quantizing can shift a block's neighbours)
                    item["stored_changed"] = n
                    stored_changed += n
                notes.append(item)
            else:
                t = ot[e["orig"]]
                writer.add_tensor(e["name"], t.data, raw_dtype=t.tensor_type if t.data.dtype == np.uint8 else None)
        log(f"Writing {out.name} ...")
        writer.write_header_to_file(path=tmp)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        check = GGUFReader(str(tmp), "r")
        ok = len(check.tensors) == len(man)
        del check
        gc.collect()  # release the check's memory map (Windows can't replace a mapped file)
        if not ok:
            raise RuntimeError("Export check failed: tensor count differs")
        try:
            os.replace(tmp, out)
        except PermissionError:
            raise RuntimeError(f"Can't overwrite {out.name}: it is open in another program "
                               f"(LM Studio, Ollama, Python...). Close it and export again.")
    finally:
        try:
            writer.close()
        except Exception:
            pass
        writer = data = None                 # drop references to the temporary arrays before deleting them
        _release_temp(tmp_files)
        if tmp.exists():
            gc.collect()
            try:
                tmp.unlink()
            except OSError:
                pass
    size = out.stat().st_size
    stats = change_stats(man, ot, counts, newp)
    stats.update(includes_script=True, stored_changed=stored_changed, stored_changed_pct=pct(stored_changed, stats["original_total"]))
    log(f"Weights you changed: {stats['edited']:,} of {stats['original_total']:,} = {stats['edited_pct']} %")
    if stored_changed != stats["edited"]:
        log(f"Values that differ in the written file: {stored_changed:,} = {stats['stored_changed_pct']} % "
            f"(re-quantizing a block shifts its neighbours)")
    if stats["added"]:
        log(f"Weights added: {stats['added']:,} = {stats['added_pct']} % of the original size")
    log(f"Done: {out} ({size / 1e6:.1f} MB)")
    result = {"path": str(out), "size": size, "metadata": mdiff, "tensors": notes, "stats": stats,
              "time": datetime.datetime.now().isoformat(timespec="seconds")}
    if record:                                         # off for internal copies (e.g. the Test tab's fast mode)
        (d / "exports" / "last_export.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LLM Decoder core (the app's engine, usable from the command line)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("gguf"); e.add_argument("--rows", type=int, default=256)
    e.add_argument("--cols", type=int, default=256); e.add_argument("--overwrite", action="store_true")
    sub.add_parser("changes").add_argument("workspace")
    x = sub.add_parser("export"); x.add_argument("workspace"); x.add_argument("--out")
    al = sub.add_parser("add-layers"); al.add_argument("workspace"); al.add_argument("--count", type=int, default=1)
    al.add_argument("--mode", choices=["identity", "copy"], default="identity"); al.add_argument("--copy-from", type=int)
    at = sub.add_parser("add-tokens"); at.add_argument("workspace"); at.add_argument("tokens", nargs="+")
    at.add_argument("--type", default="user_defined", choices=list(TOKEN_TYPES))
    at.add_argument("--init", default="mean", choices=["mean", "zeros", "copy"]); at.add_argument("--copy-from", type=int)
    sub.add_parser("undo").add_argument("workspace")
    xe = sub.add_parser("excel-export"); xe.add_argument("workspace")
    xi = sub.add_parser("excel-import"); xi.add_argument("workspace"); xi.add_argument("--file")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract_workspace(a.gguf, a.rows, a.cols, a.overwrite)
    elif a.cmd == "changes":
        print(json.dumps(pending_changes(a.workspace), indent=2))
    elif a.cmd == "export":
        print(json.dumps(export_gguf(a.workspace, a.out), indent=2))
    elif a.cmd == "add-layers":
        add_layers(a.workspace, a.count, a.mode, a.copy_from)
    elif a.cmd == "add-tokens":
        add_tokens(a.workspace, a.tokens, a.type, a.init, a.copy_from)
    elif a.cmd == "undo":
        undo_last(a.workspace)
    elif a.cmd == "excel-export":
        export_xlsx(a.workspace)
    else:
        print(json.dumps(import_xlsx(a.workspace, a.file), indent=2))
