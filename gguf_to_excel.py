"""
gguf_to_excel.py - split a GGUF model into
  1) <name>_metadata.json  (open in VS Code: architecture, hyperparameters, tokenizer, tensor list)
  2) <name>_weights.xlsx   (the numbers: one sheet per tensor + an Index sheet)

Excel limits: 1,048,576 rows x 16,384 columns per sheet. Real LLMs have millions to
billions of weights, so by default each 2D+ tensor is cropped to its top-left
--max-rows x --max-cols block (the Index sheet says which tensors were cropped).
1D tensors (biases, norms) are written whole, as one column.
Use --full to write everything that fits (huge, slow).

Layout of a weight sheet: row r = row r of the matrix (for token_embd.weight: token id r),
column c = element c of that row. Tensors with 3+ dims (e.g. MoE experts) are flattened
to 2D: rows = product of the outer dims.

Usage:
  python gguf_to_excel.py model.gguf
  python gguf_to_excel.py model.gguf --out out_dir --max-rows 500 --max-cols 500
  python gguf_to_excel.py model.gguf --filter "blk\\.0\\."      # only layer 0
  python gguf_to_excel.py model.gguf --full
"""
import argparse
import json
import re
import sys
from pathlib import Path

import xlsxwriter
from gguf import GGUFReader

# shared reading logic lives in gguf_core (also used by the app)
from gguf_core import json_safe, load_rows, read_metadata, tensor_dims

XL_MAX_ROWS = 1_048_576
XL_MAX_COLS = 16_384
BIG_CELL_WARNING = 50_000_000


def safe_sheet_name(name, used):
    base = re.sub(r"[\[\]:*?/\\]", "_", name)
    base = base[-31:]                 # keep the informative end, e.g. blk.12.attn_q.weight
    base = base.strip("'") or "tensor"  # Excel forbids a leading/trailing apostrophe
    if base.lower() == "history":       # reserved by Excel
        base = "history_"
    cand, i = base, 1
    while cand.lower() in used:
        suffix = f"~{i}"
        cand = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(cand.lower())
    return cand


def plan_tensor(t, max_rows, max_cols):
    """Decide what part of a tensor goes into Excel: (rows, cols, take_rows, take_cols, vertical)."""
    rows, cols = tensor_dims(t)
    if len(t.shape) == 1:
        # 1D vector: write as one column so long vectors (vocab-sized) fit
        return rows, cols, 1, min(cols, XL_MAX_ROWS - 1), True
    return rows, cols, min(rows, max_rows), min(cols, max_cols), False


# ---------------------------------------------------------------- main

def main():
    # Windows consoles default to cp1252; don't crash on non-English paths/names
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", help="path to .gguf file")
    ap.add_argument("--out", help="output directory (default: 'output' next to this script)")
    ap.add_argument("--max-rows", type=int, default=256, help="rows kept per tensor (default 256)")
    ap.add_argument("--max-cols", type=int, default=256, help="columns kept per tensor (default 256)")
    ap.add_argument("--filter", help="regex; only export tensors whose name matches")
    ap.add_argument("--full", action="store_true", help="no cropping beyond Excel's hard limits")
    ap.add_argument("--no-excel", action="store_true", help="only write the metadata JSON")
    args = ap.parse_args()

    if args.max_rows < 1 or args.max_cols < 1:
        ap.error("--max-rows and --max-cols must be at least 1")

    src = Path(args.gguf)
    if not src.is_file():
        sys.exit(f"File not found: {src}")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    max_rows = XL_MAX_ROWS - 1 if args.full else min(args.max_rows, XL_MAX_ROWS - 1)
    max_cols = XL_MAX_COLS if args.full else min(args.max_cols, XL_MAX_COLS)

    print(f"Reading {src} ...")
    try:
        reader = GGUFReader(str(src), "r")
    except Exception as e:
        sys.exit(f"Not a readable GGUF file ({e})")
    meta = read_metadata(reader)

    split_count = meta.get("split.count")
    if isinstance(split_count, int) and split_count > 1:
        print(f"WARNING: this is part {meta.get('split.no', 0) + 1} of a {split_count}-file split model. "
              f"Only the tensors stored in this file are exported; run the script on each part.")

    # ---- metadata JSON
    tensor_info = []
    for t in reader.tensors:
        rows, cols = tensor_dims(t)
        tensor_info.append({
            "name": t.name,
            "shape_ggml": [int(x) for x in t.shape],
            "as_matrix_rows_x_cols": [rows, cols],
            "type": t.tensor_type.name,
            "n_elements": int(t.n_elements),
            "n_bytes": int(t.n_bytes),
        })
    meta_path = out / f"{stem}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "file": src.name,
            "n_tensors": len(tensor_info),
            "n_parameters": sum(x["n_elements"] for x in tensor_info),
            "metadata": meta,
            "tensors": tensor_info,
        }), f, indent=2, ensure_ascii=False)
    print(f"Wrote {meta_path}")
    if args.no_excel:
        return 0

    # ---- weights Excel
    tensors = list(reader.tensors)
    if args.filter:
        rx = re.compile(args.filter)
        tensors = [t for t in tensors if rx.search(t.name)]
        if not tensors:
            sys.exit(f"No tensor name matches --filter {args.filter!r}")

    plans = [plan_tensor(t, max_rows, max_cols) for t in tensors]
    total_cells = sum(p[2] * p[3] for p in plans)
    print(f"Exporting {len(tensors)} tensors, ~{total_cells:,} cells to Excel")
    if total_cells > BIG_CELL_WARNING:
        print("WARNING: that is a lot of cells; the file will be very large and slow to open. "
              "Consider smaller --max-rows/--max-cols or --filter.")

    xlsx_path = out / f"{stem}_weights.xlsx"
    # nan_inf_to_errors: write NaN/Inf weights as Excel errors instead of crashing
    wb = xlsxwriter.Workbook(str(xlsx_path), {"constant_memory": True, "nan_inf_to_errors": True})
    index = wb.add_worksheet("Index")  # created first so it is the first tab
    hdr = wb.add_format({"bold": True, "bg_color": "#DDDDDD"})
    index_rows = []
    used = {"index"}

    for n, (t, (rows, cols, take_r, take_c, vertical)) in enumerate(zip(tensors, plans), 1):
        shape = str([int(x) for x in t.shape])
        sheet = safe_sheet_name(t.name, used)
        try:
            win = load_rows(t, take_r)[:, :take_c]
        except Exception as e:
            index_rows.append([t.name, shape, t.tensor_type.name, int(t.n_elements), "", "", f"SKIPPED: {e}"])
            print(f"[{n}/{len(tensors)}] {t.name}: skipped ({e})")
            continue
        ws = wb.add_worksheet(sheet)
        if vertical:
            for r, v in enumerate(win[0].tolist()):
                ws.write_number(r, 0, v)
            exported = f"{take_c} x 1 (column)"
            cropped = take_c < cols
        else:
            for r in range(win.shape[0]):
                ws.write_row(r, 0, win[r].tolist())
            exported = f"{win.shape[0]} x {win.shape[1]}"
            cropped = win.shape[0] < rows or win.shape[1] < cols
        status = (f"cropped from {rows} x {cols}" if not vertical else f"cropped from {cols}") if cropped else "complete"
        index_rows.append([t.name, shape, t.tensor_type.name, int(t.n_elements), sheet, exported, status])
        print(f"[{n}/{len(tensors)}] {t.name} -> '{sheet}' ({exported}{', cropped' if cropped else ''})")

    heads = ["tensor", "shape (ggml order)", "type", "n_elements", "sheet", "exported", "status"]
    index.write_row(0, 0, heads, hdr)
    for i, row in enumerate(index_rows, 1):
        index.write_row(i, 0, row)
    index.set_column(0, 0, 40)
    index.set_column(1, 6, 22)
    try:
        wb.close()
    except xlsxwriter.exceptions.FileCreateError:
        sys.exit(f"Could not write {xlsx_path}. Is it open in Excel? Close it and run again.")
    print(f"Wrote {xlsx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
