"""Builds a synthetic GGUF full of edge cases, runs the converter, and checks every exported number.
Run:  python test_gguf_to_excel.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import openpyxl
import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import quantize, dequantize

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def build(path):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", 2)
    w.add_float32("llama.rope.freq_base", 10000.0)
    w.add_bool("llama.attention.causal", True)
    w.add_string("general.name", "edge-case-model")
    # one token is invalid UTF-8 (happens with byte-level vocabularies)
    w.add_array("tokenizer.ggml.tokens", ["<s>", b"\xff\xfe", "hello", "▁wörld"])
    w.add_array("tokenizer.ggml.scores", [0.0, -1.5, -2.0, -3.0])

    expected = {}  # name -> 2D float array of what the full tensor should contain

    def add(name, arr, qtype=None):
        arr = arr.astype(np.float32) if arr.dtype.kind == "f" and qtype is not None else arr
        if qtype is None:
            w.add_tensor(name, arr)
            full = arr.astype(np.float64)
        else:
            qd = quantize(arr, qtype)
            w.add_tensor(name, qd, raw_dtype=qtype)  # writer derives the element shape from the byte shape
            full = dequantize(qd, qtype).astype(np.float64)
        expected[name] = full.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else full.reshape(1, -1)

    add("token_embd.weight", rng.standard_normal((300, 320)).astype(np.float32))       # cropped both ways
    add("blk.0.attn_q.weight", rng.standard_normal((64, 64)).astype(np.float32), Q.Q8_0)
    add("blk.0.attn_k.weight", rng.standard_normal((64, 64)).astype(np.float32), Q.Q4_0)
    add("blk.0.attn_v.weight", rng.standard_normal((32, 64)).astype(np.float32), Q.BF16)
    add("blk.0.attn_output.weight", rng.standard_normal((16, 32)).astype(np.float16))
    add("blk.0.ffn_up_exps.weight", rng.standard_normal((4, 8, 32)).astype(np.float32))  # 3D (MoE)
    add("blk.0.f64_test.weight", rng.standard_normal((8, 8)))                          # F64
    add("blk.0.int_test.weight", rng.integers(-1000, 1000, (8, 8)).astype(np.int32))   # I32
    nan = rng.standard_normal((4, 4)).astype(np.float32); nan[0, 0] = np.nan; nan[1, 1] = np.inf
    add("blk.0.nan_test.weight", nan)
    add("output_norm.weight", rng.standard_normal(20000).astype(np.float32))           # 1D > 16384
    # two names whose last 31 characters are identical -> sheet names must still be unique
    add("aaaa.blk.1.very_long_tensor_name_here.weight", rng.standard_normal((2, 4)).astype(np.float32))
    add("bbbb.blk.1.very_long_tensor_name_here.weight", rng.standard_normal((2, 4)).astype(np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return expected


def main():
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "edge.gguf"
    expected = build(src)
    out = tmp / "out"
    r = subprocess.run([sys.executable, str(HERE.parent / "gguf_to_excel.py"), str(src), "--out", str(out)],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout, r.stderr)
    assert r.returncode == 0, "converter failed"

    meta = json.load(open(out / "edge_metadata.json", encoding="utf-8"))
    m = meta["metadata"]
    assert m["general.name"] == "edge-case-model"
    assert m["llama.block_count"] == 2 and m["llama.attention.causal"] is True
    assert m["tokenizer.ggml.tokens"][0] == "<s>" and m["tokenizer.ggml.tokens"][3] == "▁wörld", m["tokenizer.ggml.tokens"]
    assert len(m["tokenizer.ggml.tokens"]) == 4
    assert meta["n_tensors"] == len(expected)

    wb = openpyxl.load_workbook(out / "edge_weights.xlsx", read_only=True)
    index = list(wb["Index"].iter_rows(min_row=2, values_only=True))
    assert len(index) == len(expected), index
    sheets = [row[4] for row in index]
    assert len(set(sheets)) == len(sheets), "duplicate sheet names"
    failures = []
    for name, sheet, exported, status in [(r[0], r[4], r[5], r[6]) for r in index]:
        if status.startswith("SKIPPED"):
            failures.append(f"{name}: {status}"); continue
        cells = [list(row) for row in wb[sheet].iter_rows(values_only=True)]
        got = np.array([[np.nan if isinstance(v, str) else v for v in row] for row in cells], dtype=np.float64)
        exp = expected[name]
        if "column" in exported:          # 1D written vertically
            got = got[:, 0].reshape(1, -1)
        exp_win = exp[: got.shape[0], : got.shape[1]]
        if exp_win.shape != got.shape:
            failures.append(f"{name}: shape {got.shape} vs expected window {exp_win.shape}"); continue
        finite = np.isfinite(exp_win)
        if not np.allclose(got[finite], exp_win[finite], rtol=1e-6, atol=1e-7):
            failures.append(f"{name}: values differ (max err {np.nanmax(np.abs(got - exp_win)):.3g})")
        if (~finite).any() and not np.isnan(got[~finite]).all():
            failures.append(f"{name}: NaN/Inf not marked")
        print(f"  ok  {name:48s} {exported:18s} {status}")
    # cropping rules
    st = {r[0]: (r[5], r[6]) for r in index}
    assert st["token_embd.weight"] == ("256 x 256", "cropped from 300 x 320"), st["token_embd.weight"]
    assert st["output_norm.weight"] == ("20000 x 1 (column)", "complete"), st["output_norm.weight"]
    assert st["blk.0.ffn_up_exps.weight"] == ("32 x 32", "complete"), st["blk.0.ffn_up_exps.weight"]
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures)); sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
