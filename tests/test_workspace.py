"""End-to-end test of extract -> edit -> export using a synthetic GGUF full of edge cases.
Run:  python test_workspace.py
"""
import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFReader, GGUFValueType as VT
from gguf.quants import quantize, dequantize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core
from test_gguf_to_excel import build

tmp = Path(tempfile.mkdtemp())
core.WORKSPACES = tmp / "workspaces"           # keep the real workspaces folder untouched
src = tmp / "edge.gguf"
expected = build(src)


def raw_tokens(r):
    f = r.fields["tokenizer.ggml.tokens"]
    return [bytes(f.parts[i]) for i in f.data]


def tensors(r):
    return {t.name: t for t in r.tensors}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok ", msg)


ws = core.extract_workspace(src, max_rows=256, max_cols=256, log=lambda *_: None)
d = core.WORKSPACES / ws
orig = GGUFReader(str(src))

print("1) export without edits is lossless")
res = core.export_gguf(ws, log=lambda *_: None)
out = GGUFReader(res["path"])
check(raw_tokens(out) == raw_tokens(orig), "invalid-UTF-8 token bytes preserved")
check(all(out.fields[k].types == orig.fields[k].types for k in orig.fields), "all metadata types preserved")
check(all(np.array_equal(tensors(out)[n].data, t.data, equal_nan=True) and
          tensors(out)[n].tensor_type == t.tensor_type for n, t in tensors(orig).items()), "all tensors byte-identical")
check(res["tensors"] == [], "nothing reported as modified")
del out  # release the memory map before exporting over the same file


print("1b) exporting over a file that another program has open gives a clear error")
locked = GGUFReader(res["path"])
try:
    core.export_gguf(ws, log=lambda *_: None)
    check(True, "overwrite allowed (file wasn't locked on this OS)")
except RuntimeError as ex:
    check("open in another program" in str(ex), f"clear error: {ex}")
check(GGUFReader(res["path"]).tensors is not None, "previous export still intact")
del locked
import gc; gc.collect()

print("2) CSV re-saved with rounding noise (like Excel does) is not a change")
p = d / "weights" / "token_embd.weight.csv"
np.savetxt(p, core.read_csv(p), delimiter=",", fmt="%.8g")
check(core.pending_changes(ws)["weights"] == [], "no weight changes detected")

print("3) real edits")
# CSV edits: F32 matrix, Q8_0 matrix, 1D column
a = core.read_csv(p); a[3, 5] = 42.0; np.savetxt(p, a, delimiter=",", fmt="%.9g")
q = d / "weights" / "blk.0.attn_q.weight.csv"
b = core.read_csv(q); b[0, :] = 0.5; np.savetxt(q, b, delimiter=",", fmt="%.9g")
c = d / "weights" / "output_norm.weight.csv"
v = core.read_csv(c); v[19999, 0] = -7.0; np.savetxt(c, v, delimiter=",", fmt="%.9g")
# metadata edits
m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
m["general.name"] = "edited-model"
m["llama.block_count"] = 3
m["my.new_key"] = 5
m["my.new_list"] = [1.5, 2.5]
del m["tokenizer.ggml.scores"]
(d / "metadata.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
# script edits: Q4_0 tensor (re-quantized), I32 tensor, 3D tensor beyond nothing, F16 tensor
(d / "edit_weights.py").write_text(core.EDIT_TEMPLATE.replace("    pass\n", """
    w["blk.0.attn_k.weight"][0, :] = 0.0
    w["blk.0.int_test.weight"] += 1
    w["blk.0.ffn_up_exps.weight"][3] = 1.0
    w["blk.0.attn_output.weight"] = np.ones(w.info("blk.0.attn_output.weight")["shape"])
    _ = w["blk.0.attn_v.weight"]          # read-only access must NOT count as a change
"""), encoding="utf-8")

pc = core.pending_changes(ws)
check(sorted(x["name"] for x in pc["weights"]) == ["blk.0.attn_q.weight", "output_norm.weight", "token_embd.weight"],
      f"pending CSV changes found: {pc['weights']}")
check({k: sorted(v) for k, v in pc["metadata"].items()} ==
      {"changed": ["general.name", "llama.block_count"], "added": ["my.new_key", "my.new_list"],
       "removed": ["tokenizer.ggml.scores"]}, f"pending metadata diff: {pc['metadata']}")
check(pc["script"], "script detected")

res = core.export_gguf(ws, log=lambda *_: None)
out = GGUFReader(res["path"])
ot, oo = tensors(out), tensors(orig)
check(sorted(x["name"] for x in res["tensors"]) == sorted(
    ["token_embd.weight", "blk.0.attn_q.weight", "output_norm.weight", "blk.0.attn_k.weight",
     "blk.0.int_test.weight", "blk.0.ffn_up_exps.weight", "blk.0.attn_output.weight"]),
    f"modified tensors: {[x['name'] for x in res['tensors']]}")

full = lambda n: core.full_tensor(ot[n])
before = lambda n: core.full_tensor(oo[n])
e = full("token_embd.weight"); e0 = before("token_embd.weight"); e0[3, 5] = 42.0
check(np.array_equal(e, e0), "F32 CSV edit applied exactly, rest untouched")
qv = full("blk.0.attn_q.weight")
check(np.allclose(qv[0], 0.5, atol=0.01) and np.allclose(qv[1:], before("blk.0.attn_q.weight")[1:], atol=0.05),
      "Q8_0 CSV edit applied (re-quantized)")
check(ot["blk.0.attn_q.weight"].tensor_type.name == "Q8_0", "Q8_0 stays Q8_0")
check(full("output_norm.weight")[19999] == -7.0, "1D column CSV edit applied")
check(np.allclose(full("blk.0.attn_k.weight")[0], 0, atol=1e-6), "Q4_0 script edit applied")
check(np.array_equal(full("blk.0.int_test.weight"), before("blk.0.int_test.weight") + 1)
      and ot["blk.0.int_test.weight"].tensor_type.name == "I32", "I32 script edit applied, still I32")
check(np.all(full("blk.0.ffn_up_exps.weight")[3] == 1.0), "3D tensor script edit applied")
check(np.all(full("blk.0.attn_output.weight") == 1.0) and ot["blk.0.attn_output.weight"].tensor_type.name == "F16",
      "F16 replacement via assignment")
check(np.array_equal(ot["blk.0.attn_v.weight"].data, oo["blk.0.attn_v.weight"].data), "read-only tensor untouched")
check(np.array_equal(ot["blk.0.nan_test.weight"].data, oo["blk.0.nan_test.weight"].data, equal_nan=True),
      "NaN tensor untouched")
meta = core.read_metadata(out)
check(meta["general.name"] == "edited-model", "string metadata edited")
check(meta["llama.block_count"] == 3 and out.fields["llama.block_count"].types == [VT.UINT32], "UINT32 type kept")
check(meta["my.new_key"] == 5 and out.fields["my.new_key"].types == [VT.INT32], "new int key added")
check(meta["my.new_list"] == [1.5, 2.5], "new list key added")
check("tokenizer.ggml.scores" not in meta, "key removed")
check(raw_tokens(out) == raw_tokens(orig), "untouched tokens still byte-exact")

print("4) bad edits give clear errors")


def expect_error(fn, words, msg):
    try:
        fn()
    except Exception as ex:
        check(all(w in str(ex) for w in words), f"{msg}: {str(ex).splitlines()[0]}")
        return
    raise AssertionError(f"{msg}: no error raised")


good_meta = (d / "metadata.json").read_text(encoding="utf-8")
(d / "metadata.json").write_text(good_meta.replace('"edited-model"', '"edited-model",,'), encoding="utf-8")
expect_error(lambda: core.export_gguf(ws, log=lambda *_: None), ["not valid JSON"], "broken JSON")
m["llama.block_count"] = "three"
(d / "metadata.json").write_text(json.dumps(m), encoding="utf-8")
expect_error(lambda: core.export_gguf(ws, log=lambda *_: None), ["whole number"], "wrong value type")
m["llama.block_count"] = -1
(d / "metadata.json").write_text(json.dumps(m), encoding="utf-8")
expect_error(lambda: core.export_gguf(ws, log=lambda *_: None), ["out of range"], "negative unsigned")
(d / "metadata.json").write_text(good_meta, encoding="utf-8")
np.savetxt(q, b[:, :10], delimiter=",")
expect_error(lambda: core.export_gguf(ws, log=lambda *_: None), ["don't add or remove"], "CSV shape changed")
np.savetxt(q, b, delimiter=",", fmt="%.9g")
(d / "edit_weights.py").write_text("def edit(w):\n    w['nope'] = 1\n", encoding="utf-8")
expect_error(lambda: core.export_gguf(ws, log=lambda *_: None), ["edit_weights.py failed", "nope"], "script error")

quiet = lambda *_: None
del out, ot
gc.collect()

print("5) add layers on top")
ws = core.extract_workspace(src, overwrite=True, log=quiet)
r = core.add_layers(ws, 2, "identity", 0, log=quiet)
man = {e["name"]: e for e in core.load_manifest(d)}
check(all(f"blk.{j}.{n}" in man for j in (2, 3) for n in ("attn_q.weight", "attn_output.weight", "ffn_up_exps.weight")),
      f"new tensors blk.2.* and blk.3.* exist ({r['label']})")
check(man["blk.2.attn_output.weight"]["source"] == "zeros" and man["blk.2.attn_q.weight"]["source"] == "original",
      "pass-through: attn_output zeroed, others copied")
check(core.load_user_metadata(d)["llama.block_count"] == 4, "block_count 2 -> 4")
res = core.export_gguf(ws, log=quiet)
out = GGUFReader(res["path"])
ot = tensors(out)
n_l0 = sum(1 for n in tensors(orig) if n.startswith("blk.0."))
check(len(ot) == len(tensors(orig)) + 2 * n_l0, f"exported tensor count {len(ot)} = original + 2 x {n_l0}")
check(np.array_equal(ot["blk.3.attn_q.weight"].data, oo["blk.0.attn_q.weight"].data)
      and ot["blk.3.attn_q.weight"].tensor_type.name == "Q8_0", "copied layer keeps exact bytes and Q8_0 type")
check(np.all(core.full_tensor(ot["blk.2.attn_output.weight"]) == 0), "pass-through output weights are zero")
check(core.read_metadata(out)["llama.block_count"] == 4, "exported block_count = 4")
del out, ot
gc.collect()

print("6) insert rows / columns + undo")
r = core.insert_slices(ws, "token_embd.weight", "rows", 0, 2, "zeros", log=quiet)
e = next(x for x in core.load_manifest(d) if x["name"] == "token_embd.weight")
check(e["numpy_shape"] == [302, 320] and e["source"] == "npy", "token_embd 300 -> 302 rows")
grid = core.read_csv(d / e["csv"])
check(np.all(grid[:2] == 0) and np.allclose(grid[2], expected["token_embd.weight"][0, :256]), "grid shows the new rows at the top")
r = core.insert_slices(ws, "blk.0.attn_output.weight", "cols", 1, 1, "copy", 0, log=quiet)
e2 = next(x for x in core.load_manifest(d) if x["name"] == "blk.0.attn_output.weight")
g2 = core.read_csv(d / e2["csv"])
check(e2["numpy_shape"] == [16, 33] and np.array_equal(g2[:, 1], g2[:, 0]), "column inserted as a copy of column 0")
res = core.export_gguf(ws, log=quiet)
out = GGUFReader(res["path"])
emb = core.full_tensor(tensors(out)["token_embd.weight"])
check(emb.shape == (302, 320) and np.all(emb[:2] == 0) and np.array_equal(emb[2:], expected["token_embd.weight"]),
      "exported token_embd has the inserted rows, rest exact")
del out
gc.collect()
core.undo_last(ws, log=quiet)
core.undo_last(ws, log=quiet)
e = next(x for x in core.load_manifest(d) if x["name"] == "token_embd.weight")
check(e["numpy_shape"] == [300, 320] and e["source"] == "original" and not list((d / "added").glob("*")),
      "undo x2 restores the original shape and deletes the saved arrays")
check(core.load_user_metadata(d)["llama.block_count"] == 4, "earlier change (layers) still there")
core.undo_last(ws, log=quiet)
check(core.load_user_metadata(d)["llama.block_count"] == 2 and
      not any(x["name"].startswith("blk.2.") for x in core.load_manifest(d)), "third undo removes the layers")
expect_error(lambda: core.undo_last(ws, log=quiet), ["Nothing to undo"], "nothing left to undo")

print("7) add tokens")
expect_error(lambda: core.add_tokens(ws, ["<x>"], log=quiet), ["rows but the vocabulary has"],
             "vocab/embedding mismatch is caught")
src2 = tmp / "tiny.gguf"
w2 = gguf.GGUFWriter(str(src2), "llama")
w2.add_uint32("llama.block_count", 1)
w2.add_uint32("llama.vocab_size", 6)
w2.add_string("tokenizer.ggml.model", "llama")
w2.add_array("tokenizer.ggml.tokens", ["<unk>", "<s>", "</s>", "a", "b", "c"])
w2.add_array("tokenizer.ggml.scores", [0.0] * 6)
w2.add_array("tokenizer.ggml.token_type", [2, 3, 3, 1, 1, 1])
rng = np.random.default_rng(1)
emb0 = rng.standard_normal((6, 32)).astype(np.float32)
w2.add_tensor("token_embd.weight", quantize(emb0, gguf.GGMLQuantizationType.Q8_0),
              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
w2.add_tensor("output.weight", rng.standard_normal((6, 32)).astype(np.float32))
w2.add_tensor("blk.0.attn_q.weight", rng.standard_normal((32, 32)).astype(np.float32))
w2.write_header_to_file(); w2.write_kv_data_to_file(); w2.write_tensors_to_file(); w2.close()
ws2 = core.extract_workspace(src2, log=quiet)
d2 = core.WORKSPACES / ws2
expect_error(lambda: core.add_tokens(ws2, ["a"], log=quiet), ["Already in the vocabulary"], "duplicate token refused")
r = core.add_tokens(ws2, ["<think>", "</think>"], "user_defined", "mean", log=quiet)
m2 = core.load_user_metadata(d2)
check(m2["tokenizer.ggml.tokens"][-2:] == ["<think>", "</think>"] and m2["llama.vocab_size"] == 8
      and m2["tokenizer.ggml.token_type"][-2:] == [4, 4] and len(m2["tokenizer.ggml.scores"]) == 8,
      f"tokenizer lists extended ({r['label']})")
res = core.export_gguf(ws2, log=quiet)
out2 = GGUFReader(res["path"])
t2 = tensors(out2)
e8 = core.full_tensor(t2["token_embd.weight"])
check(e8.shape == (8, 32) and t2["token_embd.weight"].tensor_type.name == "Q8_0", "token_embd 6 -> 8 rows, still Q8_0")
check(np.allclose(e8[6], dequantize(quantize(emb0, gguf.GGMLQuantizationType.Q8_0),
                                    gguf.GGMLQuantizationType.Q8_0).mean(0), atol=0.02), "new rows = mean embedding")
check(core.full_tensor(t2["output.weight"]).shape == (8, 32), "output.weight 6 -> 8 rows")
check(out2.fields["llama.vocab_size"].types == [VT.UINT32] and core.read_metadata(out2)["llama.vocab_size"] == 8,
      "vocab_size updated, type kept")
del out2, t2
gc.collect()

print("8) Excel round trip")
core.export_xlsx(ws2, log=quiet)
import openpyxl
xf = d2 / "excel" / f"{ws2}_weights.xlsx"
wbx = openpyxl.load_workbook(xf)
idx = {r[0]: r[1] for r in wbx["Index"].iter_rows(min_row=2, max_col=2, values_only=True) if r[0] and r[1]}
wbx[idx["blk.0.attn_q.weight"]].cell(row=1, column=1).value = 123.5
wbx[idx["token_embd.weight"]].cell(row=8, column=2).value = -9.0       # a new token's row
last = list(idx.values())[-1]
keep = wbx[last].cell(row=1, column=1).value
wbx[last].cell(row=1, column=1).value = "abc"                          # bad cell in the LAST sheet
wbx.save(xf)
man2 = {e["name"]: e for e in core.load_manifest(d2)}
q_before = core.read_csv(d2 / man2["blk.0.attn_q.weight"]["csv"])[0, 0]
try:
    core.import_xlsx(ws2, log=quiet); refused = False
except ValueError as ex:
    refused = "not a number" in str(ex)
check(refused, "a workbook with a non-number cell is refused")
check(core.read_csv(d2 / man2["blk.0.attn_q.weight"]["csv"])[0, 0] == q_before, "a refused workbook changes nothing (not even earlier sheets)")
wbx[last].cell(row=1, column=1).value = keep
wbx.save(xf)                                                            # fails if the refused import left the file open
r = core.import_xlsx(ws2, log=quiet)
check(r["cells"] == 2 and sorted(r["tensors"]) == ["blk.0.attn_q.weight", "token_embd.weight"], f"imported {r}")
res = core.export_gguf(ws2, log=quiet)
out2 = GGUFReader(res["path"])
check(core.full_tensor(tensors(out2)["blk.0.attn_q.weight"])[0, 0] == 123.5, "Excel edit reaches the GGUF")
check(abs(core.full_tensor(tensors(out2)["token_embd.weight"])[7, 1] + 9.0) < 0.1, "Excel edit on a new token row")
print("9) exact change percentages")
st = res["stats"]
# tiny model: token_embd 6x32 + output 6x32 + attn_q 32x32 = 1408 original weights; 2 tokens added -> 2x(2x32) = 128 new
check(st["original_total"] == 1408 and st["added"] == 128, f"totals: original {st['original_total']}, added {st['added']}")
check(st["edited"] == 1 and st["edited_in_new"] == 1,
      f"Excel edit on attn_q counts as original, edit on the new token row counts as new ({st['edited']}, {st['edited_in_new']})")
check(st["edited_pct"] == "0.0710227273", f"1/1408 = {st['edited_pct']} % (10 decimals, exact)")
check(st["added_pct"] == "9.0909090909", f"128/1408 = {st['added_pct']} %")
check(st["stored_changed"] == 1, f"F32 tensor: exactly 1 value differs in the written file ({st['stored_changed']})")
check(core.pending_changes(ws2)["stats"]["edited"] == 1, "pending (grid-only) count matches")
(d2 / "edit_weights.py").write_text(core.EDIT_TEMPLATE.replace("    pass\n",
    "    w['blk.0.attn_q.weight'][5, :10] = 7.0\n"), encoding="utf-8")
m = core.measure_changes(ws2, log=quiet)
check(m["edited"] == 11 and m["edited_pct"] == core.pct(11, 1408), f"script edits measured exactly: {m['edited']} = {m['edited_pct']} %")
check(core.pct(1, 3) == "33.3333333333" and core.pct(2, 3, 5) == "66.66667" and core.pct(0, 10) == "0.0000000000", "pct() rounding")
del out2
gc.collect()

print("10) edit ANY cell (outside the CSV block) + K-quant tensors keep their format")
import kquants
src3 = tmp / "kq.gguf"
w3 = gguf.GGUFWriter(str(src3), "llama")
w3.add_uint32("llama.block_count", 1)
kq_vals = (rng.standard_normal((8, 512)) * 0.02).astype(np.float32)
for name, qt in (("blk.0.ffn_up.weight", "Q4_K"), ("blk.0.ffn_down.weight", "Q6_K"), ("blk.0.attn_q.weight", "Q2_K")):
    w3.add_tensor(name, kquants.quantize_blocks(kq_vals.reshape(-1, 256), qt).reshape(8, -1),
                  raw_dtype=gguf.GGMLQuantizationType[qt])
w3.add_tensor("blk.0.attn_norm.weight", rng.standard_normal(1000).astype(np.float32))
w3.write_header_to_file(); w3.write_kv_data_to_file(); w3.write_tensors_to_file(); w3.close()
ws3 = core.extract_workspace(src3, max_rows=4, max_cols=4, log=quiet)      # tiny CSV block on purpose
d3 = core.WORKSPACES / ws3
core.save_cells(ws3, "blk.0.ffn_up.weight", [[7, 500, 0.125], [0, 0, 0.0625]])   # outside + inside the block
core.save_cells(ws3, "blk.0.attn_norm.weight", [[999, 0, -3.5]])
core.save_cells(ws3, "blk.0.ffn_down.weight", [[5, 300, 0.01]])
e_up = next(x for x in core.load_manifest(d3) if x["name"] == "blk.0.ffn_up.weight")
check(core.load_sparse(d3, e_up) == {(7, 500): 0.125} and core.read_csv(d3 / e_up["csv"])[0, 0] == 0.0625,
      "cell outside the block -> edits/*.json, inside -> CSV")
_, _, ot3 = core.open_source(d3)
cur, base = core.tile(d3, ot3, e_up, 6, 2, 498, 4)
check(cur[1, 2] == 0.125 and base[1, 2] != 0.125, "tile shows the edit anywhere in the tensor")
check(core.pending_changes(ws3)["stats"]["edited"] == 4, "pending count includes edits outside the block")
core.save_cells(ws3, "blk.0.ffn_down.weight", [[5, 300, float(base_v)] for base_v in
                [core.base_tile(d3, ot3, next(x for x in core.load_manifest(d3) if x["name"] == "blk.0.ffn_down.weight"), 5, 1, 300, 1)[0, 0]]])
check(core.pending_changes(ws3)["stats"]["edited"] == 3, "typing the original value back removes the edit")
res = core.export_gguf(ws3, log=quiet)
o3 = {t.name: t for t in GGUFReader(res["path"]).tensors}
s3 = {t.name: t for t in GGUFReader(str(src3)).tensors}
up = core.full_tensor(o3["blk.0.ffn_up.weight"])
check(o3["blk.0.ffn_up.weight"].tensor_type.name == "Q4_K", "edited Q4_K tensor is still Q4_K")
check(abs(up[7, 500] - 0.125) < 0.01 and abs(up[0, 0] - 0.0625) < 0.01, f"both edits reached the file ({up[7,500]:.4f}, {up[0,0]:.4f})")
blocks_changed = int((o3["blk.0.ffn_up.weight"].data.reshape(-1, 144) != s3["blk.0.ffn_up.weight"].data.reshape(-1, 144)).any(1).sum())
check(blocks_changed == 2, f"only the 2 edited blocks changed bytes ({blocks_changed} of 16)")
check(np.array_equal(o3["blk.0.ffn_down.weight"].data, s3["blk.0.ffn_down.weight"].data), "un-edited Q6_K tensor byte-identical")
check(core.full_tensor(o3["blk.0.attn_norm.weight"])[999] == -3.5, "1D edit outside the block, exact (F32)")
ex = core.export_gguf(ws3, out_path=d3 / "exports" / "exact.gguf", exact=True, log=quiet)
oe = {t.name: t for t in GGUFReader(ex["path"]).tensors}
check(oe["blk.0.ffn_up.weight"].tensor_type.name == "F32" and core.full_tensor(oe["blk.0.ffn_up.weight"])[7, 500] == 0.125,
      "exact mode: edited tensor stored as F32 with the typed value exactly")
check(np.array_equal(oe["blk.0.ffn_down.weight"].data, s3["blk.0.ffn_down.weight"].data)
      and oe["blk.0.ffn_down.weight"].tensor_type.name == "Q6_K", "exact mode leaves un-edited tensors byte-identical, in their format")
del o3, s3, oe, up
gc.collect()
core.insert_slices(ws3, "blk.0.ffn_up.weight", "rows", 0, 1, "zeros", log=quiet)
e_up = next(x for x in core.load_manifest(d3) if x["name"] == "blk.0.ffn_up.weight")
check(not core.load_sparse(d3, e_up) and core.pending_changes(ws3)["stats"]["edited"] == 3,
      "inserting rows bakes outside-block edits in (still counted)")
core.undo_last(ws3, log=quiet)
e_up = next(x for x in core.load_manifest(d3) if x["name"] == "blk.0.ffn_up.weight")
check(core.load_sparse(d3, e_up) == {(7, 500): 0.125}, "undo restores the edits file")
core.revert_tensor(ws3, "blk.0.ffn_up.weight")
check(core.pending_changes(ws3)["stats"]["edited"] == 1, "revert tensor clears its CSV and outside-block edits")

shutil.rmtree(tmp, ignore_errors=True)
print("ALL WORKSPACE CHECKS PASSED")
