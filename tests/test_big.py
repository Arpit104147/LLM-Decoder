"""Big-model behaviour on a small synthetic model: split files read as one, export that streams edited
tensors (memory stays small), pass-through layers on 3D expert tensors, disk-space checks, the logit-lens
copy cache, and the memory guard of the inspection tools.

Run:  python test_big.py
"""
import gc
import shutil
import sys
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import quantize, dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core
import inspect_core as ic
import kquants

ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok  " if cond else "  BAD ") + msg, flush=True)


quiet = lambda *_: None
tmp = Path(tempfile.mkdtemp(prefix="llmdec_big_"))
rng = np.random.default_rng(7)
E, F, H, V = 8, 512, 4096, 64            # experts, expert rows, hidden size, vocab

print("1) a 3-part split MoE-shaped model")
w = gguf.GGUFWriter(str(tmp / "moe.gguf"), "llama", split_max_tensors=2)
w.add_uint32("llama.block_count", 1)
w.add_uint32("llama.expert_count", E)
w.add_array("llama.attention.head_count_kv", [4])
w.add_string("tokenizer.ggml.model", "gpt2")
w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(V)])
src = {}
emb = rng.standard_normal((V, H)).astype(np.float16)
src["token_embd.weight"] = emb
w.add_tensor("token_embd.weight", emb)
exps = (rng.standard_normal((E, F, H)) * 0.02).astype(np.float32)          # 16.8M numbers = 67 MB as float32
qexps = quantize(exps.reshape(-1, H), Q.Q8_0).reshape(E, F, -1)
w.add_tensor("blk.0.ffn_down_exps.weight", qexps, raw_dtype=Q.Q8_0)
kv = (rng.standard_normal((4, 256, 1024)) * 0.02).astype(np.float32)
w.add_tensor("blk.0.ffn_up_exps.weight", kquants.quantize_blocks(kv.reshape(-1, 256), "Q4_K").reshape(4, 256, -1),
             raw_dtype=Q.Q4_K)
w.add_tensor("blk.0.attn_output.weight", (rng.standard_normal((H, 256)) * 0.02).astype(np.float32))
w.add_tensor("output_norm.weight", np.ones(H, np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
parts = sorted(tmp.glob("moe-*-of-*.gguf"))
check(len(parts) == 3, f"writer made {len(parts)} parts: {[p.name for p in parts]}")

ws = core.extract_workspace(parts[1], overwrite=True, log=quiet)  # open the MIDDLE part on purpose
d = core.WORKSPACES / ws
info = core.load_info(d)
man = core.load_manifest(d)
check(ws == "moe" and info["split_count"] == 3 and Path(info["gguf_path"]).name == parts[0].name,
      f"workspace '{ws}' from any part, starts at part 1, {info['split_count']} parts")
check(len(man) == 5, f"all {len(man)} tensors from all parts are in the workspace")
meta = core.load_user_metadata(d)
check(not any(k.startswith("split.") for k in meta), "split bookkeeping keys are not in metadata.json")
e_x = next(x for x in man if x["name"] == "blk.0.ffn_down_exps.weight")
_, _, ot = core.open_source(d)
cur, base = core.tile(d, ot, e_x, 7 * F + 500, 1, 4000, 3)
check(np.allclose(base[0], dequantize(qexps[7, 500:501], Q.Q8_0).reshape(-1)[4000:4003]),
      "grid reads expert 7 of the 3D tensor from its part")
missing = tmp / "lonely-00002-of-00003.gguf"
shutil.copy(parts[1], missing)
try:
    core.extract_workspace(missing, log=quiet); err = ""
except FileNotFoundError as ex:
    err = str(ex)
check("missing" in err and "lonely-00001-of-00003.gguf" in err, f"missing parts are named: {err[:110]}…")

print("2) edits anywhere, then a low-memory export")
core.save_cells(ws, "blk.0.ffn_down_exps.weight", [[7 * F + 500, 4000, 0.5], [3, 3, -0.25]])   # far away + CSV block
core.save_cells(ws, "blk.0.ffn_up_exps.weight", [[3 * 256 + 255, 1023, 0.125]])
core.save_cells(ws, "output_norm.weight", [[4095, 0, 2.0]])
check(core.pending_changes(ws)["stats"]["edited"] == 4, "4 edits pending")
core.release_sources(); gc.collect()
tracemalloc.start()
res = core.export_gguf(ws, log=quiet)
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
check(peak < 45e6, f"export peak memory {peak / 1e6:.0f} MB (the big tensor alone is 67 MB as float32)")
check(Path(res["path"]).name == "moe_edited.gguf" and not list(Path(res["path"]).parent.glob("*.tmp")),
      "one merged file, temporary files removed")
out = GGUFReader(res["path"])
ometa = core.read_metadata(out)
check(not any(k.startswith("split.") for k in ometa) and len(out.tensors) == 5, "merged file: no split keys, 5 tensors")
ot_out = {t.name: t for t in out.tensors}
srcr = core.SplitReader(parts)
ot_src = {t.name: t for t in srcr.tensors}
check(np.array_equal(ot_out["token_embd.weight"].data, ot_src["token_embd.weight"].data)
      and np.array_equal(ot_out["blk.0.attn_output.weight"].data, ot_src["blk.0.attn_output.weight"].data),
      "untouched tensors byte-identical")
tx = ot_out["blk.0.ffn_down_exps.weight"]
check(tuple(int(x) for x in tx.shape) == (H, F, E) and tx.tensor_type == Q.Q8_0, "3D shape and Q8_0 kept")
changed_blocks = int((tx.data.reshape(-1, 34) != ot_src["blk.0.ffn_down_exps.weight"].data.reshape(-1, 34)).any(1).sum())
check(changed_blocks == 2, f"only the 2 edited blocks of {E * F * H // 32:,} changed bytes ({changed_blocks})")
v = core._decode(tx.data.reshape(E * F, -1)[7 * F + 500:7 * F + 501], Q.Q8_0).reshape(-1)[4000]
check(abs(v - 0.5) < 0.01, f"far-away edit is in the file ({v:.4f})")
up = core.full_tensor(ot_out["blk.0.ffn_up_exps.weight"])
check(ot_out["blk.0.ffn_up_exps.weight"].tensor_type == Q.Q4_K and abs(up[3, 255, 1023] - 0.125) < 0.02,
      f"Q4_K expert tensor kept, edit present ({up[3, 255, 1023]:.4f})")
check(core.full_tensor(ot_out["output_norm.weight"])[4095] == 2.0, "1D edit exact")
note = next(t for t in res["tensors"] if t["name"] == "blk.0.ffn_down_exps.weight")
check("re-encoded 2 of" in note["note"] and note["stored_changed"] >= 2, f"report: {note['note']}")
del out, ot_out, tx, srcr, ot_src
gc.collect()

print("3) exact mode streams too")
core.release_sources(); gc.collect()
tracemalloc.start()
res = core.export_gguf(ws, log=quiet, exact=True)
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
out = GGUFReader(res["path"])
tx = next(t for t in out.tensors if t.name == "blk.0.ffn_down_exps.weight")
check(tx.tensor_type == Q.F32 and core.full_tensor(tx)[7, 500, 4000] == np.float32(0.5),
      f"stored as F32 with the exact value; peak memory {peak / 1e6:.0f} MB")
check(peak < 60e6, "exact-mode export peak memory stays small")
del out, tx
gc.collect()

print("4) pass-through layer on a 3D expert tensor")
for name in ("blk.0.ffn_down_exps.weight", "blk.0.ffn_up_exps.weight", "output_norm.weight"):
    core.revert_tensor(ws, name)
core.add_layers(ws, 1, "identity", log=quiet)
core.release_sources(); gc.collect()
tracemalloc.start()
res = core.export_gguf(ws, log=quiet)
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
out = GGUFReader(res["path"])
z = next(t for t in out.tensors if t.name == "blk.1.ffn_down_exps.weight")
zv = core.full_tensor(z)
check(z.tensor_type == Q.Q8_0 and zv.shape == (E, F, H) and not zv.any(), "new expert tensor is all zeros, Q8_0, 3D")
check(peak < 45e6, f"peak memory {peak / 1e6:.0f} MB")
up1 = next(t for t in out.tensors if t.name == "blk.1.ffn_up_exps.weight")
check(np.array_equal(up1.data, next(t for t in out.tensors if t.name == "blk.0.ffn_up_exps.weight").data),
      "copied layer tensors are byte copies")
del out, z, zv, up1
gc.collect()
core.undo_last(ws, log=quiet)

print("5) not enough disk space -> clear error before writing anything")
core.save_cells(ws, "blk.0.ffn_down_exps.weight", [[1, 1, 0.3]])
real = core.shutil.disk_usage
core.shutil.disk_usage = lambda p: real(p)._replace(free=50 * 1024 ** 2)
try:
    core.export_gguf(ws, log=quiet); err = ""
except RuntimeError as ex:
    err = str(ex)
finally:
    core.shutil.disk_usage = real
check("Not enough disk space" in err and "GB" in err, f"{err[:120]}…")
check(not list((d / "exports").glob("*.partial")) and not list((d / "exports").glob("*.tmp")), "nothing half-written")
core.revert_tensor(ws, "blk.0.ffn_down_exps.weight")

print("6) logit-lens copies: split-safe, size-capped cache")
t = ic._truncated(ws, 1, quiet)
tr = GGUFReader(str(t))
check(not any(k.startswith("split.") for k in core.read_metadata(tr)) and len(tr.tensors) == 5,
      "cut-down copy of a split model is one normal file")
del tr
gc.collect()
lens = d / "lens"
for i, size in enumerate((300, 300, 300)):
    p = lens / f"layers_{90 + i}.gguf"
    p.write_bytes(b"\0" * size)
    s = p.with_suffix(".stamp"); s.write_text("x")
    import os, time
    os.utime(s, (time.time() - 1000 + i, time.time() - 1000 + i))
ic._make_room(lens, 400, 700 + t.stat().st_size, quiet)        # cap leaves room for only 1 old copy + the new one
left = sorted(p.name for p in lens.glob("layers_9*.gguf"))
check(left == ["layers_92.gguf"], f"oldest copies removed to stay under the cap: {left}")
real = ic.shutil.disk_usage
ic.shutil.disk_usage = lambda p: real(p)._replace(free=10 * 1024 ** 2)
core.shutil.disk_usage = ic.shutil.disk_usage
try:
    (lens / "layers_1.stamp").unlink()
    ic._truncated(ws, 1, quiet); err = ""
except RuntimeError as ex:
    err = str(ex)
finally:
    ic.shutil.disk_usage = core.shutil.disk_usage = real
check("Not enough disk space" in err, f"{err[:110]}…")
ic.clear_lens(ws)

print("7) inspection tools refuse matrices too big for this computer's memory")
real = core.total_ram
core.total_ram = lambda: 1024 ** 2
try:
    ic.nearest_tokens(ws, 3); err = ""
except MemoryError as ex:
    err = str(ex)
finally:
    core.total_ram = real
check("would need" in err and "GB of memory" in err, f"{err[:110]}…")
check(core.total_ram() > 1e9, f"total RAM detected: {core.total_ram() / 1e9:.0f} GB")

core.release_sources(); gc.collect()
shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print("ALL BIG-MODEL CHECKS PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
