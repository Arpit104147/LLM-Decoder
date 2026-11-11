"""Tests for inspect_core: tensor roles, token decoding, nearest tokens / neuron math, logit-lens truncation.
Run:  python test_inspect.py
"""
import gc
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core
import inspect_core as ic

tmp = Path(tempfile.mkdtemp())
core.WORKSPACES = tmp / "workspaces"
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok  " if cond else "  BAD ") + msg)


print("1) tensor roles")
for name, role in [("token_embd.weight", "Token embedding table"), ("blk.3.attn_q.weight", "Query projection"),
                   ("blk.3.ffn_down.weight", "MLP down projection"), ("blk.0.shortconv.out_proj.weight", "Conv output projection"),
                   ("blk.7.ffn_down_exps.weight", "Experts' output weights"), ("output.weight", "Unembedding (LM head)"),
                   ("v.blk.2.attn_q.weight", "vision encoder Query projection"), ("mm.2.weight", "Multimodal projector")]:
    check(ic.describe_tensor(name)["role"] == role, f"{name} -> {ic.describe_tensor(name)['role']}")
check(ic.describe_tensor("blk.3.ffn_down.weight")["kind"] == "writes" and ic.describe_tensor("blk.3.ffn_up.weight")["kind"] == "reads",
      "down writes / up reads the residual stream")

print("2) token text")
check(ic.token_text("Ġthe", "gpt2") == " the" and ic.token_text("Ċ", "gpt2") == "\n", "GPT-2 byte-level decoding")
check(ic.token_text("▁the", "llama") == " the" and ic.token_text("<0x0A>", "llama") == "\n", "SentencePiece decoding")

print("3) nearest tokens + neuron projection (exact math on a tiny model)")
rng = np.random.default_rng(0)
V, E, F, L = 50, 32, 64, 4
emb = rng.standard_normal((V, E)).astype(np.float32)
emb[7] = emb[3] * 2 + 0.01 * rng.standard_normal(E)          # token 7 points the same way as token 3
down = rng.standard_normal((E, F)).astype(np.float32)
down[:, 5] = emb[11] * 3                                         # neuron 5 writes token 11's direction
src = tmp / "tiny.gguf"
w = gguf.GGUFWriter(str(src), "llama")
w.add_uint32("llama.block_count", L)
w.add_uint32("llama.embedding_length", E)
w.add_array("llama.attention.head_count_kv", [2] * L)
w.add_string("tokenizer.ggml.model", "gpt2")
w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(V)])
w.add_tensor("token_embd.weight", emb)
w.add_tensor("output_norm.weight", np.ones(E, np.float32))
for i in range(L):
    w.add_tensor(f"blk.{i}.ffn_down.weight", down)
    w.add_tensor(f"blk.{i}.ffn_up.weight", rng.standard_normal((F, E)).astype(np.float32))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(E, np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
ws = core.extract_workspace(src, log=lambda *_: None)
r = ic.nearest_tokens(ws, 3)
check(r["nearest"][0]["id"] == 7, f"nearest to token 3 is token 7 (got {r['nearest'][0]['id']})")
cos = emb @ emb[3] / np.linalg.norm(emb, axis=1) / np.linalg.norm(emb[3])
check(abs(r["nearest"][0]["score"] - cos[7]) < 1e-3 and all(t["id"] != 3 for t in r["nearest"]), "cosine value exact, token itself excluded")
n = ic.neuron(ws, "blk.2.ffn_down.weight", 5)
check(n["mode"] == "writes" and n["axis"] == "col" and n["promotes"][0]["id"] == 11, f"neuron 5 promotes token 11 (got {n['promotes'][0]['id']})")
n2 = ic.neuron(ws, "blk.0.ffn_up.weight", 9)
check(n2["mode"] == "reads" and n2["axis"] == "row", "ffn_up rows are read as detectors")
a = ic.anatomy(ws)
check(len(a["layers"]) == L and a["layers"][0]["kinds"] == ["MLP"] and a["tied_embeddings"], "anatomy: 4 MLP layers, tied embeddings")

print("4) logit-lens truncation")
t = ic._truncated(ws, 2, log=lambda *_: None)
r2 = GGUFReader(str(t))
names = {x.name for x in r2.tensors}
meta = core.read_metadata(r2)
check(meta["llama.block_count"] == 2 and meta["llama.attention.head_count_kv"] == [2, 2], "block_count and per-layer lists cut to 2")
check(names == {"token_embd.weight", "output_norm.weight"} | {f"blk.{i}.{s}" for i in range(2) for s in ("ffn_down.weight", "ffn_up.weight", "ffn_norm.weight")},
      "only layers 0-1 kept, plus embedding/output tensors")
del r2
gc.collect()
core.release_sources()
shutil.rmtree(tmp, ignore_errors=True)
print("ALL INSPECT CHECKS PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
