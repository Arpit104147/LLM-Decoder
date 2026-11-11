"""
Checks every architecture family of the chat tracer against llama.cpp, without downloading any model:
for each family a tiny model with random weights is written as a GGUF (same tensor names, shapes and settings
as the real models), then run both in llama.cpp (LM Studio's llama-server) and in trace_core.

Random weights give gibberish, but llama.cpp does exactly the same maths whatever the numbers are, so if both
engines produce the same tokens and the same probabilities here, the family's maths is the same.
It cannot catch a quirk that only a real file has (e.g. unusual metadata), which test_trace.py covers on real models.

  python test_trace_arch.py              all families
  python test_trace_arch.py qwen3 phi3   only these

Also checks the tokenizers: SentencePiece on a synthetic vocabulary, and the BPE splitting rules on the real
Qwen3.5 vocabulary (if that workspace exists).
"""
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import inspect_core as ic
import trace_core as tc

PORT = 8150
TMP = Path(tempfile.mkdtemp(prefix="llmdec_arch_"))
ok = fail = 0


def check(cond, msg):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


# ---------------------------------------------------------------- llama-server
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode())
    try:
        return json.loads(urllib.request.urlopen(req, timeout=900).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"llama-server said {e.code}: {e.read().decode(errors='replace')[:400]}")


class Server:
    def __init__(self, model, ctx=1024):
        self.log = tempfile.TemporaryFile()
        self.p = subprocess.Popen([str(ic.find_llama_server()), "-m", str(model), "--port", str(PORT), "-c", str(ctx),
                                   "-ngl", "0", "--device", "none", "--no-op-offload", "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--no-warmup", "-np", "1", "-t", "4"], stdout=subprocess.DEVNULL,
                                  stderr=self.log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                return
            except Exception:
                if self.p.poll() is not None:
                    break
                time.sleep(0.3)
        self.log.seek(0)
        msg = self.log.read().decode(errors="replace")
        self.close()
        raise RuntimeError("llama-server didn't start:\n" + "\n".join(l for l in msg.splitlines() if "error" in l.lower() or "failed" in l.lower())[-1500:])

    def close(self):
        self.p.kill()
        self.p.wait()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------- synthetic models
def byte_vocab():
    """A gpt2-type vocabulary where every token is whole characters (printable ASCII, space, newline, tab and
    two-letter pieces), so llama-server never holds a token back as an unfinished UTF-8 character and reports
    probabilities for every step. Plus a few merges and control tokens."""
    enc = {b: c for c, b in ic._BYTE_DEC.items()}
    chars = [enc[b] for b in list(range(0x20, 0x7F)) + [0x0A, 0x09]]
    letters = "etaoinshrdlu"
    toks = chars + [a + b for a in letters for b in letters][:150] + ["Ġt", "Ġthe"]
    toks += ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    types = [1] * (len(toks) - 3) + [3, 3, 3]
    merges = ["Ġ t", "h e", "Ġt he"]
    return toks, types, merges


class Spec:
    """A tiny model of one family. Options mirror what the real models have."""
    def __init__(self, arch, D=64, H=4, KV=2, hd=16, ff=96, L=4, **o):
        self.arch, self.D, self.H, self.KV, self.hd, self.ff, self.L, self.o = arch, D, H, KV, hd, ff, L, o


def write_model(spec, path, seed=0):
    rng = np.random.default_rng(seed)
    a, D, H, KV, hd, ff, L, o = spec.arch, spec.D, spec.H, spec.KV, spec.hd, spec.ff, spec.L, spec.o
    w = gguf.GGUFWriter(str(path), a)
    keys = {"block_count": L, "context_length": 4096, "embedding_length": D, "feed_forward_length": ff,
            "attention.head_count": H, "attention.head_count_kv": KV, "attention.key_length": hd,
            "attention.value_length": hd, "rope.dimension_count": o.get("n_rot", hd), "rope.freq_base": o.get("base", 10000.0),
            f"attention.layer_norm_{'epsilon' if o.get('layernorm') else 'rms_epsilon'}": 1e-5}
    if o.get("moe"):
        mo = o["moe"]
        keys.update({"expert_count": mo["E"], "expert_used_count": mo["k"], "expert_feed_forward_length": mo["ffe"]})
        if mo.get("ffs"):
            keys["expert_shared_feed_forward_length"] = mo["ffs"]
    keys.update(o.get("keys", {}))                     # a family's own keys win (DeepSeek redefines the key lengths)
    for k, v in keys.items():
        if v is None:
            continue
        if isinstance(v, list):
            w.add_array(f"{a}.{k}", v)
            continue
        add = (w.add_bool if isinstance(v, bool) else w.add_string if isinstance(v, str)
               else w.add_float32 if isinstance(v, float) else w.add_uint32)
        add(f"{a}.{k}", v)
    toks, types, merges = byte_vocab()
    V = len(toks)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(V - 3)
    w.add_eos_token_id(V - 1)
    w.add_add_bos_token(False)
    rnd = lambda *shape, s=None: (rng.standard_normal(shape) * (s if s is not None else 1.0 / np.sqrt(shape[-1]))).astype(np.float32)
    ones = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    w.add_tensor("token_embd.weight", rnd(V, D, s=1.0))
    if not o.get("tied"):
        w.add_tensor("output.weight", rnd(V, D, s=0.5))
    def norm(name, n):
        w.add_tensor(name + ".weight", ones(n))
        if o.get("ln_bias"):
            w.add_tensor(name + ".bias", rnd(n, s=0.1))
    norm(o.get("final_norm_name", "output_norm"), D)
    if o.get("rope_freqs"):
        w.add_tensor("rope_freqs.weight", (1.0 + rng.random(o.get("n_rot", hd) // 2) * 3).astype(np.float32))
    for i in range(L):
        p = f"blk.{i}."
        if not o.get("no_pre"):
            norm(p + "attn_norm", D)
        ml = o.get("mla")
        sm = o.get("ssm")
        ssm_layer = bool(sm) and i in sm.get("layers", range(L))
        conv_layer = i in o.get("conv_layers", ()) or (ssm_layer and not o.get("parallel_ssm")) or i in o.get("mlp_only", ())
        if ssm_layer and o.get("parallel_ssm"):        # Falcon-H1: Mamba 2 next to attention in the same layer
            di, ds, K, G, Hs = sm["di"], sm["ds"], sm["K"], sm["G"], sm["H"]
            w.add_tensor(p + "ssm_in.weight", rnd(di + di + 2 * G * ds + Hs, D))
            w.add_tensor(p + "ssm_conv1d.weight", rnd(di + 2 * G * ds, K, s=0.5))
            w.add_tensor(p + "ssm_conv1d.bias", rnd(di + 2 * G * ds, s=0.1))
            w.add_tensor(p + "ssm_dt.bias", (rng.random(Hs) - 1.0).astype(np.float32))
            w.add_tensor(p + "ssm_a", -np.exp(rng.standard_normal((Hs, 1)) * 0.5).astype(np.float32))
            w.add_tensor(p + "ssm_norm.weight", ones(di).reshape(G, di // G))
            w.add_tensor(p + "ssm_d", ones(Hs).reshape(-1, 1))
            w.add_tensor(p + "ssm_out.weight", rnd(D, di))
        if ssm_layer and not o.get("parallel_ssm"):    # Mamba 1 / Mamba 2 instead of attention
            di, ds, K = sm["di"], sm["ds"], sm["K"]
            if sm["v"] == 1:
                r = sm["dtr"]
                w.add_tensor(p + "ssm_in.weight", rnd(2 * di, D))
                w.add_tensor(p + "ssm_conv1d.weight", rnd(di, K, s=0.5))
                w.add_tensor(p + "ssm_conv1d.bias", rnd(di, s=0.1))
                w.add_tensor(p + "ssm_x.weight", rnd(r + 2 * ds, di))
                w.add_tensor(p + "ssm_dt.weight", rnd(di, r))
                w.add_tensor(p + "ssm_dt.bias", (rng.random(di) - 1.0).astype(np.float32))
                w.add_tensor(p + "ssm_a", -np.exp(rng.standard_normal((di, ds)) * 0.5).astype(np.float32))
                if sm.get("dbc_norms"):
                    w.add_tensor(p + "ssm_dt_norm.weight", ones(r))
                    w.add_tensor(p + "ssm_b_norm.weight", ones(ds))
                    w.add_tensor(p + "ssm_c_norm.weight", ones(ds))
            else:
                G, Hs = sm["G"], sm["H"]
                w.add_tensor(p + "ssm_in.weight", rnd(di + di + 2 * G * ds + Hs, D))
                w.add_tensor(p + "ssm_conv1d.weight", rnd(di + 2 * G * ds, K, s=0.5))
                w.add_tensor(p + "ssm_conv1d.bias", rnd(di + 2 * G * ds, s=0.1))
                w.add_tensor(p + "ssm_dt.bias", (rng.random(Hs) - 1.0).astype(np.float32))
                w.add_tensor(p + "ssm_a", -np.exp(rng.standard_normal((Hs, 1)) * 0.5).astype(np.float32))
                w.add_tensor(p + "ssm_norm.weight", ones(di).reshape(G, di // G))
            w.add_tensor(p + "ssm_d", ones(di) if sm["v"] == 1 else ones(sm["H"]).reshape(-1, 1))
            w.add_tensor(p + "ssm_out.weight", rnd(D, di))
        elif conv_layer and i in o.get("mlp_only", ()):
            pass                                       # Nemotron-H: a layer with only an MLP
        elif conv_layer:                               # LFM short convolution instead of attention
            K = o["keys"]["shortconv.l_cache"]
            w.add_tensor(p + "shortconv.in_proj.weight", rnd(3 * D, D))
            w.add_tensor(p + "shortconv.conv.weight", rnd(D, K, s=0.5))
            w.add_tensor(p + "shortconv.out_proj.weight", rnd(D, D))
        elif ml:                                       # DeepSeek multi-head latent attention
            hk = ml["nope"] + ml["rope"]
            if ml.get("q_lora"):
                w.add_tensor(p + "attn_q_a.weight", rnd(ml["q_lora"], D))
                w.add_tensor(p + "attn_q_a_norm.weight", ones(ml["q_lora"]))
                w.add_tensor(p + "attn_q_b.weight", rnd(H * hk, ml["q_lora"]))
            else:
                w.add_tensor(p + "attn_q.weight", rnd(H * hk, D))
            w.add_tensor(p + "attn_kv_a_mqa.weight", rnd(ml["kv_lora"] + ml["rope"], D))
            w.add_tensor(p + "attn_kv_a_norm.weight", ones(ml["kv_lora"]))
            if ml.get("split"):
                w.add_tensor(p + "attn_k_b.weight", rnd(H, ml["kv_lora"], ml["nope"], s=1 / np.sqrt(ml["kv_lora"])))
                w.add_tensor(p + "attn_v_b.weight", rnd(H, ml["v"], ml["kv_lora"]))
            else:
                w.add_tensor(p + "attn_kv_b.weight", rnd(H * (ml["nope"] + ml["v"]), ml["kv_lora"]))
            w.add_tensor(p + "attn_output.weight", rnd(D, H * ml["v"]))
        elif o.get("fused_qkv"):
            w.add_tensor(p + "attn_qkv.weight", rnd((H + 2 * KV) * hd, D))
        else:
            w.add_tensor(p + "attn_q.weight", rnd(H * hd, D))
            w.add_tensor(p + "attn_k.weight", rnd(KV * hd, D))
            w.add_tensor(p + "attn_v.weight", rnd(KV * hd, D))
            if o.get("qkv_bias"):
                for nm, n in (("attn_q", H * hd), ("attn_k", KV * hd), ("attn_v", KV * hd)):
                    w.add_tensor(p + nm + ".bias", rnd(n, s=0.3))
        if o.get("qk_norm") and not conv_layer:
            w.add_tensor(p + "attn_q_norm.weight", ones(hd))
            w.add_tensor(p + "attn_k_norm.weight", ones(hd))
        if o.get("qk_norm_full"):
            w.add_tensor(p + "attn_q_norm.weight", ones(H * hd))
            w.add_tensor(p + "attn_k_norm.weight", ones(KV * hd))
        if o.get("qk_norm_heads"):
            w.add_tensor(p + "attn_q_norm.weight", ones(H * hd).reshape(H, hd))
            w.add_tensor(p + "attn_k_norm.weight", ones(KV * hd).reshape(KV, hd))
        if not ml and not conv_layer:
            w.add_tensor(p + "attn_output.weight", rnd(D, H * hd))
            if o.get("head_gate"):
                w.add_tensor(p + "attn_gate.weight", rnd(H, D))
        if o.get("bias_all") or o.get("out_bias"):
            w.add_tensor(p + "attn_output.bias", rnd(D, s=0.2))
        if o.get("sinks"):
            w.add_tensor(p + "attn_sinks.weight", rnd(H, s=1.0))
        if o.get("post_norms"):
            w.add_tensor(p + "post_attention_norm.weight", ones(D))
        if o.get("no_ffn") or i not in o.get("ffn_layers", range(L)):
            continue                                   # Mamba / Nemotron-H layers without an MLP
        if o.get("ffn_norm_bare"):                     # Falcon-H1 names it "blk.N.ffn_norm" (no ".weight")
            w.add_tensor(p + "ffn_norm", ones(D))
        elif not o.get("no_pre") and not o.get("parallel") and i not in o.get("mlp_only", ()):
            norm(p + o.get("pre_ffn_name", "ffn_norm"), D)
        moe_here = o.get("moe") and i >= o["moe"].get("dense_lead", 0)
        if moe_here:
            mo = o["moe"]
            E, ffe = mo["E"], mo["ffe"]
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(E, D, s=1.0))
            if mo.get("router_bias"):
                w.add_tensor(p + "ffn_gate_inp.bias", rnd(E, s=0.5))
            if mo.get("sel_bias"):
                w.add_tensor(p + "exp_probs_b.bias", rnd(E, s=0.3))
            for nm, shp in (("ffn_gate_exps", (E, ffe, D)), ("ffn_up_exps", (E, ffe, D)), ("ffn_down_exps", (E, D, ffe))):
                x = rnd(*shp)
                if mo.get("quant"):                    # store experts quantized, as real files do (gpt-oss: MXFP4)
                    qt = getattr(gguf.GGMLQuantizationType, mo["quant"])
                    qx = gguf.quants.quantize(x.reshape(-1, shp[-1]), qt)
                    w.add_tensor(p + nm + ".weight", qx.reshape(shp[0], shp[1], -1), raw_dtype=qt)
                else:
                    w.add_tensor(p + nm + ".weight", x)
            if mo.get("expert_bias"):
                w.add_tensor(p + "ffn_gate_exps.bias", rnd(E, ffe, s=0.3))
                w.add_tensor(p + "ffn_up_exps.bias", rnd(E, ffe, s=0.3))
                w.add_tensor(p + "ffn_down_exps.bias", rnd(E, D, s=0.3))
            if o["moe"].get("ffs"):
                ffs = o["moe"]["ffs"]
                w.add_tensor(p + "ffn_gate_shexp.weight", rnd(ffs, D))
                w.add_tensor(p + "ffn_up_shexp.weight", rnd(ffs, D))
                w.add_tensor(p + "ffn_down_shexp.weight", rnd(D, ffs))
                if o["moe"].get("shared_gate"):
                    w.add_tensor(p + "ffn_gate_inp_shexp.weight", rnd(D, s=0.3))
        elif o.get("fused_up") and not o.get("moe"):
            w.add_tensor(p + "ffn_up.weight", rnd(2 * ff, D))
        else:
            if not o.get("plain_ffn"):
                w.add_tensor(p + "ffn_gate.weight", rnd(ff, D))
            w.add_tensor(p + "ffn_up.weight", rnd(ff, D))
        if not moe_here:
            w.add_tensor(p + "ffn_down.weight", rnd(D, ff))
        if o.get("bias_all"):
            w.add_tensor(p + "ffn_up.bias", rnd(ff, s=0.2))
            w.add_tensor(p + "ffn_down.bias", rnd(D, s=0.2))
        if o.get("post_norms"):
            w.add_tensor(p + "post_ffw_norm.weight", ones(D))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return V


def write_rwkv(spec, path, seed=0):
    """A tiny RWKV-6 model (rwkv6 / rwkv6 keys and tensor names as llama.cpp loads them)."""
    rng = np.random.default_rng(seed)
    a, D, L, ff, o = spec.arch, spec.D, spec.L, spec.ff, spec.o
    hs, ex, dex = o["hs"], o["extra"], o["dextra"]
    H = D // hs
    w = gguf.GGUFWriter(str(path), a)
    keys = {"block_count": L, "context_length": 4096, "embedding_length": D, "feed_forward_length": ff,
            "wkv.head_size": hs, "attention.head_count": 0}
    if a == "rwkv7":
        keys.update({f"attention.{k}_lora_rank": o["lora"] for k in ("decay", "iclr", "value_residual_mix", "gate")})
    else:
        keys.update({"time_mix_extra_dim": ex, "time_decay_extra_dim": dex, "rescale_every_n_layers": o.get("rescale", 0)})
    for k, v in keys.items():
        w.add_uint32(f"{a}.{k}", v)
    w.add_float32(f"{a}.attention.layer_norm_epsilon", 1e-5)
    toks, types, merges = byte_vocab()
    V = len(toks)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("gpt-2"); w.add_token_list(toks); w.add_token_types(types)
    w.add_token_merges(merges); w.add_bos_token_id(V - 3); w.add_eos_token_id(V - 1); w.add_add_bos_token(False)
    rnd = lambda *shape, s=None: (rng.standard_normal(shape) * (s if s is not None else 1.0 / np.sqrt(shape[-1]))).astype(np.float32)
    ones = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    lerp = lambda *shape: rng.random(shape).astype(np.float32)
    w.add_tensor("token_embd.weight", rnd(V, D, s=1.0))
    w.add_tensor("token_embd_norm.weight", ones(D)); w.add_tensor("token_embd_norm.bias", rnd(D, s=0.1))
    w.add_tensor("output_norm.weight", ones(D)); w.add_tensor("output_norm.bias", rnd(D, s=0.1))
    w.add_tensor("output.weight", rnd(V, D, s=0.5))
    for i in range(L):
        p = f"blk.{i}."
        for nm in ("attn_norm", "attn_norm_2"):
            w.add_tensor(p + nm + ".weight", ones(D)); w.add_tensor(p + nm + ".bias", rnd(D, s=0.1))
        if a == "rwkv7":
            lr = o["lora"]
            w.add_tensor(p + "time_mix_lerp_fused.weight", lerp(6, 1, 1, D))
            for nm in ("w", "a", "v", "g"):                 # (layer 0 has v0/v1/v2 too, unused)
                w.add_tensor(p + f"time_mix_{nm}1.weight", rnd(lr, D, s=0.3))
                w.add_tensor(p + f"time_mix_{nm}2.weight", rnd(D, lr, s=0.3))
                if nm != "g":
                    w.add_tensor(p + f"time_mix_{nm}0.weight", rnd(1, 1, D, s=0.5))
            w.add_tensor(p + "time_mix_k_k.weight", ones(D).reshape(1, 1, D))
            w.add_tensor(p + "time_mix_k_a.weight", ones(D).reshape(1, 1, D))
            w.add_tensor(p + "time_mix_r_k.weight", rnd(H * hs, s=0.5))
            for nm in ("receptance", "key", "value", "output"):
                w.add_tensor(p + f"time_mix_{nm}.weight", rnd(D, D))
            w.add_tensor(p + "time_mix_ln.weight", ones(D)); w.add_tensor(p + "time_mix_ln.bias", rnd(D, s=0.1))
            w.add_tensor(p + "channel_mix_lerp_k.weight", lerp(1, 1, D))
            w.add_tensor(p + "channel_mix_key.weight", rnd(ff, D))
            w.add_tensor(p + "channel_mix_value.weight", rnd(D, ff))
            continue
        w.add_tensor(p + "time_mix_lerp_x.weight", lerp(1, 1, D))
        w.add_tensor(p + "time_mix_lerp_fused.weight", lerp(5, 1, 1, D))
        w.add_tensor(p + "time_mix_w1.weight", rnd(5 * ex, D, s=0.3))
        w.add_tensor(p + "time_mix_w2.weight", rnd(5, D, ex, s=0.3))
        w.add_tensor(p + "time_mix_decay.weight", (rng.standard_normal((1, 1, D)) * 0.5 - 1.0).astype(np.float32))
        w.add_tensor(p + "time_mix_decay_w1.weight", rnd(dex, D, s=0.3))
        w.add_tensor(p + "time_mix_decay_w2.weight", rnd(D, dex, s=0.3))
        w.add_tensor(p + "time_mix_first.weight", rnd(H, hs, s=0.5))
        for nm in ("receptance", "key", "value", "gate", "output"):
            w.add_tensor(p + f"time_mix_{nm}.weight", rnd(D, D))
        w.add_tensor(p + "time_mix_ln.weight", ones(D)); w.add_tensor(p + "time_mix_ln.bias", rnd(D, s=0.1))
        w.add_tensor(p + "channel_mix_lerp_k.weight", lerp(1, 1, D))
        w.add_tensor(p + "channel_mix_lerp_r.weight", lerp(1, 1, D))
        w.add_tensor(p + "channel_mix_key.weight", rnd(ff, D))
        w.add_tensor(p + "channel_mix_value.weight", rnd(D, ff))
        w.add_tensor(p + "channel_mix_receptance.weight", rnd(D, D))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return V


def write_gemma3n(spec, path, seed=0):
    """A tiny Gemma 3n: 25 layers (so layers 20-24 share the KV cache of 18/19), 4 AltUp streams, LAuReL rank 4,
    8-wide per-layer inputs, sliding window 16 on layers where L % 5 < 4."""
    rng = np.random.default_rng(seed)
    a, D, H, KV, hd, ff, L, o = spec.arch, spec.D, spec.H, spec.KV, spec.hd, spec.ff, spec.L, spec.o
    A, dpl, rank = 4, o["dpl"], o["rank"]
    w = gguf.GGUFWriter(str(path), a)
    for k, v in {"block_count": L, "context_length": 4096, "embedding_length": D, "feed_forward_length": ff,
                 "attention.head_count": H, "attention.head_count_kv": KV, "attention.key_length": hd,
                 "attention.value_length": hd, "rope.dimension_count": hd, "attention.sliding_window": 16,
                 "altup.num_inputs": A, "altup.active_idx": 0, "embedding_length_per_layer_input": dpl,
                 "attention.shared_kv_layers": L - 20}.items():
        w.add_uint32(f"{a}.{k}", v)
    for k, v in {"rope.freq_base": 1000000.0, "rope.freq_base_swa": 10000.0, "attention.layer_norm_rms_epsilon": 1e-6,
                 "final_logit_softcapping": 30.0}.items():
        w.add_float32(f"{a}.{k}", v)
    w.add_array(f"{a}.activation_sparsity_scale", [0.95] * 10 + [0.0] * (L - 10))
    toks, types, merges = byte_vocab()
    V = len(toks)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("gpt-2"); w.add_token_list(toks); w.add_token_types(types)
    w.add_token_merges(merges); w.add_bos_token_id(V - 3); w.add_eos_token_id(V - 1); w.add_add_bos_token(False)
    rnd = lambda *shape, s=None: (rng.standard_normal(shape) * (s if s is not None else 1.0 / np.sqrt(shape[-1]))).astype(np.float32)
    ones = lambda *n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    w.add_tensor("token_embd.weight", rnd(V, D, s=0.3))
    w.add_tensor("output_norm.weight", ones(D))
    w.add_tensor("altup_proj.weight", rnd(A - 1, D, D))
    w.add_tensor("altup_unembd_proj.weight", rnd(A - 1, D, D))
    w.add_tensor("per_layer_token_embd.weight", rnd(V, dpl * L, s=0.3))
    w.add_tensor("per_layer_model_proj.weight", rnd(dpl * L, D))
    w.add_tensor("per_layer_proj_norm.weight", ones(dpl))
    for i in range(L):
        p = f"blk.{i}."
        for nm in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm", "altup_router_norm",
                   "laurel_post_norm"):
            w.add_tensor(p + nm + ".weight", ones(D))
        w.add_tensor(p + "altup_correct_scale.weight", ones(D))
        w.add_tensor(p + "attn_q.weight", rnd(H * hd, D)); w.add_tensor(p + "attn_k.weight", rnd(KV * hd, D))
        w.add_tensor(p + "attn_v.weight", rnd(KV * hd, D)); w.add_tensor(p + "attn_output.weight", rnd(D, H * hd))
        w.add_tensor(p + "attn_q_norm.weight", ones(hd)); w.add_tensor(p + "attn_k_norm.weight", ones(hd))
        w.add_tensor(p + "ffn_gate.weight", rnd(ff, D)); w.add_tensor(p + "ffn_up.weight", rnd(ff, D))
        w.add_tensor(p + "ffn_down.weight", rnd(D, ff))
        w.add_tensor(p + "inp_gate.weight", rnd(dpl, D)); w.add_tensor(p + "proj.weight", rnd(D, dpl))
        w.add_tensor(p + "altup_correct_coef.weight", rnd(A, A, s=0.3))
        w.add_tensor(p + "altup_predict_coef.weight", rnd(A * A, A, s=0.3))
        w.add_tensor(p + "altup_router.weight", rnd(A, D, s=3.0))
        w.add_tensor(p + "laurel_l.weight", rnd(rank, D)); w.add_tensor(p + "laurel_r.weight", rnd(D, rank))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return V


FAMILIES = {
    "llama": Spec("llama"),
    "llama-tied": Spec("llama", tied=True),
    "llama3-ropefreqs": Spec("llama", rope_freqs=True, base=500000.0),
    "granite": Spec("granite", keys={"embedding_scale": 2.5, "residual_scale": 0.6, "attention.scale": 0.2, "logit_scale": 3.0}),
    "qwen2": Spec("qwen2", qkv_bias=True, base=1000000.0),
    "qwen3": Spec("qwen3", qk_norm=True),
    # YaRN long-context rope (original context 32 so the interpolation actually changes these short tests)
    "qwen3-yarn": Spec("qwen3", qk_norm=True, keys={"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                                                    "rope.scaling.original_context_length": 32}),
    "llama-yarn": Spec("llama", keys={"rope.scaling.type": "yarn", "rope.scaling.factor": 8.0,
                                      "rope.scaling.original_context_length": 16}),
    "phi3": Spec("phi3", fused_qkv=True, fused_up=True),
    "phi3-window-key": Spec("phi3", fused_qkv=True, fused_up=True, keys={"attention.sliding_window": 2047}),
    "gemma": Spec("gemma", tied=True),
    "gemma2": Spec("gemma2", tied=True, post_norms=True, L=4,
                   keys={"attn_logit_softcapping": 5.0, "final_logit_softcapping": 4.0, "attention.sliding_window": 16}),
    "gemma3": Spec("gemma3", tied=True, post_norms=True, qk_norm=True, L=7, base=1000000.0,
                   keys={"attention.sliding_window": 16, "rope.scaling.type": "linear", "rope.scaling.factor": 8.0}),
    "command-r": Spec("command-r", tied=True, layernorm=True, parallel=True, keys={"logit_scale": 0.3}),
    # llama.cpp only reads Command-R+'s q/k norms when the model has 64+ layers (as the real 104B)
    "command-r-plus": Spec("command-r", tied=True, layernorm=True, parallel=True, qk_norm_heads=True, L=64, D=32, H=2, KV=1,
                           keys={"logit_scale": 0.3}),
    "starcoder2": Spec("starcoder2", layernorm=True, ln_bias=True, qkv_bias=True, bias_all=True, plain_ffn=True),
    "olmo2": Spec("olmo2", no_pre=True, post_norms=True, qk_norm_full=True),
    # mixtures of experts (8 experts, 2 per token, 48 neurons each)
    "mixtral": Spec("llama", ff=48, moe={"E": 8, "k": 2, "ffe": 48}),          # expert size = feed_forward_length
    "qwen2moe": Spec("qwen2moe", qkv_bias=True, moe={"E": 8, "k": 2, "ffe": 48, "ffs": 64, "shared_gate": True}),
    "qwen3moe": Spec("qwen3moe", qk_norm=True, moe={"E": 8, "k": 3, "ffe": 48}),
    "olmoe": Spec("olmoe", KV=4, ff=48, qk_norm_full=True, moe={"E": 8, "k": 2, "ffe": 48}),   # no grouped KV heads
    "granitemoe": Spec("granitemoe", ff=48, moe={"E": 8, "k": 2, "ffe": 48},
                       keys={"embedding_scale": 2.5, "residual_scale": 0.6, "attention.scale": 0.2, "logit_scale": 3.0}),
    # DeepSeek: latent attention (kv_lora 32, 8 rotated + 16 plain dims per head), first layer dense, shared expert
    "deepseek2-v2lite": Spec("deepseek2", L=3, H=4, KV=4, n_rot=8,
                             mla={"kv_lora": 32, "nope": 16, "rope": 8, "v": 16},
                             moe={"E": 8, "k": 2, "ffe": 48, "ffs": 48, "dense_lead": 1},
                             keys={"attention.q_lora_rank": 0, "attention.kv_lora_rank": 32, "attention.key_length": 24, "attention.value_length": 16,
                                   "leading_dense_block_count": 1, "expert_shared_count": 1, "expert_weights_scale": 1.0}),
    "deepseek3": Spec("deepseek2", L=3, H=4, KV=1, n_rot=8,
                      mla={"kv_lora": 32, "nope": 16, "rope": 8, "v": 16, "q_lora": 24, "split": True},
                      moe={"E": 8, "k": 2, "ffe": 48, "ffs": 48, "dense_lead": 1, "sel_bias": True},
                      keys={"attention.q_lora_rank": 24, "attention.kv_lora_rank": 32, "attention.key_length": 40,
                            "attention.value_length": 32, "attention.key_length_mla": 24, "attention.value_length_mla": 16,
                            "leading_dense_block_count": 1, "expert_shared_count": 1, "expert_weights_scale": 2.5,
                            "expert_weights_norm": True, "expert_gating_func": 2,
                            "rope.scaling.type": "yarn", "rope.scaling.factor": 4.0, "rope.scaling.original_context_length": 32,
                            "rope.scaling.yarn_log_multiplier": 0.1}),
    "deepseek3-groups": Spec("deepseek2", L=3, H=4, KV=1, n_rot=8,
                             mla={"kv_lora": 32, "nope": 16, "rope": 8, "v": 16, "q_lora": 24, "split": True},
                             moe={"E": 8, "k": 2, "ffe": 48, "ffs": 48, "dense_lead": 1, "sel_bias": True},
                             keys={"attention.q_lora_rank": 24, "attention.kv_lora_rank": 32, "attention.key_length": 40,
                                   "attention.value_length": 32, "attention.key_length_mla": 24, "attention.value_length_mla": 16,
                                   "leading_dense_block_count": 1, "expert_shared_count": 1, "expert_weights_scale": 2.5,
                                   "expert_weights_norm": True, "expert_gating_func": 2,
                                   "expert_group_count": 4, "expert_group_used_count": 2}),
    "gpt-oss": Spec("gpt-oss", ff=48, qkv_bias=True, out_bias=True, sinks=True, pre_ffn_name="post_attention_norm",
                    moe={"E": 8, "k": 2, "ffe": 48, "router_bias": True, "expert_bias": True},
                    keys={"attention.sliding_window": 16, "rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                          "rope.scaling.original_context_length": 32}),
    # Llama 4 is not enabled in trace_core yet: even with no experts, llama.cpp's attention for it differs from
    # every variant tried (NoPE layers, L2 q/k norm on/off, rope styles, attention temperature). Run it by name:
    #   python test_trace_arch.py llama4      (to work on it)
    "glm4": Spec("glm4", n_rot=8, qkv_bias=True, post_norms=True, fused_up=True),
    "glm4moe": Spec("glm4moe", n_rot=8, qkv_bias=True, pre_ffn_name="post_attention_norm",
                    moe={"E": 8, "k": 2, "ffe": 48, "ffs": 48, "dense_lead": 1, "sel_bias": True},
                    keys={"leading_dense_block_count": 1, "expert_shared_count": 1, "expert_weights_scale": 2.5,
                          "expert_weights_norm": True, "expert_gating_func": 2}),
    # LFM: short-convolution layers (0, 1, 3) + attention layers (2, 4); LFM3 adds head gates, post-norms, windows
    "lfm2": Spec("lfm2", L=5, qk_norm=True, conv_layers=(0, 1, 3), final_norm_name="token_embd_norm",
                 keys={"shortconv.l_cache": 3, "attention.head_count_kv": [0, 0, 2, 0, 2]}),
    "lfm3": Spec("lfm3", L=5, qk_norm=True, conv_layers=(0, 1, 3), head_gate=True, post_norms=True, rope_freqs=True,
                 keys={"shortconv.l_cache": 7, "attention.head_count_kv": [0, 0, 2, 0, 2], "attention.sliding_window": 16,
                       "attention.sliding_window_pattern": [False, False, False, False, True], "rope.freq_base_swa": 5000.0}),
    # state-space models (Mamba): 128 inner channels, 16 state dims, kernel 4
    "mamba": Spec("mamba", no_ffn=True, ssm={"v": 1, "di": 128, "ds": 16, "dtr": 8, "K": 4},
                  keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8}),
    "falcon-mamba": Spec("mamba", no_ffn=True, ssm={"v": 1, "di": 128, "ds": 16, "dtr": 8, "K": 4},
                         keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8,
                               "ssm.dt_b_c_rms": True}),
    "mamba2": Spec("mamba2", no_ffn=True, ssm={"v": 2, "di": 128, "ds": 16, "K": 4, "G": 2, "H": 8},
                   keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8,
                         "ssm.group_count": 2}),
    "jamba": Spec("jamba", L=4, ssm={"v": 1, "di": 128, "ds": 16, "dtr": 8, "K": 4, "layers": (0, 1, 3), "dbc_norms": True},
                  moe={"E": 4, "k": 2, "ffe": 96, "dense_lead": 2}, ff=96,
                  keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8,
                        "attention.head_count_kv": [0, 0, 2, 0]}),
    "granitehybrid": Spec("granitehybrid", L=4, ssm={"v": 2, "di": 128, "ds": 16, "K": 4, "G": 1, "H": 8, "layers": (0, 1, 3)},
                          moe={"E": 4, "k": 2, "ffe": 48, "ffs": 48}, ff=48,
                          keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8,
                                "ssm.group_count": 1, "attention.head_count_kv": [0, 0, 2, 0], "embedding_scale": 2.0,
                                "residual_scale": 0.5, "attention.scale": 0.25, "logit_scale": 2.0}),
    "falcon-h1": Spec("falcon-h1", L=4, parallel_ssm=True, ffn_norm_bare=True, ssm={"v": 2, "di": 128, "ds": 16, "K": 4, "G": 1, "H": 8},
                      keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8,
                            "ssm.group_count": 1}),
    # layers: 0 Mamba 2, 1 MLP, 2 attention, 3 Mamba 2, 4 MLP
    "nemotron_h": Spec("nemotron_h", L=5, ssm={"v": 2, "di": 128, "ds": 16, "K": 4, "G": 2, "H": 8, "layers": (0, 3)},
                       plain_ffn=True, ffn_layers=(1, 4), mlp_only=(1, 4),
                       keys={"ssm.conv_kernel": 4, "ssm.inner_size": 128, "ssm.state_size": 16, "ssm.time_step_rank": 8,
                             "ssm.group_count": 2, "attention.head_count_kv": [0, 0, 2, 0, 0], "attention.layer_norm_epsilon": 1e-5,
                             "feed_forward_length": [0, 96, 0, 0, 96]}),
    # as the real files: experts in MXFP4 (4-bit). llama.cpp then rounds activations to 8 bits inside those matmuls,
    # so probabilities differ a little more (the same happens with real Q8_0 / Q4 models)
    # (measured: Q8_0 experts drift 0.18, MXFP4 0.30 on this random model; the MXFP4 decoding itself is checked
    #  value for value by check_mxfp4_decode)
    "gpt-oss-mxfp4": Spec("gpt-oss", ff=64, qkv_bias=True, out_bias=True, sinks=True, pre_ffn_name="post_attention_norm", tol=0.5,
                          moe={"E": 8, "k": 2, "ffe": 64, "router_bias": True, "expert_bias": True, "quant": "MXFP4"},
                          keys={"attention.sliding_window": 16, "rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                                "rope.scaling.original_context_length": 32}),
}


FAMILIES_WIP = {          # not enabled in trace_core yet; only run when named on the command line
}
FAMILIES.update({
    "rwkv6": Spec("rwkv6", L=4, ff=96, rwkv=True, hs=16, extra=8, dextra=8),
    "rwkv6-rescale": Spec("rwkv6", L=4, ff=96, rwkv=True, hs=16, extra=8, dextra=8, rescale=2),
    # head size 64 as real RWKV-7 models: ggml's vectorized WKV7 loop steps 32 numbers at a time and reads past
    # the end of smaller heads
    "rwkv7": Spec("rwkv7", D=128, L=4, ff=192, rwkv=True, hs=64, extra=0, dextra=0, lora=8),
    # (per-layer input width 256 and LAuReL rank 64 as the real models: this llama.cpp build fixes those values)
    # Measured: every token identical; log-probabilities 0.02-0.05 apart typically, up to 0.26 for unlikely tokens.
    # Every structural alternative tried (window ±1, other KV sources, no sparsity, rope style…) is 10-100× worse.
    "gemma3n": Spec("gemma3n", D=32, H=2, KV=1, hd=16, ff=64, L=25, dpl=256, rank=64, tol=0.3),
    # Llama 4: 4 layers (layer 3 without rope), MoE top-1 with a shared expert; also a dense one (no MoE layers)
    "llama4": Spec("llama4", ff=48, moe={"E": 8, "k": 1, "ffe": 48, "ffs": 48}, keys={"interleave_moe_layer_step": 1}),
    "llama4-dense": Spec("llama4", ff=48, moe={"E": 8, "k": 1, "ffe": 48, "ffs": 48, "dense_lead": 9},
                         keys={"interleave_moe_layer_step": 99}),
    # past 8192 tokens: chunked attention on layers 0-2 and attention temperature on layer 3 kick in
    "llama4-long": Spec("llama4", ff=48, moe={"E": 8, "k": 1, "ffe": 48, "ffs": 48}, n_prompt=8250, n_gen=16,
                        keys={"interleave_moe_layer_step": 1, "rope.freq_base_swa": 20000.0, "context_length": 16384}),
    "mistral3": Spec("mistral3", keys={"attention.temperature_scale": 0.5, "attention.temperature_length": 8}),
    "mistral3-yarn": Spec("mistral3", keys={"attention.temperature_scale": 0.3, "attention.temperature_length": 16,
                                            "rope.scaling.type": "yarn", "rope.scaling.factor": 4.0,
                                            "rope.scaling.original_context_length": 32}),
})


def compare_family(name, spec, n_prompt=None, n_gen=None):
    n_prompt, n_gen = n_prompt or spec.o.get("n_prompt", 40), n_gen or spec.o.get("n_gen", 24)
    path = TMP / f"{name}.gguf"
    V = (write_rwkv if spec.o.get("rwkv") else write_gemma3n if spec.arch == "gemma3n" else write_model)(spec, path)
    model, tok, meta = tc.load_file(path)
    rng = np.random.default_rng(1)
    prompt = [int(x) for x in rng.integers(0, V - 6, n_prompt)]
    with Server(path, ctx=1024 if n_prompt + n_gen < 900 else (n_prompt + n_gen + 1024) // 1024 * 1024 + 1024):
        r = post("/completion", {"prompt": prompt, "n_predict": n_gen, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                                 "n_probs": 5, "ignore_eos": True, "cache_prompt": False, "return_tokens": True})
    theirs = r.get("tokens") or []
    tprobs = [{t["id"]: t["logprob"] for t in c.get("top_logprobs") or []} for c in r["completion_probabilities"]]
    n_checked = len(tprobs) if len(tprobs) == len(theirs) else 0       # every step must have its probabilities
    st, rec = tc.State(model.n_layer), tc.Recorder()
    lg = model.logits(model.forward(prompt, st, rec))[-1]
    eog = {"<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "</s>"}   # llama.cpp's end-of-text set
    banned = np.array([t in eog or i == tok.eos for i, t in enumerate(tok.tokens)] + [False] * (len(lg) - len(tok.tokens)))
    mine, worst, where = [], 0.0, None
    for k in range(n_gen):
        lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
        if k < len(tprobs) and tprobs[k]:
            for i, v in tprobs[k].items():
                if v > -30 and abs(float(lp[i]) - v) > worst:          # below e^-30 llama.cpp's float probabilities underflow
                    worst, where = abs(float(lp[i]) - v), (k, i, round(float(lp[i]), 4), round(v, 4))
        y = int(np.argmax(np.where(banned, -np.inf, lg)))   # ignore_eos makes llama.cpp skip end-of-text tokens
        mine.append(y)
        lg = model.logits(model.forward([y], st, rec))[-1]
    same = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
    # attribution adds up to the score, at a few positions
    A = {k: np.concatenate(v) for k, v in rec.cols.items()}
    err = 0.0
    for pos in (n_prompt - 1, n_prompt + 5, n_prompt + n_gen - 2):
        tgt = mine[pos - n_prompt + 1] if pos >= n_prompt - 1 else prompt[pos + 1]
        att = tc.attribute(model, A, pos, tgt)
        x = A["resid"][pos][-1].astype(np.float32)
        true = float(model.final(x[None])[0] @ model.U[tgt]) / model.logit_scale
        if model.out_bias is not None:
            true += float(model.out_bias[tgt]) / model.logit_scale
        err = max(err, abs(att["logit"] - true) / max(1.0, abs(true)))
    # ggml's CPU GELU reads a 16-bit lookup table (fast, ~1e-3 off); this tracer uses the exact formula
    tol = spec.o.get("tol") or (5e-3 if model.act == "gelu" else 2e-3)
    if not model.attribution_exact:
        err = 0.0                                      # (Gemma 3n: pushes are approximate by design)
    check(same == n_gen == len(theirs) and worst < tol and err < 2e-2 and n_checked == n_gen,
          f"{name:18s} {same}/{n_gen} tokens identical · max log-prob difference {worst:.1e} over {n_checked} steps · attribution error {err:.1e}"
          + ("" if worst < 2e-3 else f"  (step, token, mine, llama.cpp) = {where}"))


# ---------------------------------------------------------------- tokenizers
TEXTS = ["Hello world", "  two  spaces and\ttab\n\nnew lines\r\n", "What is 17*23? Answer: 391.", "1234567890 3.14159",
         "don't I'll they're WE'VE", "def f(x):\n    return x**2  # square", "नमस्ते दुनिया", "日本語のテキスト",
         "emoji 😀👍🏽 café", "naïve façade — “quotes” …", "   leading and trailing   ", "\n\n\n", "á è", "x" * 30,
         "CamelCaseWord HTTPServer iPhone I'M DON'T", "Price: $1,234,567.89 (+5%) x+y=z <a href='/x'>", "path/to/file.py\n\n  ",
         "東京タワーは高い。中文：你好！", "ÀÉÎõü ΑΒΓ αβγ Привет мир", "a\tb\r\nc  \n d", "!!!???...,,,;;;"]


def write_tok_model(path, meta_tok):
    """A 1-layer llama model whose only purpose is to carry a tokenizer into llama.cpp."""
    a, D, V = "llama", 32, len(meta_tok["tokens"])
    w = gguf.GGUFWriter(str(path), a)
    for k, v in (("block_count", 1), ("context_length", 512), ("embedding_length", D), ("feed_forward_length", 32),
                 ("attention.head_count", 2), ("attention.head_count_kv", 2)):
        w.add_uint32(f"{a}.{k}", v)
    w.add_float32(f"{a}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_tokenizer_model(meta_tok["model"])
    if meta_tok.get("pre"):
        w.add_tokenizer_pre(meta_tok["pre"])
    w.add_token_list(meta_tok["tokens"])
    w.add_token_types(meta_tok["types"])
    if meta_tok.get("scores") is not None:
        w.add_token_scores(meta_tok["scores"])
    if meta_tok.get("merges"):
        w.add_token_merges(meta_tok["merges"])
    for k in ("bos", "eos", "unk"):
        if meta_tok.get(k) is not None:
            getattr(w, f"add_{k}_token_id")(meta_tok[k])
    w.add_add_bos_token(False)
    z = lambda *s: np.zeros(s, np.float32) + 0.01
    w.add_tensor("token_embd.weight", z(V, D))
    w.add_tensor("output_norm.weight", z(D))
    for nm, s in (("attn_norm.weight", (D,)), ("attn_q.weight", (D, D)), ("attn_k.weight", (D, D)), ("attn_v.weight", (D, D)),
                  ("attn_output.weight", (D, D)), ("ffn_norm.weight", (D,)), ("ffn_gate.weight", (32, D)),
                  ("ffn_up.weight", (32, D)), ("ffn_down.weight", (D, 32))):
        w.add_tensor("blk.0." + nm, z(*s))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def compare_tokenizer(label, meta_tok, texts=TEXTS):
    path = TMP / f"tok_{label}.gguf"
    write_tok_model(path, meta_tok)
    meta = {"tokenizer.ggml.model": meta_tok["model"], "tokenizer.ggml.pre": meta_tok.get("pre"),
            "tokenizer.ggml.tokens": meta_tok["tokens"], "tokenizer.ggml.token_type": meta_tok["types"],
            "tokenizer.ggml.scores": meta_tok.get("scores"), "tokenizer.ggml.merges": meta_tok.get("merges"),
            "tokenizer.ggml.eos_token_id": meta_tok.get("eos"), "tokenizer.ggml.unknown_token_id": meta_tok.get("unk")}
    tok = tc.make_tokenizer(meta)
    bad = []
    with Server(path):
        for s in texts:
            theirs = [t["id"] if isinstance(t, dict) else t for t in post("/tokenize", {"content": s, "add_special": False,
                                                                                        "parse_special": True})["tokens"]]
            mine = tok.encode(s)
            if mine != theirs:
                bad.append((s, mine[:12], theirs[:12]))
    check(not bad, f"tokenizer {label:18s} matches llama.cpp on {len(texts) - len(bad)}/{len(texts)} texts" + (f"  first mismatch: {bad[0]}" if bad else ""))


def probe_vocab(texts=TEXTS, n_merges=3000):
    """A byte-level BPE vocabulary trained on the test texts themselves, WITHOUT any pre-splitting: its merges join
    characters across every place a split rule could cut (digits, contractions, case changes, spaces, punctuation).
    So if two split rules cut a text differently, the token ids differ. (A real vocabulary like Qwen's has no
    multi-digit tokens, so it can't tell digit-grouping rules apart.)"""
    enc = {b: c for c, b in ic._BYTE_DEC.items()}
    seqs = [[enc[b] for b in t.encode("utf-8")] for t in texts * 3]
    toks = [enc[b] for b in range(256)]
    have, merges = set(toks), []
    for _ in range(n_merges):
        counts = {}
        for s in seqs:
            for a, b in zip(s, s[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        (a, b), c = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append(f"{a} {b}")
        if a + b not in have:
            have.add(a + b)
            toks.append(a + b)
        for s in seqs:
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    s[i:i + 2] = [a + b]
                i += 1
    toks += ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    return {"model": "gpt2", "tokens": toks, "types": [1] * (len(toks) - 3) + [3, 3, 3], "merges": merges, "eos": len(toks) - 1}


def rwkv_vocab():
    """An RWKV-style vocabulary: every byte (escaped the way RWKV GGUFs store them) plus multi-byte pieces."""
    def esc(b):
        s = ""
        for x in b:
            ch = chr(x)
            s += {9: "\\t", 10: "\\n", 13: "\\r", 92: "\\\\"}.get(x, ch if 32 <= x < 127 else f"\\x{x:02x}")
        return s
    pieces = [bytes([b]) for b in range(256)]
    for t in TEXTS:
        raw = t.encode("utf-8")
        for n in (2, 3, 5):
            pieces += [raw[i:i + n] for i in range(0, len(raw) - n, 3)]
    seen, toks = set(), ["<|endoftext|>"]
    for p in pieces:
        if p not in seen:
            seen.add(p)
            toks.append(esc(p))
    return {"model": "rwkv", "tokens": toks, "types": [3] + [1] * (len(toks) - 1), "eos": 0, "unk": 0}


def spm_vocab():
    rng = np.random.default_rng(3)
    toks, types, scores = ["<unk>", "<s>", "</s>"], [2, 3, 3], [0.0, 0.0, 0.0]
    for b in range(256):
        toks.append(f"<0x{b:02X}>"); types.append(6); scores.append(0.0)
    words = ["▁", "▁the", "▁a", "th", "he", "in", "er", "an", "▁w", "or", "ld", "▁wor", "▁world", "ll", "▁hello", "el", "lo",
             "▁t", "▁s", "es", "is", "on", "▁is", "▁17", "1", "2", "3", "7", "▁x", "x", "xx", "xxxx", "re", "▁re", "▁de",
             "f", "(", ")", ":", "▁▁", "▁▁▁▁", "y", "▁an", "sw", "▁Answer", "W", "h", "a", "t", "▁W", "?", "."]
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,'!-é—":
        words.append(ch)
    seen = set(toks)
    for wd in words:
        if wd not in seen:
            seen.add(wd)
            toks.append(wd); types.append(1); scores.append(float(-rng.random() * 10 - len(wd) * 0.1 * rng.random()))
    toks.append("<|im_end|>"); types.append(3); scores.append(0.0)
    return {"model": "llama", "tokens": toks, "types": types, "scores": scores, "bos": 1, "eos": 2, "unk": 0}


def check_mxfp4_decode():
    """MXFP4 (gpt-oss experts), decoded straight from the format's definition, must equal what the tracer reads.
    Block of 32 values: 1 byte E8M0 exponent e, 16 bytes of 4-bit codes (low nibbles = values 0-15, high = 16-31);
    value = E2M1(code) * 2^(e-127), E2M1 = 0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives."""
    import gguf_core as core
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((16, 256)) * rng.choice([1e-3, 1.0, 40.0], (16, 1))).astype(np.float32)
    blocks = gguf.quants.quantize(x, gguf.GGMLQuantizationType.MXFP4).reshape(-1, 17)
    e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    codes = np.concatenate([blocks[:, 1:] & 0x0F, blocks[:, 1:] >> 4], 1)
    ref = e2m1[codes] * np.ldexp(np.float32(1), blocks[:, :1].astype(np.int32) - 127)
    got = core._decode(blocks.reshape(16, -1), gguf.GGMLQuantizationType.MXFP4).reshape(-1, 32)
    check(np.array_equal(ref, got), f"MXFP4 decoding = the format's definition, value for value ({ref.size} values)")


def check_lowmem(names=("llama", "qwen3moe", "deepseek3", "gpt-oss-mxfp4", "mamba2", "lfm3", "qwen2moe")):
    """Low-memory mode (weights unpacked on demand through a size-limited cache) must give exactly the same numbers
    as loading everything. A 64 KB cache forces tensors and experts to be dropped and unpacked again constantly."""
    for name in names:
        spec = FAMILIES[name]
        path = TMP / f"lm_{name}.gguf"
        V = write_model(spec, path)
        prompt = [int(x) for x in np.random.default_rng(2).integers(0, V - 6, 30)]
        outs = []
        for budget in (None, 64 * 1024):
            m, tok, meta = tc.load_file(path, lowmem_budget=budget)
            st, rec = tc.State(m.n_layer), tc.Recorder()
            lg = [m.logits(m.forward(prompt, st, rec))[-1]]
            for y in (5, 17, 33):
                lg.append(m.logits(m.forward([y], st, rec))[-1])
            A = {k: np.concatenate(v) for k, v in rec.cols.items()}
            att = tc.attribute(m, A, len(prompt) - 1, int(np.argmax(lg[0])))["logit"]
            outs.append((np.stack(lg), att, getattr(m.T, "unpacked", 0)))
        diff = float(np.abs(outs[0][0] - outs[1][0]).max())
        check(diff < 1e-5 and abs(outs[0][1] - outs[1][1]) < 1e-4,
              f"low-memory mode = full load for {name:14s} (max logit difference {diff:.1e}; {outs[1][2]:,} numbers unpacked with a 64 KB cache)")


def check_lowmem_edits():
    """Grid edits (flat index → value, as the workspace stores them) applied while unpacking, in a 2D matrix and
    inside one expert of a 3D tensor, must equal the edited full tensor."""
    import gguf_core as core
    from gguf import GGUFReader
    path = TMP / "lm_edits.gguf"
    write_model(FAMILIES["qwen3moe"], path)
    r = GGUFReader(str(path))
    ot = {t.name: t for t in r.tensors}
    shapes = {n: tc._natural_shape(t) for n, t in ot.items()}
    rng = np.random.default_rng(9)
    edits = {}
    for name in ("blk.1.attn_q.weight", "blk.2.ffn_down_exps.weight"):
        n = int(np.prod(shapes[name]))
        edits[name] = {int(i): float(v) for i, v in zip(rng.choice(n, 40, replace=False), rng.standard_normal(40) * 9)}
    lw = tc.LazyWeights(ot, list(ot), shapes, pending=edits, budget=32 * 1024)
    ok_all = True
    for name, ed in edits.items():
        full = core._apply_flat(np.array(core.full_tensor(ot[name]), dtype=np.float32), ed)
        got = np.stack([lw[name][e] for e in range(shapes[name][0])]) if len(shapes[name]) == 3 else lw[name]
        ok_all &= np.array_equal(full, got)
    check(ok_all, "grid edits are applied exactly while unpacking (a 2D matrix and single experts of a 3D tensor)")


def main():
    want = set(sys.argv[1:])
    if not want or "lowmem" in want:
        check_lowmem_edits()
    if not want or "mxfp4" in want or "gpt-oss" in want:
        check_mxfp4_decode()
    if not want or "lowmem" in want:
        check_lowmem()
    print(f"llama.cpp: {ic.find_llama_server().parent.name}\n")
    for name, spec in {**FAMILIES, **{k: v for k, v in FAMILIES_WIP.items() if k in want}}.items():
        if want and name not in want and spec.arch not in want:
            continue
        if spec.arch not in tc.ARCHS:
            tc.ARCHS[spec.arch] = getattr(tc, {"rwkv7": "Rwkv7"}.get(spec.arch, ""), None) or tc.Model
        try:
            compare_family(name, spec)
        except Exception as e:
            check(False, f"{name:18s} {type(e).__name__}: {str(e)[:300]}")
    if not want or "tokenizers" in want or "candidates" in want:
        try:
            compare_tokenizer("spm (synthetic)", spm_vocab())
        except Exception as e:
            check(False, f"spm tokenizer: {e}")
        try:
            compare_tokenizer("rwkv (synthetic)", rwkv_vocab())
        except Exception as e:
            check(False, f"rwkv tokenizer: {e}")
        import gguf_core as core
        try:
            meta = core.load_user_metadata(core.workspace_dir("Qwen3.5-0.8B-Q8_0"))
            base = {"model": "gpt2", "tokens": meta["tokenizer.ggml.tokens"], "types": meta["tokenizer.ggml.token_type"],
                    "merges": meta["tokenizer.ggml.merges"], "eos": meta["tokenizer.ggml.eos_token_id"]}
            names = sorted(tc.PRE_RULES) if "candidates" in want else sorted(tc.PRE_VERIFIED)
            saved = set(tc.PRE_VERIFIED)
            tc.PRE_VERIFIED.update(names)                # let the tokenizer build candidate rules for the check
            probe = probe_vocab()
            try:
                for pre in names:
                    compare_tokenizer(f"bpe pre={pre}", {**base, "pre": pre})
                    compare_tokenizer(f"probe pre={pre}", {**probe, "pre": pre})
            finally:
                tc.PRE_VERIFIED.clear()
                tc.PRE_VERIFIED.update(saved)
        except FileNotFoundError:
            print("skip: BPE pre-tokenizer checks need the Qwen3.5 workspace for a real vocabulary")
    print(f"\n{ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
