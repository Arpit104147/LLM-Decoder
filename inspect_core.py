"""
Layer inspection for LLM Decoder: no AI, only fixed rules and linear algebra on the weights,
plus (for the logit lens) LM Studio's own llama.cpp.

  anatomy(ws)                     what every tensor does (hard-coded by llama.cpp tensor names), layer kinds
  search_tokens(ws, text)         find tokens in the vocabulary
  nearest_tokens(ws, token_id)    tokens whose embedding vectors are most similar (cosine)
  neuron(ws, tensor, index)       which tokens a row/column of a weight matrix pushes toward or reacts to
  logit_lens(ws, prompt, layers)  next-token prediction after each chosen layer, computed by llama.cpp

Honest limits (shown in the app too): a single weight has no "topic"; knowledge is spread across many
weights and each weight takes part in many features. These tools show what parts of the model *do*.
"""
import gc
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, GGUFValueType as VT

import gguf_core as core

# ============================================================== tensor roles (hard-coded, llama.cpp naming)

# (pattern on the part after "blk.N.", group, short role, longer explanation, how it touches the residual stream)
ROLES = [
    (r"attn_norm\.(weight|bias)", "attention", "Pre-mixer norm",
     "Scales each of the hidden dimensions before the layer's attention / convolution block reads them.", "norm"),
    (r"attn_q\.(weight|bias)", "attention", "Query projection",
     "Turns each token's hidden vector into a query: 'what am I looking for in earlier tokens?'", "reads"),
    (r"attn_k\.(weight|bias)", "attention", "Key projection",
     "Turns each token into a key: 'what do I contain?' Queries are matched against keys.", "reads"),
    (r"attn_v\.(weight|bias)", "attention", "Value projection",
     "The information a token passes on when another token attends to it.", "reads"),
    (r"attn_qkv\.(weight|bias)", "attention", "Fused query/key/value projection",
     "Query, key and value projections stored as one matrix.", "reads"),
    (r"attn_output\.(weight|bias)", "attention", "Attention output",
     "Mixes the attention heads' results and ADDS them to the residual stream. Each column = one direction it can write.", "writes"),
    (r"attn_q_norm\.weight", "attention", "Query norm", "Normalizes each head's query vector (stabilizes attention).", "norm"),
    (r"attn_k_norm\.weight", "attention", "Key norm", "Normalizes each head's key vector.", "norm"),
    (r"attn_gate\.weight", "attention", "Head output gate",
     "Computes a per-head gate from the input that scales how much each attention head contributes.", "reads"),
    (r"attn_sinks\.weight", "attention", "Attention sinks", "Learned 'no-op' attention targets per head.", "other"),
    (r"attn_(q_a|q_b|kv_a_mqa|kv_b|k_b|v_b)(_norm)?\.weight", "attention", "Low-rank attention (MLA)",
     "Compressed query / key-value projections (DeepSeek-style multi-head latent attention).", "reads"),
    (r"attn_rel_b\.weight|attn_rot_embd", "attention", "Position bias", "Relative position information for attention.", "other"),
    (r"post_attention_norm\.weight", "attention", "Post-mixer norm",
     "Normalizes the attention / convolution output before it is added to the residual stream.", "norm"),
    (r"attn_output_norm\.(weight|bias)", "attention", "Post-attention norm (BERT)",
     "Layer norm applied after adding the attention output (post-norm models).", "norm"),
    (r"shortconv\.in_proj\.weight", "convolution", "Conv input projection",
     "Projects the hidden vector into the gate/input signals of the short convolution (LFM).", "reads"),
    (r"shortconv\.conv\.weight", "convolution", "Short convolution kernel",
     "A small causal filter over the last few tokens (one per channel): local mixing instead of attention.", "other"),
    (r"shortconv\.out_proj\.weight", "convolution", "Conv output projection",
     "ADDS the convolution's result to the residual stream. Each column = one direction it can write.", "writes"),
    (r"ssm_(in|x|dt|a|d|conv1d|norm|out|b|c|dt_norm|b_norm|c_norm)\.(weight|bias)", "state-space", "State-space (Mamba) part",
     "Part of a selective state-space block: a recurrent alternative to attention.", "other"),
    (r"time_mix_.*|channel_mix_.*", "rwkv", "RWKV mixing", "Part of an RWKV time-mix / channel-mix block.", "other"),
    (r"ffn_norm\.(weight|bias)", "mlp", "Pre-MLP norm", "Scales each hidden dimension before the MLP reads them.", "norm"),
    (r"ffn_gate\.(weight|bias)", "mlp", "MLP gate",
     "Each row is one neuron's 'detector': it reads the hidden vector and decides how strongly the neuron fires (through SiLU/GELU).", "reads"),
    (r"ffn_up\.(weight|bias)", "mlp", "MLP up projection",
     "Each row is one neuron's input weights; multiplied with the gate to give the neuron's activation.", "reads"),
    (r"ffn_down\.(weight|bias)", "mlp", "MLP down projection",
     "Each column is what one neuron WRITES into the residual stream when it fires. This is where much factual recall is thought to live.", "writes"),
    (r"post_ffw_norm\.weight", "mlp", "Post-MLP norm", "Normalizes the MLP output before it is added to the residual stream.", "norm"),
    (r"layer_output_norm\.(weight|bias)", "mlp", "Post-MLP norm (BERT)", "Layer norm after adding the MLP output (post-norm models).", "norm"),
    (r"ffn_gate_inp\.weight", "moe", "MoE router", "Scores every expert for each token; the top experts are used.", "reads"),
    (r"exp_probs_b\.bias", "moe", "Router bias", "Bias added to the router's expert scores.", "other"),
    (r"ffn_(gate|up)_exps\.weight", "moe", "Experts' input weights", "Gate/up matrices of all experts (3D: expert x neuron x hidden).", "reads"),
    (r"ffn_down_exps\.weight", "moe", "Experts' output weights", "What each expert's neurons write into the residual stream.", "writes"),
    (r"ffn_(gate|up)_shexp\.weight", "moe", "Shared expert input", "Input weights of the always-on shared expert.", "reads"),
    (r"ffn_down_shexp\.weight", "moe", "Shared expert output", "What the shared expert writes into the residual stream.", "writes"),
    (r"layer_out_norm\.weight|layer_output_scale|.*_scale", "other", "Scale", "A learned scaling factor.", "norm"),
]
GLOBAL_ROLES = {
    "token_embd.weight": ("embedding", "Token embedding table",
                          "Row t is the starting vector for token t: the model's 'dictionary' of what each token means.", "embed"),
    "output.weight": ("output", "Unembedding (LM head)",
                      "Row t scores token t as the next token: final hidden vector · row t = that token's logit.", "unembed"),
    "output_norm.weight": ("output", "Final norm", "Scales the final hidden vector before the LM head.", "norm"),
    "output_norm.bias": ("output", "Final norm bias", "Bias of the final layer norm.", "norm"),
    "rope_freqs.weight": ("position", "RoPE frequency factors", "Rescales rotary position frequencies (long-context support).", "other"),
    "token_embd_norm.weight": ("embedding", "Embedding norm", "Layer norm applied right after the embedding lookup (BERT-style).", "norm"),
    "token_embd_norm.bias": ("embedding", "Embedding norm bias", "Bias of the embedding layer norm.", "norm"),
    "token_types.weight": ("embedding", "Segment embeddings", "Added to tokens depending on which segment (sentence A/B) they are in.", "other"),
    "position_embd.weight": ("position", "Learned position embeddings", "Row p is added to the token at position p.", "other"),
    "cls.weight": ("output", "Classifier head", "Maps the pooled vector to class scores.", "other"),
    "cls.output.weight": ("output", "Classifier output", "Final classifier projection.", "other"),
}
LAYER_RX = re.compile(r"^((?:[a-z]+\.)?)blk\.(\d+)\.(.+)$")


def describe_tensor(name):
    m = LAYER_RX.match(name)
    if name in GLOBAL_ROLES:
        g, short, long, kind = GLOBAL_ROLES[name]
        return {"group": g, "role": short, "explain": long, "kind": kind, "layer": None}
    if m:
        prefix, layer, rest = m.group(1), int(m.group(2)), m.group(3)
        tower = {"v.": "vision encoder ", "a.": "audio encoder ", "enc.": "encoder ", "dec.": "decoder "}.get(prefix, "")
        for rx, g, short, long, kind in ROLES:
            if re.fullmatch(rx, rest):
                return {"group": g, "role": (tower + short) if tower else short, "explain": long, "kind": kind, "layer": layer}
        if tower:
            return {"group": "encoder", "role": tower.strip().capitalize() + " part", "kind": "other", "layer": layer,
                    "explain": f"Part of layer {layer} of the {tower.strip()} (a transformer that encodes images/audio)."}
        return {"group": "other", "role": "Architecture-specific", "kind": "other", "layer": layer,
                "explain": "A tensor this tool has no fixed description for (specific to this architecture)."}
    if name.startswith(("mm.", "v.", "a.")):
        return {"group": "projector" if name.startswith("mm.") else "encoder", "layer": None, "kind": "other",
                "role": "Multimodal projector" if name.startswith("mm.") else "Encoder input/output",
                "explain": "Maps the vision/audio encoder's features into the language model's embedding space."
                if name.startswith("mm.") else "Patch/position embedding or final norm of the vision/audio encoder."}
    return {"group": "other", "role": "Architecture-specific", "kind": "other", "layer": None,
            "explain": "A tensor this tool has no fixed description for."}


def _meta(d):
    return core.load_user_metadata(d)


def anatomy(ws):
    d = core.workspace_dir(ws)
    man = core.load_manifest(d)
    meta = _meta(d)
    arch = meta.get("general.architecture", "")
    swa = meta.get(f"{arch}.attention.sliding_window_pattern")
    kvh = meta.get(f"{arch}.attention.head_count_kv")
    layers, glob = {}, []
    for e in man:
        info = describe_tensor(e["name"])
        item = {"name": e["name"], "type": e["type"], "shape": e["numpy_shape"], "params": e["n_elements"],
                "added": bool(e.get("added")), **info}
        m = LAYER_RX.match(e["name"])
        if m:
            key = (m.group(1), int(m.group(2)))
            layers.setdefault(key, []).append(item)
        else:
            glob.append(item)
    out = []
    for (prefix, i), ts in sorted(layers.items()):
        rests = {LAYER_RX.match(t["name"]).group(3) for t in ts}
        kinds = []
        if any(r.startswith(("attn_q.", "attn_qkv.", "attn_q_a.")) for r in rests):
            kinds.append("attention")
        if any(r.startswith("shortconv.") for r in rests):
            kinds.append("convolution")
        if any(r.startswith("ssm_") for r in rests):
            kinds.append("state-space")
        if any(r.startswith("time_mix") for r in rests):
            kinds.append("rwkv")
        if "attention" in kinds and "state-space" in kinds and not any(r.startswith("attn_output.") for r in rests):
            # q/k/v read into a recurrent state, written out by ssm_out (Qwen3.5 / Qwen3-Next Gated DeltaNet): one
            # linear-attention mixer, not two parts (Falcon-H1 has both side by side, with attn_output)
            kinds = ["linear attention" if k == "attention" else k for k in kinds if k != "state-space"]
        if any("_exps." in r for r in rests):
            kinds.append("MoE MLP")
        elif any(r.startswith(("ffn_up.", "ffn_gate.")) for r in rests):
            kinds.append("MLP")
        notes = []
        if not prefix and isinstance(swa, list) and i < len(swa) and "attention" in kinds:
            notes.append("sliding-window" if swa[i] else "full attention")
        if not prefix and isinstance(kvh, list) and i < len(kvh) and kvh[i]:
            notes.append(f"{kvh[i]} KV heads")
        out.append({"tower": prefix.rstrip(".") or "text", "index": i, "kinds": kinds, "notes": notes,
                    "params": sum(t["params"] for t in ts), "tensors": ts, "added": all(t["added"] for t in ts)})
    by_group = {}
    for t in glob + [t for L in out for t in L["tensors"]]:
        by_group[t["group"]] = by_group.get(t["group"], 0) + t["params"]
    tied = not any(e["name"] == "output.weight" for e in man) and any(e["name"] == "token_embd.weight" for e in man)
    return {"architecture": arch, "layers": out, "globals": glob, "params_by_group": by_group,
            "total": sum(e["n_elements"] for e in man), "tied_embeddings": tied}


# ============================================================== tokens

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DEC = _bytes_to_unicode()
_VOCAB = {}


def token_text(tok, model):
    """Readable text of a vocabulary entry (GPT-2 byte-level, SentencePiece and WordPiece conventions)."""
    if model == "gpt2":
        try:
            return bytes(_BYTE_DEC[ch] for ch in tok).decode("utf-8", errors="replace")
        except KeyError:
            return tok
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", tok)
    if m:
        return bytes([int(m.group(1), 16)]).decode("latin-1")
    return tok.replace("▁", " ")


def vocab(ws):
    d = core.workspace_dir(ws)
    p = d / "metadata.json"
    key = (ws, p.stat().st_mtime_ns)
    if key not in _VOCAB:
        meta = _meta(d)
        toks = meta.get("tokenizer.ggml.tokens") or []
        model = meta.get("tokenizer.ggml.model")
        _VOCAB.clear()
        _VOCAB[key] = {"tokens": toks, "texts": [token_text(t, model) for t in toks], "model": model,
                       "types": meta.get("tokenizer.ggml.token_type")}
    return _VOCAB[key]


def search_tokens(ws, text, limit=60):
    v = vocab(ws)
    q = text.lower()
    exact, starts, contains = [], [], []
    for i, (tok, txt) in enumerate(zip(v["tokens"], v["texts"])):
        tl = txt.lower()
        if txt == text or tok == text or txt.strip() == text.strip() and text.strip():
            exact.append(i)
        elif tl.strip().startswith(q.strip()) and q.strip():
            starts.append(i)
        elif q and q in tl:
            contains.append(i)
        if len(exact) + len(starts) + len(contains) > 4000:
            break
    ids = (exact + sorted(starts, key=lambda i: len(v["texts"][i])) + sorted(contains, key=lambda i: len(v["texts"][i])))[:limit]
    return [_tok(v, i) for i in ids]


def _tok(v, i, score=None):
    out = {"id": int(i), "token": v["tokens"][i] if i < len(v["tokens"]) else f"<{i}>",
           "text": v["texts"][i] if i < len(v["texts"]) else f"<id {i}>"}
    if score is not None:
        out["score"] = float(score)
    return out


# ============================================================== matrices (cached as float16 to save memory)

_MATS = {}
_LOCK = threading.Lock()


def _matrix(ws, name, log=None):
    """Whole 2D tensor (current values incl. edits) as float16, cached. Also returns row norms."""
    d = core.workspace_dir(ws)
    man = core.load_manifest(d)
    e = next((x for x in man if x["name"] == name), None)
    if e is None:
        raise ValueError(f"No tensor named {name!r}")
    stamp = (ws, name, e.get("source"), e.get("npy"), (d / e["csv"]).stat().st_mtime_ns,
             (d / core.edits_rel(e)).stat().st_mtime_ns if (d / core.edits_rel(e)).exists() else 0)
    with _LOCK:
        hit = _MATS.get(name + "|" + ws)
        if hit and hit["stamp"] == stamp:
            return hit
        _, _, ot = core.open_source(d, check=False)
        R, C = core.ui_dims(e)
        need, ram = R * C * 2, core.total_ram()
        if ram and need > 0.4 * ram:
            raise MemoryError(f"{name} has {R * C:,} numbers; this tool would need {need / 1e9:.1f} GB of memory "
                              f"(this computer has {ram / 1e9:.0f} GB). Pick a smaller tensor.")
        if log:
            log(f"Loading {name} ({R:,} x {C:,}) into memory…")
        M = np.empty((R, C), dtype=np.float16)
        step = max(1, 2_000_000 // max(C, 1))
        for r0 in range(0, R, step):
            M[r0:r0 + step] = core.base_tile(d, ot, e, r0, step, 0, C).astype(np.float16)
        # apply the user's edits so the tools see the model you are building
        cur_csv = core.read_csv(d / e["csv"])
        wr, wc = e["window"]
        base = M[:wr, :wc].astype(np.float64)
        mask = ~np.isclose(cur_csv, base, rtol=core.CHANGE_RTOL, atol=core.CHANGE_ATOL, equal_nan=True)
        M[:wr, :wc][mask] = cur_csv[mask].astype(np.float16)
        for (r, c), val in core.load_sparse(d, e).items():
            M[r, c] = val
        norms = np.sqrt((M.astype(np.float32) ** 2).sum(1)) if R * C < 400_000_000 else \
            np.concatenate([np.sqrt((M[i:i + 8192].astype(np.float32) ** 2).sum(1)) for i in range(0, R, 8192)])
        while len(_MATS) >= 2:
            _MATS.pop(next(iter(_MATS)))
        hit = _MATS[name + "|" + ws] = {"stamp": stamp, "M": M, "norms": norms, "entry": e}
        return hit


def _vector(ws, name):
    d = core.workspace_dir(ws)
    e = next((x for x in core.load_manifest(d) if x["name"] == name), None)
    if e is None:
        return None
    _, _, ot = core.open_source(d, check=False)
    return core.current_full(d, ot, e).astype(np.float32).reshape(-1)


def _matvec(M, v):
    out = np.empty(M.shape[0], dtype=np.float32)
    for i in range(0, M.shape[0], 16384):
        out[i:i + 16384] = M[i:i + 16384].astype(np.float32) @ v
    return out


def _top(scores, k, v, largest=True, skip=()):
    s = np.where(np.isfinite(scores), scores, -np.inf if largest else np.inf)
    idx = np.argpartition(-s if largest else s, min(k + len(skip), len(s) - 1))[: k + len(skip)]
    idx = idx[np.argsort(-s[idx] if largest else s[idx])]
    return [_tok(v, int(i), s[i]) for i in idx if int(i) not in skip][:k]


def unembedding_name(ws):
    d = core.workspace_dir(ws)
    names = {e["name"] for e in core.load_manifest(d)}
    return "output.weight" if "output.weight" in names else "token_embd.weight"


def nearest_tokens(ws, token_id, k=24, log=None):
    """Tokens whose embedding rows point in the most similar direction (cosine similarity)."""
    v = vocab(ws)
    E = _matrix(ws, "token_embd.weight", log)
    M, norms = E["M"], E["norms"]
    if not 0 <= token_id < M.shape[0]:
        raise ValueError(f"Token id must be between 0 and {M.shape[0] - 1}")
    q = M[token_id].astype(np.float32)
    sims = _matvec(M, q / (np.linalg.norm(q) or 1)) / np.where(norms > 0, norms, 1)
    res = {"token": _tok(v, token_id), "norm": float(norms[token_id]), "nearest": _top(sims, k, v, skip={token_id}),
           "farthest": _top(sims, 8, v, largest=False)}
    U = unembedding_name(ws)
    if U != "token_embd.weight":                          # separate LM head: what this token's output row resembles
        Uh = _matrix(ws, U, log)
        u = Uh["M"][token_id].astype(np.float32)
        s2 = _matvec(Uh["M"], u / (np.linalg.norm(u) or 1)) / np.where(Uh["norms"] > 0, Uh["norms"], 1)
        res["nearest_output"] = _top(s2, 12, v, skip={token_id})
    return res


def neuron(ws, name, index, axis=None, k=20, log=None):
    """What one row/column of a weight matrix does, measured against the vocabulary.
    'writes' matrices (rows = hidden size): column `index` is a direction added to the residual stream;
       its logits through the LM head show which next tokens it promotes / suppresses.
    'reads' matrices (cols = hidden size): row `index` is a detector; its similarity with token embeddings
       shows which tokens' meanings excite it most (ignores the layers before it, so it is a rough guide)."""
    d = core.workspace_dir(ws)
    meta = _meta(d)
    arch = meta.get("general.architecture", "")
    n_embd = meta.get(f"{arch}.embedding_length")
    man = {e["name"]: e for e in core.load_manifest(d)}
    if name not in man:
        raise ValueError(f"No tensor named {name!r}")
    e = man[name]
    info = describe_tensor(name)
    if len(e["numpy_shape"]) != 2:
        raise ValueError("Pick a 2D weight matrix (or use the token explorer for the embedding table)")
    R, C = e["numpy_shape"]
    if axis is None:
        axis = "col" if (R == n_embd and info["kind"] == "writes") else "row" if C == n_embd else ("col" if R == n_embd else None)
    if axis is None:
        raise ValueError(f"{name} is {R} x {C}; neither side matches the hidden size {n_embd}, so it can't be "
                         f"compared with the vocabulary directly")
    v = vocab(ws)
    W = _matrix(ws, name, log)["M"]
    if axis == "col":
        if not 0 <= index < C:
            raise ValueError(f"Column must be between 0 and {C - 1}")
        vec = W[:, index].astype(np.float32)
        g = _vector(ws, "output_norm.weight")
        if g is not None and g.shape == vec.shape:
            vec = vec * g
        U = _matrix(ws, unembedding_name(ws), log)
        logits = _matvec(U["M"], vec)
        mode = "writes"
        res = {"promotes": _top(logits, k, v), "suppresses": _top(logits, k, v, largest=False),
               "explain": f"Column {index} of {name} is a direction this component adds to the hidden state. "
                          f"Multiplying it by the LM head ({unembedding_name(ws)}) gives how much it raises or lowers "
                          f"each next-token score. (Direct effect only; later layers can change it.)"}
    else:
        if not 0 <= index < R:
            raise ValueError(f"Row must be between 0 and {R - 1}")
        vec = W[index].astype(np.float32)
        m = LAYER_RX.match(name)
        if m:
            normname = f"{m.group(1)}blk.{m.group(2)}.{'ffn_norm' if '.ffn_' in name or name.split('.', 2)[-1].startswith('ffn') else 'attn_norm'}.weight"
            g = _vector(ws, normname)
            if g is not None and g.shape == vec.shape:
                vec = vec * g
        E = _matrix(ws, "token_embd.weight", log)
        sims = _matvec(E["M"], vec / (np.linalg.norm(vec) or 1)) / np.where(E["norms"] > 0, E["norms"], 1)
        mode = "reads"
        res = {"promotes": _top(sims, k, v), "suppresses": _top(sims, k, v, largest=False),
               "explain": f"Row {index} of {name} is a detector: the component fires when the hidden state points "
                          f"along it. Listed are the tokens whose embeddings point most along (or against) it. "
                          f"Only exact for layer 0; deeper layers see mixed, contextual vectors."}
    return {"tensor": name, "index": index, "axis": axis, "mode": mode, "role": info["role"],
            "norm": float(np.linalg.norm(vec)), **res}


# ============================================================== logit lens (llama.cpp on truncated copies)

def find_llama_server():
    if os.environ.get("LLAMA_SERVER") and Path(os.environ["LLAMA_SERVER"]).is_file():
        return Path(os.environ["LLAMA_SERVER"])
    root = Path.home() / ".lmstudio" / "extensions" / "backends"
    cands = []
    for exe in root.glob("*/llama-server.exe"):
        m = re.search(r"(\d+)\.(\d+)\.(\d+)$", exe.parent.name)
        ver = tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
        cands.append((ver, "cuda" not in exe.parent.name, exe))   # newest first; prefer non-CUDA builds
    if not cands:
        found = shutil.which("llama-server")
        if found:
            return Path(found)
        raise FileNotFoundError("llama.cpp's llama-server wasn't found. Install LM Studio (it ships one) or set LLAMA_SERVER.")
    return sorted(cands, reverse=True)[0][2]


def _make_room(lens_dir, need, cap, log):
    """Keep the cache of cut-down copies at most `cap` bytes (one model's size) and leave room for a new
    copy of `need` bytes: delete the least recently used copies first."""
    if not lens_dir.is_dir():
        return
    copies = []
    for p in lens_dir.glob("layers_*.gguf"):
        s = p.with_suffix(".stamp")
        copies.append((s.stat().st_mtime if s.exists() else 0, p, s))
    copies.sort()
    total = sum(p.stat().st_size for _, p, _ in copies)
    free = shutil.disk_usage(lens_dir).free
    margin = 512 * 1024 ** 2
    while copies and (total + need > max(cap, need) or free < need + margin):
        _, p, s = copies.pop(0)
        size = p.stat().st_size
        gc.collect()
        try:
            p.unlink()
            s.unlink(missing_ok=True)
        except OSError:
            continue
        total -= size
        free += size
        log(f"Freed {size / 1e9:.2f} GB: removed the cached copy {p.name}")


def _truncated(ws, L, log):
    """A copy of the original model that stops after layer L-1 (cached in lens/)."""
    d = core.workspace_dir(ws)
    info, reader, ot = core.open_source(d)
    out = d / "lens" / f"layers_{L}.gguf"
    stamp = d / "lens" / f"layers_{L}.stamp"
    want = f"{info['file_size']}:{info['file_mtime_ns']}"
    if out.is_file() and stamp.is_file() and stamp.read_text() == want:
        os.utime(stamp)                                    # mark as recently used
        return out
    out.parent.mkdir(exist_ok=True)
    meta = core.original_metadata(reader)
    arch = meta["general.architecture"]
    n = meta[f"{arch}.block_count"]
    keep = [t for t in reader.tensors if not ((m := LAYER_RX.match(t.name)) and not m.group(1) and int(m.group(2)) >= L)]
    need = sum(int(t.n_bytes) for t in keep) + 64 * 1024 ** 2
    _make_room(d / "lens", need, info["file_size"], log)
    core.check_disk_space(d / "lens", need, f"the logit-lens copy cut after layer {L - 1}",
                          "Pick fewer or earlier layers, or free up space.")
    log(f"Building a copy that stops after layer {L - 1} ({L} of {n} layers)…")
    w = GGUFWriter(str(out) + ".partial", arch)
    for key, field in reader.fields.items():
        if key in core.PSEUDO_KEYS or key == "general.architecture":
            continue
        if key == "general.alignment":
            w.add_custom_alignment(int(core.field_value(field)))
        elif key == f"{arch}.block_count":
            w.add_key_value(key, L, field.types[0])
        elif key.startswith(arch + ".") and field.types[0] == VT.ARRAY and len(field.data) == n:
            vals = core._raw_array(field)[:L]                       # per-layer lists keep only the first L
            w.add_key_value(key, vals, VT.ARRAY, sub_type=field.types[-1])
        else:
            core._copy_kv(w, key, field)
    for t in keep:
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type if t.data.dtype == np.uint8 else None)
    w.write_header_to_file(path=Path(str(out) + ".partial"))
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    gc.collect()
    os.replace(str(out) + ".partial", out)
    stamp.write_text(want)
    return out


def _predict(server, model, prompt, k, log, port=8131):
    # llama.cpp's log goes to a temporary FILE: an unread pipe fills up and freezes the server
    errlog = tempfile.TemporaryFile()
    p = subprocess.Popen([str(server), "-m", str(model), "--port", str(port), "-c", "2048", "-ngl", "0", "--no-warmup",
                          "-np", "1"], stdout=subprocess.DEVNULL, stderr=errlog,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for _ in range(600):
            if p.poll() is not None:
                errlog.seek(0)
                raise RuntimeError("llama.cpp couldn't load the model:\n" + errlog.read().decode(errors="replace")[-1200:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.4)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": k,
                                                      "cache_prompt": False}).encode())
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        first = r["completion_probabilities"][0]
        tops = first.get("top_logprobs") or first.get("probs") or []
        out = []
        for t in tops:
            lp = t.get("logprob")
            pr = float(np.exp(lp)) if lp is not None else float(t.get("prob", 0))
            out.append({"id": t.get("id"), "text": t.get("token", t.get("tok_str", "")), "prob": pr})
        return out
    finally:
        p.kill()
        p.wait()
        errlog.close()


def lens_layers(ws):
    d = core.workspace_dir(ws)
    meta = _meta(d)
    n = meta.get(f"{meta.get('general.architecture')}.block_count")
    return n if isinstance(n, int) else None


def lens_disk(ws, layers):
    d = core.workspace_dir(ws)
    n = lens_layers(ws)
    size = core.load_info(d)["file_size"]
    cached = {int(p.stem.split("_")[1]) for p in (d / "lens").glob("layers_*.gguf")} if (d / "lens").is_dir() else set()
    return {"new_bytes": int(sum(size * L / n for L in layers if L < n and L not in cached)),
            "cached_bytes": int(sum(p.stat().st_size for p in (d / "lens").glob("*.gguf"))) if (d / "lens").is_dir() else 0}


def logit_lens(ws, prompt, layers, k=8, log=print):
    """Next-token top-k after each chosen number of layers. layers: list of ints in 1..n (n = full model)."""
    if not prompt:
        raise ValueError("Type a prompt")
    d = core.workspace_dir(ws)
    info = core.load_info(d)
    n = lens_layers(ws)
    if not n:
        raise ValueError("This model has no block_count, so the logit lens can't split it into layers")
    meta = _meta(d)
    if meta.get("general.architecture") in ("bert", "nomic-bert", "jina-bert-v2", "clip", "t5encoder"):
        raise ValueError("This is an encoder/embedding model; it doesn't predict next tokens")
    layers = sorted({int(L) for L in layers if 1 <= int(L) <= n})
    if not layers:
        raise ValueError(f"Choose layers between 1 and {n}")
    server = find_llama_server()
    log(f"Using {server.parent.name}")
    rows = []
    for L in layers:
        model = Path(info["gguf_path"]) if L == n else _truncated(ws, L, log)
        log(f"Running after layer {L - 1} ({L}/{n})…")
        rows.append({"layers": L, "top": _predict(server, model, prompt, k, log)})
        log(f"  → {rows[-1]['top'][0]['text']!r} ({rows[-1]['top'][0]['prob'] * 100:.1f} %)" if rows[-1]["top"] else "  → (none)")
    core.release_sources()
    return {"prompt": prompt, "n_layers": n, "rows": rows,
            "note": "Uses the ORIGINAL model file (your unsaved/unexported edits are not included)."}


def clear_lens(ws):
    d = core.workspace_dir(ws)
    gc.collect()
    shutil.rmtree(d / "lens", ignore_errors=True)
    return {"ok": True}
