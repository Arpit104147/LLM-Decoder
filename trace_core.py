"""
Chat + tracing for LLM Decoder: runs the model itself in numpy, from the workspace's weights (your grid,
CSV, Excel and edit_weights.py changes included), and records what happens inside it for every token.

What a trace holds, per position t (the model reading token t and predicting token t+1):
  act[t, L, j]       activation of MLP neuron j in layer L   (silu(gate) * up, the number ffn_down multiplies)
  heads[t, L, h, :]  output of attention / linear-attention head h in layer L, before the output projection
  resid[t, L, :]     the hidden state (residual stream) entering layer L (L = n_layer: after the last layer)
  attn[t, L]         for full-attention layers: how much each head looked at every earlier token
  top-k predictions, the probability of the token that really came next, attention/MLP output sizes

From these, `position_detail` works out *direct logit attribution*: how much each neuron, head and layer
pushed the score of a chosen next token up or down. These pushes add up exactly to that token's score
(logit), and each one points at concrete weights: neuron j of layer L = column j of blk.L.ffn_down and
row j of blk.L.ffn_up / ffn_gate.

Supported architectures are registered in ARCHS. Each is checked against llama.cpp (test_trace.py).
"""
import base64
import gc
import json
import math
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path

import numpy as np

import gguf_core as core
import inspect_core as ic

MAX_POSITIONS = 4096          # longest conversation a trace keeps
TOPK = 10

# ============================================================== tokenizer (byte-level BPE, as llama.cpp)

_UCLASS = {}


def _uclass(cat):
    """Regex character-class body with every code point of a Unicode category: 'L' (all letters), 'Lu', 'M', 'N',
    'P', 'S'… what \\p{L} etc. mean in llama.cpp's regexes."""
    if not _UCLASS:
        ranges = {}
        prev, start = None, 0
        for cp in range(0x110001):
            c = unicodedata.category(chr(cp)) if cp < 0x110000 else None
            if c != prev:
                if prev:
                    ranges.setdefault(prev, []).append((start, cp - 1))
                    ranges.setdefault(prev[0], []).append((start, cp - 1))
                prev, start = c, cp
        esc = lambda c: "\\U%08x" % c
        for k, rs in ranges.items():
            _UCLASS[k] = "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in rs)
    return _UCLASS[cat]


def _to_python_regex(rx):
    """llama.cpp regex (std::regex syntax with \\p{..} classes) → Python re: \\p{X} becomes the class body inside
    [...] and [body] outside it; $ outside a class means end of the piece (\\Z)."""
    out, i, in_cls = [], 0, False
    while i < len(rx):
        ch = rx[i]
        if ch == "\\" and rx.startswith("\\p{", i):
            j = rx.index("}", i)
            body = _uclass(rx[i + 3:j])
            out.append(body if in_cls else f"[{body}]")
            i = j + 1
            continue
        if ch == "\\":
            out.append(rx[i:i + 2])
            i += 2
            continue
        if ch == "[" and not in_cls:
            in_cls = True
        elif ch == "]" and in_cls:
            in_cls = False
        elif ch == "$" and not in_cls:
            ch = r"\Z"
        out.append(ch)
        i += 1
    return re.compile("".join(out))


# llama.cpp's split rules (llama-vocab.cpp), by tokenizer.ggml.pre name. A rule is a list of regexes applied in turn:
# each one splits every piece further, and text a regex doesn't match stays a piece of its own.
# Every name listed here is checked against llama.cpp by test_trace_arch.py; a model with another name is refused.
_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
_LLAMA3 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
           r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_QWEN2 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
          r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_QWEN35 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
           r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_GPT4O = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
          r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
          r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_TEKKEN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
           r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
           r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_DEEPSEEK3 = [r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+",
              r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]
PRE_RULES = {
    "gpt-2": [_GPT2], "mpt": [_GPT2], "olmo": [_GPT2], "jais": [_GPT2],
    "llama-bpe": [_LLAMA3], "llama3": [_LLAMA3], "llama-v3": [_LLAMA3], "smaug-bpe": [_LLAMA3], "dbrx": [_LLAMA3],
    "qwen2": [_QWEN2], "deepseek-r1-qwen": [_QWEN2],
    "qwen35": [_QWEN35],
    "gpt-4o": [_GPT4O], "llama4": [_GPT4O],
    "tekken": [_TEKKEN],
    "deepseek-v3": _DEEPSEEK3,
    "default": [r"[\p{P}\$\+<=>\^~\|]+", _GPT2, r"\p{N}+", r"[0-9][0-9][0-9]"],
    "falcon": [r"[\p{P}\$\+<=>\^~\|`]+", _GPT2, r"[0-9][0-9][0-9]"],
    "starcoder": [r"\p{N}", _GPT2], "refact": [r"\p{N}", _GPT2], "command-r": [r"\p{N}", _GPT2],
    "smollm": [r"\p{N}", _GPT2], "codeshell": [r"\p{N}", _GPT2], "exaone": [r"\p{N}", _GPT2],
    "lfm2": [_LLAMA3], "glm4": [_LLAMA3],
}
# all of them proven identical to llama.cpp (test_trace_arch.py: the Qwen vocabulary plus a probe vocabulary whose
# merges cross every split point, which catches a wrong rule)
PRE_VERIFIED = set(PRE_RULES)
_PRE_CACHE = {}


def _pre_rule(pre):
    if pre not in PRE_VERIFIED:
        raise ValueError(f"the tokenizer's splitting rule '{pre}' isn't supported by the tracer yet "
                         f"(supported: {', '.join(sorted(PRE_VERIFIED))})")
    if pre not in _PRE_CACHE:
        _PRE_CACHE[pre] = [_to_python_regex(r) for r in PRE_RULES[pre]]
    return _PRE_CACHE[pre]


def _split(text, rule):
    """llama.cpp's unicode_regex_split: apply each regex in turn to every piece; unmatched text stays a piece."""
    pieces = [text]
    for rx in rule:
        out = []
        for p in pieces:
            at = 0
            for m in rx.finditer(p):
                if m.start() > at:
                    out.append(p[at:m.start()])
                if m.end() > m.start():
                    out.append(m.group())
                at = m.end()
            if at < len(p):
                out.append(p[at:])
        pieces = out
    return pieces


PRE_ALL = PRE_VERIFIED


class Tokenizer:
    def __init__(self, meta):
        model = meta.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError(f"Chat tracing needs a byte-level BPE tokenizer (GGUF type 'gpt2'); this model uses {model!r}.")
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.types = meta.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {}
        for r, m in enumerate(meta.get("tokenizer.ggml.merges") or []):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = r
        special = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4) and t), key=len, reverse=True)
        self.special_rx = re.compile("(" + "|".join(map(re.escape, special)) + ")") if special else None
        self.byte_enc = {b: c for c, b in ic._BYTE_DEC.items()}
        self.pre = _pre_rule(meta.get("tokenizer.ggml.pre"))
        self.cache = {}
        self.eos = meta.get("tokenizer.ggml.eos_token_id")
        self.bos = meta.get("tokenizer.ggml.bos_token_id") if meta.get("tokenizer.ggml.add_bos_token") else None
        self.texts = [self.piece(i) for i in range(len(self.tokens))]

    def _bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best, br = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (br is None or r < br):
                    best, br = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        ids = []
        for p in parts:
            if p in self.ids:
                ids.append(self.ids[p])
            else:                                   # not in the vocabulary: fall back to single bytes
                ids.extend(self.ids[ch] for ch in p if ch in self.ids)
        if len(self.cache) < 200_000:
            self.cache[word] = ids
        return ids

    def encode(self, text, special=True):
        out = []
        chunks = self.special_rx.split(text) if (special and self.special_rx) else [text]
        for k, chunk in enumerate(chunks):
            if not chunk:
                continue
            if special and self.special_rx and k % 2 == 1:
                out.append(self.ids[chunk])
                continue
            for piece in _split(chunk, self.pre):
                word = "".join(self.byte_enc[b] for b in piece.encode("utf-8"))
                out.extend(self._bpe(word))
        return out

    def piece(self, i):
        t = self.tokens[i]
        if self.types[i] in (3, 4, 5):
            return t
        return ic.token_text(t, "gpt2")

    def decode(self, ids):
        bs = bytearray()
        for i in ids:
            t = self.tokens[i]
            if self.types[i] in (3, 4, 5):
                bs += t.encode()
            else:
                bs += bytes(ic._BYTE_DEC.get(ch, 63) for ch in t)
        return bs.decode("utf-8", errors="replace")

    def is_stop(self, i):
        return i == self.eos or self.types[i] == 3 and self.tokens[i] in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "</s>")


class SPMTokenizer(Tokenizer):
    """SentencePiece (GGUF type 'llama'): Llama 2, Mistral 7B, Gemma, Phi-3… As llama.cpp's llm_tokenizer_spm:
    spaces become '▁' (and one is prepended), then the adjacent pair whose merged text is a token with the highest
    score is merged, repeatedly; leftovers that are not tokens become <0xXX> byte tokens."""

    def __init__(self, meta):
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.types = meta.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.scores = meta.get("tokenizer.ggml.scores") or [0.0] * len(self.tokens)
        self.ids = {}
        for i, t in enumerate(self.tokens):
            self.ids.setdefault(t, i)
        special = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4) and t), key=len, reverse=True)
        self.special_rx = re.compile("(" + "|".join(map(re.escape, special)) + ")") if special else None
        self.add_prefix = meta.get("tokenizer.ggml.add_space_prefix", True)
        self.unk = meta.get("tokenizer.ggml.unknown_token_id", 0)
        self.eos = meta.get("tokenizer.ggml.eos_token_id")
        self.bos = meta.get("tokenizer.ggml.bos_token_id") if meta.get("tokenizer.ggml.add_bos_token", True) else None
        self.byte_id = {b: self.ids.get(f"<0x{b:02X}>") for b in range(256)}
        self.cache = {}
        self.texts = [self.piece(i) for i in range(len(self.tokens))]

    def _spm(self, text):
        import heapq
        syms = list(text)
        n = len(syms)
        prev, nxt = list(range(-1, n - 1)), list(range(1, n + 1))
        nxt[-1] = -1
        heap = []

        def add(l, r):
            if l < 0 or r < 0:
                return
            tid = self.ids.get(syms[l] + syms[r])
            if tid is not None:
                heapq.heappush(heap, (-self.scores[tid], l, r, len(syms[l]) + len(syms[r])))
        for i in range(1, n):
            add(i - 1, i)
        while heap:
            _, l, r, size = heapq.heappop(heap)
            if not syms[l] or not syms[r] or nxt[l] != r or len(syms[l]) + len(syms[r]) != size:
                continue
            syms[l], syms[r] = syms[l] + syms[r], ""
            nxt[l] = nxt[r]
            if nxt[r] >= 0:
                prev[nxt[r]] = l
            add(prev[l], l)
            add(l, nxt[l])
        out, i = [], 0
        while 0 <= i < n:
            s = syms[i]
            if s in self.ids:
                out.append(self.ids[s])
            else:
                out.extend(self.byte_id[b] if self.byte_id[b] is not None else self.unk for b in s.encode("utf-8"))
            i = nxt[i]
        return out

    def encode(self, text, special=True):
        out, prev_special = [], True
        chunks = self.special_rx.split(text) if (special and self.special_rx) else [text]
        for k, chunk in enumerate(chunks):
            if special and self.special_rx and k % 2 == 1:
                out.append(self.ids[chunk])
                prev_special = True
                continue
            if not chunk:
                continue
            t = (" " if self.add_prefix and prev_special else "") + chunk
            out.extend(self._spm(t.replace(" ", "▁")))
            prev_special = False
        return out

    def piece(self, i):
        t = self.tokens[i]
        if self.types[i] == 6:
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t)
            if m:
                return bytes([int(m.group(1), 16)]).decode("latin-1")
        return t if self.types[i] in (3, 4, 5) else t.replace("▁", " ")

    def decode(self, ids):
        bs = bytearray()
        for i in ids:
            t = self.tokens[i]
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t) if self.types[i] == 6 else None
            bs += bytes([int(m.group(1), 16)]) if m else (t if self.types[i] in (3, 4, 5) else t.replace("▁", " ")).encode()
        s = bs.decode("utf-8", errors="replace")
        return s[1:] if self.add_prefix and s.startswith(" ") else s

    def is_stop(self, i):
        return i == self.eos or self.types[i] == 3 and self.tokens[i] in ("</s>", "<end_of_turn>", "<|end|>", "<|im_end|>", "<|eot_id|>")


def _unescape_rwkv(s):
    """llama.cpp's llama_unescape_rwkv_token: \\t \\n \\r \\xNN and \\<char> escapes → bytes."""
    out, i = bytearray(), 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "x" and i + 3 < len(s) + 1:
                out.append(int(s[i + 2:i + 4], 16))
                i += 4
                continue
            out += {"t": b"\t", "n": b"\n", "r": b"\r"}.get(n, n.encode())
            i += 2
            continue
        out += c.encode()
        i += 1
    return bytes(out)


class RwkvTokenizer(Tokenizer):
    """RWKV world tokenizer (GGUF type 'rwkv'): greedy longest match over the bytes of the text, as llama.cpp."""

    def __init__(self, meta):
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.types = meta.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.raw = [_unescape_rwkv(t) for t in self.tokens]
        self.trie = {}
        for i, b in enumerate(self.raw):
            if self.types[i] in (3, 4) or not b:
                continue
            node = self.trie
            for ch in b:
                node = node.setdefault(ch, {})
            node.setdefault(None, i)
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        special = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4) and t), key=len, reverse=True)
        self.special_rx = re.compile("(" + "|".join(map(re.escape, special)) + ")") if special else None
        self.unk = meta.get("tokenizer.ggml.unknown_token_id", 0) or 0
        self.eos = meta.get("tokenizer.ggml.eos_token_id")
        self.bos = meta.get("tokenizer.ggml.bos_token_id") if meta.get("tokenizer.ggml.add_bos_token") else None
        self.texts = [self.piece(i) for i in range(len(self.tokens))]

    def _greedy(self, data):
        out, pos = [], 0
        while pos < len(data):
            node = self.trie.get(data[pos])
            if node is None:
                out.append(self.unk)
                pos += 1
                continue
            tid, length, p = 0, 0, pos
            while node is not None:
                if None in node:
                    tid, length = node[None], p + 1
                p += 1
                node = node.get(data[p]) if p < len(data) else None
            out.append(tid)
            pos = length if length else pos + 1
        return out

    def encode(self, text, special=True):
        out = []
        chunks = self.special_rx.split(text) if (special and self.special_rx) else [text]
        for k, chunk in enumerate(chunks):
            if special and self.special_rx and k % 2 == 1:
                out.append(self.ids[chunk])
            elif chunk:
                out.extend(self._greedy(chunk.encode("utf-8")))
        return out

    def piece(self, i):
        return self.tokens[i] if self.types[i] in (3, 4) else self.raw[i].decode("utf-8", errors="replace")

    def decode(self, ids):
        return b"".join(self.tokens[i].encode() if self.types[i] in (3, 4) else self.raw[i] for i in ids).decode("utf-8", errors="replace")


TOKENIZERS = {"gpt2": Tokenizer, "llama": SPMTokenizer, "rwkv": RwkvTokenizer}


def make_tokenizer(meta):
    kind = meta.get("tokenizer.ggml.model")
    if kind not in TOKENIZERS:
        raise ValueError(f"Chat tracing can't tokenize for this model: tokenizer type {kind!r} (supported: {', '.join(TOKENIZERS)}).")
    return TOKENIZERS[kind](meta)


def prompt_ids(tok, prompt):
    """Token ids of a rendered prompt, with the model's start token in front unless the template already wrote it."""
    ids = tok.encode(prompt)
    if tok.bos is not None and (not ids or ids[0] != tok.bos):
        ids = [tok.bos] + ids
    return ids


def _generation_tag():
    """transformers' {% generation %}…{% endgeneration %} tag (marks the assistant's text for training masks):
    renders its body unchanged."""
    from jinja2 import nodes
    from jinja2.ext import Extension

    class Generation(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.Scope(body, lineno=lineno)
    return Generation


def render_chat(meta, messages, thinking=False):
    """Prompt text from the model's own chat template (Jinja, like llama.cpp / transformers)."""
    tpl = meta.get("tokenizer.chat_template")
    if tpl:
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def raise_exception(msg):
                raise ValueError(msg)
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_generation_tag()])
            env.globals["raise_exception"] = raise_exception
            toks = meta.get("tokenizer.ggml.tokens") or []
            tok = lambda k: toks[meta[k]] if isinstance(meta.get(k), int) and meta[k] < len(toks) else ""
            return env.from_string(tpl).render(messages=messages, add_generation_prompt=True, enable_thinking=bool(thinking),
                                               bos_token=tok("tokenizer.ggml.bos_token_id"),
                                               eos_token=tok("tokenizer.ggml.eos_token_id"))
        except Exception as e:                       # fall back to plain ChatML
            print("chat template failed, using ChatML:", e)
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n"


# ============================================================== math helpers

def _rms(x, w, eps):
    x = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))
    return x * w if w is not None else x


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rope_neox(x, pos, n_rot, base):
    """Rotary position embedding, NEOX style (first n_rot dims of each head, halves rotated). x: [n, H, d]."""
    half = n_rot // 2
    inv = base ** (-np.arange(half, dtype=np.float64) * 2.0 / n_rot)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    out = x.copy()
    a, b = x[..., :half], x[..., half:n_rot]
    out[..., :half] = a * cos - b * sin
    out[..., half:n_rot] = b * cos + a * sin
    return out


def _rope_norm(x, pos, n_rot, base):
    """Rotary position embedding, original/GPT-J style (adjacent pairs): llama.cpp's LLAMA_ROPE_TYPE_NORM."""
    half = n_rot // 2
    inv = base ** (-np.arange(half, dtype=np.float64) * 2.0 / n_rot)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    out = x.copy()
    a, b = x[..., 0:n_rot:2], x[..., 1:n_rot:2]
    out[..., 0:n_rot:2] = a * cos - b * sin
    out[..., 1:n_rot:2] = b * cos + a * sin
    return out


ATTN_KEEP = 32                 # per head and position, the earlier tokens with the most attention are kept


def _record_attention(rec, L, p):
    """p: [H, n, t] attention weights. Keeps the ATTN_KEEP strongest earlier tokens per head and position."""
    H, n, t = p.shape
    k = min(ATTN_KEEP, t)
    idx = np.argpartition(-p, k - 1, axis=-1)[..., :k]
    w = np.take_along_axis(p, idx, -1)
    o = np.argsort(-w, -1)
    idx, w = np.take_along_axis(idx, o, -1), np.take_along_axis(w, o, -1)
    idx = np.where(w > 0, idx, -1)            # masked keys (later tokens, outside the window) get exactly 0: not "looked at"
    I = np.full((n, H, ATTN_KEEP), -1, dtype=np.int32)
    W = np.zeros((n, H, ATTN_KEEP), dtype=np.float16)
    I[:, :, :k] = idx.transpose(1, 0, 2)
    W[:, :, :k] = w.transpose(1, 0, 2)
    rec.add(f"attn{L}_i", I)
    rec.add(f"attn{L}_w", W)


def _softmax_rows(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


# ============================================================== weights

def _ws_stamp(d):
    """Changes whenever any edit that affects the weights changes."""
    parts = []
    for p in (d / "tensors.json", d / "metadata.json", d / "edit_weights.py", d / "source.json"):
        parts.append(p.stat().st_mtime_ns if p.exists() else 0)
    for sub in ("weights", "edits", "added", "computed"):
        q = d / sub
        if q.is_dir():
            with os.scandir(q) as it:
                parts.append(max((e.stat().st_mtime_ns for e in it), default=0))
    q = d / "computed"                                  # names too: deleting an edit must change the stamp
    if q.is_dir():
        parts.append("|".join(sorted(p.name for p in q.glob("*.npz"))))
    return tuple(parts)


def _load(ws, which, log):
    """name -> float32 array, metadata. which = 'edited' (workspace, all edits) or 'original' (source file).
    Everything is unpacked into memory (models that fit; bigger ones use _load_lazy)."""
    d = core.workspace_dir(ws)
    info, reader, ot = core.open_source(d, check=False)
    if which == "original":
        meta = core.original_metadata(reader)
        names = [t.name for t in reader.tensors]
        n_params = sum(int(np.prod(t.shape)) for t in reader.tensors)
    else:
        meta = core.load_user_metadata(d)
        man = core.load_manifest(d)
        names = [e["name"] for e in man]
        n_params = sum(e["n_elements"] for e in man)
    names = [n for n in names if not n.startswith(("v.", "a.", "mm."))]
    ram = core.total_ram()
    if ram and n_params * 4 > 0.6 * ram:
        raise MemoryError(f"Tracing runs the model in full precision: about {n_params * 4 / 1e9:.1f} GB of memory for "
                          f"{n_params / 1e9:.1f}B weights, and this computer has {ram / 1e9:.0f} GB. Use a smaller model.")
    T = {}
    if which == "original":
        tmap = {t.name: t for t in reader.tensors}
        for i, n in enumerate(names):
            T[n] = np.ascontiguousarray(core.full_tensor(tmap[n]), dtype=np.float32)
            if i % 40 == 0:
                log(f"Loading weights {i + 1}/{len(names)}…")
    else:
        man = core.load_manifest(d)
        w, edits, modified, _counts, _newp = core._apply_all_edits(d, ot, man, log)
        w._pending = edits
        for i, n in enumerate(names):
            T[n] = np.ascontiguousarray(w[n], dtype=np.float32)
            w._cache.pop(n, None)
            if i % 40 == 0:
                log(f"Loading weights {i + 1}/{len(names)}…")
        meta["_edited_tensors"] = sorted(modified)
    return T, meta


# ============================================================== low-memory mode (big models)
#
# The model file is memory-mapped and stays compressed (Q4_K, Q8_0, MXFP4…). A tensor is unpacked to float32
# only when a layer needs it, into a cache with a size limit; the least recently used are dropped first. In a
# mixture-of-experts layer only the experts a token was routed to are unpacked. Grid / CSV / Excel edits are
# applied as a tensor is unpacked; tensors edit_weights.py touched are held whole. Small models fit in the
# cache, so after the first token they run at full speed; big ones unpack layers again for every token.

class ExpertSlices:
    """A 3D expert tensor [experts, rows, cols] that unpacks one expert at a time (through the cache)."""
    ndim = 3

    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.shape = tuple(int(x) for x in owner.shapes[name])

    def __getitem__(self, e):
        return self.owner._get_slice(self.name, int(e))


class LazyWeights:
    def __init__(self, ot, names, shapes, pending=None, pinned=None, budget=None):
        self.ot, self.names, self.shapes = ot, set(names), shapes
        self.pending, self.pinned = pending or {}, dict(pinned or {})
        ram = core.total_ram() or 8e9
        self.budget = budget if budget is not None else int(0.45 * ram)
        self.cache, self.used = {}, 0                 # (name, expert or None) -> array, insertion order = LRU order
        self.unpacked = 0                             # numbers unpacked so far (for speed estimates)

    def __contains__(self, name):
        return name in self.names

    def get(self, name, default=None):
        return self[name] if name in self.names else default

    def keys(self):
        return iter(self.names)

    def _remember(self, key, arr):
        self.unpacked += arr.size
        if arr.nbytes > self.budget:
            return arr
        while self.cache and self.used + arr.nbytes > self.budget:
            old = self.cache.pop(next(iter(self.cache)))
            self.used -= old.nbytes
        self.cache[key] = arr
        self.used += arr.nbytes
        return arr

    def _decode(self, t, data):
        a = core._decode(data, t.tensor_type)
        return np.ascontiguousarray(a, dtype=np.float32)

    def __getitem__(self, name):
        if name in self.pinned:
            return self.pinned[name]
        if name not in self.names:
            raise KeyError(name)
        shape = self.shapes[name]
        if len(shape) == 3 and "_exps." in name:
            return ExpertSlices(self, name)
        key = (name, None)
        if key in self.cache:
            arr = self.cache.pop(key)
            self.cache[key] = arr                     # most recently used
            return arr
        t = self.ot[name]
        arr = self._decode(t, t.data).reshape(shape)
        ed = self.pending.get(name)
        if ed:
            arr = core._apply_flat(arr.copy(), ed).astype(np.float32, copy=False)
        return self._remember(key, arr)

    def _get_slice(self, name, e):
        if name in self.pinned:
            return self.pinned[name][e]
        key = (name, e)
        if key in self.cache:
            arr = self.cache.pop(key)
            self.cache[key] = arr
            return arr
        t, shape = self.ot[name], self.shapes[name]
        arr = self._decode(t, t.data[e]).reshape(shape[1:])
        ed = self.pending.get(name)
        if ed:
            per = int(np.prod(shape[1:]))
            sub = {k - e * per: v for k, v in ed.items() if e * per <= k < (e + 1) * per}
            if sub:
                arr = core._apply_flat(arr.copy(), sub).astype(np.float32, copy=False)
        return self._remember(key, arr)


def _natural_shape(t):
    return tuple(int(x) for x in reversed([int(v) for v in t.shape]))


def _load_lazy(ws, which, log):
    """Weights for low-memory mode: memory-mapped, unpacked on demand. Same metadata as _load."""
    d = core.workspace_dir(ws)
    info, reader, ot = core.open_source(d, check=False)
    if which == "original":
        meta = core.original_metadata(reader)
        names = [t.name for t in reader.tensors if not t.name.startswith(("v.", "a.", "mm."))]
        return LazyWeights(ot, names, {n: _natural_shape(ot[n]) for n in names}), meta
    meta = core.load_user_metadata(d)
    man = core.load_manifest(d)
    if any(e["source"] != "original" for e in man):
        raise ValueError("Low-memory mode reads the weights from the model file, so it can't include grown or added "
                         "tensors yet. Trace the original weights, or export the grown model and open that.")
    w, edits, modified, _c, _n = core._apply_all_edits(d, ot, man, log)    # runs edit_weights.py if it has code
    names = [e["name"] for e in man if not e["name"].startswith(("v.", "a.", "mm."))]
    pinned = {n: np.ascontiguousarray(a, dtype=np.float32) for n, a in w._cache.items()}
    meta["_edited_tensors"] = sorted(modified)
    log(f"Low-memory mode: {len(pinned)} tensor(s) held whole for edit_weights.py, grid edits applied on unpacking")
    return LazyWeights({e["name"]: ot[e["orig"]] for e in man if e["name"] in names}, names,
                       {e["name"]: tuple(e["numpy_shape"]) for e in man}, pending=edits, pinned=pinned), meta


def _sizes(ws):
    """(all weights, weights used per token) in numbers, for memory and speed estimates."""
    d = core.workspace_dir(ws)
    meta = core.load_user_metadata(d)
    arch = meta.get("general.architecture")
    E, k = meta.get(f"{arch}.expert_count") or 0, meta.get(f"{arch}.expert_used_count") or 0
    total = active = 0
    for e in core.load_manifest(d):
        if e["name"].startswith(("v.", "a.", "mm.")):
            continue
        total += e["n_elements"]
        active += e["n_elements"] * (k / E if E and "_exps." in e["name"] else 1)
    return total, int(active)


UNPACK_RATE = 1.5e8          # numbers per second numpy unpacks from Q4/Q8 blocks (rough, measured on this PC class)


# ============================================================== architectures
#
# One generic decoder (Model) covers most families; each family only says what differs from it. The rules
# follow llama.cpp's graph for that architecture (src/models/*.cpp), and every family is checked against
# llama.cpp by test_trace_arch.py on a small synthetic model.

# ---- stopping a long task: the app sets the flag of the thread running it; long loops call check_stop()
import threading as _threading
_JOB = _threading.local()


class Stopped(RuntimeError):
    pass


def set_stop_flag(event):
    _JOB.cancel = event


def check_stop():
    ev = getattr(_JOB, "cancel", None)
    if ev is not None and ev.is_set():
        raise Stopped("Stopped")


class State:
    def __init__(self, n_layer):
        self.pos = 0
        self.L = [dict() for _ in range(n_layer)]


class Recorder:
    """Collects per-position arrays during a forward pass."""
    def __init__(self):
        self.cols = {}

    def add(self, key, arr):
        self.cols.setdefault(key, []).append(arr)


class NullRecorder(Recorder):
    """Keeps nothing: for re-runs whose activations nobody reads (also skips sorting the attention weights)."""
    null = True

    def add(self, key, arr):
        pass


def _per(v, L, dflt=None):
    """A hyperparameter that may be one value or a per-layer list."""
    if isinstance(v, list):
        return v[L] if L < len(v) else dflt
    return dflt if v is None else v


def _layernorm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    y = xc * (1.0 / np.sqrt(np.mean(xc * xc, axis=-1, keepdims=True) + eps))
    if w is not None:
        y = y * w
    return y + b if b is not None else y


def _gelu_exact(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _gelu(x):
    """GELU (tanh approximation) exactly as llama.cpp's CPU computes it: through a 16-bit lookup table
    (input rounded to float16, result stored as float16), 0 below -10 and x above 10."""
    x = np.asarray(x, np.float32)
    y = _gelu_exact(x.astype(np.float16).astype(np.float32)).astype(np.float16).astype(np.float32)
    return np.where(x <= -10.0, np.float32(0), np.where(x >= 10.0, x, y)).astype(np.float32)


ACTS = {"silu": _silu, "gelu": _gelu, "relu": lambda x: np.maximum(x, 0), "relu2": lambda x: np.square(np.maximum(x, 0))}


class Model:
    """Generic decoder-only transformer, as llama.cpp builds it:
        x = embed(tokens) * embedding_scale
        for each layer:  h = norm(x);  a = mixer(h);  a = post_norm(a);  x = x + a * residual_scale
                         h = norm(x);  m = ffn(h);    m = post_norm(m);  x = x + m * residual_scale
        logits = (norm(x) @ output.T) / logit_scale  (optionally soft-capped)
    Tensors that exist switch features on (biases, q/k norms, post-norms, fused qkv / gate-up, rope factors)."""
    rope_style = "norm"            # "norm": adjacent pairs (LLaMA); "neox": first and second half (Qwen, Gemma, Phi-3…)
    act = "silu"
    norm_kind = "rms"              # "rms" or "layer" (LayerNorm with mean)
    qk_norm_kind = "rms"
    emb_scale_sqrt = False         # Gemma multiplies embeddings by sqrt(n_embd)
    parallel = False               # attention and MLP read the same normed input (Command-R, Falcon, GPT-J…)

    def __init__(self, T, meta, arch):
        self.T, self.meta, self.name = T, meta, arch
        g = self.g = lambda k, dflt=None: meta.get(f"{arch}.{k}", dflt)
        self.n_layer = int(g("block_count"))
        self.D = int(g("embedding_length"))
        self.eps = float(g("attention.layer_norm_rms_epsilon") or g("attention.layer_norm_epsilon") or 1e-6)
        self.n_head_l = [int(_per(g("attention.head_count"), L, 0)) for L in range(self.n_layer)]
        self.n_head = max(self.n_head_l)
        self.n_kv_l = [int(_per(g("attention.head_count_kv"), L, self.n_head_l[L])) for L in range(self.n_layer)]
        self.hd = int(g("attention.key_length") or self.D // max(1, self.n_head))
        self.vd = int(g("attention.value_length") or self.hd)
        self.n_rot = int(g("rope.dimension_count") or self.hd)
        self.base = float(g("rope.freq_base") or 10000.0)
        self.base_swa = float(g("rope.freq_base_swa") or self.base)
        self.rope_pos_scale = 1.0
        self.yarn = None
        if g("rope.scaling.type") == "linear" and g("rope.scaling.factor"):
            self.rope_pos_scale = 1.0 / float(g("rope.scaling.factor"))
        if g("rope.scaling.type") == "yarn" and g("rope.scaling.factor"):
            # YaRN (long-context rope): low-frequency dims are interpolated, high-frequency ones kept (ggml rope_yarn)
            self.yarn = {"freq_scale": 1.0 / float(g("rope.scaling.factor")),
                         "n_ctx_orig": int(g("rope.scaling.original_context_length") or g("context_length") or 4096),
                         "beta_fast": float(g("rope.scaling.yarn_beta_fast") or 32.0),
                         "beta_slow": float(g("rope.scaling.yarn_beta_slow") or 1.0),
                         "ext_factor": float(g("rope.scaling.yarn_ext_factor") or 1.0),
                         "attn_factor": float(g("rope.scaling.attn_factor") or 1.0)}
        self.freq_factors = T.get("rope_freqs.weight")
        self.attn_scale = float(g("attention.scale") or 0) or 1.0 / math.sqrt(self.hd)
        self.softcap_attn = float(g("attn_logit_softcapping") or 0)
        self.softcap_final = float(g("final_logit_softcapping") or 0)
        self.emb_scale = float(g("embedding_scale") or 0) or (math.sqrt(self.D) if self.emb_scale_sqrt else 1.0)
        self.res_scale = float(g("residual_scale") or 0) or 1.0
        self.logit_scale = float(g("logit_scale") or 0) or 1.0
        self.swa = int(g("attention.sliding_window") or 0)
        self.swa_l = self._swa_layers()
        self.ff = int(_per(g("feed_forward_length"), 0, 0) or 0)
        self.E = T["token_embd.weight"]
        self.U = T.get("output.weight", self.E)
        self.out_norm, self.out_norm_b = T.get("output_norm.weight"), T.get("output_norm.bias")
        self.out_bias = T.get("output.bias")

    # ---- per-layer facts
    def _swa_layers(self):
        pat = self.g("attention.sliding_window_pattern")
        if isinstance(pat, list):
            return [bool(x) for x in pat] + [False] * (self.n_layer - len(pat))
        return [False] * self.n_layer

    def w(self, L, part):
        return self.T.get(f"blk.{L}.{part}")

    def is_attention(self, L):
        return self.w(L, "attn_q.weight") is not None or self.w(L, "attn_qkv.weight") is not None

    def kinds(self):
        return [("sliding attention" if self.swa_l[L] else "attention") if self.is_attention(L) else "other"
                for L in range(self.n_layer)]

    def n_heads(self, L):
        return self.n_head_l[L]

    def head_dim(self, L):
        return self.vd

    def out_proj(self, L):
        return self.w(L, "attn_output.weight")

    def rope_base(self, L):
        return self.base_swa if self.swa_l[L] else self.base

    def uses_rope(self, L):
        return True

    # ---- building blocks
    def norm(self, x, name_or_w, b=None):
        w = self.T.get(name_or_w) if isinstance(name_or_w, str) else name_or_w
        if isinstance(name_or_w, str) and b is None:
            b = self.T.get(name_or_w[:-len("weight")] + "bias") if name_or_w.endswith("weight") else None
        if self.norm_kind == "layer":
            return _layernorm(x, w, b, self.eps)
        y = _rms(x, w, self.eps)
        return y + b if b is not None else y

    def lin(self, x, L, name):
        """x @ W.T (+ bias) for blk.L.<name>.weight"""
        y = x @ self.w(L, name + ".weight").T
        b = self.w(L, name + ".bias")
        return y + b if b is not None else y

    def rope(self, x, pos, L):
        if not self.uses_rope(L) or self.n_rot == 0:
            return x
        half = self.n_rot // 2
        inv = self.rope_base(L) ** (-np.arange(half, dtype=np.float64) * 2.0 / self.n_rot)
        ff = self.rope_factors(L)
        if ff is not None:
            inv = inv / ff[:half].astype(np.float64)
        y = self.yarn if self.uses_yarn(L) else None
        if y:
            ext = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]               # theta_extrap
            fs = y["freq_scale"]
            corr = lambda beta: self.n_rot * math.log(y["n_ctx_orig"] / (beta * 2 * math.pi)) / (2 * math.log(self.rope_base(L)))
            lo, hi = max(0.0, math.floor(corr(y["beta_fast"]))), min(self.n_rot - 1.0, math.ceil(corr(y["beta_slow"])))
            ramp = (1 - np.clip((np.arange(half) - lo) / max(0.001, hi - lo), 0, 1)) * y["ext_factor"]
            ang = ext * fs * (1 - ramp) + ext * ramp
            msc = self.yarn_attn_factor(L) * ((1.0 + 0.1 * math.log(1.0 / fs)) if y["ext_factor"] else 1.0)
        else:
            ang = (np.asarray(pos, dtype=np.float64) * self.rope_scale(L))[:, None] * inv[None, :]
            msc = 1.0
        cos = (np.cos(ang) * msc).astype(np.float32)[:, None, :]
        sin = (np.sin(ang) * msc).astype(np.float32)[:, None, :]
        out = x.copy()
        if self.rope_style == "neox":
            a, b = x[..., :half], x[..., half:self.n_rot]
            out[..., :half], out[..., half:self.n_rot] = a * cos - b * sin, b * cos + a * sin
        else:
            a, b = x[..., 0:self.n_rot:2], x[..., 1:self.n_rot:2]
            out[..., 0:self.n_rot:2], out[..., 1:self.n_rot:2] = a * cos - b * sin, b * cos + a * sin
        return out

    def rope_factors(self, L):
        return self.freq_factors

    def rope_scale(self, L):
        return self.rope_pos_scale

    def uses_yarn(self, L):
        return self.yarn is not None

    def yarn_attn_factor(self, L):
        return self.yarn["attn_factor"]

    def qkv(self, L, h):
        n, H, KV, hd, vd = h.shape[0], self.n_head_l[L], self.n_kv_l[L], self.hd, self.vd
        if self.w(L, "attn_qkv.weight") is not None:
            y = self.lin(h, L, "attn_qkv")
            q, k, v = y[:, :H * hd], y[:, H * hd:H * hd + KV * hd], y[:, H * hd + KV * hd:]
        else:
            q, k, v = self.lin(h, L, "attn_q"), self.lin(h, L, "attn_k"), self.lin(h, L, "attn_v")
        qn, kn = self.w(L, "attn_q_norm.weight"), self.w(L, "attn_k_norm.weight")
        if qn is not None and qn.shape[0] == H * hd:                 # norm over all heads at once (OLMo 2, OLMoE)
            q = self._qk_norm(q, qn, self.w(L, "attn_q_norm.bias"))
        if kn is not None and kn.shape[0] == KV * hd:
            k = self._qk_norm(k, kn, self.w(L, "attn_k_norm.bias"))
        q, k, v = q.reshape(n, H, hd), k.reshape(n, KV, hd), v.reshape(n, KV, vd)
        if qn is not None and qn.shape[-1] == hd:                    # per-head norm (Qwen3, Gemma 3; 2D = own weights per head, Command-R+)
            q = self._qk_norm(q, qn, self.w(L, "attn_q_norm.bias"))
        if kn is not None and kn.shape[-1] == hd:
            k = self._qk_norm(k, kn, self.w(L, "attn_k_norm.bias"))
        return q, k, v, None

    def _qk_norm(self, x, w, b):
        return _layernorm(x, w, b, self.eps) if self.qk_norm_kind == "layer" else _rms(x, w, self.eps)

    def attend(self, L, q, k, v, pos, st, rec):
        """Causal (optionally sliding-window, soft-capped) attention with a KV cache. q [n,H,d], k/v [n,KV,d]."""
        S = st.L[L]
        S["K"] = k if "K" not in S else np.concatenate([S["K"], k])
        S["V"] = v if "V" not in S else np.concatenate([S["V"], v])
        S["P"] = np.asarray(pos) if "P" not in S else np.concatenate([S["P"], pos])
        H, KV = q.shape[1], k.shape[1]
        r = H // KV
        Kr, Vr = np.repeat(S["K"], r, axis=1), np.repeat(S["V"], r, axis=1)
        s = np.matmul(q.transpose(1, 0, 2), Kr.transpose(1, 2, 0)) * np.float32(self.attn_scale)    # [H, n, t]
        if self.softcap_attn:
            c = np.float32(self.softcap_attn)
            s = np.tanh(s / c) * c
        kp, qp = S["P"][None, :], np.asarray(pos)[:, None]
        mask = kp > qp
        if self.swa_l[L] and self.swa:
            mask |= (qp - kp) >= self.swa
        em = self.extra_mask(L, qp, kp)
        if em is not None:
            mask |= em
        s = np.where(mask[None], -np.inf, s)
        sink = self.w(L, "attn_sinks.weight")
        if sink is not None:                                          # gpt-oss: a learned extra logit per head
            s = np.concatenate([s, np.broadcast_to(sink.reshape(H, 1, 1), (H, s.shape[1], 1))], -1)
            p = _softmax_rows(s)[..., :-1]
        else:
            p = _softmax_rows(s)
        o = np.matmul(p, Vr.transpose(1, 0, 2)).transpose(1, 0, 2)                                  # [n, H, vd]
        if not getattr(rec, "null", False):
            _record_attention(rec, L, p)
        return o.astype(np.float32)

    def attn_temp(self, L, pos):
        """Attention temperature tuning (Mistral 3, Llama 4 NoPE layers): queries × (1 + s·log(1 + floor((pos+1)/len))).
        Changes nothing until the position reaches temperature_length."""
        s, n = float(self.g("attention.temperature_scale") or 0), float(self.g("attention.temperature_length") or 0)
        if not s or not n or not self.uses_attn_temp(L):
            return None
        return (np.log(np.floor((np.asarray(pos, np.float64) + 1.0) / n) + 1.0) * s + 1.0).astype(np.float32)[:, None, None]

    def uses_attn_temp(self, L):
        return True

    def extra_mask(self, L, qp, kp):
        return None

    def mixer(self, L, h, pos, st, rec):
        q, k, v, gate = self.qkv(L, h)
        q, k = self.rope(q, pos, L), self.rope(k, pos, L)
        t = self.attn_temp(L, pos)
        if t is not None:
            q = q * t
        o = self.attend(L, q, k, v, pos, st, rec)
        if gate is not None:
            o = o * gate
        n = h.shape[0]
        return self.lin(o.reshape(n, -1), L, "attn_output"), o

    def ffn(self, L, h, rec=None):
        """Returns (neuron activations, output) of the layer's MLP: dense, mixture of experts, or none (Mamba)."""
        if self.is_moe(L):
            return self.moe(L, h, rec)
        if self.w(L, "ffn_up.weight") is None:
            return np.zeros((h.shape[0], 1), np.float32), np.zeros_like(h)
        return self.dense_ffn(L, h)

    def dense_ffn(self, L, h, sfx=""):
        """Gated (SwiGLU / GeGLU), fused gate+up (Phi-3), or plain MLP. sfx='_shexp' for a shared expert."""
        f = ACTS[self.act]
        up = self.lin(h, L, "ffn_up" + sfx)
        if self.w(L, f"ffn_gate{sfx}.weight") is not None:
            act = f(self.lin(h, L, "ffn_gate" + sfx)) * up
        elif self.w(L, f"ffn_down{sfx}.weight").shape[1] * 2 == up.shape[1]:   # fused [gate; up]
            half = up.shape[1] // 2
            act = f(up[:, :half]) * up[:, half:]
        else:
            act = f(up)
        return act, self.lin(act, L, "ffn_down" + sfx)

    # ---- mixture of experts
    moe_gating = "softmax"          # how router scores become weights: softmax | sigmoid | softmax_weight (gpt-oss)
    moe_norm = False                # renormalize the chosen experts' weights to sum to 1
    moe_weight_before = False       # multiply the expert's input (not its output) by its weight (Llama 4)

    def is_moe(self, L):
        return self.w(L, "ffn_gate_inp.weight") is not None

    def moe_params(self):
        gf = self.g("expert_gating_func")
        gating = {1: "softmax", 2: "sigmoid", 3: "softmax_weight"}.get(gf, self.moe_gating)
        norm = self.g("expert_weights_norm")
        return gating, self.moe_norm if norm is None else bool(norm), float(self.g("expert_weights_scale") or 0) or 1.0

    def moe(self, L, h, rec):
        """Router → top-k experts → weighted sum of their MLPs (+ shared expert). Activations returned are the
        chosen experts' neurons, slot by slot (slot 0 = highest weight), then the shared expert's."""
        f = ACTS[self.act]
        n = h.shape[0]
        k = int(self.g("expert_used_count") or 2)
        gating, norm, scale = self.moe_params()
        logits = self.lin(h, L, "ffn_gate_inp")                               # [n, E]
        if gating == "sigmoid":
            probs = _sigmoid(logits)
        elif gating == "softmax":
            probs = _softmax_rows(logits)
        else:
            probs = logits
        if self.w(L, "exp_probs_b.bias") is not None:
            sel = probs + self.w(L, "exp_probs_b.bias")
        elif gating == "sigmoid":
            sel = logits            # same order as the sigmoid, but no float32 ties where the sigmoid saturates at 1.0
        else:
            sel = probs
        sel = self.limit_groups(sel, k)
        idx = np.argsort(-sel, axis=-1, kind="stable")[:, :k]
        wts = np.take_along_axis(probs, idx, -1)
        if gating == "softmax_weight":
            wts = _softmax_rows(np.take_along_axis(logits, idx, -1))
        if norm:
            wts = wts / wts.sum(-1, keepdims=True)
        wts = (wts * np.float32(scale)).astype(np.float32)
        G, U, Dn = self.w(L, "ffn_gate_exps.weight"), self.w(L, "ffn_up_exps.weight"), self.w(L, "ffn_down_exps.weight")
        bG, bU, bD = self.w(L, "ffn_gate_exps.bias"), self.w(L, "ffn_up_exps.bias"), self.w(L, "ffn_down_exps.bias")
        ffe = U.shape[1]
        acts = np.zeros((n, k, ffe), np.float32)
        out = np.zeros((n, self.D), np.float32)
        for e in np.unique(idx):
            rows, slots = np.nonzero(idx == e)
            x = h[rows]
            if self.moe_weight_before:                                       # Llama 4: weight scales the expert's input
                x = x * wts[rows, slots][:, None]
            u = x @ U[e].T + (bU[e] if bU is not None else 0)
            if G is not None:
                a = self.expert_act(x @ G[e].T + (bG[e] if bG is not None else 0), u)
            else:
                a = f(u)
            acts[rows, slots] = a
            y = a @ Dn[e].T + (bD[e] if bD is not None else 0)
            out[rows] += y if self.moe_weight_before else wts[rows, slots][:, None] * y
        act = acts.reshape(n, k * ffe)
        sg = np.ones(n, np.float32)
        if self.w(L, "ffn_up_shexp.weight") is not None:
            sa, so = self.dense_ffn(L, h, "_shexp")
            gi = self.w(L, "ffn_gate_inp_shexp.weight")
            if gi is not None:                                                # Qwen2-MoE: a sigmoid gate on the shared expert
                sg = _sigmoid(h @ gi.reshape(-1)).astype(np.float32)
            out += sg[:, None] * so
            act = np.concatenate([act, sa], 1)
        if rec is not None:
            rec.add(f"moe{L}_ids", idx.astype(np.int32))
            rec.add(f"moe{L}_w", wts)
            rec.add(f"moe{L}_sg", sg)
            rec.add(f"moe{L}_router", probs.astype(np.float16) if gating != "softmax_weight" else _softmax_rows(logits).astype(np.float16))
        return act, out

    def expert_act(self, gate, up):
        return ACTS[self.act](gate) * up

    def limit_groups(self, sel, k):
        """Group-limited routing (DeepSeek V3, Kimi K2): experts are split into groups, each group is scored by the
        sum of its 2 best experts, and only the best `expert_group_used_count` groups may be chosen."""
        G, used = int(self.g("expert_group_count") or 0), int(self.g("expert_group_used_count") or 0)
        if G <= 1 or not used or used >= G:
            return sel
        n, E = sel.shape
        grp = sel.reshape(n, G, E // G)
        score = np.sort(grp, -1)[..., -2:].sum(-1)                              # [n, G]
        keep = np.argsort(-score, -1, kind="stable")[:, :used]
        mask = np.ones((n, G), bool)
        np.put_along_axis(mask, keep, False, -1)
        return np.where(np.repeat(mask, E // G, axis=1), -np.inf, sel)

    def moe_layout(self):
        """Per MoE layer: experts, experts per token, neurons per expert, shared-expert neurons (for the UI's links)."""
        out = {}
        for L in range(self.n_layer):
            if self.is_moe(L):
                U, ds = self.w(L, "ffn_up_exps.weight"), self.w(L, "ffn_down_shexp.weight")
                out[L] = {"E": int(U.shape[0]), "k": int(self.g("expert_used_count") or 2), "ffe": int(U.shape[1]),
                          "ffs": int(ds.shape[1]) if ds is not None else 0, "D": self.D}
        return out

    # ---- the layer stack
    def pre_ffn_norm(self, L):
        for nm in ("ffn_norm.weight", "post_attention_norm.weight"):
            if self.w(L, nm) is not None:
                return f"blk.{L}.{nm}"
        return None

    def post_norm_names(self, L):
        """(norm applied to the mixer's output, norm applied to the MLP's output) or None."""
        has = lambda nm: self.w(L, nm) is not None
        sandwich = has("ffn_norm.weight") and has("post_attention_norm.weight")
        no_pre = not has("attn_norm.weight") and has("post_attention_norm.weight")
        pa = f"blk.{L}.post_attention_norm.weight" if (sandwich or no_pre) else None   # Gemma 2/3, OLMo 2, LFM
        pf = f"blk.{L}.post_ffw_norm.weight" if has("post_ffw_norm.weight") else None
        return pa, pf

    def _post(self, y, name):
        """Apply an output norm; also return the divisor it used (so attribution can stay exact)."""
        if not name:
            return y, np.ones(y.shape[0], np.float32)
        return self.norm(y, name), self.final_scale(y)

    def block(self, L, x, pos, st, rec):
        """One layer. Returns (x, neuron activations, head outputs, mixer output added, MLP output added,
        mixer post-norm divisor, MLP post-norm divisor)."""
        an = f"blk.{L}.attn_norm.weight"
        h = self.norm(x, an) if an in self.T else x
        a, hv = self.mixer(L, h, pos, st, rec)
        pa, pf = self.post_norm_names(L)
        a, a_div = self._post(a, pa)
        rs = np.float32(self.res_scale)
        if self.parallel:
            act, m = self.ffn(L, h, rec)
            m, m_div = self._post(m, pf)
            return x + (a + m) * rs, act, hv, a * rs, m * rs, a_div, m_div
        x = x + a * rs
        if pa and self.w(L, "ffn_norm.weight") is not None:
            fn = f"blk.{L}.ffn_norm.weight"
        elif self.w(L, "attn_norm.weight") is None:
            fn = None                                                  # OLMo 2: no pre-norms at all
        else:
            fn = self.pre_ffn_norm(L)
        h2 = self.norm(x, fn) if fn else x
        act, m = self.ffn(L, h2, rec)
        m, m_div = self._post(m, pf)
        return x + m * rs, act, hv, a * rs, m * rs, a_div, m_div

    def embed(self, ids):
        x = self.E[np.asarray(ids)].astype(np.float32)
        return x * np.float32(self.emb_scale) if self.emb_scale != 1.0 else x

    def final(self, x):
        if self.out_norm is None:
            return x
        if self.norm_kind == "layer":
            return _layernorm(x, self.out_norm, self.out_norm_b, self.eps)
        return _rms(x, self.out_norm, self.eps)

    def final_scale(self, x):
        """The divisor the final norm applies (for attribution): RMS, or the std for LayerNorm."""
        if self.norm_kind == "layer":
            xc = x - x.mean(-1, keepdims=True)
            return np.sqrt(np.mean(xc * xc, axis=-1) + self.eps)
        return np.sqrt(np.mean(x * x, axis=-1) + self.eps)

    def logits(self, xf):
        y = xf @ self.U.T
        if self.out_bias is not None:
            y = y + self.out_bias
        if self.logit_scale != 1.0:
            y = y / np.float32(self.logit_scale)
        if self.softcap_final:
            c = np.float32(self.softcap_final)
            y = np.tanh(y / c) * c
        return y

    def logit_direction(self, target, scale):
        """d such that (a part's vector) · d = its direct contribution to the target's score, before soft-capping."""
        w = self.out_norm if self.out_norm is not None else 1.0
        d = self.U[target] * w / float(scale) / self.logit_scale
        if self.norm_kind == "layer":
            d = d - d.mean()                                          # LayerNorm subtracts the mean first
        return d.astype(np.float32)

    # ---- attribution hooks: how a layer's outputs reach the final score (see attribute())
    embed_factor = 1.0
    attribution_exact = True        # False where layer pushes can't add up exactly (Gemma 3n's merged streams)

    def mlp_down(self, L):
        return self.w(L, "ffn_down.weight")

    def mlp_direction(self, L, dm, A, pos):
        return dm

    def mixer_direction(self, L, da, A, pos):
        return da

    def head_layout(self, L):
        if not self.is_attention(L):
            return {}
        H, KV, hd, vd = self.n_head_l[L], self.n_kv_l[L], self.hd, self.vd
        r = max(1, H // max(1, KV))
        if self.w(L, "attn_qkv.weight") is not None:
            nm = f"blk.{L}.attn_qkv.weight"
            reads = [[nm, "rows", hd, 1, "query", 0], [nm, "rows", hd, r, "key", H * hd], [nm, "rows", vd, r, "value", H * hd + KV * hd]]
        else:
            reads = [[f"blk.{L}.attn_q.weight", "rows", hd, 1, "query"], [f"blk.{L}.attn_k.weight", "rows", hd, r, f"key (shared by {r})"],
                     [f"blk.{L}.attn_v.weight", "rows", vd, r, f"value (shared by {r})"]]
        return {"kind": "attention", "out": f"blk.{L}.attn_output.weight", "out_cols": vd, "reads": reads}

    def forward(self, ids, st, rec):
        x = self.embed(ids)
        n = len(ids)
        pos = np.arange(st.pos, st.pos + n)
        resid = [x.astype(np.float16)]
        acts, heads, a_norm, m_norm, a_vec, m_vec, a_div, m_div = [], [], [], [], [], [], [], []
        for L in range(self.n_layer):
            x, act, hv, a_out, m_out, ad, md = self.block(L, x, pos, st, rec)
            x = x.astype(np.float32, copy=False)
            assert x.dtype == np.float32, f"layer {L} produced {x.dtype}"
            resid.append(x.astype(np.float16))
            acts.append(act.astype(np.float16))
            heads.append(hv.reshape(n, -1).astype(np.float16))
            a_norm.append(np.linalg.norm(a_out, axis=-1))
            m_norm.append(np.linalg.norm(m_out, axis=-1))
            a_vec.append(a_out.astype(np.float16))
            m_vec.append(m_out.astype(np.float16))
            a_div.append(ad)
            m_div.append(md)
        st.pos += n
        rec.add("resid", np.stack(resid, 1))
        rec.add("a_vec", np.stack(a_vec, 1))                          # what each layer's mixer added   [n, L, D]
        rec.add("m_vec", np.stack(m_vec, 1))                          # what each layer's MLP added
        rec.add("a_div", np.stack(a_div, 1).astype(np.float32))       # post-norm divisors (1 = no post-norm)
        rec.add("m_div", np.stack(m_div, 1).astype(np.float32))
        aw = max(a.shape[1] for a in acts)
        rec.add("act", np.stack([np.pad(a, ((0, 0), (0, aw - a.shape[1]))) for a in acts], 1))
        hw = max(h.shape[1] for h in heads)
        rec.add("heads", np.stack([np.pad(h, ((0, 0), (0, hw - h.shape[1]))) for h in heads], 1))
        rec.add("attn_norm", np.stack(a_norm, 1).astype(np.float32))
        rec.add("mlp_norm", np.stack(m_norm, 1).astype(np.float32))
        rec.add("final_rms", self.final_scale(x).astype(np.float32))
        return self.final(x)


# ---------------------------------------------------------------- families (only what differs)

class Llama(Model):
    """LLaMA 1/2/3/3.x, Mistral, Yi, DeepSeek-R1-Distill-Llama, SmolLM 1/2… (rope: adjacent pairs).
    With experts in the file it is Mixtral: softmax router, top-k, weights renormalized."""
    moe_norm = True


class Mistral3(Llama):
    """Mistral 3 family (Ministral 3, Mistral Small 3.2+, Mistral Large 3 text): LLaMA layers with YaRN and
    attention temperature tuning on every layer: queries × (1 + s·log(1 + floor(pos / original_context))),
    s = attention.temperature_scale (as llama.cpp: only together with YaRN)."""

    def attn_temp(self, L, pos):
        s = float(self.g("attention.temperature_scale") or 0)
        if not s or not self.yarn:
            return None
        n = float(self.yarn["n_ctx_orig"])
        return (np.log(np.floor(np.asarray(pos, np.float64) / n) + 1.0) * s + 1.0).astype(np.float32)[:, None, None]


class Granite(Model):
    """IBM Granite 3.x: LLaMA plus embedding / residual / attention / logit multipliers (read from metadata).
    granitemoe: the same with a Mixtral-style mixture of experts."""
    moe_norm = True


class Qwen2(Model):
    """Qwen 2 / 2.5 (incl. DeepSeek-R1-Distill-Qwen), Qwen3 (per-head q/k norm), Phi-3/4 (fused qkv and gate+up).
    qwen2moe: softmax router without renormalizing, plus a shared expert behind a sigmoid gate."""
    rope_style = "neox"


class Qwen3Moe(Qwen2):
    """Qwen3-MoE (30B-A3B, 235B-A22B…): softmax router, top-k weights renormalized."""
    moe_norm = True


class OLMoE(Model):
    """OLMoE: q/k norm over all heads, softmax router without renormalizing, NEOX rope."""
    rope_style = "neox"


class Phi3(Qwen2):
    """Phi-3 / 3.5 / 4. Its sliding window (2047 tokens or more in real models) is not applied: shorter chats
    are unaffected, and llama.cpp's own window for Phi-3 does not behave like a plain window in tests."""

    def rope_factors(self, L):
        return self.T.get("rope_factors_short.weight", self.freq_factors)


class Gemma(Model):
    """Gemma 1: embeddings × sqrt(n_embd), GeGLU MLP, NEOX rope, tied output."""
    rope_style, act, emb_scale_sqrt = "neox", "gelu", True


class Gemma2(Gemma):
    """Gemma 2: norms before AND after attention and MLP, soft-capped attention and final scores, sliding window
    on every other layer (even layers)."""
    big_layers = 46                                                   # the 27B uses n_embd/n_head for the query scale

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        if self.n_layer == self.big_layers:
            self.attn_scale = 1.0 / math.sqrt(self.D / self.n_head)

    def _swa_layers(self):
        return [bool(self.swa) and L % 2 == 0 for L in range(self.n_layer)]


class Gemma3(Gemma2):
    """Gemma 3: Gemma 2 without soft-capping, plus q/k norm; 5 sliding-window layers per global layer. Sliding
    layers use rope base 10000 without scaling; global layers use the file's base and (linear) scaling."""
    big_layers = 62

    def _swa_layers(self):
        return [bool(self.swa) and L % 6 < 5 for L in range(self.n_layer)]

    def rope_base(self, L):
        return 10000.0 if self.swa_l[L] else self.base

    def rope_scale(self, L):
        return 1.0 if self.swa_l[L] else self.rope_pos_scale


class CommandR(Model):
    """Cohere Command-R / R+: LayerNorm (no bias), attention and MLP in parallel from the same norm,
    scores multiplied by logit_scale, per-head LayerNorm on q/k in R+."""
    norm_kind = qk_norm_kind = "layer"
    parallel = True

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        self.logit_scale = 1.0 / (float(self.g("logit_scale") or 1.0))   # multiplies (Granite divides)


class StarCoder2(Model):
    """StarCoder 2: LayerNorm with bias, biases everywhere, plain (not gated) GELU MLP, NEOX rope."""
    norm_kind, act, rope_style = "layer", "gelu", "neox"


class Olmo2(Model):
    """OLMo 2: no norms before attention / MLP, RMS norms after them, q/k norm over all heads at once, NEOX rope."""
    rope_style = "neox"


class DeepSeek2(Model):
    """DeepSeek V2 / V2.5 / V3 / R1, Kimi K2 (and other models built on them).
    Multi-head latent attention: keys and values come from a small shared latent vector c (kv_lora_rank wide) that
    each head expands with its own matrices (attn_k_b / attn_v_b, or the older fused attn_kv_b); only the last
    `rope.dimension_count` dims of each query/key are rotated, and that rotated key part is shared by all heads.
    Queries may go through a low-rank pair (attn_q_a → norm → attn_q_b). The first layers are dense; the rest are
    MoE with shared experts, a softmax (V2) or sigmoid + bias (V3) router, optional group-limited routing."""

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        g = self.g
        self.kv_lora = int(g("attention.kv_lora_rank"))
        kmla, vmla = g("attention.key_length_mla"), g("attention.value_length_mla")
        if kmla and vmla:                     # newer GGUFs: key/value_length describe the latent, *_mla the heads
            self.hd, self.vd = int(kmla), int(vmla)
        self.nope = self.hd - self.n_rot
        self.n_kv_l = list(self.n_head_l)     # after expanding the latent, every head has its own key and value
        fs = self.yarn["freq_scale"] if self.yarn else 1.0
        mscale = 1.0 + float(g("rope.scaling.yarn_log_multiplier") or 0.0) * math.log(1.0 / fs)
        self.attn_scale = mscale * mscale / math.sqrt(self.hd)

    def yarn_attn_factor(self, L):            # rope's own YaRN magnitude scaling is cancelled; it's in attn_scale
        return 1.0 / (1.0 + 0.1 * math.log(1.0 / self.yarn["freq_scale"]))

    def mixer(self, L, h, pos, st, rec):
        n, H, R = h.shape[0], self.n_head_l[L], self.n_rot
        if self.w(L, "attn_q_a.weight") is not None:
            q = self.lin(_rms(self.lin(h, L, "attn_q_a"), self.w(L, "attn_q_a_norm.weight"), self.eps), L, "attn_q_b")
        else:
            q = self.lin(h, L, "attn_q")
        q = q.reshape(n, H, self.hd)
        kva = self.lin(h, L, "attn_kv_a_mqa")
        c, k_pe = kva[:, :self.kv_lora], kva[:, self.kv_lora:].reshape(n, 1, R)
        c = _rms(c, self.w(L, "attn_kv_a_norm.weight"), self.eps)
        if self.w(L, "attn_k_b.weight") is not None:
            k_nope = np.einsum("nc,hcd->nhd", c, self.w(L, "attn_k_b.weight"))     # attn_k_b: [heads, latent, nope]
            v = np.einsum("nc,hvc->nhv", c, self.w(L, "attn_v_b.weight"))
        else:
            kv = (c @ self.w(L, "attn_kv_b.weight").T).reshape(n, H, self.nope + self.vd)
            k_nope, v = kv[..., :self.nope], kv[..., self.nope:]
        q_pe, k_pe = self.rope(q[..., self.nope:], pos, L), self.rope(k_pe, pos, L)
        q = np.concatenate([q[..., :self.nope], q_pe], -1).astype(np.float32)
        k = np.concatenate([k_nope, np.broadcast_to(k_pe, (n, H, R))], -1).astype(np.float32)
        o = self.attend(L, q, k, np.ascontiguousarray(v, dtype=np.float32), pos, st, rec)
        return self.lin(o.reshape(n, -1), L, "attn_output"), o

    def head_layout(self, L):
        H, hd, vd = self.n_head_l[L], self.hd, self.vd
        q = [f"blk.{L}.attn_q_b.weight" if self.w(L, "attn_q_b.weight") is not None else f"blk.{L}.attn_q.weight", "rows", hd, 1, "query"]
        if self.w(L, "attn_k_b.weight") is not None:
            kb, vb = self.w(L, "attn_k_b.weight"), self.w(L, "attn_v_b.weight")
            reads = [q, [f"blk.{L}.attn_k_b.weight", "rows", kb.shape[1], 1, "key from latent (head's rows)"],
                     [f"blk.{L}.attn_v_b.weight", "rows", vb.shape[1], 1, "value from latent"]]
        else:
            reads = [q, [f"blk.{L}.attn_kv_b.weight", "rows", self.nope + vd, 1, "key + value from latent"]]
        return {"kind": "latent attention", "out": f"blk.{L}.attn_output.weight", "out_cols": vd, "reads": reads}

    def kinds(self):
        return ["latent attention" for _ in range(self.n_layer)]


class GptOss(Model):
    """OpenAI gpt-oss 20B / 120B: attention with a learned 'sink' per head, sliding window on even layers, YaRN,
    biases on attention, router and experts; softmax over the chosen experts only; clamped SwiGLU
    (gate capped at 7, up clamped to ±7, out = gate·sigmoid(1.702·gate)·(up+1))."""
    rope_style = "neox"
    moe_gating = "softmax_weight"

    def _swa_layers(self):
        return [bool(self.swa) and L % 2 == 0 for L in range(self.n_layer)]

    def expert_act(self, gate, up):
        x = np.minimum(gate, np.float32(7.0))
        return x * _sigmoid(np.float32(1.702) * x) * (np.clip(up, -7.0, 7.0) + 1)


class Llama4(Model):
    """Llama 4 Scout / Maverick: every 4th layer has no rope (NoPE); the other layers RMS-normalize q and k (no
    weights) after rope (not Maverick). MoE layers: sigmoid router, top-1, the weight multiplies the expert's
    input; plus a shared expert. Checked against llama.cpp for chats up to 8192 tokens. Beyond that the rope
    layers attend only within 8192-token chunks and NoPE layers scale queries by 1 + 0.1·log(1 + floor((pos+1)/8192))
    (Meta's reference code); those two parts are not checked against llama.cpp."""
    moe_gating = "sigmoid"
    moe_weight_before = True
    CHUNK, TEMP_FLOOR, TEMP_SCALE, TEMP_OFFSET = 8192, 8192.0, 0.1, 1.0      # fixed in llama.cpp's llama4 loader

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        swa = self.g("attention.sliding_window")
        self.mobile = swa is not None and int(swa) == 0       # MobileLLM-style: no chunking, no temperature
        pat = self.g("attention.sliding_window_pattern")
        self.chunked = ([False] * self.n_layer if self.mobile else
                        [bool(x) for x in pat] if isinstance(pat, list) else [L % 4 < 3 for L in range(self.n_layer)])
        self.base_chunk = float(self.g("rope.freq_base_swa") or self.base)

    def uses_rope(self, L):
        step = self.n_layer if self.mobile else 4
        return (L + 1) % step != 0

    def rope_base(self, L):
        return self.base_chunk if self.chunked[L] else self.base

    def kinds(self):
        return [("chunked attention" if self.chunked[L] else "attention") + ("" if self.uses_rope(L) else " (no rope)")
                for L in range(self.n_layer)]

    def _swa_layers(self):
        return [False] * self.n_layer

    def attn_temp(self, L, pos):
        if self.mobile or self.uses_rope(L):
            return None
        p = np.asarray(pos, np.float64)
        return (np.log(np.floor((p + self.TEMP_OFFSET) / self.TEMP_FLOOR) + 1.0) * self.TEMP_SCALE + 1.0
                ).astype(np.float32)[:, None, None]

    def extra_mask(self, L, qp, kp):
        """Chunked layers: a token only sees earlier tokens of its own 8192-token chunk (llama.cpp's CHUNKED mask)."""
        return kp < (qp // self.CHUNK) * self.CHUNK if self.chunked[L] else None

    def use_kq_norm(self):
        return int(self.g("expert_count") or 0) != 128                # llama.cpp: off for Maverick (128 experts)

    def mixer(self, L, h, pos, st, rec):
        q, k, v, _ = self.qkv(L, h)
        if self.uses_rope(L):
            q, k = self.rope(q, pos, L), self.rope(k, pos, L)
            if self.use_kq_norm():                    # Llama 4's "L2Norm" divides by the RMS (mean of squares)
                q, k = _rms(q, None, self.eps), _rms(k, None, self.eps)
        else:
            t = self.attn_temp(L, pos)
            if t is not None:
                q = q * t
        o = self.attend(L, q.astype(np.float32), k.astype(np.float32), v, pos, st, rec)
        return self.lin(o.reshape(h.shape[0], -1), L, "attn_output"), o


class Glm4(Model):
    """GLM-4 (0414) / GLM-Z1: norms before and after attention and MLP, fused gate+up MLP, rope on part of each head.
    glm4moe (GLM-4.5 / 4.6 / Air): sigmoid router with bias, dense first layers, shared expert. Their extra
    'next-token prediction' layer at the end (nextn_predict_layers) is skipped, as llama.cpp does.
    Rope: adjacent pairs for GLM-4, NEOX halves for GLM-4.5 (checked against llama.cpp)."""

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        self.n_layer -= int(self.g("nextn_predict_layers") or 0)
        self.rope_style = "neox" if arch == "glm4moe" else "norm"


class Lfm(Model):
    """Liquid LFM2 / LFM2.5 / LFM3: most layers are a gated short convolution instead of attention
    (in_proj → B, C, x; y = C · causal_conv(B · x); out_proj), the rest are grouped-query attention with per-head
    q/k norm. LFM3 adds a sigmoid gate per head (attn_gate), norms after attention and MLP, sliding-window
    layers with their own rope base, and rope frequency factors."""
    rope_style = "neox"

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        self.l_cache = int(self.g("shortconv.l_cache") or 3)
        if self.out_norm is None:                                      # LFM2 stores its final norm as token_embd_norm
            self.out_norm = T.get("token_embd_norm.weight")

    def is_attention(self, L):
        return self.w(L, "attn_q.weight") is not None

    def rope_factors(self, L):
        return None if self.swa_l[L] else self.freq_factors        # LFM3: factors only on the global layers

    def kinds(self):
        return [("sliding attention" if self.swa_l[L] else "attention") if self.is_attention(L) else "short conv"
                for L in range(self.n_layer)]

    def n_heads(self, L):
        return self.n_head_l[L] if self.is_attention(L) else 1

    def head_dim(self, L):
        return self.vd if self.is_attention(L) else self.D

    def out_proj(self, L):
        return self.w(L, "attn_output.weight") if self.is_attention(L) else self.w(L, "shortconv.out_proj.weight")

    def head_layout(self, L):
        if self.is_attention(L):
            return super().head_layout(L)
        D = self.D
        return {"kind": "short conv", "out": f"blk.{L}.shortconv.out_proj.weight", "out_cols": D,
                "reads": [[f"blk.{L}.shortconv.in_proj.weight", "rows", D, 1, "B (input gate)", 0],
                          [f"blk.{L}.shortconv.in_proj.weight", "rows", D, 1, "C (output gate)", D],
                          [f"blk.{L}.shortconv.in_proj.weight", "rows", D, 1, "x (value)", 2 * D]]}

    def mixer(self, L, h, pos, st, rec):
        if not self.is_attention(L):
            return self.shortconv(L, h, st)
        q, k, v, _ = self.qkv(L, h)
        q, k = self.rope(q, pos, L), self.rope(k, pos, L)
        o = self.attend(L, q.astype(np.float32), k.astype(np.float32), v, pos, st, rec)
        g = self.w(L, "attn_gate.weight")
        if g is not None:                                              # LFM3: one sigmoid gate per head
            o = o * _sigmoid(h @ g.T)[:, :, None]
        return self.lin(o.reshape(h.shape[0], -1), L, "attn_output"), o

    def shortconv(self, L, h, st):
        n, D, K = h.shape[0], self.D, self.l_cache
        bcx = h @ self.w(L, "shortconv.in_proj.weight").T
        b, c, x = bcx[:, :D], bcx[:, D:2 * D], bcx[:, 2 * D:]
        bx = b * x
        S = st.L[L]
        xin = np.concatenate([S.get("conv", np.zeros((K - 1, D), np.float32)), bx])
        S["conv"] = xin[-(K - 1):]
        wc = self.w(L, "shortconv.conv.weight")                        # [D, K]
        conv = sum(xin[j:j + n] * wc[:, j] for j in range(K))
        y = (c * conv).astype(np.float32)
        return y @ self.w(L, "shortconv.out_proj.weight").T, y


def _gated_delta(q, k, v, beta, g, M0, C=64):
    """Gated delta rule, chunk by chunk (exact; the same as stepping token by token):
        M_t = g_t M_{t-1} + k_t δ_tᵀ,   δ_t = β_t (v_t − k_t·(g_t M_{t-1})),   out_t = q_t·M_t       (per head)
    Within a chunk, with γ_t = g_1…g_t:  g_t M_{t-1} = γ_t M0 + Σ_{s<t} (γ_t/γ_s) k_s δ_sᵀ, so the δ solve one unit
    lower-triangular system  (I + A) Δ = β (V − γ K M0),  A_ts = β_t (γ_t/γ_s) k_t·k_s (s<t),  and
        out = γ Q M0 + (R ∘ Q Kᵀ) Δ,   M_C = γ_C M0 + Kᵀ diag(γ_C/γ) Δ,   R_ts = γ_t/γ_s (s<=t).
    q, k: [n, H, dk], v: [n, H, dv], beta, g: [n, H], M0: [H, dk, dv]. Returns (out [n, H, dv], new state)."""
    n, H, dk = q.shape
    dv = v.shape[-1]
    out = np.empty((n, H, dv), dtype=np.float32)
    M = M0.astype(np.float64)
    logg = np.log(np.maximum(g.astype(np.float64), 1e-300))
    for c0 in range(0, n, C):
        sl = slice(c0, min(n, c0 + C))
        m = sl.stop - sl.start
        Q, K, V = (a[sl].transpose(1, 0, 2).astype(np.float64) for a in (q, k, v))     # [H, m, d]
        B = beta[sl].T.astype(np.float64)                                              # [H, m]
        lg = np.cumsum(logg[sl].T, axis=1)                                             # log γ_t, [H, m]
        gam = np.exp(lg)
        low = np.tril(np.ones((m, m), dtype=bool))
        Rt = np.exp(np.where(low, lg[:, :, None] - lg[:, None, :], -np.inf))           # γ_t/γ_s for s <= t, else 0
        strict = Rt * (~np.eye(m, dtype=bool))
        A = B[:, :, None] * strict * (K @ K.transpose(0, 2, 1))
        rhs = B[:, :, None] * (V - gam[:, :, None] * (K @ M))
        D = np.linalg.solve(np.eye(m) + A, rhs)                                        # Δ, [H, m, dv]
        O = gam[:, :, None] * (Q @ M) + (Rt * (Q @ K.transpose(0, 2, 1))) @ D
        out[sl] = O.transpose(1, 0, 2)
        w = np.exp(lg[:, -1:] - lg)                                                    # γ_C/γ_s
        M = gam[:, -1][:, None, None] * M + K.transpose(0, 2, 1) @ (w[:, :, None] * D)
    return out, M.astype(np.float32)


def _softplus_ggml(x):
    return np.where(x <= 20.0, np.log1p(np.exp(np.minimum(x, 20.0))), x).astype(np.float32)


class SSM(Model):
    """State-space (Mamba) layers, alone or mixed with attention:
      mamba   Mamba 1, Falcon-Mamba: in_proj → (x, z); causal conv + SiLU on x; x_proj → (dt, B, C);
              per channel a state h ← h·exp(dt·A) + dt·B·x, y = h·C + D·x; y·SiLU(z) → out_proj. No MLP.
      mamba2  Mamba 2: in_proj → (z, xBC, dt); conv + SiLU; heads share B/C per group; one decay A per head;
              y·SiLU(z) then RMS norm per group → out_proj.
    Hybrids use these classes too: jamba (Mamba 1 + attention without rope + MoE), granitehybrid (Granite 4:
    Mamba 2 + attention + MoE, Granite multipliers), nemotron_h (Mamba 2 / attention / MLP-only layers)."""
    attn_rope = True

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        g = self.g
        self.d_conv = int(g("ssm.conv_kernel") or 4)
        self.d_inner = int(g("ssm.inner_size") or 2 * self.D)
        self.d_state = int(g("ssm.state_size") or 16)
        self.dt_rank = int(g("ssm.time_step_rank") or 0)
        self.n_group = int(g("ssm.group_count") or 1)
        self.dt_bc_rms = bool(g("ssm.dt_b_c_rms"))

    def is_ssm(self, L):
        return self.w(L, "ssm_in.weight") is not None

    def is_mamba2(self, L):
        return self.is_ssm(L) and self.w(L, "ssm_x.weight") is None       # Mamba 2 has no x_proj

    def is_attention(self, L):
        return self.w(L, "attn_q.weight") is not None or self.w(L, "attn_qkv.weight") is not None

    def kinds(self):
        return [("mamba 2" if self.is_mamba2(L) else "mamba") if self.is_ssm(L) else
                ("attention" if self.is_attention(L) else "mlp only") for L in range(self.n_layer)]

    def n_heads(self, L):
        if self.is_ssm(L):
            return self.w(L, "ssm_a").shape[0] if self.is_mamba2(L) else 1
        return self.n_head_l[L] if self.is_attention(L) else 1

    def head_dim(self, L):
        if self.is_ssm(L):
            return self.d_inner // self.n_heads(L)
        return self.vd if self.is_attention(L) else self.D

    def out_proj(self, L):
        if self.is_ssm(L):
            return self.w(L, "ssm_out.weight")
        return self.w(L, "attn_output.weight")

    def head_layout(self, L):
        if not self.is_ssm(L):
            return super().head_layout(L) if self.is_attention(L) else {}
        H, hd = self.n_heads(L), self.head_dim(L)
        return {"kind": self.kinds()[L], "out": f"blk.{L}.ssm_out.weight", "out_cols": hd,
                "reads": [[f"blk.{L}.ssm_in.weight", "rows", hd, 1, "input projection (this head's x)",
                           self.d_inner if self.is_mamba2(L) else 0]]}

    def uses_rope(self, L):
        return self.attn_rope

    def mixer(self, L, h, pos, st, rec):
        if self.is_ssm(L):
            return self.mamba2(L, h, st) if self.is_mamba2(L) else self.mamba1(L, h, st)
        if not self.is_attention(L):                                   # Nemotron-H MLP-only layer
            return np.zeros_like(h), np.zeros((h.shape[0], 1, self.D), np.float32)
        return super().mixer(L, h, pos, st, rec)

    def _conv(self, L, x, st):
        """Causal depthwise convolution over time with the state of the last d_conv-1 inputs, + bias, SiLU."""
        n, K = x.shape[0], self.d_conv
        S = st.L[L]
        xin = np.concatenate([S.get("conv", np.zeros((K - 1, x.shape[1]), np.float32)), x])
        S["conv"] = xin[-(K - 1):]
        wc = self.w(L, "ssm_conv1d.weight")                             # [channels, K]
        y = sum(xin[j:j + n] * wc[:, j] for j in range(K))
        b = self.w(L, "ssm_conv1d.bias")
        return _silu(y + b if b is not None else y).astype(np.float32)

    def mamba1(self, L, h, st):
        n, di, ds = h.shape[0], self.d_inner, self.d_state
        xz = h @ self.w(L, "ssm_in.weight").T
        x, z = xz[:, :di], xz[:, di:]
        x = self._conv(L, x, st)
        xdb = x @ self.w(L, "ssm_x.weight").T
        r = xdb.shape[1] - 2 * ds
        dt, B, C = xdb[:, :r], xdb[:, r:r + ds], xdb[:, r + ds:]
        for nm, val in (("ssm_dt_norm", "dt"), ("ssm_b_norm", "B"), ("ssm_c_norm", "C")):
            wn = self.w(L, nm + ".weight")                              # Jamba: weighted RMS norms on dt, B, C
            if wn is not None or self.dt_bc_rms:                         # Falcon-Mamba: the same without weights
                v = {"dt": dt, "B": B, "C": C}[val]
                v = _rms(v, wn, self.eps)
                dt, B, C = (v, B, C) if val == "dt" else (dt, v, C) if val == "B" else (dt, B, v)
        dt = _softplus_ggml(dt @ self.w(L, "ssm_dt.weight").T + self.w(L, "ssm_dt.bias"))
        A, Dk = self.w(L, "ssm_a"), self.w(L, "ssm_d")                   # A: [di, ds] (negative)
        S = st.L[L]
        s = S.get("ssm", np.zeros((di, ds), np.float32))
        y = np.empty((n, di), np.float32)
        for t in range(n):
            s = s * np.exp(dt[t][:, None] * A) + (x[t] * dt[t])[:, None] * B[t][None, :]
            y[t] = s @ C[t]
        S["ssm"] = s
        y = (y + x * Dk) * _silu(z)
        return y @ self.w(L, "ssm_out.weight").T, y.reshape(n, 1, di)

    def mamba2(self, L, h, st):
        n, di, ds, G = h.shape[0], self.d_inner, self.d_state, self.n_group
        A, Dk = self.w(L, "ssm_a").reshape(-1), self.w(L, "ssm_d").reshape(-1)   # one decay and one skip per head
        H = A.shape[0]
        hd = di // H
        zxbcdt = h @ self.w(L, "ssm_in.weight").T
        z, xbc, dt = zxbcdt[:, :di], zxbcdt[:, di:di + di + 2 * G * ds], zxbcdt[:, di + di + 2 * G * ds:]
        xbc = self._conv(L, xbc, st)
        x = xbc[:, :di].reshape(n, H, hd)
        B = xbc[:, di:di + G * ds].reshape(n, G, ds)
        C = xbc[:, di + G * ds:].reshape(n, G, ds)
        dt = _softplus_ggml(dt + self.w(L, "ssm_dt.bias"))               # [n, H]
        grp = np.arange(H) // (H // G)
        S = st.L[L]
        s = S.get("ssm", np.zeros((H, hd, ds), np.float32))
        y = np.empty((n, H, hd), np.float32)
        for t in range(n):
            dA = np.exp(dt[t] * A)                                       # [H]
            s = s * dA[:, None, None] + (x[t] * dt[t][:, None])[:, :, None] * B[t][grp][:, None, :]
            y[t] = np.einsum("hds,hs->hd", s, C[t][grp])
        S["ssm"] = s
        y = y + x * Dk[:, None]
        y = (y.reshape(n, di) * _silu(z)).reshape(n, G, di // G)
        y = _rms(y, self.w(L, "ssm_norm.weight").reshape(G, di // G), self.eps).reshape(n, di).astype(np.float32)
        return y @ self.w(L, "ssm_out.weight").T, y.reshape(n, H, hd)


class Jamba(SSM):
    """AI21 Jamba: Mamba 1 layers with RMS norms on dt/B/C, attention layers without any position encoding,
    dense or MoE MLP in every layer."""
    attn_rope = False


class GraniteHybrid(SSM):
    """IBM Granite 4.0 hybrids (H-Tiny, H-Small, H-Micro…): Mamba 2 + attention, MoE with a shared expert,
    Granite's embedding / residual / attention / logit multipliers."""
    moe_norm = True


class NemotronH(SSM):
    """NVIDIA Nemotron-H / Nemotron Nano 2: each layer is ONE of Mamba 2, attention, or an MLP (squared-ReLU),
    each reading the layer's single norm (attn_norm). Its attention has no rope (the Mamba layers carry position)."""
    act = "relu2"
    attn_rope = False

    def block(self, L, x, pos, st, rec):
        h = self.norm(x, f"blk.{L}.attn_norm.weight")
        one = np.ones(x.shape[0], np.float32)
        if self.is_ssm(L) or self.is_attention(L):
            a, hv = self.mixer(L, h, pos, st, rec)
            return x + a, np.zeros((x.shape[0], 1), np.float32), hv, a, np.zeros_like(x), one, one
        act, m = self.ffn(L, h, rec)
        return x + m, act, np.zeros((x.shape[0], 1, self.D), np.float32), np.zeros_like(x), m, one, one


class FalconH1(SSM):
    """TII Falcon-H1: in every layer attention and Mamba 2 read the same normed input side by side and both
    outputs are added; then the MLP. (Its µP multipliers are folded into the weights when the GGUF is made.)
    Heads listed for attribution are the attention heads; the Mamba 2 part counts in the layer's total."""
    rope_style = "neox"

    def kinds(self):
        return ["attention + mamba 2" for _ in range(self.n_layer)]

    def pre_ffn_norm(self, L):
        return f"blk.{L}.ffn_norm" if f"blk.{L}.ffn_norm" in self.T else super().pre_ffn_norm(L)

    def n_heads(self, L):
        return self.n_head_l[L]

    def head_dim(self, L):
        return self.vd

    def out_proj(self, L):
        return self.w(L, "attn_output.weight")

    def head_layout(self, L):
        return Model.head_layout(self, L)

    def mixer(self, L, h, pos, st, rec):
        a, hv = Model.mixer(self, L, h, pos, st, rec)
        s, _ = self.mamba2(L, h, st)
        return a + s, hv


class Rwkv6(Model):
    """RWKV-6 (Finch): no attention. Each layer: time mix (a recurrent memory per head, fed by a token-shifted,
    data-dependent mix of the current and previous token) then channel mix (squared-ReLU MLP with a sigmoid
    'receptance' gate), both after LayerNorm. Embeddings get a LayerNorm first."""
    norm_kind = "layer"

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        self.hs = int(self.g("wkv.head_size") or 64)
        self.H = self.D // self.hs
        self.rescale = int(self.g("rescale_every_n_layers") or 0)
        self.eps = float(self.g("attention.layer_norm_epsilon") or 1e-5)

    def kinds(self):
        return ["time mix" for _ in range(self.n_layer)]

    def n_heads(self, L):
        return self.H

    def head_dim(self, L):
        return self.hs

    def out_proj(self, L):
        return self.w(L, "time_mix_output.weight")

    def head_layout(self, L):
        return {"kind": "time mix", "out": f"blk.{L}.time_mix_output.weight", "out_cols": self.hs,
                "reads": [[f"blk.{L}.time_mix_receptance.weight", "rows", self.hs, 1, "receptance"],
                          [f"blk.{L}.time_mix_key.weight", "rows", self.hs, 1, "key"],
                          [f"blk.{L}.time_mix_value.weight", "rows", self.hs, 1, "value"]]}

    def embed(self, ids):
        x = super().embed(ids)
        return self.norm(x, "token_embd_norm.weight")

    def _shift(self, L, key, cur, st):
        S = st.L[L]
        prev = np.concatenate([S.get(key, np.zeros((1, self.D), np.float32)), cur[:-1]])
        S[key] = cur[-1:]
        return prev - cur

    def time_mix(self, L, cur, st):
        n, H, hs = cur.shape[0], self.H, self.hs
        w = lambda nm: self.w(L, nm)
        sx = self._shift(L, "tm", cur, st)
        xxx = cur + sx * w("time_mix_lerp_x.weight").reshape(-1)
        w1, w2 = w("time_mix_w1.weight"), w("time_mix_w2.weight")         # [5*extra, D], [5, D, extra]
        extra = w1.shape[0] // 5
        t = np.tanh(xxx @ w1.T).reshape(n, 5, extra)
        mix = np.einsum("nke,kde->nkd", t, w2)                                # [n, 5, D]
        lerp = w("time_mix_lerp_fused.weight").reshape(5, self.D)
        xw, xk, xv, xr, xg = [cur + sx * (lerp[i] + mix[:, i]) for i in range(5)]
        r = (xr @ w("time_mix_receptance.weight").T).reshape(n, H, hs)
        k = (xk @ w("time_mix_key.weight").T).reshape(n, H, hs)
        v = (xv @ w("time_mix_value.weight").T).reshape(n, H, hs)
        g = _silu(xg @ w("time_mix_gate.weight").T)
        dw = w("time_mix_decay.weight").reshape(-1) + np.tanh(xw @ w("time_mix_decay_w1.weight").T) @ w("time_mix_decay_w2.weight").T
        decay = np.exp(-np.exp(dw)).reshape(n, H, hs)
        u = w("time_mix_first.weight").reshape(H, hs)
        S = st.L[L]
        s = S.get("wkv", np.zeros((H, hs, hs), np.float32))
        out = np.empty((n, H, hs), np.float32)
        for i in range(n):
            kv = k[i][:, :, None] * v[i][:, None, :]                           # [H, hs_k, hs_v]
            out[i] = np.einsum("hk,hkv->hv", r[i], u[:, :, None] * kv + s)
            s = s * decay[i][:, :, None] + kv
        S["wkv"] = s
        o = out.reshape(n, H, hs)
        mu = o.mean(-1, keepdims=True)
        var = ((o - mu) ** 2).mean(-1, keepdims=True)
        o = ((o - mu) / np.sqrt(var + 64e-5)).reshape(n, self.D)               # group norm, one group per head
        o = o * w("time_mix_ln.weight").reshape(-1) + w("time_mix_ln.bias").reshape(-1)
        o = (o * g).astype(np.float32)
        return o @ w("time_mix_output.weight").T, o.reshape(n, H, hs)

    def channel_mix(self, L, cur, st):
        w = lambda nm: self.w(L, nm)
        sx = self._shift(L, "cm", cur, st)
        xk = cur + sx * w("channel_mix_lerp_k.weight").reshape(-1)
        xr = cur + sx * w("channel_mix_lerp_r.weight").reshape(-1)
        r = _sigmoid(xr @ w("channel_mix_receptance.weight").T)
        act = np.square(np.maximum(xk @ w("channel_mix_key.weight").T, 0))
        return act, r * (act @ w("channel_mix_value.weight").T), r

    def block(self, L, x, pos, st, rec):
        one = np.ones(x.shape[0], np.float32)
        a, hv = self.time_mix(L, self.norm(x, f"blk.{L}.attn_norm.weight"), st)
        x = x + a
        act, m, r = self.channel_mix(L, self.norm(x, f"blk.{L}.attn_norm_2.weight"), st)
        x = x + m
        rec.add(f"cmr{L}", r.astype(np.float16))                          # receptance gate, for attribution
        if self.rescale and (L + 1) % self.rescale == 0:
            x = x * np.float32(0.5)                                        # llama.cpp halves the stream here
        f = np.float32(self.later_factor(L))
        return x, act, hv, a * f, m * f, one, one                          # what each part adds to the FINAL stream

    def later_factor(self, L):
        """How much the halvings at and after layer L scale what layer L adds."""
        if not self.rescale:
            return 1.0
        return 0.5 ** sum(1 for p in range(L, self.n_layer) if (p + 1) % self.rescale == 0)

    @property
    def embed_factor(self):
        return self.later_factor(0)

    def mlp_down(self, L):
        return self.w(L, "channel_mix_value.weight")

    def mlp_direction(self, L, dm, A, pos):
        return dm * self.later_factor(L) * A[f"cmr{L}"][pos].astype(np.float32)

    def mixer_direction(self, L, da, A, pos):
        return da * self.later_factor(L)


class Rwkv7(Rwkv6):
    """RWKV-7 (Goose): the memory update becomes S ← S·diag(w) + v·kᵀ + (S·a)·bᵀ with a = −κ̂, b = κ̂·iclr (an
    in-context learning rate), low-rank (LoRA) paths make w, the learning rate a, the gate g and a mix of each
    layer's value with the FIRST layer's value; a bonus term (r·k·r_k)·v is added; the channel mix has no gate."""

    def time_mix(self, L, cur, st):
        n, H, hs, D = cur.shape[0], self.H, self.hs, self.D
        w_ = lambda nm: self.w(L, nm)
        flat = lambda nm: w_(nm).reshape(-1)
        sx = self._shift(L, "tm", cur, st)
        lerp = w_("time_mix_lerp_fused.weight").reshape(-1, D)
        xs = [cur + sx * lerp[i] for i in range(lerp.shape[0])]
        xr, xw, xk, xv, xa = xs[:5]
        xg = xs[5] if len(xs) > 5 else None
        r = xr @ w_("time_mix_receptance.weight").T
        wv = np.tanh(xw @ w_("time_mix_w1.weight").T) @ w_("time_mix_w2.weight").T + flat("time_mix_w0.weight")
        decay = np.exp(np.float32(-0.606531) * _sigmoid(wv))
        k = xk @ w_("time_mix_key.weight").T
        v = xv @ w_("time_mix_value.weight").T
        S = st.L[0]
        if L == 0:
            st.v_first = v
        else:
            mix = _sigmoid((xv @ w_("time_mix_v1.weight").T) @ w_("time_mix_v2.weight").T + flat("time_mix_v0.weight"))
            v = v + (st.v_first - v) * mix
        g = None
        if w_("time_mix_g1.weight") is not None:
            g = _sigmoid(xg @ w_("time_mix_g1.weight").T) @ w_("time_mix_g2.weight").T
        a = _sigmoid((xa @ w_("time_mix_a1.weight").T) @ w_("time_mix_a2.weight").T + flat("time_mix_a0.weight"))
        kk = (k * flat("time_mix_k_k.weight")).reshape(n, H, hs)
        kk = kk / np.maximum(np.sqrt(np.sum(kk * kk, -1, keepdims=True)), np.float32(1e-12))
        ka = k * flat("time_mix_k_a.weight")
        k = k + a * ka - ka
        r, decay, k, v, a = (t.reshape(n, H, hs).astype(np.float32) for t in (r, decay, k, v, a))
        bb = kk * a
        aa = -kk
        Sl = st.L[L]
        s = Sl.get("wkv", np.zeros((H, hs, hs), np.float32))                 # [head, value dim i, key dim j]
        out = np.empty((n, H, hs), np.float32)
        for t in range(n):
            sa = np.einsum("hij,hj->hi", s, aa[t])
            s = s * decay[t][:, None, :] + v[t][:, :, None] * k[t][:, None, :] + sa[:, :, None] * bb[t][:, None, :]
            out[t] = np.einsum("hij,hj->hi", s, r[t])
        Sl["wkv"] = s
        o = out
        if w_("time_mix_ln.weight") is not None:
            mu = o.mean(-1, keepdims=True)
            var = ((o - mu) ** 2).mean(-1, keepdims=True)
            o = ((o - mu) / np.sqrt(var + 64e-5)).reshape(n, D) * flat("time_mix_ln.weight") + flat("time_mix_ln.bias")
            o = o.reshape(n, H, hs)
        rk = np.sum(k * r * w_("time_mix_r_k.weight").reshape(H, hs), -1, keepdims=True)
        o = (o + v * rk).reshape(n, D)
        if g is not None:
            o = o * g
        o = o.astype(np.float32)
        return o @ w_("time_mix_output.weight").T, o.reshape(n, H, hs)

    def channel_mix(self, L, cur, st):
        w_ = lambda nm: self.w(L, nm)
        sx = self._shift(L, "cm", cur, st)
        xk = cur + sx * w_("channel_mix_lerp_k.weight").reshape(-1)
        act = np.square(np.maximum(xk @ w_("channel_mix_key.weight").T, 0))
        return act, act @ w_("channel_mix_value.weight").T, np.ones((cur.shape[0], self.D), np.float32)


class Gemma3n(Model):
    """Gemma 3n (E2B / E4B), following llama.cpp's gemma3n graph:
      - AltUp: the hidden state is 4 parallel streams; each layer PREDICTS all streams from a learned mix (router
        = tanh of a normed projection), runs attention + MLP on the active stream only, then CORRECTS every stream
        with the 'innovation' (what the layer added) times learned coefficients;
      - LAuReL: a low-rank residual (laurel_l → laurel_r → norm) added next to attention, (attn + laurel)/√2;
      - per-layer embeddings: a second embedding table gives each layer its own small input, gated into stream 1-3;
      - activation sparsity: in the first 10 layers MLP gates below mean + 1.645·std are cut to zero;
      - layers from 20 on reuse the key/value cache of layer 18 (sliding) or 19 (global);
      - q/k RMS norm with weights, v RMS norm without, attention scale 1, final soft-cap.
    Attribution is approximate here: the 4 streams are merged at the end with a magnitude normalization, so the
    per-layer pushes of the active stream don't add up exactly to the score."""
    rope_style, act, emb_scale_sqrt = "neox", "gelu", True
    attribution_exact = False
    KV_FROM_START, N_SPARSE, SPARSE_STD = 20, 10, 1.6448533535003662

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        self.n_altup = int(self.g("altup.num_inputs") or 4)
        self.i_act = int(self.g("altup.active_idx") or 0)
        self.dpl = int(self.g("embedding_length_per_layer_input") or 256)
        self.attn_scale = 1.0
        self.softcap_final = float(self.g("final_logit_softcapping") or 30.0)

    def _swa_layers(self):
        pat = self.g("attention.sliding_window_pattern")
        if isinstance(pat, list):
            return [bool(x) for x in pat] + [False] * (self.n_layer - len(pat))
        return [L % 5 < 4 for L in range(self.n_layer)]

    def has_kv(self, L):
        return L < self.KV_FROM_START

    def kv_source(self, L):
        return L if self.has_kv(L) else self.KV_FROM_START - (2 if self.swa_l[L] else 1)

    def kinds(self):
        return [("sliding attention" if self.swa_l[L] else "attention") + ("" if self.has_kv(L) else " (shared KV)")
                for L in range(self.n_layer)]

    def _rows(self, name, ids):
        T = self.T
        if isinstance(T, LazyWeights) and name not in T.pinned:
            t = T.ot[name]
            return T._decode(t, t.data[np.asarray(ids)]).reshape(len(ids), -1)
        return T[name][np.asarray(ids)]

    def _router(self, L, x):
        r = _rms(x, self.w(L, "altup_router_norm.weight"), self.eps) * np.float32(1.0 / self.D)
        return np.tanh(r @ self.w(L, "altup_router.weight").T)                       # [n, A]

    @staticmethod
    def _match_mag(y, target):
        return y * (target / np.sqrt(np.sum(y * y, -1, keepdims=True)))

    def attend_shared(self, L, q, pos, rec):
        """Attention of a layer without its own KV: queries against the source layer's cache."""
        S = self._st.L[self.kv_source(L)]
        H, KV = q.shape[1], S["K"].shape[1]
        r = H // KV
        Kr, Vr = np.repeat(S["K"], r, axis=1), np.repeat(S["V"], r, axis=1)
        s = np.matmul(q.transpose(1, 0, 2), Kr.transpose(1, 2, 0)) * np.float32(self.attn_scale)
        kp, qp = S["P"][None, :], np.asarray(pos)[:, None]
        mask = kp > qp
        if self.swa_l[L] and self.swa:
            mask |= (qp - kp) >= self.swa
        p = _softmax_rows(np.where(mask[None], -np.inf, s))
        if not getattr(rec, "null", False):
            _record_attention(rec, L, p)
        return np.matmul(p, Vr.transpose(1, 0, 2)).transpose(1, 0, 2).astype(np.float32)

    def attention(self, L, h, pos, st, rec):
        n, H, KV, hd = h.shape[0], self.n_head_l[L], self.n_kv_l[L], self.hd
        q = _rms(self.lin(h, L, "attn_q").reshape(n, H, hd), self.w(L, "attn_q_norm.weight"), self.eps)
        q = self.rope(q, pos, L)
        if self.has_kv(L):
            k = _rms(self.lin(h, L, "attn_k").reshape(n, KV, hd), self.w(L, "attn_k_norm.weight"), self.eps)
            v = _rms(self.lin(h, L, "attn_v").reshape(n, KV, self.vd), None, self.eps)
            k = self.rope(k, pos, L)
            o = self.attend(L, q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), pos, st, rec)
        else:
            o = self.attend_shared(L, q.astype(np.float32), pos, rec)
        return self.lin(o.reshape(n, -1), L, "attn_output"), o

    def forward(self, ids, st, rec):
        n, D, A, a = len(ids), self.D, self.n_altup, self.i_act
        self._st = st
        pos = np.arange(st.pos, st.pos + n)
        x0 = self.embed(ids)
        Lc = self.n_layer
        ple = self._rows("per_layer_token_embd.weight", ids).reshape(n, Lc, self.dpl) * np.float32(math.sqrt(self.dpl))
        proj = (x0 @ self.T["per_layer_model_proj.weight"].T * np.float32(1.0 / math.sqrt(D))).reshape(n, Lc, self.dpl)
        proj = _rms(proj, self.T["per_layer_proj_norm.weight"], self.eps)
        ple = ((proj + ple) * np.float32(1.0 / math.sqrt(2.0))).astype(np.float32)
        mag = np.sqrt(np.sum(x0 * x0, -1, keepdims=True))
        ap = self.T["altup_proj.weight"]
        X = np.stack([x0] + [self._match_mag(x0 @ ap[i].T, mag) for i in range(A - 1)]).astype(np.float32)   # [A, n, D]
        resid = [X[a].astype(np.float16)]
        acts, heads, a_norm, m_norm, a_vec, m_vec = [], [], [], [], [], []
        one = np.ones(n, np.float32)
        for L in range(Lc):
            w = lambda nm: self.w(L, nm)
            # AltUp predict
            mod = self._router(L, X[a])
            coefs = (mod @ w("altup_predict_coef.weight").T).reshape(n, A, A)              # [n, j, k]
            pred = np.einsum("njk,knd->jnd", coefs, X) + X
            active = pred[a]
            h = _rms(active, w("attn_norm.weight"), self.eps)
            lau = _rms((h @ w("laurel_l.weight").T) @ w("laurel_r.weight").T, w("laurel_post_norm.weight"), self.eps) + h
            att, hv = self.attention(L, h, pos, st, rec)
            att = _rms(att, w("post_attention_norm.weight"), self.eps)
            attn_gated = att + active
            attn_laurel = (attn_gated + lau) * np.float32(1.0 / math.sqrt(2.0))
            h2 = _rms(attn_laurel, w("ffn_norm.weight"), self.eps)
            up, gate = h2 @ w("ffn_up.weight").T, h2 @ w("ffn_gate.weight").T
            if L < self.N_SPARSE:                                                        # activation sparsity
                mu = gate.mean(-1, keepdims=True)
                sd = np.sqrt(np.sum((gate - mu) ** 2, -1, keepdims=True) / (gate.shape[-1] - 1))
                gate = np.maximum(gate - (mu + sd * np.float32(self.SPARSE_STD)), 0)
            act = _gelu(gate) * up
            m = _rms(act @ w("ffn_down.weight").T, w("post_ffw_norm.weight"), self.eps)
            afl = m + attn_laurel
            # AltUp correct
            cmod = self._router(L, afl)
            cco = cmod @ w("altup_correct_coef.weight").T + 1.0                           # [n, A]
            innov = afl - pred[a]
            corr = innov[None] * cco.T[:, :, None] + pred
            first = _gelu((corr[a] * w("altup_correct_scale.weight")) @ w("inp_gate.weight").T) * ple[:, L]
            first = _rms(first @ w("proj.weight").T, w("post_norm.weight"), self.eps)
            corr[1:] += first
            X = corr.astype(np.float32)
            resid.append(X[a].astype(np.float16))
            acts.append(act.astype(np.float16))
            heads.append(hv.reshape(n, -1).astype(np.float16))
            a_vec.append(att.astype(np.float16))
            m_vec.append(m.astype(np.float16))
            a_norm.append(np.linalg.norm(att, axis=-1))
            m_norm.append(np.linalg.norm(m, axis=-1))
        # merge the streams back into one
        tmag = np.sqrt(np.sum(X[a] * X[a], -1, keepdims=True))
        un = self.T["altup_unembd_proj.weight"]
        cur = X[0] + sum(self._match_mag(X[1 + i] @ un[i].T, tmag) for i in range(A - 1))
        cur = (cur * np.float32(1.0 / A)).astype(np.float32)
        resid[-1] = cur.astype(np.float16)
        st.pos += n
        rec.add("resid", np.stack(resid, 1))
        rec.add("act", np.stack(acts, 1))
        rec.add("heads", np.stack(heads, 1))
        rec.add("a_vec", np.stack(a_vec, 1))
        rec.add("m_vec", np.stack(m_vec, 1))
        rec.add("a_div", np.stack([one] * Lc, 1))
        rec.add("m_div", np.stack([one] * Lc, 1))
        rec.add("attn_norm", np.stack(a_norm, 1).astype(np.float32))
        rec.add("mlp_norm", np.stack(m_norm, 1).astype(np.float32))
        rec.add("final_rms", self.final_scale(cur).astype(np.float32))
        return self.final(cur)


class Qwen35(Model):
    """Qwen3.5 / Qwen3-Next: every 4th layer is gated full attention, the others are Gated DeltaNet
    (linear attention with a recurrent state). Dense MLP."""
    rope_style = "neox"

    def __init__(self, T, meta, arch):
        super().__init__(T, meta, arch)
        g = self.g
        self.conv_k = int(g("ssm.conv_kernel", 4))
        self.dk = int(g("ssm.state_size"))
        self.nk = int(g("ssm.group_count"))
        self.nv = int(g("ssm.time_step_rank"))
        self.inner = int(g("ssm.inner_size"))
        self.dv = self.inner // self.nv
        self.full = [self.w(L, "attn_q.weight") is not None for L in range(self.n_layer)]

    def kinds(self):
        return ["full attention" if f else "linear attention" for f in self.full]

    def n_heads(self, L):
        return self.n_head_l[L] if self.full[L] else self.nv

    def head_dim(self, L):
        return self.vd if self.full[L] else self.dv

    def out_proj(self, L):
        return self.w(L, "attn_output.weight") if self.full[L] else self.w(L, "ssm_out.weight")

    def head_layout(self, L):
        if self.full[L]:
            r = self.n_head_l[L] // self.n_kv_l[L]
            return {"kind": "full", "out": f"blk.{L}.attn_output.weight", "out_cols": self.vd,
                    "reads": [[f"blk.{L}.attn_q.weight", "rows", 2 * self.hd, 1, "query + gate"],
                              [f"blk.{L}.attn_k.weight", "rows", self.hd, r, "key (shared by %d heads)" % r],
                              [f"blk.{L}.attn_v.weight", "rows", self.vd, r, "value (shared by %d heads)" % r]]}
        r = self.nv // self.nk
        return {"kind": "linear", "out": f"blk.{L}.ssm_out.weight", "out_cols": self.dv,
                "reads": [[f"blk.{L}.attn_qkv.weight", "rows", self.dk, r, "query", 0],
                          [f"blk.{L}.attn_qkv.weight", "rows", self.dk, r, "key", self.nk * self.dk],
                          [f"blk.{L}.attn_qkv.weight", "rows", self.dv, 1, "value", 2 * self.nk * self.dk],
                          [f"blk.{L}.attn_gate.weight", "rows", self.dv, 1, "output gate (z)"]]}

    def qkv(self, L, h):
        n, H, KV, hd = h.shape[0], self.n_head_l[L], self.n_kv_l[L], self.hd
        qg = self.lin(h, L, "attn_q").reshape(n, H, 2 * hd)          # per head: [query | gate]
        q, gate = qg[..., :hd], qg[..., hd:]
        k = self.lin(h, L, "attn_k").reshape(n, KV, hd)
        v = self.lin(h, L, "attn_v").reshape(n, KV, self.vd)
        q = _rms(q, self.w(L, "attn_q_norm.weight"), self.eps)
        k = _rms(k, self.w(L, "attn_k_norm.weight"), self.eps)
        return q, k, v, _sigmoid(gate)

    def mixer(self, L, h, pos, st, rec):
        return super().mixer(L, h, pos, st, rec) if self.full[L] else self._linear(L, h, st)

    def _linear(self, L, h, st):
        n, K = h.shape[0], self.conv_k
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        qkv = h @ self.w(L, "attn_qkv.weight").T
        z = (h @ self.w(L, "attn_gate.weight").T).reshape(n, nv, dv)
        b = h @ self.w(L, "ssm_beta.weight").T
        a = h @ self.w(L, "ssm_alpha.weight").T
        S = st.L[L]
        tail = S.get("conv", np.zeros((K - 1, qkv.shape[1]), dtype=np.float32))
        xin = np.concatenate([tail, qkv])
        S["conv"] = xin[-(K - 1):]
        wc = self.w(L, "ssm_conv1d.weight")                                         # [C, K]
        conv = sum(xin[j:j + n] * wc[:, j] for j in range(K))
        conv = _silu(conv)
        q = conv[:, :nk * dk].reshape(n, nk, dk)
        k = conv[:, nk * dk:2 * nk * dk].reshape(n, nk, dk)
        v = conv[:, 2 * nk * dk:].reshape(n, nv, dv)
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + self.eps)
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + self.eps)
        q = q / math.sqrt(dk)
        if nv != nk:
            q, k = np.repeat(q, nv // nk, axis=1), np.repeat(k, nv // nk, axis=1)
        beta = _sigmoid(b)
        gdec = np.exp(np.logaddexp(0, a + self.w(L, "ssm_dt.bias")) * self.w(L, "ssm_a"))   # decay in (0, 1]
        M = S.get("S", np.zeros((nv, dk, dv), dtype=np.float32))
        out, M = _gated_delta(q, k, v, beta, gdec, M)
        S["S"] = M
        o = _rms(out, self.w(L, "ssm_norm.weight"), self.eps) * _silu(z)
        return o.reshape(n, nv * dv) @ self.w(L, "ssm_out.weight").T, o


# architecture name in the GGUF -> class. Aliases are families llama.cpp builds with the same graph.
ARCHS = {
    "llama": Llama, "granite": Granite, "granitemoe": Granite,
    "qwen2": Qwen2, "qwen3": Qwen2, "phi3": Phi3,
    "qwen2moe": Qwen2, "qwen3moe": Qwen3Moe, "olmoe": OLMoE,
    "deepseek2": DeepSeek2, "gpt-oss": GptOss, "glm4": Glm4, "glm4moe": Glm4,
    "lfm2": Lfm, "lfm3": Lfm,
    "mamba": SSM, "mamba2": SSM, "jamba": Jamba, "granitehybrid": GraniteHybrid, "nemotron_h": NemotronH,
    "falcon-h1": FalconH1, "mistral3": Mistral3, "rwkv6": Rwkv6, "rwkv7": Rwkv7,
    "llama4": Llama4,
    "gemma": Gemma, "gemma2": Gemma2, "gemma3": Gemma3, "gemma3n": Gemma3n,
    "command-r": CommandR, "starcoder2": StarCoder2, "olmo2": Olmo2,
    "qwen35": Qwen35, "qwen3next": Qwen35,
}


def build(T, meta):
    arch = meta.get("general.architecture")
    if arch not in ARCHS:
        raise ValueError(f"Chat tracing doesn't support the '{arch}' architecture yet "
                         f"(supported: {', '.join(sorted(ARCHS))}).")
    return ARCHS[arch](T, meta, arch)


def load_file(path, log=lambda m: None, lowmem_budget=None):
    """Model + tokenizer straight from a .gguf file (no workspace): used by the tests.
    lowmem_budget (bytes): use the low-memory mode with this cache size instead of loading everything."""
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    meta = {k: core.json_safe(v) for k, v in core.read_metadata(reader).items()}
    if lowmem_budget is not None:
        ot = {t.name: t for t in reader.tensors}
        T = LazyWeights(ot, list(ot), {n: _natural_shape(t) for n, t in ot.items()}, budget=lowmem_budget)
        T._reader = reader
    else:
        T = {t.name: np.ascontiguousarray(core.full_tensor(t), dtype=np.float32) for t in reader.tensors}
    return build(T, meta), make_tokenizer(meta), meta


# ============================================================== loaded model cache

_MODEL = {"key": None, "model": None, "tok": None, "meta": None}
_MLOCK = threading.Lock()


def _key(ws, which):
    d = core.workspace_dir(ws)
    return (ws, which, list(_ws_stamp(d)) if which == "edited" else core.load_info(d).get("file_mtime_ns"))


def status(ws):
    """Can this workspace be traced, how much memory would it need, and is it loaded?"""
    d = core.workspace_dir(ws)
    meta = core.load_user_metadata(d)
    arch = meta.get("general.architecture")
    n, active = _sizes(ws)
    ram = core.total_ram()
    reason = None
    lowmem = needs_lowmem(n)
    if arch not in ARCHS:
        reason = f"The '{arch}' architecture isn't supported by the tracer yet (supported: {', '.join(sorted(ARCHS))})."
    elif meta.get("tokenizer.ggml.model") not in TOKENIZERS:
        reason = (f"The tracer can't tokenize for this model yet: its tokenizer type is {meta.get('tokenizer.ggml.model')!r} "
                  f"(supported: {', '.join(sorted(TOKENIZERS))}).")
    elif meta.get("tokenizer.ggml.model") == "gpt2" and meta.get("tokenizer.ggml.pre") not in PRE_ALL:
        reason = (f"The tracer can't tokenize for this model yet: its splitting rule '{meta.get('tokenizer.ggml.pre')}' "
                  f"isn't implemented (supported: {', '.join(sorted(PRE_ALL))}).")
    loaded = None
    k = _MODEL["key"]
    if k and k[0] == ws:
        loaded = k[1] if k == _key(ws, k[1]) else None
    note = None
    if lowmem:
        sec = active / UNPACK_RATE
        note = (f"Big model ({n / 1e9:.1f}B weights; {active / 1e9:.1f}B used per token): it runs in low-memory mode. "
                f"The weights stay compressed in the file and are unpacked as each layer needs them, so expect "
                f"roughly {sec:.0f} s per token{' (fewer once parts are cached)' if n * 4 < 2 * 0.45 * (ram or 0) else ''}. "
                f"Fine for short questions; long chats take a while.")
    return {"supported": reason is None, "reason": reason, "architecture": arch,
            "need_gb": round(min(n * 4, 0.45 * (ram or 8e9)) / 1e9, 1) if lowmem else round(n * 4 / 1e9, 1),
            "ram_gb": round(ram / 1e9) if ram else None, "loaded": loaded, "lowmem": lowmem, "note": note,
            "params_b": round(n / 1e9, 2)}


FORCE_LOWMEM = False      # tests set this to run small models through the low-memory path


def needs_lowmem(n_params):
    ram = core.total_ram() or 8e9
    return FORCE_LOWMEM or n_params * 4 > 0.5 * ram


def ready_for(ws, tid):
    info, _ = _open_trace(ws, tid)
    return _MODEL["key"] == _key(ws, info["weights"])


def trace_weights(ws, tid):
    return _open_trace(ws, tid)[0]["weights"]


def get_model(ws, which="edited", log=print):
    key = _key(ws, which)
    with _MLOCK:
        if _MODEL["key"] == key:
            return _MODEL["model"], _MODEL["tok"]
        _MODEL.update(key=None, model=None, tok=None, meta=None)
        gc.collect()
        t0 = time.time()
        log(f"Loading the {'workspace (with your edits)' if which == 'edited' else 'original'} weights into memory…")
        n, active = _sizes(ws)
        lowmem = needs_lowmem(n)
        if lowmem:
            log(f"Low-memory mode: {n / 1e9:.1f}B weights stay compressed and are unpacked when needed")
            T, meta = _load_lazy(ws, which, log)
        else:
            T, meta = _load(ws, which, log)
        model = build(T, meta)
        tok = make_tokenizer(meta)
        if lowmem:
            log(f"Model ready in {time.time() - t0:.0f} s (cache limit {T.budget / 1e9:.1f} GB)")
        else:
            log(f"Model ready in {time.time() - t0:.0f} s ({sum(a.nbytes for a in T.values()) / 1e9:.1f} GB in memory)")
            core.release_sources()
        _MODEL.update(key=key, model=model, tok=tok, meta=meta)
        return model, tok


def unload():
    with _MLOCK:
        _MODEL.update(key=None, model=None, tok=None, meta=None)
    gc.collect()


# ============================================================== run (prefill + generation) with recording

def _logits(model, xf):
    return model.logits(xf)


def _record_preds(rec, logits, targets):
    """top-k, log-sum-exp and the log-probability of the known next token for each row of logits."""
    m = logits.max(-1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(logits - m).sum(-1))).astype(np.float32)
    k = np.argpartition(-logits, TOPK, axis=-1)[:, :TOPK]
    kv = np.take_along_axis(logits, k, -1)
    o = np.argsort(-kv, -1)
    rec.add("top_ids", np.take_along_axis(k, o, -1).astype(np.int32))
    rec.add("top_lp", (np.take_along_axis(kv, o, -1) - lse[:, None]).astype(np.float32))
    tl = np.array([logits[i, t] - lse[i] if t >= 0 else np.nan for i, t in enumerate(targets)], dtype=np.float32)
    rec.add("next_lp", tl)


def _sample(logits, temperature, top_p, top_k, rng):
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    if top_k and 0 < top_k < len(z):
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p and top_p < 1:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[:max(1, int(np.searchsorted(cum, top_p) + 1))]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(len(p), p=p))


def run(model, tok, prompt_ids, max_new=256, temperature=0.0, top_p=1.0, top_k=0, seed=0, emit=None, cancel=None, log=print):
    """Prefill the prompt, then generate. Returns (all token ids, number of prompt tokens, recorder, timings)."""
    if len(prompt_ids) + max_new > MAX_POSITIONS:
        raise ValueError(f"Too long: the prompt has {len(prompt_ids)} tokens; with {max_new} new ones that is over "
                         f"{MAX_POSITIONS}. Start a new chat or ask for fewer tokens.")
    rng = np.random.default_rng(seed)
    st, rec = State(model.n_layer), Recorder()
    t0 = time.time()
    out_ids = list(prompt_ids)
    step = 64
    xf = None
    for s0 in range(0, len(prompt_ids), step):
        chunk = prompt_ids[s0:s0 + step]
        xf = model.forward(chunk, st, rec)
        targets = [prompt_ids[s0 + i + 1] if s0 + i + 1 < len(prompt_ids) else -1 for i in range(len(chunk))]
        lg = _logits(model, xf)
        _record_preds(rec, lg[:-1] if targets[-1] < 0 else lg, targets[:-1] if targets[-1] < 0 else targets)
        log(f"Read {min(s0 + step, len(prompt_ids))}/{len(prompt_ids)} prompt tokens")
    last = lg[-1]
    t_prompt = time.time() - t0
    t1 = time.time()
    n_gen = 0
    for i in range(max_new):
        y = _sample(last, temperature, top_p, top_k, rng)
        _record_preds(rec, last[None], [y])                 # the prediction row for the previous position
        out_ids.append(y)
        n_gen += 1
        if emit:
            emit({"id": y, "text": tok.texts[y]})
        if tok.is_stop(y) or (cancel is not None and cancel.is_set()) or i == max_new - 1:
            break
        xf = model.forward([y], st, rec)
        last = _logits(model, xf)[-1]
    t_gen = time.time() - t1
    return out_ids, len(prompt_ids), rec, {"prompt_s": round(t_prompt, 2), "gen_s": round(t_gen, 2), "n_gen": n_gen,
                                           "tok_per_s": round(n_gen / t_gen, 2) if t_gen > 0 else None}


# ============================================================== traces on disk

def traces_dir(ws):
    return core.workspace_dir(ws) / "traces"


def _pack(model, rec):
    arr = {k: np.concatenate(v) for k, v in rec.cols.items()}
    n = arr["act"].shape[0]
    for k in ("top_ids", "top_lp", "next_lp"):
        arr[k] = arr[k][:n]
    return arr


def chat(ws, messages, max_new=256, temperature=0.0, top_p=0.8, top_k=20, seed=0, thinking=False, which="edited",
         emit=None, cancel=None, log=print):
    """Run one chat turn and save its trace. messages: [{role, content}], last one from the user."""
    if not messages or messages[-1].get("role") != "user" or not str(messages[-1].get("content", "")).strip():
        raise ValueError("Type a message")
    model, tok = get_model(ws, which, log)
    loaded = (_MODEL["meta"], _MODEL["key"])           # as loaded now: unloading during the reply must not lose it
    prompt = render_chat(loaded[0], messages, thinking)
    ids = prompt_ids(tok, prompt)
    log(f"Prompt: {len(ids)} tokens")
    all_ids, n_prompt, rec, timing = run(model, tok, ids, max_new, temperature, top_p, top_k, seed, emit, cancel, log)
    settings = {"max_new": max_new, "temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed, "thinking": thinking}
    tid = _save_trace(ws, model, tok, messages, all_ids, n_prompt, rec, which, settings, timing, loaded)
    log(f"Reply: {timing['n_gen']} tokens at {timing['tok_per_s']} tokens/s. Trace saved as {tid}")
    return summary(ws, tid)


def trace_ids(ws, messages, all_ids, n_prompt, which="edited", settings=None, log=print):
    """Save the trace of a reply written elsewhere (e.g. by llama.cpp): prompt + reply are read in one pass, which
    records exactly what generating them token by token would have (same positions, same predictions)."""
    model, tok = get_model(ws, which, log)
    loaded = (_MODEL["meta"], _MODEL["key"])
    all_ids = [int(i) for i in all_ids]
    if len(all_ids) > MAX_POSITIONS:
        raise ValueError(f"Too long: {len(all_ids)} tokens (at most {MAX_POSITIONS})")
    st, rec = State(model.n_layer), Recorder()
    t0 = time.time()
    n = len(all_ids) - 1
    for s0 in range(0, n, 64):
        chunk = all_ids[s0:min(s0 + 64, n)]
        lg = _logits(model, model.forward(chunk, st, rec))
        _record_preds(rec, lg, all_ids[s0 + 1:s0 + 1 + len(chunk)])
    dt = time.time() - t0
    timing = {"prompt_s": round(dt, 2), "gen_s": 0.0, "n_gen": len(all_ids) - n_prompt, "tok_per_s": None, "replayed": True}
    return _save_trace(ws, model, tok, messages, all_ids, n_prompt, rec, which, settings or {}, timing, loaded)


def _save_trace(ws, model, tok, messages, all_ids, n_prompt, rec, which, settings, timing, loaded=None):
    meta, key = loaded or (_MODEL["meta"], _MODEL["key"])
    arr = _pack(model, rec)
    reply_ids = all_ids[n_prompt:]
    reply = tok.decode([i for i in reply_ids if not tok.is_stop(i)])
    tid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    out = traces_dir(ws) / tid
    tmp = traces_dir(ws) / (tid + ".partial")         # written whole, then renamed: a failed save leaves no half trace
    tmp.mkdir(parents=True, exist_ok=True)
    np.savez(tmp / "trace.npz", **arr)
    info = {"id": tid, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "messages": messages, "reply": reply,
            "ids": [int(i) for i in all_ids], "n_prompt": n_prompt, "weights": which,
            "edited_tensors": meta.get("_edited_tensors", []) if which == "edited" else [],
            "settings": settings,
            "timing": timing, "architecture": model.name, "n_layer": model.n_layer,
            "kinds": model.kinds() if hasattr(model, "kinds") else None, "weights_key": key[2],
            "moe": {str(L): v for L, v in model.moe_layout().items()}}
    try:
        (tmp / "trace.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tid


_TRACES = {}


def _open_trace(ws, tid):
    if not re.fullmatch(r"[\w-]+", tid or ""):
        raise ValueError("Bad trace id")
    p = traces_dir(ws) / tid
    if not (p / "trace.json").is_file():
        raise FileNotFoundError(f"No trace {tid!r}")
    key = (ws, tid)
    if key not in _TRACES:
        while len(_TRACES) >= 3:
            _TRACES.pop(next(iter(_TRACES)))
        info = json.loads((p / "trace.json").read_text(encoding="utf-8"))
        with np.load(p / "trace.npz") as z:                      # read once; NpzFile re-reads the zip on every access
            arrays = {k: z[k] for k in z.files}
        _TRACES[key] = (info, arrays)
    return _TRACES[key]


def neuron_ref(info, A, pos, L, j):
    """What column j of the recorded activations of layer L is at position pos, and where its weights are.
    Dense layer: neuron j. MoE layer: slot j // ffe holds the expert chosen at that rank for this token."""
    m = (info.get("moe") or {}).get(str(L))
    if not m:
        return {"expert": None, "local": j, "label": f"#{j}",
                "links": {"down": [f"blk.{L}.ffn_down.weight", "col", j], "up": [f"blk.{L}.ffn_up.weight", "row", j],
                          "gate": [f"blk.{L}.ffn_gate.weight", "row", j]}}
    k, ffe, D = m["k"], m["ffe"], m["D"]
    if j < k * ffe:
        s, jl = divmod(j, ffe)
        e = int(A[f"moe{L}_ids"][pos][s])
        return {"expert": e, "local": jl, "slot": s, "weight": float(A[f"moe{L}_w"][pos][s]), "label": f"expert {e} · #{jl}",
                "links": {"down": [f"blk.{L}.ffn_down_exps.weight", "col", jl, e * D, D],
                          "up": [f"blk.{L}.ffn_up_exps.weight", "row", e * ffe + jl],
                          "gate": [f"blk.{L}.ffn_gate_exps.weight", "row", e * ffe + jl]}}
    jl = j - k * ffe
    return {"expert": -1, "local": jl, "label": f"shared · #{jl}",
            "links": {"down": [f"blk.{L}.ffn_down_shexp.weight", "col", jl], "up": [f"blk.{L}.ffn_up_shexp.weight", "row", jl],
                      "gate": [f"blk.{L}.ffn_gate_shexp.weight", "row", jl]}}


def neuron_over_trace(ws, tid, layer, neuron, expert=None):
    """Activation of one MLP neuron at every position of a trace (for highlighting tokens). In an MoE layer,
    neuron = the neuron inside `expert` (-1 = shared expert); positions that didn't use that expert show 0."""
    info, A = _open_trace(ws, tid)
    act = A["act"]
    m = (info.get("moe") or {}).get(str(layer))
    if m and expert is not None:
        ids, ffe, k = A[f"moe{layer}_ids"], m["ffe"], m["k"]
        width = m.get("ffs", 0) if expert == -1 else ffe
        if not (-1 <= expert < m["E"]) or (expert == -1 and not width):
            raise ValueError(f"Expert must be 0–{m['E'] - 1}" + (" or -1 (shared)" if m.get("ffs") else ""))
        if not 0 <= neuron < width:
            raise ValueError(f"Neuron must be 0–{width - 1}")
        a = np.zeros(act.shape[0], np.float32)
        for p in range(act.shape[0]):
            if expert == -1:
                a[p] = act[p, layer, k * ffe + neuron]
            else:
                hit = np.nonzero(ids[p] == expert)[0]
                if len(hit):
                    a[p] = act[p, layer, hit[0] * ffe + neuron]
        return {"layer": layer, "neuron": neuron, "expert": expert, "act": [round(float(x), 4) for x in a],
                "max": float(np.abs(a).max()) if len(a) else 0.0, "used": int((a != 0).sum())}
    if not (0 <= layer < act.shape[1] and 0 <= neuron < act.shape[2]):
        raise ValueError(f"Layer must be 0–{act.shape[1] - 1} and neuron 0–{act.shape[2] - 1}")
    a = act[:, layer, neuron].astype(np.float32)
    return {"layer": layer, "neuron": neuron, "act": [round(float(x), 4) for x in a],
            "max": float(np.abs(a).max()) if len(a) else 0.0}


def _trace_tests(ws):
    """trace id -> name of the saved test that made it (Test tab)."""
    d = core.workspace_dir(ws) / "tests"
    out = {}
    for p in d.glob("*.json") if d.is_dir() else []:
        if p.name.endswith(".checks.json"):
            continue
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for x in t.get("items", []):
            if x.get("trace"):
                out[x["trace"]] = t.get("name") or t.get("id")
    return out


def list_traces(ws):
    root = traces_dir(ws)
    out = []
    from_tests = _trace_tests(ws)
    if root.is_dir():
        for p in sorted(root.iterdir(), reverse=True):
            j = p / "trace.json"
            if j.is_file():
                try:
                    info = json.loads(j.read_text(encoding="utf-8"))
                except Exception:
                    continue
                user = next((m["content"] for m in reversed(info["messages"]) if m["role"] == "user"), "")
                test = from_tests.get(info["id"])
                replayed = bool((info.get("timing") or {}).get("replayed"))
                out.append({"id": info["id"], "created": info["created"], "question": user[:120], "reply": info["reply"][:120],
                            "weights": info["weights"], "n_tokens": len(info["ids"]), "edited": bool(info.get("edited_tensors")),
                            "source": "test" if test else ("replayed" if replayed else "chat"), "test": test,
                            "replayed": replayed})
    return out


def delete_trace(ws, tid):
    _open_trace(ws, tid)
    _TRACES.pop((ws, tid), None)
    gc.collect()
    shutil.rmtree(traces_dir(ws) / tid, ignore_errors=True)
    return {"ok": True}


def summary(ws, tid):
    """Everything the chat view needs for a trace, except per-position details."""
    info, A = _open_trace(ws, tid)
    v = ic.vocab(ws)
    ids = info["ids"]
    n = A["act"].shape[0]                           # positions traced = len(ids) - 1
    act = A["act"]
    npr = info["n_prompt"]
    reply_pos = np.arange(max(0, npr - 1), n)
    heat = np.abs(act[reply_pos].astype(np.float32)).mean(0) if len(reply_pos) else np.zeros(act.shape[1:], np.float32)
    toks = []
    for i, t in enumerate(ids):
        p = i - 1                                   # position whose prediction produced this token
        toks.append({"id": int(t), "text": v["texts"][t] if t < len(v["texts"]) else f"<{t}>", "reply": i >= npr,
                     "p": float(np.exp(A["next_lp"][p])) if 0 <= p < n and np.isfinite(A["next_lp"][p]) else None,
                     "top1": int(A["top_ids"][p][0]) if 0 <= p < n else None})
    moe = info.get("moe") or {}
    expert_heat = None
    if moe:                                          # how much weight each expert got over the reply
        Emax = max(m["E"] for m in moe.values())
        eh = np.zeros((info["n_layer"], Emax), np.float32)
        for Ls, m in moe.items():
            L = int(Ls)
            ids, w = A[f"moe{L}_ids"][reply_pos], A[f"moe{L}_w"][reply_pos]
            np.add.at(eh[L], ids.ravel(), w.ravel())
        expert_heat = eh / max(1, len(reply_pos))
    return {**{k: info[k] for k in ("id", "created", "messages", "reply", "n_prompt", "weights", "edited_tensors",
                                     "settings", "timing", "architecture", "n_layer")},
            "moe": moe, "expert_heat": _b64(expert_heat) if expert_heat is not None else None,
            "expert_shape": list(expert_heat.shape) if expert_heat is not None else None,
            "tokens": toks, "n_positions": int(n), "ff": int(act.shape[2]),
            "heat": _b64(heat), "heat_shape": list(heat.shape),
            "attn_norm": A["attn_norm"].round(3).tolist(), "mlp_norm": A["mlp_norm"].round(3).tolist(),
            "kinds": info.get("kinds"), "stale": _stale(ws, info)}


def _stale(ws, info):
    """True when the weights changed after this trace was recorded (its pushes would then mix old and new)."""
    try:
        return info.get("weights_key") != _key(ws, info["weights"])[2]
    except Exception:
        return False


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def attribute(model, A, pos, target):
    """Direct logit attribution at one position: how much each part of the model added to the target's score.
    A: the recorded arrays (a trace, or a Recorder's columns joined). Every number is exact, not an estimate:
        score = embedding + constant + sum over layers (mixer push + MLP push)
        MLP push of a layer   = sum of its neurons' pushes (+ the ffn_down bias)
        mixer push of a layer = sum of its heads' pushes   (+ the output bias)
    Output norms (Gemma, OLMo 2, LFM) and residual multipliers (Granite) are linear once their divisor is
    known, and the divisor is recorded, so they are included exactly. The score is before any final soft-cap."""
    act = A["act"][pos].astype(np.float32)                    # [L, ff]
    heads = A["heads"][pos].astype(np.float32)                # [L, H*hd]
    d = model.logit_direction(target, A["final_rms"][pos])
    rs = np.float32(model.res_scale)
    nd, hd_rows = np.zeros_like(act), []
    for L in range(model.n_layer):
        pa, pf = model.post_norm_names(L)
        dm = model.mlp_direction(L, d * rs * (model.T[pf] / A["m_div"][pos, L] if pf else 1), A, pos)
        down = model.mlp_down(L)
        if model.is_moe(L):                                  # chosen experts, slot by slot, each × its router weight
            ids, wts, sg = A[f"moe{L}_ids"][pos], A[f"moe{L}_w"][pos], float(A[f"moe{L}_sg"][pos])
            Dn = model.w(L, "ffn_down_exps.weight")
            ffe = Dn.shape[2]
            for s, e in enumerate(ids):
                wf = 1.0 if model.moe_weight_before else wts[s]          # Llama 4: the weight is already inside act
                nd[L, s * ffe:(s + 1) * ffe] = act[L, s * ffe:(s + 1) * ffe] * wf * (dm @ Dn[e])
            ds = model.w(L, "ffn_down_shexp.weight")
            if ds is not None:
                o = len(ids) * ffe
                nd[L, o:o + ds.shape[1]] = act[L, o:o + ds.shape[1]] * sg * (dm @ ds)
        elif down is not None:
            nd[L, :down.shape[1]] = act[L, :down.shape[1]] * (dm @ down)
        da = model.mixer_direction(L, d * rs * (model.T[pa] / A["a_div"][pos, L] if pa else 1), A, pos)
        op = model.out_proj(L)
        H, dh = model.n_heads(L), model.head_dim(L)
        if op is not None and H * dh == op.shape[1]:
            hd_rows.append((heads[L][:H * dh].reshape(H, dh) * (da @ op).reshape(H, dh)).sum(1))
        else:
            hd_rows.append(np.zeros(max(H, 1), np.float32))
    layer_attn = A["a_vec"][pos].astype(np.float32) @ d
    layer_mlp = A["m_vec"][pos].astype(np.float32) @ d
    emb = float(A["resid"][pos][0].astype(np.float32) @ d) * model.embed_factor
    const = 0.0
    if model.norm_kind == "layer" and model.out_norm_b is not None:
        const += float(model.out_norm_b @ model.U[target]) / model.logit_scale
    if model.out_bias is not None:
        const += float(model.out_bias[target]) / model.logit_scale
    return {"act": act, "neuron_push": nd, "head_push": hd_rows, "layer_attn": layer_attn, "layer_mlp": layer_mlp,
            "embedding": emb, "constant": const, "logit": float(emb + const + layer_attn.sum() + layer_mlp.sum())}


def position_detail(ws, tid, pos, target=None, which=None, log=print):
    """Why the model predicted what it did at position `pos` (reading token pos, predicting token pos+1).
    target: the token id to explain (default: the token that really came next)."""
    info, A = _open_trace(ws, tid)
    n = A["act"].shape[0]
    if not 0 <= pos < n:
        raise ValueError(f"Position must be between 0 and {n - 1}")
    ids = info["ids"]
    model, tok = get_model(ws, which or info["weights"], log)
    v = ic.vocab(ws)
    if target is None:
        target = ids[pos + 1]
    target = int(target)
    if not 0 <= target < model.U.shape[0]:
        raise ValueError(f"Token id must be between 0 and {model.U.shape[0] - 1}")
    att = attribute(model, A, pos, target)
    act, nd, hd_rows = att["act"], att["neuron_push"], att["head_push"]
    layer_attn, layer_mlp, emb, logit = att["layer_attn"], att["layer_mlp"], att["embedding"], att["logit"]
    resid = A["resid"][pos].astype(np.float32)
    # logit lens: what the hidden state after each layer would predict
    lg = model.logits(model.final(resid))
    lens = []
    for L in range(lg.shape[0]):
        row = lg[L]
        m = row.max()
        lse = m + np.log(np.exp(row - m).sum())
        top = np.argpartition(-row, 5)[:5]
        top = top[np.argsort(-row[top])]
        lens.append({"after": L, "top": [ic._tok(v, int(i), float(np.exp(row[i] - lse))) for i in top],
                     "target_p": float(np.exp(row[target] - lse)),
                     "target_rank": int((row > row[target]).sum()) + 1})
    flat = nd.ravel()
    ff = nd.shape[1]
    pick = lambda idx: [{"layer": int(i // ff), "neuron": int(i % ff), "push": float(flat[i]), "act": float(act.ravel()[i]),
                         **neuron_ref(info, A, pos, int(i // ff), int(i % ff))} for i in idx]
    experts = {}
    for Ls, m in (info.get("moe") or {}).items():
        L = int(Ls)
        rp = A[f"moe{L}_router"][pos].astype(np.float32)
        top = np.argsort(-rp)[:8]
        experts[L] = {**m, "ids": [int(e) for e in A[f"moe{L}_ids"][pos]], "w": [round(float(x), 4) for x in A[f"moe{L}_w"][pos]],
                      "router_top": [[int(e), round(float(rp[e]), 4)] for e in top]}
    up = np.argsort(-flat)[:20]
    down = np.argsort(flat)[:20]
    loud = np.argsort(-np.abs(act.ravel()))[:20]
    heads_out = []
    for L, row in enumerate(hd_rows):
        for h, val in enumerate(row):
            heads_out.append({"layer": L, "head": h, "push": float(val)})
    heads_out.sort(key=lambda x: -abs(x["push"]))
    attn = {}
    for L in range(model.n_layer):
        if f"attn{L}_i" in A:
            I, W = A[f"attn{L}_i"][pos], A[f"attn{L}_w"][pos].astype(np.float32)   # [H, keep]
            attn[L] = [{"head": h, "top": [[int(j), round(float(w), 4)] for j, w in zip(I[h], W[h]) if 0 <= j <= pos and w > 0]}
                       for h in range(I.shape[0])]
    top = [{"id": int(i), "text": v["texts"][i], "p": float(np.exp(lp))} for i, lp in zip(A["top_ids"][pos], A["top_lp"][pos])]
    return {"pos": pos, "reading": ic._tok(v, ids[pos]), "actual_next": ic._tok(v, ids[pos + 1]),
            "target": ic._tok(v, target), "target_logit": logit, "top": top,
            "embedding_push": emb + att["constant"], "layer_attn": layer_attn.tolist(), "layer_mlp": layer_mlp.tolist(),
            "neuron_push": _b64(nd), "neuron_act": _b64(act), "shape": list(act.shape),
            "top_up": pick(up), "top_down": pick(down), "loudest": pick(loud),
            "heads": heads_out[:24], "head_layout": {L: model.head_layout(L) for L in range(model.n_layer)},
            "kinds": model.kinds() if hasattr(model, "kinds") else None,
            "attention": attn, "lens": lens, "stale": _stale(ws, info), "experts": experts,
            "note": "Pushes are direct effects on the final score (direct logit attribution): a neuron's push = its "
                    "activation × (its ffn_down column · the target's unembedding direction, after the final norm). "
                    "They add up exactly to the target's logit; indirect effects through later layers are not split out."}
