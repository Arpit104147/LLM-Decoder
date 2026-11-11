"""
Encoders for llama.cpp's K-quant formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K), written in numpy.

The gguf Python package can only *decode* these formats. These encoders produce valid blocks that
gguf/llama.cpp decode correctly. They use a small scale search per sub-block (not llama.cpp's exact
optimizer), so accuracy is close to, but not bit-identical with, llama.cpp's quantizer. LLM Decoder only
uses them for blocks you actually edited; every other block keeps its original bytes.

Each encoder takes float32 values shaped (n_blocks, 256) and returns uint8 bytes (n_blocks, block_bytes).
"""
import numpy as np

QK = 256
_CANDS = np.linspace(0.80, 1.20, 21, dtype=np.float32)


def _f16(x):
    return x.astype(np.float16)


def _min_quant(x, nmax):
    """Asymmetric sub-block quantization: x ~ scale*q - mn, q in [0, nmax]. x: (n, g, s). Returns scale, mn (n, g)."""
    xmin = np.minimum(x.min(-1, keepdims=True), 0)
    rng = x.max(-1, keepdims=True) - xmin
    best_err = np.full(xmin.shape, np.inf, dtype=np.float32)
    best = rng / nmax
    for f in _CANDS:
        sc = rng / nmax * f
        safe = np.where(sc > 0, sc, 1)
        q = np.clip(np.rint((x - xmin) / safe), 0, nmax)
        err = ((q * sc + xmin - x) ** 2).sum(-1, keepdims=True)
        take = err < best_err
        best = np.where(take, sc, best)
        best_err = np.where(take, err, best_err)
    return best[..., 0], -xmin[..., 0]


def _sym_quant(x, nmax):
    """Symmetric sub-block quantization: x ~ scale*q, q in [-nmax, nmax-1]. x: (n, g, s). Returns scale (n, g)."""
    idx = np.abs(x).argmax(-1)[..., None]
    mx = np.take_along_axis(x, idx, -1)                       # signed value with the largest magnitude
    base = -mx / nmax                                         # maps it to -nmax (uses the wider side)
    best_err = np.full(mx.shape, np.inf, dtype=np.float32)
    best = base
    for f in _CANDS:
        sc = base * f
        safe = np.where(sc != 0, sc, 1)
        q = np.clip(np.rint(x / safe), -nmax, nmax - 1)
        err = ((q * sc - x) ** 2).sum(-1, keepdims=True)
        take = err < best_err
        best = np.where(take, sc, best)
        best_err = np.where(take, err, best_err)
    return best[..., 0]


def _super(scale, mn, smax):
    """Quantize per-sub-block scale/min to integers with f16 super-block factors d, dmin."""
    d = _f16(scale.max(1) / smax)
    dmin = _f16(mn.max(1) / smax)
    df, dmf = d.astype(np.float32)[:, None], dmin.astype(np.float32)[:, None]
    sc = np.clip(np.rint(np.where(df > 0, scale / np.where(df > 0, df, 1), 0)), 0, smax).astype(np.uint8)
    m = np.clip(np.rint(np.where(dmf > 0, mn / np.where(dmf > 0, dmf, 1), 0)), 0, smax).astype(np.uint8)
    return d, dmin, sc, m


def _q_from(x, eff_scale, eff_min, lo, hi):
    safe = np.where(eff_scale != 0, eff_scale, 1)
    q = np.rint((x + eff_min) / safe)
    q = np.where(eff_scale != 0, q, 0 if lo == 0 else 0)
    return np.clip(q, lo, hi).astype(np.int16)


def _pack_k4_scales(sc, m):
    out = np.zeros((sc.shape[0], 12), dtype=np.uint8)
    out[:, 0:4] = (sc[:, 0:4] & 63) | ((sc[:, 4:8] >> 4) << 6)
    out[:, 4:8] = (m[:, 0:4] & 63) | ((m[:, 4:8] >> 4) << 6)
    out[:, 8:12] = (sc[:, 4:8] & 0xF) | ((m[:, 4:8] & 0xF) << 4)
    return out


def _min_type(x, nsub, nmax, smax):
    n = x.shape[0]
    sub = x.reshape(n, nsub, QK // nsub)
    scale, mn = _min_quant(sub, nmax)
    d, dmin, sc, m = _super(scale, mn, smax)
    eff_s = d.astype(np.float32)[:, None, None] * sc[..., None]
    eff_m = dmin.astype(np.float32)[:, None, None] * m[..., None]
    q = _q_from(sub, eff_s, eff_m, 0, nmax).reshape(n, QK)
    return d, dmin, sc, m, q


def quantize_q4_k(x):
    n = x.shape[0]
    d, dmin, sc, m, q = _min_type(x, 8, 15, 63)
    qs = np.zeros((n, 128), dtype=np.uint8)
    for c in range(4):
        qs[:, c * 32:(c + 1) * 32] = q[:, c * 64:c * 64 + 32] | (q[:, c * 64 + 32:c * 64 + 64] << 4)
    return np.concatenate([d.view(np.uint8).reshape(n, 2), dmin.view(np.uint8).reshape(n, 2),
                           _pack_k4_scales(sc, m), qs], axis=1)


def quantize_q5_k(x):
    n = x.shape[0]
    d, dmin, sc, m, q = _min_type(x, 8, 31, 63)
    qs = np.zeros((n, 128), dtype=np.uint8)
    qh = np.zeros((n, 32), dtype=np.uint8)
    for c in range(4):
        lo, hi = q[:, c * 64:c * 64 + 32], q[:, c * 64 + 32:c * 64 + 64]
        qs[:, c * 32:(c + 1) * 32] = (lo & 0xF) | ((hi & 0xF) << 4)
        qh |= (((lo >> 4) & 1) << (2 * c)).astype(np.uint8) | (((hi >> 4) & 1) << (2 * c + 1)).astype(np.uint8)
    return np.concatenate([d.view(np.uint8).reshape(n, 2), dmin.view(np.uint8).reshape(n, 2),
                           _pack_k4_scales(sc, m), qh, qs], axis=1)


def quantize_q2_k(x):
    n = x.shape[0]
    d, dmin, sc, m, q = _min_type(x, 16, 3, 15)
    scales = (sc & 0xF) | ((m & 0xF) << 4)
    qs = np.zeros((n, 64), dtype=np.uint8)
    for h in range(2):
        for s in range(4):
            qs[:, h * 32:(h + 1) * 32] |= (q[:, h * 128 + s * 32:h * 128 + s * 32 + 32] << (2 * s)).astype(np.uint8)
    return np.concatenate([scales, qs, d.view(np.uint8).reshape(n, 2), dmin.view(np.uint8).reshape(n, 2)], axis=1)


def _sym_type(x, nsub, nmax, smin, smax):
    n = x.shape[0]
    sub = x.reshape(n, nsub, QK // nsub)
    scale = _sym_quant(sub, nmax)
    amax = np.abs(scale).max(1)
    d = _f16(amax / (-smin))
    df = d.astype(np.float32)[:, None]
    sc = np.clip(np.rint(np.where(df > 0, scale / np.where(df > 0, df, 1), 0)), smin, smax).astype(np.int16)
    eff = df[..., None] * sc[..., None]
    q = _q_from(sub, eff, 0, -nmax, nmax - 1).reshape(n, QK)
    return d, sc, q


def quantize_q6_k(x):
    n = x.shape[0]
    d, sc, q = _sym_type(x, 16, 32, -128, 127)
    u = (q + 32).astype(np.uint8)
    ql = np.zeros((n, 128), dtype=np.uint8)
    qh = np.zeros((n, 64), dtype=np.uint8)
    for half in range(2):
        o = half * 128
        q1, q2, q3, q4 = (u[:, o + k * 32:o + k * 32 + 32] for k in range(4))
        ql[:, half * 64:half * 64 + 32] = (q1 & 0xF) | ((q3 & 0xF) << 4)
        ql[:, half * 64 + 32:half * 64 + 64] = (q2 & 0xF) | ((q4 & 0xF) << 4)
        qh[:, half * 32:half * 32 + 32] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
    return np.concatenate([ql, qh, sc.astype(np.int8).view(np.uint8), d.view(np.uint8).reshape(n, 2)], axis=1)


def quantize_q3_k(x):
    n = x.shape[0]
    d, sc, q = _sym_type(x, 16, 4, -32, 31)
    ql = np.where(q >= 0, q, q + 4).astype(np.uint8)          # low 2 bits
    hb = (q >= 0).astype(np.uint8)                            # high bit set = no -4 offset
    hmask = np.zeros((n, 32), dtype=np.uint8)
    for b in range(8):
        hmask |= hb[:, b * 32:(b + 1) * 32] << b
    qs = np.zeros((n, 64), dtype=np.uint8)
    for h in range(2):
        for s in range(4):
            qs[:, h * 32:(h + 1) * 32] |= ql[:, h * 128 + s * 32:h * 128 + s * 32 + 32] << (2 * s)
    su = (sc + 32).astype(np.uint8)                           # 6-bit unsigned
    low, high = su & 0xF, su >> 4
    scales = np.zeros((n, 12), dtype=np.uint8)
    scales[:, 0:8] = low[:, 0:8] | (low[:, 8:16] << 4)
    for k in range(4):
        scales[:, 8:12] |= high[:, 4 * k:4 * k + 4] << (2 * k)
    return np.concatenate([hmask, qs, scales, d.view(np.uint8).reshape(n, 2)], axis=1)


ENCODERS = {"Q2_K": quantize_q2_k, "Q3_K": quantize_q3_k, "Q4_K": quantize_q4_k,
            "Q5_K": quantize_q5_k, "Q6_K": quantize_q6_k}


def quantize_blocks(values, type_name):
    """values: float array (n_blocks, 256). Returns (n_blocks, block_bytes) uint8."""
    v = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, QK)
    v = np.nan_to_num(v, nan=0.0, posinf=6.5e4, neginf=-6.5e4)
    return ENCODERS[type_name](v)
