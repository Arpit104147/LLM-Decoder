"""Checks the K-quant encoders: valid bytes (decoded by the official gguf decoder), accuracy, and that
values a format can already represent survive re-encoding. Run:  python test_kquants.py
"""
import sys

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import kquants

rng = np.random.default_rng(0)
ok = True
# relative RMS error limits (error / std of the data) for gaussian weights
LIMITS = {"Q6_K": 0.02, "Q5_K": 0.045, "Q4_K": 0.09, "Q3_K": 0.2, "Q2_K": 0.4}
for name, lim in LIMITS.items():
    qt = QT[name]
    bs, ts = GGML_QUANT_SIZES[qt]
    x = (rng.standard_normal((400, 256)) * 0.02).astype(np.float32)
    x[3] *= 50                     # a block with outliers
    x[7] = 0                       # an all-zero block
    x[9] = np.abs(x[9])            # all-positive block
    enc = kquants.quantize_blocks(x, name)
    good = enc.shape == (400, ts) and enc.dtype == np.uint8
    y = dequantize(enc, qt).reshape(400, 256)
    rel = float(np.sqrt(((y - x) ** 2).mean()) / x.std())
    zero_ok = bool(np.all(y[7] == 0))
    # re-encoding already-quantized values should change (almost) nothing
    y2 = dequantize(kquants.quantize_blocks(y, name), qt).reshape(400, 256)
    stable = float(np.sqrt(((y2 - y) ** 2).mean()) / x.std())
    good = good and rel < lim and zero_ok and stable < lim / 2 and np.isfinite(y).all()
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} {name}: {ts} bytes/block, rel. error {rel:.4f} (limit {lim}), "
          f"zeros kept {zero_ok}, re-encode drift {stable:.4f}")
print("ALL K-QUANT CHECKS PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
