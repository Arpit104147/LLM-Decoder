"""Checks that the raw-storage decoder (stored integers x scales) reproduces gguf's official dequantization
for every block type it explains. Uses the bundled LM Studio model plus synthetic Q8_0/Q4_0/Q4_1 data.
Run:  python test_raw_storage.py [model.gguf]
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import quantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import gguf_core as core

model = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".lmstudio/.internal/bundled-models/nomic-ai/"
                                                  "nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf")
ok = True
r = GGUFReader(model)
seen = {}
for t in r.tensors:
    seen.setdefault(t.tensor_type, t)
rng = np.random.default_rng(0)
for t in seen.values():
    rows, cols = core.tensor_dims(t)
    for _ in range(25):
        rr, cc = int(rng.integers(rows)), int(rng.integers(cols))
        res = core.raw_storage(t, rr, cc)
        exact = float(core.load_rows(t, rr + 1)[rr, cc])
        good = res["value"] == exact and res.get("decode_matches", True)
        ok &= good
    print(f"  {'ok ' if good else 'BAD'} {t.tensor_type.name:5s} {t.name:32s} e.g. value {res['value_repr']}"
          + (f" = {res['formula']}" if "formula" in res else ""))

for qt in (QT.Q8_0, QT.Q4_0, QT.Q4_1):
    data = quantize(rng.standard_normal((4, 64)).astype(np.float32), qt)

    class T:  # minimal stand-in for a GGUF tensor
        tensor_type, shape = qt, [64, 4]
        data_ = data
    T.data = data
    res = core.raw_storage(T, 2, 40)
    good = res["decode_matches"]
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} {qt.name:5s} synthetic  {res['formula']}")
print("ALL RAW-STORAGE CHECKS PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
