# Computed edits on GSM8K: compute from problems 0-149, test on 150-299 (stopped early)

Workspace weights (Qwen3.5-0.8B Q8_0 + the laser23_k25 edit), llama.cpp greedy, thinking off, 1024-token cap.

| Step | Result |
|---|---|
| Baseline, problems 0-299 | 153/300 (51.0 %) · 0-149: 72/150 · held-out 150-299: **81/150** |
| Wrong answers in 0-149 | 78, all located: 188 token fixes (9 at a wrong calculation step, the rest at the final answer) |
| Layer (old "cheapest" rule) | layer 23 (cost 0.27 vs 0.98 for layer 21) |
| Exact edit of blk.23.ffn_down | solved in 2 h 48 min; **size of the change 357 % of the matrix** |
| Held-out with the edit | not run: stopped while the edit was being checked |

Stopped on purpose. Forcing 188 different exact fixes (and 72 protected right answers) into one layer needs a change
3.6x the size of the matrix, which would very likely damage other answers; and the 24-chat layer check
(layer_effects.md) showed layer 23 is rarely the layer that decides an answer. Next: the general (low-rank) mode at
the most responsible layer, with fewer token fixes per answer. The baseline run is kept (gsm_fix_experiment.json).
