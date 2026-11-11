# Computed edits on GSM8K, checked on questions they never saw (partial: stopped before the held-out check)

Qwen3.5-0.8B Q8_0 workspace weights · edits computed from the wrong answers in problems 1-150 (65 located, 159 token
fixes, 8 at a wrong calculation step, 13 skipped) · baseline 153/300 right, 47 replies cut off at 1024 tokens.

| Run | Setting | Part edited | Size of change | Token fixes reached | Other answers' top token changed |
|---|---|---|---:|---:|---:|
| 1 | most responsible part, general rank 1, keep 1 | attention L0 (`blk.0.ssm_out`) | 78,760 % | 5/159 | 98.4 % |
| 1 | most responsible part, general rank 4, keep 1 | attention L0 | 78,769 % | 37/159 | 98.5 % |
| 2 | most responsible part, general rank 4, keep 4 | attention L0 | 27,955 % | 39/159 | 96.7 % |
| 2 | cheapest layer, general rank 1 | MLP L23 (push needed 0.24 x its output; L21: 0.76 x) | stopped while computing | | |

Finding so far: the part whose switching-off changes the mistakes most (attention of layer 0, where the question is
read) cannot be edited to correct them: it would need changes hundreds of times the matrix and would change almost
every other answer. The size guard refuses these edits. Next: spread the correction over the layers that decide it
(MEMIT-style), then the held-out check. Logs: gsm_search_experiment_run1.log, gsm_search_experiment_run2_partial.log.
