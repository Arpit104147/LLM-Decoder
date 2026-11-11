# Qwen3.5-0.8B — weight-edit experiments (2026-11-06)

Edits are real weight changes: a head ×k scales its columns of `attn_output` (full attention) or `ssm_out`
(linear attention); a neuron ×k scales its column of `ffn_down`. ×0 removes it, ×2 doubles what it writes.

## 1. Replay of the 100 study chats (tracer, teacher-forced) — `edit_tests.md`

| Edit | Δ log-prob of original reply | Top token changed | First reply token changed |
|---|---|---|---|
| L23 heads 2,4,5,6 ×0 | -0.160 | 10.1% | 43/100 |
| L23 head 6 ×0 | -0.034 | 4.1% | 25/100 |
| L23 #9 + L19 #0 + L21 #0 ×0 | -0.028 | 5.2% | 18/100 |
| L23 #9 ×0 | -0.025 | 3.3% | 3/100 |
| L19 #0 ×0 | -0.003 | 2.5% | 14/100 |
| L21 head 15 ×0 | -0.004 | 1.2% | 8/100 |
| control: L23 head 0 / head 1 ×0 | -0.009 / -0.004 | 1.9% / 1.8% | 2 / 6 |
| "specialist" neurons (L23 #223 math, L17 #603 / L23 #165 code) ×0 | 0.000 | ≤0.1% | 0/100 |
| control: random neurons L23 #1000, #2000, L19 #1500 ×0 | 0.000 | ≤0.1% | 0/100 |

Neurons picked by *direct push on the answer* matter; neurons picked by *activation difference between
categories* (the "specialists") do nothing when removed — they are correlated with the topic, not causal.

## 2. Benchmarks in llama.cpp (exported GGUFs, greedy, thinking off)

Study = the 100 questions with the traced chats' exact prompt tokens. "Hand-checked" re-reads every answer whose
grade changed versus the original and drops keyword-matching mistakes (wrong answers that contain the keyword,
or right answers in another format). GSM8K = test problems 0-29, 1024-token cap.

| Model | Study (grader) | Study (hand-checked) | Real gains / losses | GSM8K 0-29 | GSM8K 0-49 |
|---|---|---|---|---|---|
| original | 78 | 78 | – | 16/30 | 25/50 |
| L23 head 6 ×0 | 82 | **81** | +5 / −2 | 15/30 | |
| L23 heads 2,4,5,6 ×0 | 87 | **82** | +7 / −3 | **10/30** (11 capped) | 19/50 |
| L23 #9 ×0 | 79 | 77 | +2 / −3 | 13/30 | 25/50 |
| control: L23 head 1 ×0 | 77 | 77 | +2 / −3 | 13/30 | |
| L23 #9 ×2 | 77 | 76 | +1 / −3 | 15/30 | |
| L19 #0 ×0 | 75 | 75 | +2 / −5 | 14/30 | |
| L23 head 6 ×2 | 75 | 74 | +1 / −5 | 16/30 | |

## What it means

- **Any edit moves about ±3 answers** (the control head did), so only bigger moves count. m01 (17×23) flips to
  right under almost every edit: it was a near-tie.
- **L23 heads 2,4,5,6 enforce "answer briefly".** Without them replies are 60 % longer and the model works problems
  out: reasoning 5/15 → 11/15 (7 genuine fixes: 32, 16 legs, 7 apples, 4 days, 391, "went", `[0, 1, 2]`). But it
  loses recall (Mars → Jupiter, Spring → Summer) and rambles on GSM8K (11 of 30 hit the cap, 16 → 10).
- **L23 head 6 alone** is the best single edit: +3 hand-checked on the study set, GSM8K within noise (15 vs 16).
  Doubling it goes the other way (74). Its effect is format + short-answer selection, not knowledge.
- **L19 #0 carries facts**: removing it gives "Alfred D. Davis" for 1984, 5 continents, "Cabbage" (−5).
- **L23 #9** (the top "push" neuron everywhere) is mostly a confidence / format neuron: removing or doubling it
  changes half the replies' wording but the score stays within noise.
- With 30-50 GSM8K problems, differences under ~4 are not significant.

## 3. GSM8K problems 0-299: original vs L23 head 6 ×0 (llama.cpp, greedy, thinking off, 1024-token cap)

| Model | Correct | Hit the cap | Mean tokens |
|---|---|---|---|
| original | **169/300 (56.3 %)** | 38 | 475 |
| L23 head 6 ×0 | 156/300 (52.0 %) | 30 | 461 |

Only the original right: 35 · only the edit right: 22 · exact McNemar p = 0.11. The best edit on the short-answer
questions (+3) is **4.3 points worse** on multi-step math (not significant at 0.05, but clearly not a gain):
removing head 6 shortens and simplifies answers, which helps one-word answers and hurts long working.
