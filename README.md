# LLM Decoder

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg) ![Tested on Windows 11](https://img.shields.io/badge/tested%20on-Windows%2011-lightgrey.svg)

**Open a GGUF language model and look inside it.** See and edit its metadata as code and its weights as exact
numbers in an Excel-style grid, grow it (new layers, new tokens), chat with it while every token is traced through
every layer, test it on a question set, compute weight edits that fix wrong answers and check them on questions they
never saw, then export a new `.gguf` that runs in LM Studio, llama.cpp or Ollama.

Everything runs locally. Your original model file is never changed.

> **New here?** [GUIDE.md](GUIDE.md) walks through installing, running and a first session step by step.

## Contents

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [What you can do](#what-you-can-do)
- [How it works](#how-it-works)
  - [Chat & trace](#chat--trace-how-it-works-and-its-limits)
  - [Fix wrong answers](#fix-wrong-answers-how-the-computed-edit-works)
  - [Questions and grading](#questions-and-grading-every-common-dataset-format)
- [Notes and limits](#notes-and-limits)
- [Studies](#studies)
- [Folder layout](#folder-layout)
- [Command line](#command-line)
- [Tests](#tests)
- [License](#license)

## Quick start

```bash
git clone <this repository's URL> llm-decoder
cd llm-decoder
pip install -r requirements.txt
python app.py
```

The app opens at http://127.0.0.1:8765. On Windows you can also double-click **`start.bat`** (it installs the
requirements the first time). Paste the path of a `.gguf` file and press **Extract**.

## Requirements

| Needed for | What |
|---|---|
| Everything | Python 3.12 (the version it is tested with) and the packages in `requirements.txt`: `gguf`, `numpy`, `flask`, `xlsxwriter`, `openpyxl` |
| A model | any `.gguf` file, for example one downloaded with LM Studio |
| Logit lens, Test tab fast mode, the held-out check, tests that compare with llama.cpp | llama.cpp's `llama-server`: found automatically in LM Studio or on `PATH`, or set `LLAMA_SERVER` to its path |
| Excel round trip (optional) | Microsoft Excel or another program that edits `.xlsx` |
| Memory | about 4 bytes per weight to trace a model fully in RAM (0.8B ≈ 3 GB); bigger models switch to a slower low-memory mode |

Developed and tested on Windows 11. The Python code is not Windows-specific, but other systems are untested.

## What you can do

| Where | What |
|---|---|
| **Open a model** | Paste a `.gguf` path or Browse; press *Extract*. A workspace is created in `workspaces/<model>/`. |
| **Metadata** | *Table* view (edit one setting; long text such as the chat template opens with real line breaks) or *Code* view (the whole `metadata.json` in the VS Code editor; Ctrl+S saves and validates). |
| **Weights** | Excel-style grid with the exact float32 values. Click a cell and type, arrows to move, Ctrl+C / Ctrl+V with Excel, Del = 0, Ctrl+Z undo, Ctrl+S save. *Raw storage* shows how a number is really stored (quantized integers × scales). |
| **Grow model** | Add layers on top (pass-through or copy), add tokens to the vocabulary, insert rows/columns. Every change can be undone. |
| **Inspect** | *Anatomy*: what every tensor does (fixed rules on llama.cpp tensor names) and what kind each layer is. *Tokens*: tokens whose embeddings are closest to a token. *Neurons*: which words a row/column of a weight matrix pushes toward or reacts to. *Logit lens*: what the model would predict after layer N (real llama.cpp runs on cut-down copies). No AI involved. |
| **Chat & trace** | Talk to the model. It runs here in Python from the workspace's weights (your edits included, or the original file), and every token is recorded. Click any token to see: the model's top guesses; how much each layer's attention and MLP pushed it; a map of all neurons × layers coloured by push or activation; the top neurons and heads with buttons that open **their exact weights** in the grid (ffn_down column, ffn_up / ffn_gate row, head slices); the logit lens; and which earlier tokens each attention head looked at. **Which layers really decided it**: switch each layer's attention / MLP off (output replaced by its average over the text), re-run everything after it, and see how much the token loses its lead over a competitor (default: the strongest *different* answer). Push is only the direct effect and always favours the last layers; on 24 study chats the two named the same layer once (`studies/qwen35-0.8b/layer_effects.md`). On the reply overview, **Which layers decided this reply** does the same for every reply token at once (one switched-off re-run per part over the whole text). The neuron maps have *one scale / per layer*. *Saved traces → Manage* searches, filters (chats / from tests / replayed) and deletes traces in bulk. Click a neuron to colour every token by how strongly it fired. Traces are saved in `workspaces/<model>/traces/`. |
| **Test** | Paste questions as `question \| answer` (or load JSON / JSONL / CSV: GSM8K, MATH, MMLU, ARC, BoolQ… work as they are). The model answers each one and every answer is graded (numbers by value, multiple-choice letters, yes / no, text). **Fix wrong answers** in three steps: **1 · Analyse** finds the token where each wrong answer went wrong and which layers decided it (switch-off total effect on wrong − right); **2 · Compute** works out new weight values, by default as **one correction spread over the layers that decided it** (or in one part you choose), keeping right answers' outputs exactly; **3 · Check** asks questions the edit never saw, without and with it, and says whether it helps (McNemar test). *Apply* is the main button only for an edit that helps, and an edit bigger than half a matrix is blocked unless you choose *Apply anyway*. **Try 7 settings, keep what helps** does steps 2–3 for several settings. Tests are saved in `workspaces/<model>/tests/`. |
| **Changes & Export** | Exact % of weights changed (10 decimal places), build the edited GGUF, or round-trip the weights through an Excel workbook. |

Other ways to edit the same workspace: VS Code (`metadata.json`, `weights/*.csv`, `edit_weights.py` for
numpy code on whole tensors), or any AI chat (*Copy AI prompt* gives it the rules; see `README_FOR_AI.md`
in each workspace).

## How it works

### Chat & trace: how it works and its limits

- The model's maths is written out in numpy (`trace_core.py`): one generic decoder plus a few lines per family.
  Supported now (GGUF architecture names in brackets):

  | Family | Covers |
  |---|---|
  | Llama (`llama`) | Llama 1/2/3/3.1+/3.2/3.3, Mistral 7B / Nemo / Small 3 (files whose architecture is `llama`; the newer `mistral3` isn't yet), Mixtral (MoE), Yi, SmolLM 1/2, DeepSeek-R1-Distill-Llama |
  | Qwen (`qwen2`, `qwen3`, `qwen2moe`, `qwen3moe`, `qwen35`, `qwen3next`) | Qwen 2/2.5/3, DeepSeek-R1-Distill-Qwen, Qwen-MoE, Qwen3-30B-A3B / 235B-A22B, Qwen3.5 / Qwen3-Next |
  | Gemma (`gemma`, `gemma2`, `gemma3`) | Gemma 1/2/3 (soft-capping, sliding windows, post-norms) |
  | Phi (`phi3`) | Phi-3, Phi-3.5, Phi-4 |
  | Granite (`granite`, `granitemoe`) | IBM Granite 3.x dense and MoE |
  | DeepSeek (`deepseek2`) | DeepSeek V2 / V2-Lite / V2.5 / V3 / R1 / V3.1, Kimi K2 (latent attention, group-limited expert routing, YaRN) |
  | gpt-oss (`gpt-oss`) | gpt-oss-20b / 120b (attention sinks, clamped SwiGLU, MXFP4 experts) |
  | GLM (`glm4`, `glm4moe`) | GLM-4 (0414), GLM-Z1, GLM-4.5 / 4.6 / Air |
  | Others | Command-R / R+ (`command-r`), StarCoder 2 (`starcoder2`), OLMo 2 (`olmo2`), OLMoE (`olmoe`) |
  | Llama 4 (`llama4`) | Llama 4 Scout / Maverick (NoPE layers, top-1 sigmoid experts, 8192-token chunked attention and attention temperature — checked with an 8,250-token prompt) |
  | Mistral 3 (`mistral3`) | Ministral 3, Mistral Small 3.2+, Mistral Large 3 (text) |
  | LFM (`lfm2`, `lfm3`) | Liquid LFM2 / LFM2.5 / LFM3 (short-convolution layers) — LFM3 also checked on the real LFM3-VL-2B file |
  | State-space (`mamba`, `mamba2`, `jamba`, `granitehybrid`, `nemotron_h`, `falcon-h1`) | Mamba 1/2, Falcon-Mamba, Jamba, Granite 4.0 H, Nemotron-H / Nano 2, Falcon-H1 |
  | RWKV (`rwkv6`, `rwkv7`) | RWKV-6 (Finch), RWKV-7 (Goose), with RWKV's own tokenizer |
  | Gemma 3n (`gemma3n`) | Gemma 3n E2B / E4B (AltUp streams, LAuReL, per-layer embeddings, activation sparsity, KV sharing). Pushes are approximate: the 4 streams are merged non-linearly at the end. Matches llama.cpp token for token; probabilities within ~0.05 typically (0.26 at most in tests) |
  | **Not yet** | Newer families your llama.cpp also knows: Gemma 4, Qwen3.5-MoE, Qwen3-VL, DeepSeek V3.2 / V4, GLM-5, Kimi K3 / Linear, Mistral 4, MiniMax M2/M3, ERNIE 4.5, EXAONE 4, Seed-OSS, SmolLM3, LFM2-MoE, Nemotron-H-MoE, Cohere 2, and older ones (Falcon 1, GPT-NeoX, StableLM, Phi-2, BLOOM…). Vision/audio encoders: only the text model is traced |

  Models of other architectures show which one is missing. Tokenizers: SentencePiece (Llama 2, Mistral 7B,
  Gemma, Phi-3) and byte-level BPE with 26 of llama.cpp's splitting rules (gpt-2, llama-bpe/llama3, qwen2,
  qwen35, gpt-4o, llama4, tekken, deepseek-v3, default, falcon, starcoder, refact, command-r, smollm, olmo, mpt,
  glm4, lfm2, exaone, codeshell, jais, dbrx, smaug-bpe, deepseek-r1-qwen…). Each is checked against llama.cpp
  with a probe vocabulary built so that a wrong rule changes the tokens. A model with another rule is refused
  rather than tokenized wrong.
- Checked against llama.cpp:
  - `tests/test_trace.py` on a real model: identical tokenizer and chat template, the same greedy replies until two
    candidates are almost tied, next-token probabilities within ~0.3 percentage points on average (llama.cpp
    rounds activations to 8 bits inside Q8_0 matmuls; this engine computes in float32).
  - `tests/test_trace_arch.py` for every family, with no download: it writes a tiny random-weight GGUF of each
    architecture, runs it in llama.cpp (CPU, 32-bit cache) and here, and requires identical tokens and
    log-probabilities within 0.002. (The tracer computes GELU through the same 16-bit lookup table as llama.cpp's
    CPU code, so GELU models match too.) Random weights exercise the same maths as real ones; quirks only a real
    file has are covered by `tests/test_trace.py` / `tests/test_trace_real.py`.
  - Where llama.cpp's behaviour couldn't be pinned down by testing alone, its source (github.com/ggml-org/llama.cpp,
    `src/models/*.cpp`) was read and followed.
- Mixture-of-experts layers: the trace records which experts the router chose for each token and their weights.
  A neuron there is "expert e · #j"; its buttons open the right slice of the 3D expert tensor in the grid.
- Phi-3's sliding window (2,047 tokens or more) isn't applied, so Phi-3 traces of longer chats may differ.
- **Push** = direct logit attribution: a neuron's push is its activation × (its `ffn_down` column · the chosen
  token's output direction after the final norm). All pushes plus the embedding add up exactly to the token's
  score. Indirect effects (a neuron changing what later layers do) are not split out.
- Memory: models that fit are unpacked to float32 (about 4 bytes per weight; 0.8B ≈ 3 GB, 5–10 tokens/s).
  Bigger models switch to **low-memory mode** automatically: the file is memory-mapped and stays compressed,
  each tensor is unpacked when a layer needs it into a cache capped at ~45 % of RAM, and in MoE layers only the
  experts a token uses are unpacked. Results are bit-for-bit the same; it is just slower when the model doesn't
  fit in the cache (it unpacks again every token): roughly 20 s/token for Qwen3-30B-A3B, minutes per token for
  70B dense or DeepSeek-V3 671B on a 34 GB PC. The Chat tab shows the estimate before you start.
- Activations at position *t* are what the model computed while **reading** token *t*, to predict token *t+1*.
- A trace keeps the 32 strongest attention targets per head and position, not the full attention matrix.

### Fix wrong answers: how the computed edit works

**Where a wrong answer went wrong.** For a number answer with working, every whole calculation `a op b … = c` in the
reply is checked (not the tail of a longer one such as `(2 + 3) × 4 + 5 = 25`, nor fractions, powers, clock times or
divisions with a remainder); the first wrong result is the mistake (e.g. `1000 - 367 = 733` → fix it to `633`, so the rest of
the working follows from the right value). With every step right, the final answer is the mistake. For text answers
it is the first word of the answer. A `solution` field in the questions is used as the whole right reply instead.
The reply up to the mistake + the right text is replayed, and **every** token there the model wouldn't write on
its own becomes a *token fix* (all digits of `391`, `W` + `ent`…), each reading the corrected text before it.

For each token fix:
1. **Key** `k`: what the edited matrix reads at that position (the layer's neuron activations), times the factor
   its output is scaled by (an expert's router weight, the shared expert's gate; 1 for a dense MLP).
2. **Goal** `r`: the change the layer's output needs so the right token beats every other by the margin. It is
   found with the real model: the exact gradient of (right − wrong) score along the direct path (final norm,
   output rows, softcap, the layer's own output norm and, for attention edits, the MLP that re-reads it), scaled
   by a line search through the later layers and refined a few times. Settings that edit the same layer share
   their goals (computed once).
3. **Least squares**: the smallest change `ΔW` to `ffn_down` (all columns, or the N most fitting neurons), measured
   on the activations of the other saved chats so their outputs move as little as possible, with `ΔW·k = r`
   exactly for every fix and `ΔW·k = 0` exactly at the right answers' answer tokens:
   `ΔW = R (K A⁻¹ Kᵀ)⁻¹ K A⁻¹`, `A` = covariance of the kept activations (+ a small ridge).
4. **Where**: by default the correction is spread over the parts that decided the mistakes (next section). For one
   part, *the most responsible* is the attention or MLP with the biggest total effect, and *the smallest change* is the
   layer needing the smallest push ÷ its typical output (on Qwen3.5-0.8B the last layer needed ~25× less than layer
   21). If some token fixes can't be reached in one layer (behind an output norm a layer can only move the residual
   so far), its edit is kept and up to 2 earlier layers get an edit for the rest.

What is edited in each kind of layer:

| Layer | Edited matrix | Models |
|---|---|---|
| Dense MLP | `ffn_down` | Llama, Qwen, Mistral, Phi, Granite, StarCoder 2, LFM… |
| Dense MLP + output norm | `ffn_down`, solved through the norm | Gemma 2 / 3, OLMo 2 |
| Attention and MLP side by side | `ffn_down` | Command-R, Falcon-style |
| Mixture of experts with a shared expert | `ffn_down_shexp` (every token uses it) | Qwen-MoE, DeepSeek, GLM-MoE, Llama 4 |
| Mixture of experts, routed only | `ffn_down_exps` of the expert each fixed token is sent to first (one solve per expert) | Qwen3-MoE, Mixtral, OLMoE, GPT-OSS, Granite-MoE, Jamba |
| No MLP (Mamba, RWKV) / merged streams (Gemma 3n) / low-memory mode | not supported (refused with the reason) | |

Saved edits are `computed/<name>.npz` (patches: tensor, expert or −1, columns, values to **add**) plus a `.json`
with what they fix. They are added after grid / CSV / `edit_weights.py` edits, in Chat & trace, Test, measure and
Export; delete one to undo it exactly. Compute an edit on the same weights you will use it with (the Test tab warns
when a test ran on the original file).

#### One correction spread over several layers (the default)

Forcing a whole correction into one matrix needs big changes there. **Spread over the layers that decided it** shares
one correction among the parts (attention and MLP) whose measured total effect on the mistakes is largest, like
MIT's MEMIT method:

1. **Which parts**: those that together hold 80 % of the measured effect (2–6 of them), outside the first quarter of
   the layers (they read the input: on GSM8K, attention L0 decided the most but would have needed ~28,000 % of its
   matrix). Or type a range, e.g. `12-20`.
2. **Goal**: with the real model, where the residual after the last of them must be for each wrong token, so the
   right one wins by the margin.
3. **Shares**: in model order, each part moves the residual after its own layer by *weight ÷ (its weight + the
   weights of the parts after it)* of what is still missing; behind an output norm or for an attention output (which
   this layer's MLP re-reads) the needed change is found with Gauss–Newton steps on the exact Jacobian. Each part is
   then solved by least squares in its own matrix as above (protected answers exactly 0).
4. **Re-run** after every part, so the next one only covers what is still missing; the last part takes the rest with
   the exact goal.

On the test models the biggest change per matrix drops from 8 % to 3 % (Llama), 19 % to 10 % (Command-R) and
39,805 % to 70 % (Gemma 2, output norms) for the same fixes. All the parts go into one LoRA adapter for the check.

#### Does it help questions it never saw? (the held-out check)

An edit reliably fixes the answers it was computed from. That alone says nothing about **other** questions of the
same kind, which is what matters, so the Test tab measures it before you apply anything:

1. **Compute from** 50–80 % of the questions; the rest are *held out* (never used to compute the edit). They are the
   first ones in the file's order, or (*More → Pick them at random*) a random pick that is the same every time; use
   that when the file is sorted by topic or difficulty, so the held-out questions are of the same kind.
2. **Check on the held-out questions**: llama.cpp asks them once without and once with the edit, with the test's own
   prompt, instruction and token limit, and grades both. The edit is loaded as a **LoRA adapter** (`ΔW = B·A`, the
   exact SVD of the edit), so one server switches it on and off; edits to routed experts, which LoRA can't hold, use
   a temporary exported model. It can also re-ask the wrong answers it was computed from: were they fixed in real
   replies, not just at the corrected tokens?
3. **Verdict** on the paired answers: *fixed* = wrong → right, *broken* = right → wrong. McNemar's exact test gives the
   chance of a split at least that uneven if the edit did nothing. **Helps** = more fixed than broken with p < 0.05;
   **hurts** = more broken, p < 0.05; else **no clear effect**, with the 95 % range of the change.
4. **Apply** is the main button only for an edit that helps; otherwise it says *Apply anyway* / *Apply without
   checking* and asks first.

The card of a computed edit also shows **Right answers: top token changed**, a quick look at side effects before the
real check. With 10 or more right answers, the last fifth of them is kept out of the fit and used for this number;
with fewer, it is measured on answers the edit was fitted to keep (the card says which), so expect more change
elsewhere. An edit can only be applied to the weights it was computed for.

**Try 7 settings, keep what helps** does all of it for 7 settings (spread over the deciding layers: shared correction and exact; the most responsible part: rank 1, and rank 4 with a high keep strength; the smallest-change layer: rank 1, rank 4 and exact with one token per answer), skips edits too big to apply safely, and
multiplies each p-value by the number of edits checked (Bonferroni), so picking the best of several can't make
chance look like an improvement. If every edit is too big to apply safely, none is checked and it says so. If none helps, it says so too: an edit that only re-fixes its own questions isn't
worth applying. More questions give the check more power (with 150 held out, roughly 13 more fixed than broken are
needed to be sure).

### Questions and grading (every common dataset format)

| Format | Example | Graded as |
|---|---|---|
| `question \| answer` lines, CSV, TSV, JSON, JSONL | any field names: question / q / prompt / input / problem, answer / a / target / output / label | number, yes/no or text |
| GSM8K | `"answer": "... #### 42"` | number |
| MATH | `"problem"`, `"solution": "... \boxed{3/4}"` | number, solution kept as the reference working |
| Multiple choice | MMLU (`choices` + index), ARC (`{text, label}` + `answerKey`), MedQA (`options` dict), HellaSwag (`endings` + `label`), or a question listing `A) … B) …` with a letter | the letter chosen ("answer is (B)", `**B**`, a reply starting "B."); or the one choice whose text the reply contains |
| Yes / no | BoolQ `true`, `"no"` | the first yes / no / true / false of the answer |

A number answer is the reply's *final* answer: `\boxed{}` > `#### n` > after "answer is / answer:" (for "the answer
is 9 × 2 = 18", the result) > the last bold number (not a "Step 3:" heading) > the last number. Thousands separators, `$` and `%`, fractions (`3/4`,
`\frac{3}{4}`, `2 1/2`), number words (`eight`, `twenty-five`, `a dozen`) and lists (`1,2,3` is three numbers, not 123)
are understood; a rounded value counts when it rounds the same (`33.33` for 33.3333). Text answers ignore case and
accents, accept plurals and don't count a negated answer ("not Paris"); "that is not true" is a no. In multiple
choice a stated answer ("the answer is C") wins over options discussed after it. A reply cut off by the token limit is marked
**cut off**; its number counts only if the reply marked it as its answer, so a number it happened to be writing is
never graded (or "fixed") as an answer. On the 300 saved GSM8K replies the grader agrees with a careful reading on
all but unfinished ones; 47 of them were cut off at 1024 tokens.

## Notes and limits

- Long tasks show progress; those that can be interrupted have a **Stop** button. While one runs, starting another says which task is busy (and offers to stop it), and the workspace can't be changed until it has finished.
- **The model file was moved, deleted or downloaded again.** A workspace reads its numbers from the original `.gguf`. If the file is missing, the workspace says so and keeps your tests, results and edits; **Locate file…** points it at the file's new place. A file with a new date is accepted when it holds the same model (same tensors, and the same sampled contents), so downloading a model again doesn't cost you the workspace. **Extract again** replaces the model data and edits but keeps the saved tests and traces.
- The app has no login and only answers this computer (127.0.0.1): requests from other web sites' pages open in the same browser are refused. The Metadata code view loads the Monaco editor from a CDN (jsDelivr) when you are online and falls back to a plain text box when you aren't; nothing about your models is sent anywhere.
- Fast mode keeps one copy of the workspace weights for llama.cpp in `workspaces/<model>/cache/` (shown in the Test tab, rewritten when the weights change, ✕ deletes it).

- The grid scrolls over the **whole** tensor (e.g. all 128,000 × 2,048 values of an embedding). The first
  256×256 cells are also kept as CSV for VS Code/AI tools; edits elsewhere are saved in `edits/<tensor>.json`.
- Edited tensors **keep their format** (Q8_0, Q4_0/1, Q5_0/1, Q2_K…Q6_K, BF16, F16, F32). Only the blocks you
  edited are re-encoded; every other block keeps its original bytes. K-quants use the encoders in `kquants.py`.
- Quantized formats can only store certain values per block, so a typed number snaps to the nearest one
  (and its block neighbours can shift slightly). Pick **Keep my exact values** on export to store edited
  tensors as F32 instead (every digit kept; edited tensors get bigger). IQ* formats (IQ2/IQ3/IQ4) have no encoder, so edited IQ tensors are stored as F16.
- Big models:
  - **Split downloads** (`model-00001-of-00004.gguf` …): open any part and all parts are read as one model.
    Export writes one merged `.gguf`.
  - **Export memory**: edited tensors are processed a few million numbers at a time, so memory stays small
    even for multi-billion-number expert tensors. The exception is `edit_weights.py`, which loads the tensors
    it touches.
  - **Disk space**: export and the logit lens check free disk space first and say how much is needed.
  - **Logit lens cache**: cut-down copies are capped at one model's size; the least recently used copy is
    removed first.
  - **Hardware limit**: running a model (logit lens, LM Studio) still needs RAM for it.
- Vision models: the `mmproj-*.gguf` encoder next to the model is detected; open it as its own workspace.
  **Add to LM Studio** puts the edited model and its encoder (edited or original) into LM Studio's models folder.
- The Excel workbook holds the same top-left block of each tensor as the CSV files (256 × 256 by default), not
  whole tensors: a 2B model has ~2 billion numbers, far more than Excel can work with. Edit other cells in the grid.
- The logit lens uses LM Studio's bundled `llama-server.exe` (newest version found). To use another
  llama.cpp build, set the `LLAMA_SERVER` environment variable to its `llama-server` path. Its cut-down model
  copies are cached in `workspaces/<model>/lens/` (Inspect → Logit lens → *Clear cache* frees the space).

## Studies

Experiments on Qwen3.5-0.8B (Q8_0) are in `studies/qwen35-0.8b/`: the scripts, their result files and a write-up
for each. They are reported as they came out, including the edits that did not help.

| Write-up | What it looked at | Result |
|---|---|---|
| [report_original.md](studies/qwen35-0.8b/report_original.md) | 100 traced chats in 6 categories | 78/100 right; confidence and the layer where the answer first leads, right vs wrong; specialist neurons |
| [layer_effects.md](studies/qwen35-0.8b/layer_effects.md) | direct push vs total effect (switch-off) on 24 chats | the two named the same top part in only 1 of 24 chats, which is why fixes use the total effect |
| [edit_tests.md](studies/qwen35-0.8b/edit_tests.md), [edit_results.md](studies/qwen35-0.8b/edit_results.md) | removing or doubling heads and neurons, replayed on the 100 chats | how much each edit moves the original replies |
| [bench_results.md](studies/qwen35-0.8b/bench_results.md), [gsm8k_original_vs_l23_head_6_x0.md](studies/qwen35-0.8b/gsm8k_original_vs_l23_head_6_x0.md) | edited models run in llama.cpp on the 100 questions and GSM8K | L23 head 6 ×0: 82/100 vs 78/100 on the study set, but 156/300 vs 169/300 on GSM8K (McNemar p = 0.11): no reliable gain |
| [gsm_fix_experiment.md](studies/qwen35-0.8b/gsm_fix_experiment.md) | one exact computed edit for 78 GSM8K mistakes in one layer | needed a change of 357 % of the matrix; stopped |
| [gsm_search_experiment.md](studies/qwen35-0.8b/gsm_search_experiment.md) | computed edits from GSM8K 1–150, to be checked on 151–300 | partial: the most responsible part (attention L0) would need ~28,000 % changes; the held-out check was not finished |

The study scripts expect the Qwen3.5-0.8B workspace (`workspaces/Qwen3.5-0.8B-Q8_0`, made by extracting the model)
and, for the trace-based ones, its saved traces.

## Folder layout

| Path | What it is |
|---|---|
| `app.py`, `static/index.html` | the web app (Flask server + the single-page UI) |
| `start.bat`, `requirements.txt` | Windows launcher; Python packages |
| `README.md`, `GUIDE.md`, `LICENSE` | this overview; step-by-step guide to running it; MIT license |
| `gguf_core.py` | workspaces: extract a GGUF, edit, add layers / tokens, undo, export |
| `inspect_core.py` | layer inspection on the weights, logit lens (llama.cpp) |
| `effects_core.py` | which layers really decided a token (switching parts off) |
| `trace_core.py` | Chat & trace: the exact forward pass of every architecture family, with attribution |
| `test_core.py`, `grade_core.py` | the Test tab: running question sets, grading every dataset format (an app module, not a test) |
| `fix_core.py`, `validate_core.py` | Fix wrong answers: computed edits, spread edits, held-out check (LoRA + McNemar) |
| `kquants.py`, `gguf_to_excel.py` | K-quant encoders; read-only Excel dump |
| `tests/` | the test scripts (each runs on its own: `python tests/<name>.py`) |
| `studies/qwen35-0.8b/` | the Qwen3.5-0.8B studies: scripts, result files and write-ups (`*.md`) |
| `workspaces/` | created by the app: one folder per extracted model (weights, tests, traces, exports); not in git |

## Command line

```
python gguf_core.py extract model.gguf
python gguf_core.py add-layers <workspace> --count 2 --mode identity
python gguf_core.py add-tokens <workspace> "<think>" "</think>"
python gguf_core.py export <workspace>
python gguf_to_excel.py model.gguf          # quick read-only Excel dump
```

## Tests

Each test is a plain script; run it from the project folder. Tests that compare with llama.cpp need `llama-server`
(see [Requirements](#requirements)); `test_trace.py`, `test_trace_real.py`, `test_e2e_api.py` and
`test_excel_real.py` also need a real model.

```
python tests/test_workspace.py      # extract / edit / grow / undo / export / Excel / exact percentages
python tests/test_gguf_to_excel.py  # Excel exporter edge cases
python tests/test_raw_storage.py    # raw-storage decoder vs. the official gguf decoder
python tests/test_kquants.py        # K-quant encoders
python tests/test_inspect.py        # layer inspection tools
python tests/test_big.py            # split models, low-memory export, disk checks, lens cache (synthetic MoE model)
python tests/test_trace.py          # chat tracer vs. llama.cpp on a real model: tokenizer, template, replies, probabilities, attribution
python tests/test_trace_arch.py     # every architecture family vs. llama.cpp on tiny synthetic models (no download)
python tests/test_trace_real.py model.gguf   # any real model file vs. llama.cpp (tokenizer, template, replies)
python tests/test_fix.py            # Test tab + Fix answers: parsing, grading, calculation steps, token fixes, every layer kind (dense, output norm, parallel, shared / routed experts), save / export / delete
python tests/test_math.py           # the maths, each against an independent method: least squares (optimality conditions, Eckart–Young), McNemar (brute force), LoRA form, residual prediction and goal gradient on 6 layer kinds, llama.cpp LoRA = edited file, the held-out check and search end to end
python tests/test_grade.py          # grading: 100+ reply styles and dataset formats (numbers, fractions, multiple choice, yes/no, text, cut-off replies, answer positions)
python tests/test_app.py            # the app's own rules: who may call it, what is refused while a task runs, a model file moved / downloaded again / extracted again, bad input
python studies/qwen35-0.8b/gsm_search_experiment.py   # real run: edits from GSM8K 1-150, checked on 151-300 (gsm_search_experiment.md)
python studies/qwen35-0.8b/verify_traces.py original   # every saved study chat: what Chat & trace highlights vs. a fresh pass and llama.cpp
python studies/qwen35-0.8b/edit_tests.py                # ablate / amplify heads and neurons, replay the 100 chats (edit_tests.md)
python studies/qwen35-0.8b/bench.py export "L23 head 6 ×0"   # edited GGUF in exports/ (your edit_weights.py is put back)
python studies/qwen35-0.8b/bench.py run l23_head_6_x0   # llama.cpp: 100 study questions + GSM8K; bench.py report -> bench_results.md
python tests/test_e2e_api.py        # full run on a real model through the running app (start the app first)
python tests/test_excel_real.py     # Excel round trip on a real model, edited in real Microsoft Excel (app running, Excel installed)
```

## License

[MIT](LICENSE)
