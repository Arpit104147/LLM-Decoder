# Guide: running LLM Decoder

This guide takes you from a fresh download to a first session: install, start the app, open a model, chat with it,
test it and export an edited copy. For what each feature does in depth, see the [README](README.md).

## 1. What you need

| | Required? | Notes |
|---|---|---|
| **Python 3.12** | yes | https://www.python.org/downloads/. On Windows, tick *Add python.exe to PATH* in the installer. Check with `python --version`. |
| **A GGUF model** | yes | Any `.gguf` file. A small one is best to start with (0.5B–2B parameters, Q8_0 or Q4_K_M, about 0.5–2 GB). |
| **LM Studio** | recommended | https://lmstudio.ai. It downloads models and ships llama.cpp's `llama-server`, which the logit lens, Test-tab fast mode and the held-out check use. Instead of LM Studio you can install llama.cpp yourself (see step 7). |
| **Microsoft Excel** | optional | Only for the Excel round trip in *Changes & Export*. |
| **VS Code** | optional | *Open in VS Code* opens the workspace folder for editing `metadata.json`, CSVs and `edit_weights.py`. |

RAM: tracing a model in Chat & trace unpacks it to float32, about 4 bytes per weight (a 0.8B model needs about 3 GB).
Larger models still work, but in a slower low-memory mode.

## 2. Get the code

With git:

```bash
git clone <this repository's URL> llm-decoder
cd llm-decoder
```

Or download the repository as a ZIP from GitHub (*Code → Download ZIP*) and unpack it.

## 3. Install the Python packages

A virtual environment keeps the packages separate from the rest of your system. It's optional but recommended.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `gguf`, `numpy`, `flask`, `xlsxwriter` and `openpyxl`.

If PowerShell refuses to run `Activate.ps1`, use Command Prompt instead (`.venv\Scripts\activate.bat`), or skip the
virtual environment and run `pip install -r requirements.txt` directly.

## 4. Start the app

```bash
python app.py
```

Your browser opens **http://127.0.0.1:8765**. The terminal shows `LLM Decoder running at http://127.0.0.1:8765`.
Keep the terminal open while you use the app, and press **Ctrl+C** in it to stop.

| Option | How |
|---|---|
| Windows shortcut | Double-click **`start.bat`**. It installs the packages if they are missing, then starts the app. |
| Don't open a browser | `python app.py --no-browser`, then open http://127.0.0.1:8765 yourself. |

The app only listens on your own computer (127.0.0.1); nothing is reachable from the network.

## 5. Open a model

1. Find your `.gguf` file. Models downloaded with LM Studio are under `C:\Users\<you>\.lmstudio\models\` on Windows,
   or `~/.lmstudio/models/` on macOS and Linux (LM Studio → *My Models* → *Show in folder* also finds them).
2. On the app's start page, paste the full path into **GGUF file** (or press **Browse…**) and press **Extract**.
3. The app creates a **workspace** in `workspaces/<model name>/` with the metadata and the weights. This takes from a few seconds to a few minutes depending on the model's size. Your original file is
   only read, never changed.

Workspaces you opened before are listed under the start page's open box; click one to continue where you left off.

For a split model (`model-00001-of-00004.gguf` …), open any of the parts. For a vision model, the
`mmproj-*.gguf` next to it is the image encoder and opens as its own workspace.

## 6. A first session

The workspace page has these tabs, roughly in the order you'd use them:

| Tab | Try this first |
|---|---|
| **Overview** | Architecture, size, layer count and what kind each layer is. |
| **Metadata** | Settings such as the context length and the chat template. *Table* edits one value; *Code (JSON)* edits the whole file (Ctrl+S saves). |
| **Weights** | Pick a tensor, e.g. `blk.0.ffn_down.weight`, and see its exact numbers. Click a cell and type to change it; Ctrl+Z undoes. |
| **Inspect** | *Anatomy* explains every tensor. *Logit lens* shows what the model would say after each layer (needs `llama-server`). |
| **Chat & trace** | Send a message. When the reply is done, click any reply token to see the top guesses, how much each layer pushed it, the neurons and heads involved, and **Which layers really decided it**. |
| **Test** | Paste a few questions, one per line as `question \| answer` (e.g. `What is 17*23? \| 391`), or load a JSON / JSONL / CSV dataset such as GSM8K or MMLU. Press **Run test**: every answer is graded. |
| **Changes & Export** | Shows exactly how much changed, then **Export** writes a new `.gguf` to `workspaces/<model>/exports/`. **Add to LM Studio** copies it into LM Studio's models folder so you can load it there. |

### Fixing wrong answers (Test tab)

After a test has run and some answers are wrong:

1. **1 · Analyse** finds where each wrong answer went wrong and which layers decided it. Keep *Compute from* at
   70 %: the other 30 % of the questions are held out to check the edit honestly.
2. **2 · Compute edit** works out the weight change. The defaults (*spread over the layers that decided it*,
   *one shared correction*) are the recommended starting point.
3. **3 · Check** asks the held-out questions without and with the edit and tells you whether it **helps**,
   **hurts** or has **no clear effect** (a statistical test, not a guess).
4. **Apply** only an edit marked *helps*. **Try 7 settings, keep what helps** runs steps 2–3 for several
   settings at once, which takes longer.

Small question sets give the check little power: with 20 held-out questions, only a large improvement can be
told apart from chance. A hundred or more questions are better.

Every edit can be undone: applied computed edits are listed under *Applied computed edits* in the Test tab with a
**Delete** button, and grid, layer and token changes have undo.

## 7. Using llama.cpp without LM Studio

The app looks for `llama-server` in this order:

1. the `LLAMA_SERVER` environment variable;
2. LM Studio's bundled builds on Windows (`~/.lmstudio/extensions/backends/*/llama-server.exe`, newest first);
3. `llama-server` on your `PATH`.

To use your own build, download a release from https://github.com/ggml-org/llama.cpp/releases (or build it), then
before starting the app:

```powershell
$env:LLAMA_SERVER = "C:\path\to\llama-server.exe"     # Windows PowerShell
```

```bash
export LLAMA_SERVER=/path/to/llama-server             # macOS / Linux
```

Use a recent build: newer model families need a recent llama.cpp.

## 8. Where your data goes and how to free space

Everything the app makes is inside `workspaces/<model>/`. That folder is not part of the repository (see
`.gitignore`).

| Folder | What | Safe to delete? |
|---|---|---|
| `weights/`, `metadata.json`, `tensors.json` | the extracted model you edit | only together with the whole workspace |
| `traces/` | saved Chat & trace recordings (often ~100 MB each) | yes. In the app: *Saved traces → Manage*. Saved tests keep their tokens and re-trace when needed. |
| `tests/` | saved tests and their results | yes, in the Test tab |
| `exports/` | edited `.gguf` files you exported | yes; export again to rebuild |
| `cache/`, `lens/` | llama.cpp copies for fast mode and the logit lens | yes. The Test tab's ✕ and *Inspect → Logit lens → Clear cache* do it. |

To remove a model completely, close the app and delete its `workspaces/<model>/` folder.

**Deleting the model file itself** (for example in LM Studio, to free space) is safe for the workspace: your tests,
results and edits stay. The workspace shows a notice that the file is missing. Download the model again to the
same place and it is picked up by itself, or press **Locate file…** if it is somewhere else now. Until then the
tabs that need the weights (Weights, Chat & trace, Analyse, Export) can't work.

## 9. Running the tests

From the project folder:

```bash
python tests/test_grade.py
```

```bash
python tests/test_workspace.py
```

```bash
python tests/test_math.py
```

```bash
python tests/test_app.py
```

Each test prints its checks and ends with a pass count or `ALL … CHECKS PASSED`. None of them needs a model download.
`test_math.py` and the `test_trace*.py` scripts compare with llama.cpp, so they need `llama-server`. The README's
[Tests](README.md#tests) section lists every script and what it needs.

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| `python` is not recognised | Python is not on `PATH`. Reinstall with *Add python.exe to PATH* ticked, or use `py app.py` on Windows. |
| `ModuleNotFoundError: No module named 'gguf'` (or flask, numpy…) | Run `pip install -r requirements.txt` in the same environment you start the app from. If you made a virtual environment, activate it first. |
| "The model file isn't where it was" | The `.gguf` the workspace was extracted from was moved or deleted. Put it back or press **Locate file…** (step 8). Don't use *Extract again* for this: it removes your edits (saved tests and traces are kept). |
| "Another task is still running" | One background task runs at a time, and the workspace can't be changed while it does. Wait, or stop the task if it has a **Stop** button. |
| The page doesn't open | Open http://127.0.0.1:8765 yourself. If port 8765 is in use, stop the other program, or change `PORT` near the top of `app.py`. |
| "llama.cpp's llama-server wasn't found" | Install LM Studio, or set `LLAMA_SERVER` (step 7). Only the logit lens, fast mode and the held-out check need it. |
| "The '…' architecture isn't supported by the tracer yet" in Chat & trace | The tracer doesn't know that model family yet (the README lists the supported ones). Metadata, weights, Grow and Export still work. |
| Chat & trace is very slow | The model doesn't fit in RAM and runs in low-memory mode. Use a smaller model or a smaller quantization. |
| A task seems stuck | Long tasks show progress and have a **Stop** button. Starting another task tells you which one is busy. |
| Export says there isn't enough disk space | Free space (step 8) or choose another drive. The message says how much is needed. |
| An edited tensor's values moved slightly after typing | Quantized formats can only store certain values per block. Choose *Keep my exact values* on export to store edited tensors as F32. |
