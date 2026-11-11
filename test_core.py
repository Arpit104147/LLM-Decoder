"""
Test: ask the model a list of questions with known answers, grade each traced reply, find where each wrong answer
went wrong and which layer pushed it, and turn that into a computed edit (fix_core) with one click.

Question formats (pasted text or a file):
  question | answer                    one per line (text answers: several right answers as  a / b / c)
  JSON list / JSONL of {"question", "answer"}   also {"q", "kw": [...]} or {"q", "num"} and GSM8K's
  {"question", "answer": "... #### 42"} (or {"row": {...}})
  CSV with question,answer columns
"""
import copy
import csv
import io
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

import gguf_core as core
import inspect_core as ic
import trace_core as tc
import fix_core as fc
import effects_core as ec

MARKUP = re.compile(r"^[\s*_`\"'“”‘’()\[\]#>:.,-]*$")


def tests_dir(ws):
    return core.workspace_dir(ws) / "tests"


# ------------------------------------------------------------------ questions and grading (grade_core)
from grade_core import parse_questions, grade, item as _item, _num, numbers as _numbers  # noqa: E402,F401


# ------------------------------------------------------------------ running
def _meta(ws, which):
    d = core.workspace_dir(ws)
    if which == "original":
        return core.original_metadata(core.open_source(d, check=False)[1])
    return core.load_user_metadata(d)


def _prompt(it, instruction):
    return it["question"] + (" " + instruction.strip() if instruction and instruction.strip() else "")


def _stamp(ws):
    return [str(x) for x in tc._ws_stamp(core.workspace_dir(ws))]


def _model_file(ws, which, log):
    """The GGUF llama.cpp runs: the source file, or the workspace weights (all edits) written once per change."""
    d = core.workspace_dir(ws)
    if which == "original":
        return Path(core.load_info(d)["gguf_path"])
    for old in (d / "exports" / "_test_model.gguf", d / "exports" / "_test_model.json"):   # where earlier versions kept it
        old.unlink(missing_ok=True)
    out = d / "cache" / "test_model.gguf"
    sf = out.with_suffix(".json")
    st = _stamp(ws)
    if out.exists() and sf.exists() and json.loads(sf.read_text(encoding="utf-8")) == st:
        return out
    log("Fast mode: writing the workspace weights (with your edits) to cache/test_model.gguf for llama.cpp…")
    out.parent.mkdir(exist_ok=True)
    core.export_gguf(core.workspace_dir(ws).name, out_path=out, log=lambda m: None, exact=True, record=False)
    sf.write_text(json.dumps(st), encoding="utf-8")
    return out


class _Llama:
    """llama-server on a free port, CPU only, `par` slots."""
    def __init__(self, model, ctx, par, log, extra=()):
        import socket
        import subprocess
        import tempfile
        import urllib.request
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.logf = tempfile.TemporaryFile()
        self.p = subprocess.Popen([str(ic.find_llama_server()), "-m", str(model), "--port", str(self.port), "-c", str(ctx * par),
                                   "-np", str(par), "-ngl", "0", "--device", "none", "--no-warmup", *map(str, extra)],
                                  stdout=self.logf, stderr=self.logf, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return
            except Exception:
                if self.p.poll() is not None:
                    self.logf.seek(0)
                    raise RuntimeError("llama.cpp didn't start: " + self.logf.read().decode(errors="replace")[-800:])
                time.sleep(0.3)
        self.close()
        raise RuntimeError("llama.cpp didn't start in time")

    def complete(self, ids, n):
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1,
                                                      "samplers": ["top_k"], "cache_prompt": False, "return_tokens": True}).encode())
        return json.loads(urllib.request.urlopen(req, timeout=3600).read())

    def set_adapters(self, scales):
        """Scale of each LoRA adapter (in the order loaded); 0 = off. Only between batches (no request running)."""
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/lora-adapters", headers={"Content-Type": "application/json"},
                                     data=json.dumps([{"id": i, "scale": float(s)} for i, s in enumerate(scales)]).encode())
        urllib.request.urlopen(req, timeout=60).read()
        got = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/lora-adapters", timeout=60).read())
        if [round(float(a["scale"]), 6) for a in got] != [round(float(s), 6) for s in scales]:
            raise RuntimeError(f"llama.cpp didn't switch the edit on/off as asked: {got}")

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        self.logf.close()


def _stop_id(tok):
    for name in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "</s>"):
        i = getattr(tok, "ids", {}).get(name)
        if i is not None and tok.is_stop(i):
            return int(i)
    return int(tok.eos) if getattr(tok, "eos", None) is not None else None


def _grade_row(it, reply, unfinished):
    """grade(); a reply cut off by the token limit hasn't finished its working, so for a number answer only an
    answer it marks as one counts (not whatever number it happened to write last)."""
    return grade(it, reply, unfinished)


def run(ws, text, instruction="Answer briefly.", max_tokens=160, which="edited", name="", engine="tracer", parallel=3,
        emit=None, cancel=None, log=print):
    """Ask every question (greedy) and grade it. engine="tracer": the app's Python engine writes each reply and traces
    it. engine="llama": llama.cpp writes the replies (several at once, much faster); a reply is traced later, by
    replaying it, only when something needs its trace (Analyse, or the Trace button)."""
    items = parse_questions(text)
    tid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    out = tests_dir(ws) / f"{tid}.json"
    out.parent.mkdir(exist_ok=True)
    res = {"id": tid, "name": name or f"{len(items)} questions", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "weights": which, "computed_edits": [p.stem for p in core.computed_patches(core.workspace_dir(ws))],
           "instruction": instruction, "max_tokens": int(max_tokens), "engine": engine, "weights_key": _stamp(ws), "items": []}
    log(f"{len(items)} question(s), weights: {'workspace (with your edits)' if which == 'edited' else 'original'}, "
        f"{'llama.cpp' if engine == 'llama' else 'Python tracer'}")
    save = lambda: out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def add(row):
        res["items"].append(row)
        res["right"] = sum(x["correct"] for x in res["items"])
        save()
        if emit:
            emit(row)
        log(f"[{len(res['items'])}/{len(items)}] {'right' if row['correct'] else 'WRONG'}  {row['reply'][:60]!r}")

    stopped = False
    if engine == "llama":
        meta = _meta(ws, which)                                # the same tokenizer and chat template as the tracer
        tok = tc.make_tokenizer(meta)
        prompts = [tc.prompt_ids(tok, tc.render_chat(meta, [{"role": "user", "content": _prompt(it, instruction)}]))
                   for it in items]
        stop = _stop_id(tok)
        srv = _Llama(_model_file(ws, which, log), max(len(p) for p in prompts) + int(max_tokens) + 16, int(parallel), log)
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(int(parallel)) as ex:
                futs = {}
                for i, it in enumerate(items):
                    futs[ex.submit(lambda i=i: (time.time(), srv.complete(prompts[i], int(max_tokens))))] = i
                for fu in as_completed(futs):
                    i = futs[fu]
                    if cancel is not None and cancel.is_set():
                        log("Stopped")
                        stopped = True
                        for f2 in futs:
                            f2.cancel()
                        srv.close()                    # ends the replies being written now (they are dropped)
                        break
                    t, r = fu.result()
                    gen = [int(x) for x in r.get("tokens", [])]
                    if r.get("stop_type") == "eos" and stop is not None and (not gen or not tok.is_stop(gen[-1])):
                        gen.append(stop)
                    reply = tok.decode([x for x in gen if not tok.is_stop(x)])
                    cut = r.get("stop_type") == "limit" or (len(gen) >= int(max_tokens) and not (gen and tok.is_stop(gen[-1])))
                    ok, got, off = _grade_row(items[i], reply, cut)
                    add({**items[i], "i": i, "reply": reply, "correct": ok, "got": got, "offset": off, "trace": None, "unfinished": cut,
                         "ids": prompts[i] + gen, "n_prompt": len(prompts[i]), "n_reply": len(gen),
                         "seconds": round(time.time() - t, 1)})
        except Exception:
            if not stopped:
                raise
        finally:
            srv.close()
        res["items"].sort(key=lambda x: x["i"])
    else:
        _model, tok = tc.get_model(ws, which, log)
        for i, it in enumerate(items):
            if cancel is not None and cancel.is_set():
                log("Stopped")
                stopped = True
                break
            t = time.time()
            r = tc.chat(ws, [{"role": "user", "content": _prompt(it, instruction)}], max_new=int(max_tokens), temperature=0,
                        which=which, log=lambda m: None)
            ids = [int(x["id"]) for x in r["tokens"]]
            # cut off = the token limit was reached before the model ended its reply (ending exactly at the limit isn't)
            cut = r["timing"]["n_gen"] >= int(max_tokens) and not tok.is_stop(ids[-1])
            ok, got, off = _grade_row(it, r["reply"], cut)
            add({**it, "i": i, "reply": r["reply"], "correct": ok, "got": got, "offset": off, "trace": r["id"], "unfinished": cut,
                 "ids": ids, "n_prompt": int(r["n_prompt"]),      # kept, so the reply can be traced again if its trace is deleted
                 "n_reply": r["timing"]["n_gen"], "seconds": round(time.time() - t, 1)})
    res["done"] = not stopped
    if stopped:
        res["stopped"] = True
        res["name"] += f" (stopped at {len(res['items'])}/{len(items)})"
    save()
    log(f"Score: {res.get('right', 0)}/{len(res['items'])}")
    return res


def cache_info(ws):
    """The fast-mode copy of the workspace weights (rewritten whenever the weights change)."""
    f = core.workspace_dir(ws) / "cache" / "test_model.gguf"
    st = f.with_suffix(".json")
    fresh = f.exists() and st.exists() and json.loads(st.read_text(encoding="utf-8")) == _stamp(ws)
    return {"path": str(f), "exists": f.exists(), "bytes": f.stat().st_size if f.exists() else 0, "up_to_date": fresh}


def clear_cache(ws):
    d = core.workspace_dir(ws)
    for f in (d / "cache" / "test_model.gguf", d / "cache" / "test_model.json", d / "exports" / "_test_model.gguf",
              d / "exports" / "_test_model.json"):
        f.unlink(missing_ok=True)
    return cache_info(ws)


def ensure_traces(ws, tid, which_items, log=print):
    """Trace the replies (by index) of a fast-mode test that have no trace yet, by replaying them through the tracer
    with the weights the test ran on. Returns the updated test."""
    r = get_test(ws, tid)
    need = [x for x in r["items"] if x["i"] in set(which_items) and not x.get("trace")]
    if not need:
        return r
    if r.get("weights_key") and r["weights"] == "edited" and r["weights_key"] != _stamp(ws):
        raise ValueError("The workspace weights changed since this test ran (an edit was applied or removed), so its "
                         "replies can't be traced as they were written. Run the test again.")
    log(f"Tracing {len(need)} reply(ies) (replaying them through the tracer)…")
    t0 = time.time()
    for n, x in enumerate(need, 1):
        tc.check_stop()
        if x.get("ids") is None:
            x["ids"], x["n_prompt"] = _rebuild_ids(ws, r, x)
        x["trace"] = tc.trace_ids(ws, [{"role": "user", "content": _prompt(x, r.get("instruction", ""))}], x["ids"],
                                  x["n_prompt"], which=r["weights"], settings={"engine": r.get("engine") or "tracer", "temperature": 0},
                                  log=lambda m: None)
        if n % 10 == 0 or n == len(need):
            log(f"  traced {n}/{len(need)} ({time.time() - t0:.0f}s)")
            (tests_dir(ws) / f"{tid}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def _rebuild_ids(ws, r, x):
    """(token ids, prompt length) of a reply that was saved without its tokens (tests run with the Python tracer by
    earlier versions kept only the trace): the prompt is rendered again and the reply text tokenized. Refused if
    that doesn't give the saved reply back exactly."""
    meta = _meta(ws, r["weights"])
    tok = tc.make_tokenizer(meta)
    prompt = tc.prompt_ids(tok, tc.render_chat(meta, [{"role": "user", "content": _prompt(x, r.get("instruction", ""))}]))
    reply = tok.encode(x["reply"], special=False)
    if tok.decode(reply) != x["reply"]:
        raise ValueError(f"Question {x['i'] + 1}: this reply was saved without its tokens and its trace was deleted, and its "
                         "text can't be turned back into the same tokens. Run the test again.")
    stop = _stop_id(tok)
    if not x.get("unfinished") and stop is not None and len(reply) < int(r.get("max_tokens") or 10 ** 9):
        reply.append(stop)                              # it ended on its own: the trace includes the stop token
    return [int(i) for i in prompt + reply], len(prompt)


def split_of(r, split, shuffle=False):
    """(indexes the edit is computed from, held-out indexes). split = fraction of the questions used for the edit
    (None / 0 / 1 = all, nothing held out): the first ones in the file's order, or with shuffle a random pick that
    is the same every time for this test (a file sorted by topic or difficulty would otherwise hold out questions
    of another kind than the edit was computed from)."""
    n = len(r["items"])
    if not split or float(split) >= 1:
        return [x["i"] for x in r["items"]], []
    k = max(1, min(n - 1, int(round(n * float(split)))))
    idx = sorted(x["i"] for x in r["items"])
    if shuffle:
        import random
        random.Random(1000003 * n + 17).shuffle(idx)
    return sorted(idx[:k]), sorted(idx[k:])


def list_tests(ws):
    d = tests_dir(ws)
    out = []
    for p in sorted(d.glob("*.json"), reverse=True) if d.is_dir() else []:
        if p.name.endswith(".checks.json"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": r["id"], "name": r["name"], "created": r["created"], "weights": r["weights"],
                        "right": r.get("right", 0), "n": len(r["items"]), "computed_edits": r.get("computed_edits", [])})
        except Exception:
            continue
    return out


def get_test(ws, tid):
    if not re.fullmatch(r"[\w-]+", tid or ""):
        raise ValueError("Bad test id")
    p = tests_dir(ws) / f"{tid}.json"
    if not p.exists():
        raise FileNotFoundError(f"No test {tid!r}")
    return json.loads(p.read_text(encoding="utf-8"))


def forget_traces(ws, ids):
    """Deleted traces: drop them from the saved tests (fast-mode replies keep their tokens and can be traced again)."""
    ids = set(ids)
    d = tests_dir(ws)
    for p in d.glob("*.json") if d.is_dir() else []:
        if p.name.endswith(".checks.json"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        hit = [x for x in r["items"] if x.get("trace") in ids]
        if hit:
            for x in hit:
                x["trace"] = None
            p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


def delete_test(ws, tid):
    r = get_test(ws, tid)
    (tests_dir(ws) / f"{tid}.json").unlink()
    (tests_dir(ws) / f"{tid}.checks.json").unlink(missing_ok=True)
    return {"deleted": r["id"]}


# ------------------------------------------------------------------ where a wrong answer went wrong
MAX_TOKENS_PER_ANSWER = 8          # token fixes per wrong answer (16 when a reference working is given)

_NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_OP = r"(?:\+|-|−|–|\*|×|·|x|X|/|÷|\\times|\\cdot|\\div)"
_EQ = re.compile(rf"(?<![\w.,])(\$?{_NUM}\$?(?:\s*{_OP}\s*\$?{_NUM}\$?)+)\s*=\s*\$?({_NUM})(?![\d])")


_PARTIAL = re.compile(r"(?:[)}\]^]|\d:|[\d)}\]%]\s*(?:[+\-−–*×·/÷^xX]|\\times|\\cdot|\\div)\s*)$")


def _calc(expr):
    e = expr.replace("$", "").replace(",", "")
    for a, b in (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"), ("–", "-")):
        e = e.replace(a, b)
    e = re.sub(r"(?<=[\d\s])[xX](?=[\s\d])", "*", e)
    if not re.fullmatch(r"[\d.+\-*/ ]+", e):
        return None
    try:
        return float(eval(e, {"__builtins__": {}}, {}))
    except (ZeroDivisionError, SyntaxError, ValueError):
        return None


def arithmetic_errors(text):
    """Every 'a op b … = c' in the text whose result is wrong: [(offset of c, c as written, the right value text,
    the equation)], in order."""
    out = []
    text = text or ""
    for m in _EQ.finditer(text):
        before, after = text[:m.start(1)], text[m.end(2):]
        # only a whole calculation: not the tail of a longer one ("(2 + 3) * 4 + 5 = 25", "2^3 * 2 = 16"), a clock
        # time ("3:00 + 2 = 5:00"), a result that goes on as a fraction ("= 3/4") or a division with a remainder
        if _PARTIAL.search(before) or re.match(r"\s*[/:^]\s*\d", after) or re.match(r"\s*(?:remainder|rem\b|r\s*\d)", after, re.I):
            continue
        v = _calc(m.group(1))
        if v is None:
            continue
        c_txt = m.group(2)
        c = float(c_txt.replace(",", ""))
        d = len(c_txt.split(".")[1]) if "." in c_txt else 0
        if abs(v - c) < 1e-9 * max(1.0, abs(v)) or (d and round(v, d) == round(c, d)):
            continue
        right = str(int(round(v))) if abs(v - round(v)) < 1e-9 else (f"{v:.{d}f}" if d else f"{v:.2f}".rstrip("0").rstrip("."))
        out.append((m.start(2), c_txt, right, m.group(0)))
    return out


def _char_to_token(tok, reply_ids, off):
    """Index of the reply token that holds character `off` of the decoded reply."""
    for k in range(len(reply_ids)):
        if len(tok.decode(reply_ids[:k + 1])) > off:
            return k
    return None


def _what_to_fix(tok, row, reply_ids):
    """(kind, token index j where the corrected text starts, the corrected text from token j, what was said,
    detail, end: the reply should stop right after it) or a skip reason."""
    reply = tok.decode(reply_ids)
    if row.get("solution"):
        return "reference working", 0, row["solution"], reply[:60], "", True
    if row["kind"] == "number":
        errs = arithmetic_errors(reply)
        if errs and (row["offset"] is None or errs[0][0] < row["offset"]):
            off, said, want, eq = errs[0]
            kind, detail = "calculation step", f"{eq.strip()} (should be {want})"
        elif row["offset"] is not None:
            off, said, want, kind, detail = row["offset"], row["got"] or "", row["answer"], "final answer", ""
        else:
            return ("the reply was cut off by the token limit before it gave an answer (raise Max tokens)"
                    if row.get("unfinished") else "the reply has no number to correct")
        j = _char_to_token(tok, reply_ids, off)
        if j is None:
            return "couldn't find the answer in the reply"
        pre = reply[len(tok.decode(reply_ids[:j])):off]          # the part of token j before the number
        # a final answer that was the last thing in the reply: the right one must also end it, or the model
        # carries on and writes its old answer again (its working still says so)
        end = kind == "final answer" and not re.sub(r"[\s.*$}\)\]]", "", reply[off + len(said):])
        return kind, j, pre + want, said, detail, end
    j = next((k for k, t in enumerate(reply_ids) if not MARKUP.match(tok.decode([t]))), 0)
    lead = tok.decode([reply_ids[j]])
    want = row["keywords"][0]
    if lead.strip()[:1].isupper() and want[:1].islower():
        want = want[:1].upper() + want[1:]
    ws_ = lead[:len(lead) - len(lead.lstrip())]
    rest = reply[len(tok.decode(reply_ids[:j])):].strip()
    return "final answer", j, ws_ + want, rest.split()[0] if rest else "", "", False


def locate(ws, model, tok, row):
    """The token fixes for one wrong answer: the reply up to the mistake + the right text, replayed; a fix at every
    token there the model still wouldn't write. {row summary, "fixes": [...]} or {"skip": reason}."""
    info, _A = tc._open_trace(ws, row["trace"])
    ids, n_prompt = info["ids"], info["n_prompt"]
    reply_ids = [i for i in ids[n_prompt:] if not tok.is_stop(i)]
    if not reply_ids:
        return {"skip": "empty reply"}
    w = _what_to_fix(tok, row, reply_ids)
    if isinstance(w, str):
        return {"skip": w}
    kind, j, text, said, detail, end = w
    corrected = list(ids[:n_prompt + j]) + tok.encode(text, special=False)
    stop = ids[-1] if tok.is_stop(ids[-1]) else getattr(tok, "eos", None)
    if end and stop is not None:
        corrected.append(int(stop))
    A = fc.forced(model, corrected)
    tops = A["top_ids"][:, 0]
    cap = 2 * MAX_TOKENS_PER_ANSWER if kind == "reference working" else MAX_TOKENS_PER_ANSWER
    fixes = []
    for p in range(max(0, n_prompt + j - 1), len(corrected) - 1):
        if int(tops[p]) != corrected[p + 1]:
            target, wrong = int(corrected[p + 1]), int(tops[p])
            at_t, at_w = tc.attribute(model, A, p, target), tc.attribute(model, A, p, wrong)
            fixes.append({"i": row["i"], "trace": row["trace"], "ids": corrected[:p + 2], "pos": p, "target": target,
                          "target_text": tok.decode([target]), "wrong": wrong, "wrong_text": tok.decode([wrong]),
                          "gap": float(at_w["logit"] - at_t["logit"]),
                          "mlp_gap": [round(float(x), 4) for x in at_w["layer_mlp"] - at_t["layer_mlp"]],
                          "attn_gap": [round(float(x), 4) for x in at_w["layer_attn"] - at_t["layer_attn"]]})
            if len(fixes) >= cap:
                break
    if not fixes:
        return {"skip": "with the right text in place the model already writes it (nothing to change)"}
    blame = np.sum([np.maximum(np.array(f["mlp_gap"]), 0) for f in fixes], 0)
    return {"i": row["i"], "question": row["question"], "kind": kind, "said": said, "should": text.strip(),
            "detail": detail, "n_tokens": len(fixes), "gap": fixes[0]["gap"], "blame_layer": int(np.argmax(blame)),
            "blame": [round(float(b), 4) for b in blame], "fixes": fixes}


def _strip_ids(rows):
    return [{**x, "fixes": [{k: v for k, v in f.items() if k != "ids"} for f in x["fixes"]]} for x in rows]


_AN = {}            # the last analysis (key -> result with token ids), reused by Compute right after Analyse


def analyze(ws, tid, log=print, with_ids=False, split=None, measure=8, shuffle=False):
    """Locate every wrong answer (of the questions the edit is computed from) and rank the layers whose MLPs pushed
    the wrong tokens (summed). split: fraction of the questions, from the start, to compute the edit from; the
    rest are held out to check it. measure: for up to this many wrong answers, also switch every layer off at the
    first wrong token and measure its total effect on (wrong − right): the layers that really decided the mistake,
    including what they do through later layers (direct push favours the last layer)."""
    r = get_test(ws, tid)
    train, held = split_of(r, split, shuffle)
    r = ensure_traces(ws, tid, train, log)             # traces that are missing (never made, or deleted) are made first
    # the same analysis again (Analyse, then Compute): reuse it. The key holds everything it depends on: the test's
    # answers and their traces, the split, how many answers are measured, and the weights.
    key = (ws, tid, tuple(train), int(measure), tuple(_stamp(ws)),
           tuple((x["i"], bool(x["correct"]), x.get("trace")) for x in r["items"]))
    if _AN.get("key") == key:
        log("Using the analysis already done for this test (same questions, split and weights)")
        out = copy.deepcopy(_AN["out"])
        return out if with_ids else {**out, "rows": _strip_ids(out["rows"])}
    tr = set(train)
    model, tok = tc.get_model(ws, r["weights"], log)
    rows, skipped = [], []
    for row in r["items"]:
        if row["correct"] or row["i"] not in tr:
            continue
        tc.check_stop()
        try:
            f = locate(ws, model, tok, row)
        except tc.Stopped:
            raise
        except Exception as e:
            f = {"skip": f"{type(e).__name__}: {e}"}
        if "skip" in f:
            skipped.append({"i": row["i"], "question": row["question"], "reason": f["skip"]})
        else:
            rows.append(f)
    ok = fc.editable_layers(model)
    blame = np.zeros(model.n_layer)
    for x in rows:
        blame += np.array(x["blame"])
    # total effects (causal) at the first wrong token of up to `measure` answers, spread over the test
    tot_m, tot_a, measured = np.zeros(model.n_layer), np.zeros(model.n_layer), 0
    pick = rows if len(rows) <= int(measure) else [rows[int(k * len(rows) / int(measure))] for k in range(int(measure))]
    for n, x in enumerate(pick if int(measure) > 0 else [], 1):
        f = x["fixes"][0]
        log(f"Total effects {n}/{len(pick)}: switching each layer off at the first wrong token of answer #{x['i'] + 1}")
        e = ec.layer_effects(model, f["ids"], f["pos"], f["wrong"], f["target"], log=lambda m: None)
        em = np.array([v if v is not None else 0.0 for v in e["effect"]["mlp"]])
        ea = np.array([v if v is not None else 0.0 for v in e["effect"]["attn"]])
        x["total_mlp"], x["total_attn"] = [round(float(v), 4) for v in em], [round(float(v), 4) for v in ea]
        x["flip_after_layer"] = e["flip_after_layer"]
        tops = ec.top_parts(e, 3)
        x["total_top"] = [[k, int(L), round(float(v), 3)] for v, k, L in tops]
        tot_m += np.maximum(em, 0)
        tot_a += np.maximum(ea, 0)
        measured += 1
    ranked = sorted(ok, key=lambda L: -blame[L])
    ranked_total = sorted(ok, key=lambda L: -tot_m[L]) if measured else []
    ok_attn = fc.editable_layers(model, "attn")
    parts_total = sorted([("mlp", L, tot_m[L]) for L in ok] + [("attn", L, tot_a[L]) for L in ok_attn],
                         key=lambda x: -x[2]) if measured else []
    rec = (ranked_total or ranked)[0] if (ranked_total or ranked) and rows else (ok[-1] if ok else None)
    nf = sum(x["n_tokens"] for x in rows)
    steps = sum(x["kind"] == "calculation step" for x in rows)
    log(f"{len(rows)} wrong answer(s) located ({nf} token fix(es){f', {steps} at a wrong calculation step' if steps else ''}), "
        f"{len(skipped)} skipped" + (f"; most blame: layer {rec} MLP" if rows and rec is not None else ""))
    protect = []
    for row in r["items"]:
        if row["correct"] and row["i"] in tr and row.get("offset") is not None:
            try:
                info, _A = tc._open_trace(ws, row["trace"])
                reply_ids = [i for i in info["ids"][info["n_prompt"]:] if not tok.is_stop(i)]
                j = _char_to_token(tok, reply_ids, row["offset"])
                if j is not None:
                    protect.append({"trace": row["trace"], "pos": info["n_prompt"] + j - 1, "i": row["i"]})
            except Exception:
                pass
    out = {"test": tid, "weights": r["weights"], "rows": rows, "skipped": skipped, "editable_layers": ok,
           "protect": protect, "layer_blame": [round(float(b), 4) for b in blame], "recommended_layer": rec,
           "top_layers": [{"layer": L, "blame": round(float(blame[L]), 4)} for L in ranked[:5]],
           "measured": measured, "total_blame_mlp": [round(float(v), 4) for v in tot_m],
           "total_blame_attn": [round(float(v), 4) for v in tot_a],
           "top_layers_total": [{"layer": L, "blame": round(float(tot_m[L]), 4)} for L in ranked_total[:5]],
           "top_parts_total": [{"part": k, "layer": int(L), "blame": round(float(v), 4)} for k, L, v in parts_total[:6]],
           "editable_attn": ok_attn,
           "right_traces": [x["trace"] for x in r["items"] if x["correct"] and x["i"] in tr], "n_token_fixes": nf,
           "train": train, "held": held, "shuffled": bool(shuffle and held)}
    _AN.update(key=key, out=copy.deepcopy(out))
    return out if with_ids else {**out, "rows": _strip_ids(rows)}


def spread_parts(a, n_layer, spec="auto", cover=0.8, max_parts=6):
    """[(part, layer, weight)] to spread one correction over, weighted by how much each part decided the mistakes
    (total effect when measured, else the MLPs' direct push). spec="auto": the parts that together hold `cover` of
    the effect, outside the first quarter of the layers (they read the input: editing them changes everything, as
    attention L0 on GSM8K needed ~28,000 % of its matrix), 2 to `max_parts` of them. spec="12-20": every editable
    part in those layers with a positive effect."""
    measured = bool(a.get("measured"))
    eff = {}
    for L in a["editable_layers"]:
        eff[("mlp", L)] = (a["total_blame_mlp"] if measured else a["layer_blame"])[L]
    if measured:
        for L in a.get("editable_attn") or []:
            eff[("attn", L)] = a["total_blame_attn"][L]
    eff = {k: float(v) for k, v in eff.items() if v and v > 0}
    if spec and spec != "auto":
        try:
            lo, hi = (int(x) for x in str(spec).replace("–", "-").split("-"))
        except ValueError:
            raise ValueError("Layer range: write it as  first-last  (e.g. 12-20), or auto")
        out = [(k[0], k[1], v) for k, v in eff.items() if lo <= k[1] <= hi]
        if len(out) < 2:
            raise ValueError(f"Layers {lo}-{hi} have fewer than 2 editable parts that pushed the mistakes")
        return out
    cand = sorted(((v, k) for k, v in eff.items() if k[1] >= n_layer // 4), reverse=True)
    if len(cand) < 2:
        raise ValueError("Fewer than 2 parts outside the first layers pushed the mistakes: use one layer instead")
    total, out = sum(v for v, _k in cand), []
    for v, k in cand:
        out.append((k[0], k[1], v))
        if len(out) >= max_parts or (len(out) >= 2 and sum(x[2] for x in out) >= cover * total):
            break
    return out


def recommend(ws, tid, layer=None, neurons=0, margin=1.0, use=None, split=None, mode="exact", rank=4, strength=1.0,
              measure=8, choose="responsible", part="mlp", per_answer=None, analysis=None, spread=None, shuffle=False,
              log=print):
    """Analyze, then solve the computed edit for the chosen (or recommended) layer. use = indexes of the
    wrong answers to fix (default all). The test's right answers are what must stay the same.
    mode="exact": every token fix reached exactly (fixes these answers). mode="general": one shared low-rank
    correction (rank `rank`) for all of them, more likely to carry over to other questions of the same kind."""
    a = analysis or analyze(ws, tid, log, with_ids=True, split=split, shuffle=shuffle,
                            measure=measure if choose in ("responsible", "second") else 0)
    rows = [x for x in a["rows"] if use is None or x["i"] in set(use)]
    cap = int(per_answer) if per_answer else None          # token fixes per answer (the first ones)
    fx = [{"trace": f["trace"], "ids": f["ids"], "pos": f["pos"], "target": f["target"]} for x in rows for f in x["fixes"][:cap]]
    if not fx:
        raise ValueError("No wrong answers to fix" + (" (all were skipped)" if a["skipped"] else ""))
    if len(fx) > fc.MAX_FIXES:                         # keep every answer, drop the last tokens of the longest ones
        per = {x["i"]: list(x["fixes"][:cap]) for x in rows}
        while sum(map(len, per.values())) > fc.MAX_FIXES:
            longest = max(per, key=lambda i: len(per[i]))
            per[longest].pop()
        fx = [{"trace": f["trace"], "ids": f["ids"], "pos": f["pos"], "target": f["target"]} for x in rows for f in per[x["i"]]]
        log(f"Too many token fixes: kept {len(fx)} (the first ones of each answer)")
    costs = []
    # Side effects are measured on right answers the edit was NOT fitted to keep (the last fifth of them, when there
    # are 10 or more): measured on the ones it was fitted to, the number would look better than it is.
    rt = list(a["right_traces"])
    unseen = rt[-max(3, len(rt) // 5):] if len(rt) >= 10 else []
    fit = dict(keep=[t for t in rt if t not in set(unseen)] or None, check=(unseen or rt)[:20], exclude=unseen,
               protect=[p_ for p_ in a["protect"] if p_.get("trace") not in set(unseen)])
    if spread:
        model, _tok = tc.get_model(ws, a["weights"], log)
        sp = spread_parts(a, model.n_layer, spread)
        log(f"Spreading the edit over {len(sp)} parts, weighted by how much each decided the mistakes")
        sol = fc.solve_spread(ws, fx, sp, neurons=neurons, margin=margin, which=a["weights"], mode=mode, rank=rank,
                              strength=strength, log=log, **fit)
        return _finish_rec(ws, tid, a, sol, costs, cap, bool(unseen))
    if layer in (None, ""):
        ok = a["editable_layers"]
        if not ok:
            raise ValueError("This model has no layer Fix answers can edit")
        if choose in ("responsible", "second") and a["measured"] and a["top_parts_total"]:
            # the attention or MLP whose switching off moves the wrong-vs-right lead most (total effect, through later layers)
            top = a["top_parts_total"][min(1 if choose == "second" else 0, len(a["top_parts_total"]) - 1)]
            L, part = top["layer"], top["part"]
            log(f"{'Second most' if choose == 'second' else 'Most'} responsible part (total effect on wrong − right, {a['measured']} answer(s) measured): "
                f"{'attention' if part == 'attn' else 'MLP'} of layer {L}")
            costs = fc.goal_costs(ws, fx[:24], [L], margin, a["weights"], log, part=part)
        else:
            # cheapest: the layers that pushed the wrong tokens most, and the last layer; the smallest change wins
            cands = list(dict.fromkeys([ok[-1]] + [r["layer"] for r in a["top_layers"][:2]]))
            log(f"Choosing the cheapest layer: trying {', '.join(f'L{L}' for L in cands)}")
            part = "mlp"
            costs = fc.goal_costs(ws, fx[:24], cands, margin, a["weights"], log)
            L = min(costs, key=lambda c: c["cost"])["layer"]
    else:
        L = int(layer)
    log(f"Computing an edit for the {'attention' if part == 'attn' else 'MLP'} of layer {L}: {len(rows)} answer(s), {len(fx)} token fix(es)")
    sol = fc.solve(ws, fx, layer=L, neurons=neurons, margin=margin, which=a["weights"], mode=mode, rank=rank, strength=strength,
                   chain=2 if mode == "exact" else 0, part=part, log=log, **fit)
    return _finish_rec(ws, tid, a, sol, costs, cap, bool(unseen))


def _finish_rec(ws, tid, a, sol, costs, cap, check_unseen=False):
    sol["split"] = {"train": a["train"], "held": a["held"]}
    sol["check_unseen"] = check_unseen                 # were the side effects measured on answers it wasn't fitted to?
    sol["test"] = tid
    sol["layer_costs"] = costs
    sol["weights_key"] = _stamp(ws)
    sol["per_answer"] = cap
    if sol["sid"] in fc._SOL:
        fc._SOL[sol["sid"]]["res"].update(layer_costs=costs, split=sol["split"], test=tid, weights_key=sol["weights_key"], per_answer=cap)
    return {"analysis": {**a, "rows": _strip_ids(a["rows"])}, "solution": sol}
