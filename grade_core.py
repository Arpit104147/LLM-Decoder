"""
Questions and grading for the Test tab.

Question formats (pasted text or a file):
  question | answer                      one per line (text answers: several right answers as  a / b / c)
  JSON list / JSONL / CSV with a question field (question, q, prompt, input, problem, ...) and an answer field
  (answer, a, kw, num, expected, output, target, label, answerKey, ...)
  GSM8K        {"question", "answer": "... #### 42"}  (the working before #### is kept as the reference solution)
  MATH         {"problem", "solution": "... \\boxed{3/4}"}
  multiple choice (MMLU, ARC, MedQA, HellaSwag ...): "choices" / "options" / "endings" as a list, an {"A": ...}
               dict or ARC's {"text": [...], "label": [...]}, answer as an index, a letter or the choice's text;
               or a question that already lists "A) ... B) ..." with a letter answer
  yes / no, true / false (BoolQ ...)

Answer kinds and how a reply is graded:
  number   the reply's final answer: \\boxed{} > "#### n" > after "answer is / answer:" > the last bold number >
           the last number. Thousands separators, $ / % signs, fractions (3/4, \\frac{3}{4}, 2 1/2), number words
           (eight, twenty-five). Equal when exact; a rounded value (2+ decimals) counts when it rounds the same
  choice   the letter the reply chose ("answer is (B)", "**B**", a reply starting "B." ...) or, when it gives no
           letter, the one choice whose text it contains
  bool     the first yes / no / true / false in the answer part of the reply
  text     any right answer as a whole word or phrase (plurals too, accents and case ignored), not negated
           ("not Paris"), in the answer part of the reply; a "not" list of wrong words makes it wrong
"""
import csv
import io
import json
import re
import unicodedata

LETTERS = "ABCDEFGHIJ"
MAX_QUESTIONS = 2000

Q_FIELDS = ("question", "q", "prompt", "input", "problem", "query", "instruction", "text", "ctx", "goal", "sentence")
A_FIELDS = ("answer", "a", "kw", "expected", "output", "target", "label", "answerkey", "answer_idx", "correct",
            "gold", "final_answer", "response")
C_FIELDS = ("choices", "options", "endings", "candidates", "answers_list")


# ------------------------------------------------------------------ numbers
_SEQ = (("\\!", ""), ("\\,", ""), ("\\$", "$"), ("\\%", "%"), ("{,}", ","))        # LaTeX spacing / escapes
_CHAR = {"\u2212": "-", "\u2013": "-", "\u2009": "", "\u202f": ""}                    # minus signs, thin spaces


def _clean_map(s):
    """(cleaned text, idx): the text with look-alike characters made plain (NFKC per character, minus signs, thin
    spaces and LaTeX escapes removed) and, for every cleaned character, the index of the character of `s` it came
    from. Offsets found in the cleaned text are turned back into offsets in the reply with idx."""
    s = str(s)
    if s.isascii() and "\\" not in s and "{,}" not in s:
        return s, range(len(s))
    out, idx, i, n = [], [], 0, len(s)
    while i < n:
        for a, b in _SEQ:
            if s.startswith(a, i):
                rep, step = b, len(a)
                break
        else:
            ch = s[i]
            rep = _CHAR[ch] if ch in _CHAR else "".join(_CHAR.get(c, c) for c in unicodedata.normalize("NFKC", ch))
            step = 1
        out.append(rep)
        idx.extend([i] * len(rep))
        i += step
    return "".join(out), idx


def _clean(s):
    return _clean_map(s)[0]


def _raw_span(reply, idx, off, text):
    """(offset in the reply, the reply's own characters) for `text` found at cleaned offset `off`."""
    if off is None or not len(idx):
        return off, text
    a = idx[min(off, len(idx) - 1)]
    b = idx[min(off + max(len(text), 1) - 1, len(idx) - 1)] + 1
    return a, reply[a:b]


def _num(s):
    """A number written as an answer -> float, else None: 1,234 · $5 · 3.50 · -2 · 3/4 · \\frac{3}{4} · 2 1/2 ·
    50% · 1e5 · x = 5 · "12 apples"."""
    s = _clean(s).strip()
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()                                    # $…$ math or a dollar sign
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)                                      # x = 5
    s = re.sub(r"^[$€£₹¥]\s*|\s*(%|percent|\\text\{[^}]*\})$", "", s).strip().rstrip(".").strip()
    m = re.fullmatch(r"(-?)\\[dt]?frac\{\s*([\d.]+)\s*\}\{\s*([\d.]+)\s*\}", s)
    if m:
        b = float(m.group(3))
        return None if b == 0 else (-1 if m.group(1) else 1) * float(m.group(2)) / b
    m = re.fullmatch(r"(-?)(\d+)\s+(\d+)/(\d+)", s)                             # mixed number 2 1/2
    if m and float(m.group(4)):
        return (-1 if m.group(1) else 1) * (float(m.group(2)) + float(m.group(3)) / float(m.group(4)))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if m:
        b = float(m.group(2))
        return None if b == 0 else float(m.group(1)) / b
    if re.fullmatch(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?", s) and re.search(r"\d", s):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None
    m = re.fullmatch(r"(\S+)(?:\s+[A-Za-z][A-Za-z.]*){1,2}", s)                 # a number and a unit: 12 apples
    if m and re.search(r"\d", m.group(1)):
        return _num(m.group(1))
    return None


_W1 = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
      "seventeen eighteen nineteen".split()
_W10 = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORDS = {w: i for i, w in enumerate(_W1)}
WORDS.update(_W10)
WORDS.update({"hundred": 100, "thousand": 1000, "dozen": 12})
_WORD_RE = re.compile(r"(?<![\w-])(?:(" + "|".join(_W10) + r")[- ](" + "|".join(_W1[1:10]) + r")|(" +
                      "|".join(sorted(WORDS, key=len, reverse=True)) + r"))(?![\w-])", re.I)
_NUM_RE = re.compile(
    r"(?P<frac>-?\\[dt]?frac\{\s*[\d.]+\s*\}\{\s*[\d.]+\s*\})"
    r"|(?<![\w.])(?P<n>(?:-(?=\.?\d))?(?:\d+\s+\d+/\d+(?![\d/])|(?:\d{1,3}(?:,\d{3})+(?![\d,])|\d+)(?:\.\d+)?"
    r"(?:\s*/\s*\d+(?:\.\d+)?)?(?:[eE][-+]?\d+)?|\.\d+))(?![\d])")


def numbers(text):
    """[(offset, text as written, value)] for every number in the text: digits (with separators, fractions,
    \\frac{}{}) and, when there are no digits at all, number words. Offsets are positions in `text`."""
    text = text or ""
    t, idx = _clean_map(text)
    return [(*_raw_span(text, idx, o, g), v) for o, g, v in _numbers(t)]


def _numbers(t):
    """numbers() on text that is already cleaned (offsets in it)."""
    out = []
    for m in _NUM_RE.finditer(t):
        g = m.group("frac") or m.group("n")
        v = _num(g)
        if v is not None:
            out.append((m.start(), g, v))
    if not out:
        for m in _WORD_RE.finditer(t):
            if m.group(1):
                v = _W10[m.group(1).lower()] + WORDS[m.group(2).lower()]
            else:
                v = WORDS[m.group(3).lower()]
            out.append((m.start(), m.group(0), float(v)))
    return out


def same_number(v, want):
    """Equal, or one of them is the other rounded to the decimals it was written with (2+ decimals)."""
    if abs(v - want) <= 1e-6 * max(1.0, abs(want)):
        return True
    for a, b in ((v, want), (want, v)):
        s = repr(float(a))
        d = len(s.split(".")[1]) if "." in s and "e" not in s else 0
        if d >= 2 and abs(round(b, d) - a) <= 10 ** (-d - 3):
            return True
    return False


# ------------------------------------------------------------------ where the reply gives its answer
_CUE = re.compile(r"answer\s*(?:is|:|=|would\s+be|will\s+be)\s*[:=\-]?\s*", re.I)


def _answer_region(reply):
    """(start, text) of the part of the reply after the last "answer is / answer:" cue, to the end of that line;
    or None when there is no cue."""
    best = None
    for m in _CUE.finditer(reply):
        best = m
    if not best:
        return None
    s = best.end()
    e = reply.find("\n", s)
    e = len(reply) if e < 0 else e
    return s, reply[s:e]


def _boxed(reply):
    """(start, content) of the last \\boxed{...} (nested braces allowed)."""
    i = reply.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(reply) and depth:
        depth += {"{": 1, "}": -1}.get(reply[j], 0)
        j += 1
    return i + 7, reply[i + 7:j - 1]


_N = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_OPS = r"(?:\+|-|\*|×|·|x|X|/|÷|\\times|\\cdot|\\div)"
_CALC = rf"\$?{_N}\$?(?:\s*[A-Za-z]{{0,12}})?(?:\s*{_OPS}\s*\$?{_N}\$?(?:\s*[A-Za-z]{{0,12}})?)+\s*=\s*"


def _after_calculation(region):
    """An answer written as a calculation ("the answer is 9 * 2 = 18"): where its result starts (after the last
    '=' of the calculation that opens the region), else 0."""
    m = re.match(r"[\s*_`(]*" + _CALC, region)
    pos = 0
    while m:
        pos += m.end()
        m = re.match(_CALC, region[pos:])
    return pos


def final_number(reply, explicit_only=False):
    """(offset in the reply, text as written, value) of the number the reply gives as its answer, or None.
    explicit_only: only an answer the reply marks as one (\\boxed, ####, "answer is", bold), not just its last number
    (for a reply that was cut off)."""
    reply = reply or ""
    clean, idx = _clean_map(reply)
    f = _final_number(clean, explicit_only)
    if f is None:
        return None
    return (*_raw_span(reply, idx, f[0], f[1]), f[2])


def _final_number(reply, explicit_only):
    """final_number() on cleaned text (offsets in it)."""
    b = _boxed(reply)
    if b:
        ns = _numbers(b[1])
        if ns:
            o, t, v = ns[-1]
            return b[0] + o, t, v
    i = reply.rfind("####")
    if i >= 0:
        ns = _numbers(reply[i + 4:])
        if ns:
            o, t, v = ns[0]
            return i + 4 + o, t, v
    r = _answer_region(reply)
    if r:
        skip = _after_calculation(r[1])                  # "answer is 9 * 2 = 18": the result, not the first operand
        for start in dict.fromkeys((skip, 0)):
            ns = _numbers(r[1][start:])
            if ns:
                o, t, v = ns[0]
                return r[0] + start + o, t, v
    # a bold label ("**Buy 4 more:**", "**Total eggs**: ...", "**Step 2: ...**") is a heading, not an answer
    bold = [m for m in re.finditer(r"\*\*([^*]+)\*\*", reply) if _numbers(m.group(1))
            and not m.group(1).rstrip().endswith(":") and reply[m.end():m.end() + 1] != ":"
            and not re.match(r"\s*(?:(?:step|part|case|method|option|question|problem|stage|phase)\s*\d+\b|\d+\s*[:.)]\s*$)",
                             m.group(1), re.I)]
    if bold:
        m = bold[-1]
        o, t, v = _numbers(m.group(1))[-1]
        return m.start(1) + o, t, v
    if explicit_only:
        return None
    ns = _numbers(reply)
    return ns[-1] if ns else None


# ------------------------------------------------------------------ text
def _norm_map(s):
    """(normalized text, idx): lower case, accents removed, curly quotes straightened, runs of white space as one
    space; idx[k] = index in `s` of the character that normalized character k came from."""
    s = str(s)
    out, idx, space = [], [], False
    for i, ch in enumerate(s):
        if ch.isascii():
            d = ch.lower()
        else:
            d = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            d = d.replace("’", "'").replace("‘", "'").lower()
        for c in d:
            if c.isspace():
                if space:
                    continue
                c, space = " ", True
            else:
                space = False
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def norm_text(s):
    return _norm_map(s)[0]


_NEG = re.compile(r"(?:\bnot|n't|\bnever|\bno)\s+(?:the\s+|a\s+|an\s+|in\s+|at\s+)?$")


def find_phrase(text, kw):
    """Offset in `text` of kw as a whole word / phrase (case, accents and spacing ignored, plural allowed), not
    right after "not"; or None."""
    (low, idx), k = _norm_map(text), norm_text(kw).strip()
    if not k:
        return None
    wordy = re.match(r"^[\w ]+$", k) is not None
    pat = (r"(?<![\w])" + re.escape(k) + r"(?:s|es)?(?![\w])") if wordy else re.escape(k)
    for m in re.finditer(pat, low):
        if not _NEG.search(low[max(0, m.start() - 16):m.start()]):
            return idx[m.start()]
    return None


# ------------------------------------------------------------------ questions
def _choice_list(c):
    """choices in any common shape -> (labels, texts)."""
    if isinstance(c, dict):
        if "text" in c:                                                    # ARC
            texts = [str(x) for x in c["text"]]
            labels = [str(x) for x in c.get("label", LETTERS[:len(texts)])]
            return labels, texts
        keys = list(c)
        return [str(k).strip().upper() for k in keys], [str(c[k]) for k in keys]
    texts = [str(x) for x in c]
    return list(LETTERS[:len(texts)]), texts


def _choices_in_text(q):
    """Options already written in the question ("A) ...", "(B) ...", "C. ..." one per line)."""
    found = re.findall(r"(?m)^\s*\(?([A-J])[.):]\s+(.+?)\s*$", q)
    letters = [l for l, _t in found]
    if len(found) >= 2 and letters == list(LETTERS[:len(letters)]):
        return letters, [t for _l, t in found]
    return None


def _bool(a):
    if isinstance(a, bool):
        return a
    s = norm_text(a).strip().rstrip(".")
    return {"yes": True, "true": True, "no": False, "false": False}.get(s)


def _pick(d, names):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None


def item(d, i):
    """One question (a dict with any of the common field names) -> a graded item."""
    if "row" in d and isinstance(d["row"], dict):
        d = d["row"]
    d = {k.lower().strip() if isinstance(k, str) else k: v for k, v in d.items()}
    q = _pick(d, Q_FIELDS)
    if q is None or not str(q).strip():
        raise ValueError(f"Item {i + 1}: no question (use a 'question' field)")
    q = str(q).strip()
    sol = d.get("solution") or d.get("working") or d.get("rationale")
    if "num" in d:
        v = _num(d["num"])
        if v is None:
            raise ValueError(f"Item {i + 1}: 'num' isn't a number")
        it = {"question": q, "answer": str(d["num"]), "kind": "number", "value": v}
    else:
        ch = _pick(d, C_FIELDS)
        a = _pick(d, A_FIELDS)
        if a is None and sol is not None:                                   # MATH: the answer is in the solution
            b = _boxed(_clean(sol))
            a = b[1] if b else None
        if a is None:
            raise ValueError(f"Item {i + 1}: no answer")
        found = _choice_list(ch) if ch is not None else (_choices_in_text(q) if isinstance(a, str) and
                                                          re.fullmatch(r"\(?[A-Ja-j]\)?\.?", a.strip()) else None)
        if found:
            it = _choice_item(q, found, a, i, written=ch is None)
        elif isinstance(a, bool) or (isinstance(a, str) and _bool(a) is not None):
            v = _bool(a)
            it = {"question": q, "answer": "yes" if v else "no", "kind": "bool", "value": v,
                  "keywords": ["Yes" if v else "No"]}
        elif isinstance(a, list):
            it = {"question": q, "answer": " / ".join(map(str, a)), "kind": "text", "keywords": [str(x) for x in a],
                  "not": d.get("not", [])}
        else:
            a = str(a)
            if "####" in a:                                                 # GSM8K: working, then #### final number
                a = a.rsplit("####", 1)[1].strip()
            v = _num(a)
            if v is not None:
                it = {"question": q, "answer": re.sub(r"[,$\s]", "", a) if re.fullmatch(r"[\d,.$\s-]+", a) else a,
                      "kind": "number", "value": v}
            else:
                it = {"question": q, "answer": a, "kind": "text",
                      "keywords": [x.strip() for x in a.split(" / ") if x.strip()], "not": d.get("not", [])}
    ref = d.get("solution") or d.get("working")
    if ref:
        it["solution"] = str(ref)                                           # the right reply, used as-is for the fix
    return it


def _choice_item(q, found, a, i, written):
    labels, texts = found
    up = [str(l).strip().upper() for l in labels]
    k = None
    if isinstance(a, bool):
        raise ValueError(f"Item {i + 1}: a multiple-choice answer can't be true / false")
    if isinstance(a, (int, float)) or (isinstance(a, str) and a.strip().isdigit() and a.strip() not in up):
        n = int(a)
        k = n if 0 <= n < len(texts) else None                             # an index (0-based, as MMLU / HellaSwag)
    else:
        s = str(a).strip().strip("().").upper()
        if s in up:
            k = up.index(s)
        else:
            m = [n for n, t in enumerate(texts) if norm_text(t).strip() == norm_text(a).strip()]
            k = m[0] if m else None
    if k is None:
        raise ValueError(f"Item {i + 1}: the answer {a!r} isn't one of the {len(texts)} choices")
    letters = list(LETTERS[:len(texts)])
    question = q if written else q + "\n" + "\n".join(f"{letters[n]}. {t}" for n, t in enumerate(texts))
    return {"question": question, "answer": letters[k], "kind": "choice", "letter": letters[k], "choices": texts,
            "keywords": [letters[k]], "choice_text": texts[k]}


def parse_questions(text):
    text = (text or "").strip().lstrip("\ufeff")
    if not text:
        raise ValueError("Add some questions: one 'question | answer' per line, or load a JSON / JSONL / CSV file")
    rows = None
    first = text.splitlines()[0]
    if text[0] in "[{":
        try:
            v = json.loads(text)
            rows = v if isinstance(v, list) else v.get("rows") or v.get("questions") or v.get("data") or [v]
        except json.JSONDecodeError:
            rows = [json.loads(l) for l in text.splitlines() if l.strip()]          # JSONL
    elif "|" not in first and ("," in first or "\t" in first) and any(f in norm_text(first) for f in Q_FIELDS):
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t" if "\t" in first else ","))
    if rows is None:
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "|" not in line:
                raise ValueError(f"Line {n}: write it as  question | answer")
            q, a = line.rsplit("|", 1)
            rows.append({"question": q.strip(), "answer": a.strip()})
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError("Each question must be an object with a question and an answer")
    items = [item(r, i) for i, r in enumerate(rows)]
    if len(items) > MAX_QUESTIONS:
        raise ValueError(f"At most {MAX_QUESTIONS} questions per test")
    return items


# ------------------------------------------------------------------ grading
def _letter(reply, letters, texts):
    """(letter, offset in the reply) the reply chose, or (None, None). A stated answer ("the answer is C") wins over
    a later mention of another option ("Option A is wrong because ...")."""
    reply = reply or ""
    r, idx = _clean_map(reply)
    L = "".join(letters)
    pats = [(rf"\\boxed\{{\s*(?:\\text\{{)?\(?([{L}])\)?", -1),
            (rf"(?i:answer)\s*(?i:is|:|would\s+be)?\s*:?\s*(?i:option\s*|choice\s*)?\**\(?([{L}])\)?(?![\w])", -1),
            (rf"(?i:option|choice)\s*(?i:is|:|would\s+be)?\s*:?\s*\**\(?([{L}])\)?(?![\w])", -1),
            (rf"\*\*\s*\(?([{L}])\)?\s*(?:[.):]|\*\*)", -1),
            (rf"^[\s*_]*(?:\(([{L}])\)|([{L}])(?:[.):]|\s*$|\s*\n))", 0),
            (rf"\(([{L}])\)", -1)]
    for p, which in pats:
        ms = list(re.finditer(p, r))
        if ms:
            m = ms[which]
            k = 1 if m.group(1) else 2
            return m.group(k).upper(), _raw_span(reply, idx, m.start(k), m.group(k))[0]
    hits = [(n, find_phrase(reply, t)) for n, t in enumerate(texts)]
    hits = [(n, o) for n, o in hits if o is not None]
    if len(hits) == 1:
        return letters[hits[0][0]], hits[0][1]
    return None, None


_NOT = re.compile(r"(?:\bnot|n't|\bnever)\s+(?:\w+ly\s+|quite\s+|at\s+all\s+)?$", re.I)


def grade(item, reply, unfinished=False):
    """(correct, what the reply gave, character offset of the answer in the reply or None). unfinished: the reply
    was cut off by the token limit, so a number answer counts only where the reply marks it as its answer."""
    reply = reply or ""
    kind = item["kind"]
    if kind == "number":
        f = final_number(reply, explicit_only=unfinished)
        if f is None:
            return False, None, None
        off, t, v = f
        return same_number(v, float(item["value"])), t.strip(), off
    if kind == "choice":
        letters = list(LETTERS[:len(item["choices"])])
        got, off = _letter(reply, letters, item["choices"])
        return got == item["letter"], got, off
    clean, idx = _clean_map(reply)
    reg = _answer_region(clean)
    start, part = reg if reg and re.search(r"\w", reg[1]) else (0, clean)
    raw = lambda o, text="x": _raw_span(reply, idx, start + o, text)[0]       # offset in `part` -> offset in the reply
    if kind == "bool":
        m = re.search(r"(?<![\w])(yes|no|true|false|correct|incorrect)(?![\w])", part, re.I)
        if not m:
            return False, None, None
        v = m.group(1).lower() in ("yes", "true", "correct")
        got = m.group(1)
        if m.group(1).lower() not in ("yes", "no") and _NOT.search(part[max(0, m.start() - 24):m.start()]):
            v, got = not v, "not " + got                 # "that is not true", "isn't correct"
        return v == item["value"], got, raw(m.start(), m.group(1))
    for bad in item.get("not", []):
        o = find_phrase(part, bad)
        if o is not None:
            return False, bad, raw(o)
    for kw in item["keywords"]:
        o = find_phrase(part, kw)
        if o is not None:
            return True, kw, raw(o)
    return False, None, None
