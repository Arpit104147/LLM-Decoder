"""
Grading and question-format checks (no model needed):  python test_grade.py
Every case is a realistic reply style or dataset format; each must be graded the way a person would.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # the project folder (app modules)
import grade_core as g

ok = fail = 0


def check(cond, msg):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


def one(d):
    return g.item(d, 0)


def G(it, reply):
    return g.grade(it, reply)[0]


# ------------------------------------------------------------------ number answers
N = lambda a: one({"question": "q", "answer": a})
cases = [  # (answer, reply, expected)
    ("391", "17*23 = 391", True),
    ("391", "391? no, it is 442", False),
    ("391", "so \\boxed{391}. Check: 17*23=391, 1", True),
    ("1234", "It is 1,234.", True),
    ("8", "A spider has **eight** legs.", True),
    ("25", "There are twenty-five students.", True),
    ("12", "She bought a dozen eggs.", True),
    ("18", "16 - 3 - 4 = 9 eggs left. 9 * $2 = $18.\n#### 18", True),
    ("18", "She makes $18 every day.", True),
    ("18", "The answer is 18 dollars. (She started with 16 eggs and used 7.)", True),
    ("18", "**18** dollars a day, since 9 eggs x 2", True),
    ("18", "**Step 1:** 16-3-4 = 9\n**Step 2:** 9*2 = 18", True),
    ("18", "**Step 3:** done. Total: 17", False),
    ("5", "**Step 2: Calculate the total** 3 + 2 = 5", True),
    ("7", "1.  **Start:** 5 apples.\n2.  **Eat 2:** 5 - 2 = 3 apples remain.\n3.  **Buy 4 more:** $3 + 4 = 7$ apples are added.\n\nTotal: $7$\n\n7", True),
    ("9", "**Total eggs**: 16 - 3 - 4 = 9", True),
    ("0.75", "The probability is 3/4.", True),
    ("0.75", "$\\frac{3}{4}$", True),
    ("3/4", "It is 0.75", True),
    ("2.5", "2 1/2 cups", True),
    ("-4", "x = -4", True),
    ("4", "between 3-4 hours, so 4", True),
    ("50", "50% of them", True),
    ("50%", "So 50 percent.", True),
    ("$1,200.50", "Total: $1,200.50", True),
    ("0.333", "1/3", True),
    ("33.33", "about 33.3333 percent", True),
    ("18", "18.0", True),
    ("18", "18.4", False),
    ("5", "5kg of flour", True),
    ("3", "the lists 1,2,3", True),
    ("123", "the lists 1,2,3", False),
    ("1000000", "one million is 1,000,000.", True),
    ("7", "There is no number here", False),
    ("12 apples", "12", True),
    ("x = 5", "so x=5", True),
    ("42", "The answer is: \\boxed{\\text{42}}", True),
    ("10", "Answer: 10\n\nExplanation: 4 + 6 = 10, not 11", True),
]
for a, reply, want in cases:
    it = N(a)
    check(it["kind"] == "number" and G(it, reply) == want, f"number {a!r} in {reply[:50]!r} -> {'right' if want else 'wrong'}")

# GSM8K and MATH formats
it = one({"question": "How many?", "answer": "Janet has 16 <<16-3=13>>13 ... #### 1,234"})
check(it["kind"] == "number" and it["value"] == 1234 and "solution" not in it, "GSM8K: '#### n' is the answer, <<calc>> working not used as a reply")
it = one({"problem": "Simplify.", "solution": "We get $\\boxed{\\frac{3}{4}}$."})
check(it["kind"] == "number" and abs(it["value"] - 0.75) < 1e-12 and it["solution"].startswith("We get"),
      "MATH: problem + solution with \\boxed{} -> number answer and reference solution")
it = one({"question": "q", "num": 12.5})
check(it["value"] == 12.5 and G(it, "12.5"), "'num' field")

# ------------------------------------------------------------------ multiple choice
mmlu = one({"question": "Which is a prime?", "choices": ["4", "6", "7", "9"], "answer": 2})
check(mmlu["kind"] == "choice" and mmlu["letter"] == "C" and "C. 7" in mmlu["question"],
      "MMLU: choices list + 0-based index; the options are added to the question")
for reply, want in [("C", True), ("C. 7", True), ("The answer is (C).", True), ("Answer: C", True), ("**C**", True),
                    ("\\boxed{C}", True), ("The correct option is C", True), ("I think it is 7.", True),
                    ("A", False), ("The answer is A", False), ("A spider... the answer is C.", True),
                    ("a prime is a number... answer is c", False), ("It's not (B); the answer is (C).", True), ("(C) 7", True),
                    ("Not sure", False)]:
    check(G(mmlu, reply) == want, f"choice: {reply!r} -> {'right' if want else 'wrong'}")
arc = one({"question": "Sky colour?", "choices": {"text": ["red", "blue", "green"], "label": ["A", "B", "C"]}, "answerKey": "B"})
check(arc["letter"] == "B" and G(arc, "B) blue") and G(arc, "It is blue."), "ARC: {text, label} choices + answerKey")
arc1 = one({"question": "Pick", "choices": {"text": ["x", "y"], "label": ["1", "2"]}, "answerKey": "2"})
check(arc1["letter"] == "B", "ARC with numeric labels: answerKey '2' -> B")
medqa = one({"question": "Dx?", "options": {"A": "flu", "B": "cold", "C": "covid"}, "answer_idx": "C"})
check(medqa["letter"] == "C" and G(medqa, "The answer is C. covid"), "MedQA: options dict + answer_idx letter")
hs = one({"ctx": "He opens the door and", "endings": ["sleeps", "walks in", "flies", "sings"], "label": "1"})
check(hs["letter"] == "B", "HellaSwag: ctx + endings + label index")
byt = one({"question": "Which?", "choices": ["Paris", "Rome"], "answer": "Rome"})
check(byt["letter"] == "B", "choices with the answer given as the choice's text")
written = one({"question": "Which is largest?\nA) 3\nB) 9\nC) 5", "answer": "B"})
check(written["kind"] == "choice" and written["letter"] == "B" and written["question"].count("B)") == 1 and G(written, "B"),
      "options already written in the question + letter answer (not added twice)")
try:
    one({"question": "Which?", "choices": ["a", "b"], "answer": 5})
    check(False, "an answer index outside the choices is refused")
except ValueError:
    check(True, "an answer index outside the choices is refused")

# ------------------------------------------------------------------ yes / no
bq = one({"question": "Is the sky blue?", "answer": True})
check(bq["kind"] == "bool" and G(bq, "Yes, it is.") and G(bq, "True.") and not G(bq, "No.") and not G(bq, "Hmm."),
      "BoolQ: answer true -> yes / true right, no wrong, no answer wrong")
bn = one({"question": "Is fire cold?", "answer": "no"})
check(bn["kind"] == "bool" and G(bn, "No, fire is hot.") and not G(bn, "Yes"), "'no' answers")

# ------------------------------------------------------------------ text
T = lambda a, **k: one({"question": "q", "answer": a, **k})
check(G(T("cheetah"), "Cheetahs.") and not G(T("cat"), "concatenate"), "text: plurals, whole words only")
check(not G(one({"question": "q", "answer": ["same"], "not": ["heavier"]}), "the same, iron is heavier"), "text: a 'not' word makes it wrong")
check(G(T("São Paulo"), "Sao Paulo is the largest") and G(T("Paris"), "PARIS"), "text: accents and case ignored")
check(not G(T("Paris"), "It is not Paris, it is Lyon.") and G(T("Lyon"), "It is not Paris, it is Lyon."), "text: a negated answer doesn't count")
check(G(T("Paris / paris city"), "paris city"), "text: several right answers")
check(G(T("Lyon"), "Paris is famous, but the answer is Lyon.") and not G(T("Paris"), "Paris is famous, but the answer is Lyon."),
      "text: after 'the answer is', only the stated answer counts")
check(G(T("H2O"), "The formula is H2O."), "text: answers with digits")
check(G(T("e = mc^2"), "Einstein: e = mc^2"), "text: answers with symbols (substring)")

# ------------------------------------------------------------------ files
items = g.parse_questions("What is 2+2? | 4\nCapital of France? | Paris / paris city\n# comment\n")
check(len(items) == 2 and items[0]["value"] == 4 and items[1]["keywords"] == ["Paris", "paris city"], "pipe lines")
items = g.parse_questions("question,answer\n\"Hi, there?\",hello\nNum?,7\n")
check(len(items) == 2 and items[0]["question"] == "Hi, there?" and items[1]["value"] == 7, "CSV")
items = g.parse_questions("Question\tAnswer\nHow many legs?\t4\n")
check(len(items) == 1 and items[0]["value"] == 4, "TSV with capitalised headers")
items = g.parse_questions("\n".join(json.dumps(x) for x in [{"Question": "a?", "Answer": "1"}, {"prompt": "b?", "target": "yes"},
                                                           {"input": "c?", "output": "blue"}]))
check([x["kind"] for x in items] == ["number", "bool", "text"], "JSONL with other field names (Question/Answer, prompt/target, input/output)")
items = g.parse_questions(json.dumps({"data": [{"question": "a?", "answer": "x"}]}))
check(len(items) == 1, "JSON object with a 'data' list")
items = g.parse_questions(json.dumps([{"row": {"question": "a?", "answer": "#### 3"}}]))
check(items[0]["value"] == 3, "Hugging Face viewer rows ({'row': {...}})")
for bad, why in (("no separator here", "a line without |"), ('[{"question": "a?"}]', "an item without an answer"),
                 ('[{"answer": "1"}]', "an item without a question"), ("[1, 2]", "a list of non-objects")):
    try:
        g.parse_questions(bad)
        check(False, f"{why} is refused")
    except ValueError:
        check(True, f"{why} is refused")

# a reply cut off by the token limit: only an answer it marks as one counts
it = N("18")
check(not g.grade(it, "16 - 3 = 13, then 13 + 5 = 18 and then we", unfinished=True)[0]
      and g.grade(it, "16 - 3 = 13, then 13 + 5 = 18 and then we", unfinished=False)[0]
      and g.grade(it, "So the answer is 18. Let me double check: 16 - 3 =", unfinished=True)[0]
      and g.grade(T("Paris"), "Paris is the capital and it is known for", unfinished=True)[0],
      "cut-off replies: a number counts only when marked as the answer; text answers are graded as usual")

# offsets point at the answer as written, also after characters the grader reads differently (LaTeX escapes,
# "…", "²", thin spaces, accents, blank lines)
for a, reply in (("18", "so 9 * 2 = 18 dollars"), ("8", "it has eight legs"), ("C", "The answer is (C)"),
                 ("18", "Each egg costs \\$2, so she makes \\$18"), ("18", "x² … so the total is 18"),
                 ("18", "9\\,000 is wrong; the total is 18"), ("1234", "It is 1 234 km"),
                 ("0.75", "so $\\frac{3}{4}$ of it"), ("C", "Hmm… “well” — the answer is (C)")):
    it = N(a) if a != "C" else mmlu
    ok_, got, off = g.grade(it, reply)
    check(ok_ and reply[off:off + len(got)].lower() == got.lower(), f"offset of {got!r} in {reply!r}")
ok_, got, off = g.grade(T("Paris"), "Hmm.\n\n\nThé capital  is Paris")
check(ok_ and "Hmm.\n\n\nThé capital  is Paris"[off:off + 5] == "Paris", "offset of a text answer after accents and blank lines")

# an answer written as a calculation: its result, not its first number
for a, reply, want in (("18", "So the answer is 9 * 2 = 18 dollars.", True), ("18", "The answer is 16 eggs - 7 eggs = 9 eggs x 2 = 18", True),
                       ("18", "The answer is 18. Check: 20 - 2 = 17", True), ("5", "the answer is x = 5", True),
                       ("9", "So the answer is 9 * 2 = 18 dollars.", False)):
    check(G(N(a), reply) == want, f"number {a!r} in {reply!r} -> {'right' if want else 'wrong'}")

# multiple choice: the stated answer wins over options discussed afterwards
for reply, want in (("The correct answer is C. Option A is wrong because 4 is even.", True),
                    ("Answer: C\nOption B (6) is divisible by 2 and option D is 3 squared.", True),
                    ("I would pick option C.", True), ("The answer is A. Option C is not prime.", False)):
    check(G(mmlu, reply) == want, f"choice: {reply!r} -> {'right' if want else 'wrong'}")

# yes / no: "not true" is a no
for reply, want in (("That is not true.", True), ("It isn't really correct to say that.", True), ("No.", True),
                    ("That is true.", False), ("Not false, it is true", False)):
    check(G(bn, reply) == want, f"yes/no (answer no): {reply!r} -> {'right' if want else 'wrong'}")

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
