"""List every study question whose grade differs from the original model, for hand checking."""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
q = {x["id"]: x for x in json.loads((H / "questions.json").read_text(encoding="utf-8"))}
o = json.loads((H / "bench_original.json").read_text(encoding="utf-8"))["study"]
for name in sys.argv[1:]:
    e = json.loads((H / f"bench_{name}.json").read_text(encoding="utf-8"))["study"]
    print(f"######## {name}: {sum(v['correct'] for v in e.values())}/100")
    for k in o:
        if o[k]["correct"] != e[k]["correct"]:
            print(f"== {k} {'+' if e[k]['correct'] else '-'} Q: {q[k]['q'][:80]} | key {q[k].get('kw', q[k].get('num'))}")
            for t, v in (("O", o[k]), ("E", e[k])):
                r = v["reply"].replace("\n", " ")
                print(f"   {t}: {r[:110]!r}" + (f" ... {r[-90:]!r}" if len(r) > 200 else ""))
