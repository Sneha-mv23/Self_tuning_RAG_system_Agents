import json
import sys
from pathlib import Path

from app.evals.golden import load_golden
from app.rag.loader import load_corpus

sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path("experiments/runs")
DIR = Path("data/judge_validation")
NAMES = {"strict": "cs256_ov32_k4_strict_t0.0", "basic": "cs256_ov32_k4_basic_t0.0"}

sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
judge_by_key = {(t["task"], t["run"], t["id"]): t["judge"] for t in sample["tasks"]}
labels = [json.loads(x) for x in (DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
items = {i.id: i for i in load_golden(docs=load_corpus())}
gen = {run: {r["id"]: r for r in json.loads(
    (RUNS / f"generation_dev_{n}.json").read_text(encoding="utf-8"))["records"]} for run, n in NAMES.items()}
jud = {run: {j["id"]: j for j in json.loads(
    (RUNS / f"judged_dev_{n}.json").read_text(encoding="utf-8"))["judgments"]} for run, n in NAMES.items()}

for l in labels:
    key = (l["task"], l["run"], l["id"])
    if judge_by_key[key] == l["human"]:
        continue
    it, rec, j = items[l["id"]], gen[l["run"]][l["id"]], jud[l["run"]][l["id"]]
    print("=" * 78)
    print(f"{l['task'].upper()} | {l['run']} {l['id']} [{it.type}] | judge={'yes' if judge_by_key[key] else 'no'} human={'yes' if l['human'] else 'no'}")
    print(f"Q: {it.question}")
    if it.reference_answer:
        print(f"Reference: {it.reference_answer}")
    print(f"Answer ({len(rec['answer'])} chars): {rec['answer'][:600]}")
    print(f"Judge reason: {j['correct_reason'] if l['task'] == 'correct' else j['faithful_reason']}\n")