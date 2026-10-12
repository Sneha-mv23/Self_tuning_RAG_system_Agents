import json
import sys
from pathlib import Path

from app.evals.golden import load_golden
from app.evals.judge import context_from_spans
from app.rag.loader import load_corpus

sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path("experiments/runs")
DIR = Path("data/classifier_validation")
NAMES = {"strict": "cs256_ov32_k4_strict_t0.0", "basic": "cs256_ov32_k4_basic_t0.0"}

sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
human = {}
if (DIR / "labels.jsonl").exists():
    for line in (DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            human[(d["run"], d["id"])] = d["human"]

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = {i.id: i for i in load_golden(docs=docs)}
records = {
    run: {r["id"]: r for r in json.loads(
        (RUNS / f"generation_dev_{name}.json").read_text(encoding="utf-8"))["records"]}
    for run, name in NAMES.items()
}

mismatch = 0
for t in sample["tasks"]:
    rec = records[t["run"]][t["id"]]
    ctx = context_from_spans(rec["used_chunks"], docs_by_id)
    found = [ev.quote in ctx for ev in items[t["id"]].evidence]
    if all(found) != t["predicted_in_prompt"]:
        mismatch += 1
        print(f"MISMATCH {t['run']} {t['id']}: classifier in_prompt={t['predicted_in_prompt']}, "
              f"quotes verbatim in context={found}")
print(f"\n{len(sample['tasks'])} items checked, {mismatch} where the classifier and the plain text check disagree")

print("\nitems where the classifier said NOT in prompt but you said the key point was in the context:")
for t in sample["tasks"]:
    if not t["predicted_in_prompt"] and human.get((t["run"], t["id"])) == "y":
        rec = records[t["run"]][t["id"]]
        seen = [c["doc_id"].split("/")[-1] + f"@{c['start']}" for c in rec["used_chunks"]]
        print(f"  {t['run']} {t['id']}: gold evidence is in {[e.doc_id.split('/')[-1] for e in items[t['id']].evidence]}; "
              f"prompt held {seen}")