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
labels_path = DIR / "labels.jsonl"
done: set[tuple[str, str]] = set()
if labels_path.exists():
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            done.add((d["run"], d["id"]))

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = {i.id: i for i in load_golden(docs=docs)}
records = {
    run: {r["id"]: r for r in json.loads(
        (RUNS / f"generation_dev_{name}.json").read_text(encoding="utf-8"))["records"]}
    for run, name in NAMES.items()
}

todo = [t for t in sample["tasks"] if (t["run"], t["id"]) not in done]
total = len(sample["tasks"])
print(f"{len(done)} of {total} labeled, {len(todo)} to go.  y / p / n, s = skip, q = quit and save.\n")
print("RULES")
print("- Judge ONLY the context text below. Ignore whether a fact is true in real life.")
print("- The reference answer may contain extra detail. Look for the KEY POINT that answers the question.")
print("- The key point may be worded differently, and may sit anywhere in the context.\n")

for t in todo:
    item = items[t["id"]]
    rec = records[t["run"]][t["id"]]
    print("=" * 78)
    print(f"[{len(done) + 1}/{total}]")
    print(f"Question: {item.question}\n")
    print(f"Reference answer: {item.reference_answer}\n")
    print(f"Context the model saw:\n{context_from_spans(rec['used_chunks'], docs_by_id)}\n")
    while True:
        a = input("Is the key point that answers the question in the context? "
                  "[y = yes / p = only a fragment of it / n = no / s / q]: ").strip().lower()
        if a in ("y", "p", "n", "s", "q"):
            break
    if a == "q":
        break
    if a == "s":
        continue
    with labels_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run": t["run"], "id": t["id"], "human": a}) + "\n")
    done.add((t["run"], t["id"]))

print(f"\n{len(done)} of {total} labeled. Run the same command to continue.")