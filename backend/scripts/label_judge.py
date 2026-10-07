import json
import sys
from pathlib import Path

from app.evals.golden import load_golden
from app.evals.judge import context_from_spans
from app.rag.loader import load_corpus

sys.stdout.reconfigure(encoding="utf-8")

RUNS = Path("experiments/runs")
DIR = Path("data/judge_validation")
NAMES = {"strict": "cs256_ov32_k4_strict_t0.0", "basic": "cs256_ov32_k4_basic_t0.0"}

sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
labels_path = DIR / "labels.jsonl"
done: set[tuple[str, str, str]] = set()
if labels_path.exists():
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            done.add((d["task"], d["run"], d["id"]))

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = {i.id: i for i in load_golden(docs=docs)}
answers = {
    run: {r["id"]: r for r in json.loads(
        (RUNS / f"generation_dev_{name}.json").read_text(encoding="utf-8"))["records"]}
    for run, name in NAMES.items()
}

todo = [t for t in sample["tasks"] if (t["task"], t["run"], t["id"]) not in done]
total = len(sample["tasks"])
print(f"{len(done)} of {total} labeled, {len(todo)} to go.  y = yes, n = no, s = skip, q = quit and save.\n")

for t in todo:
    rec = answers[t["run"]][t["id"]]
    item = items[t["id"]]
    print("=" * 78)
    print(f"[{len(done) + 1}/{total}]  {'CORRECTNESS' if t['task'] == 'correct' else 'FAITHFULNESS'}")
    print(f"Question: {item.question}\n")
    if t["task"] == "correct" and item.type == "unanswerable":
        print("The documentation does NOT answer this question.")
        print(f"\nCandidate answer:\n{rec['answer']}\n")
        prompt = "Does the answer decline, without presenting an invented answer? [y/n/s/q]: "
    elif t["task"] == "correct":
        print(f"Reference answer: {item.reference_answer}")
        print(f"\nCandidate answer:\n{rec['answer']}\n")
        prompt = ("Does it give the key information of the reference? Wording and extra detail are fine; "
                  "no if it contradicts the reference, misses the key point, or only says it doesn't know. [y/n/s/q]: ")
    else:
        print(f"Context the model saw:\n{context_from_spans(rec['used_chunks'], docs_by_id)}\n")
        print(f"Answer:\n{rec['answer']}\n")
        prompt = ("Is EVERY factual claim in the answer supported by the context? Ignore [n] markers "
                  "and any 'I don't know' sentence. [y/n/s/q]: ")
    while True:
        a = input(prompt).strip().lower()
        if a in ("y", "n", "s", "q"):
            break
    if a == "q":
        break
    if a == "s":
        continue
    with labels_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"task": t["task"], "run": t["run"], "id": t["id"], "human": a == "y"}) + "\n")
    done.add((t["task"], t["run"], t["id"]))

print(f"\n{len(done)} of {total} labeled. Run the same command to continue.")