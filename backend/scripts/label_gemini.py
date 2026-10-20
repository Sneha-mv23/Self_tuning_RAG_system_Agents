import json
import sys
from pathlib import Path

from app.evals.golden import load_golden
from app.evals.judge import context_from_spans
from app.rag.loader import load_corpus

sys.stdout.reconfigure(encoding="utf-8")

GEN = Path("experiments/runs/gemini/generation_dev_cs256_ov32_k4_strict_t0.0.json")
DIR = Path("data/judge_validation_gemini")

sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
labels_path = DIR / "labels.jsonl"
done: set[tuple[str, str]] = set()
if labels_path.exists():
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            done.add((d["task"], d["id"]))

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = {i.id: i for i in load_golden(docs=docs)}
records = {r["id"]: r for r in json.loads(GEN.read_text(encoding="utf-8"))["records"]}

todo = [t for t in sample["tasks"] if (t["task"], t["id"]) not in done]
total = len(sample["tasks"])
print(f"{len(done)} of {total} labeled, {len(todo)} to go.  y = yes, n = no, s = skip, q = quit and save.\n")
print("RULES")
print("- Decide only from the text on screen. Don't try to guess what the judge said.")
print("- Correctness: does the answer give the KEY POINT of the reference? Wording and extra detail are fine.")
print("  Answer 'n' if it contradicts the reference or misses the key point.")
print("- Faithfulness: is EVERY factual claim in the answer supported by the context shown? Ignore [n] markers.\n")

for t in todo:
    item, rec = items[t["id"]], records[t["id"]]
    print("=" * 78)
    print(f"[{len(done) + 1}/{total}]  {'CORRECTNESS' if t['task'] == 'correct' else 'FAITHFULNESS'}")
    print(f"Question: {item.question}\n")
    if t["task"] == "correct":
        if item.type == "unanswerable":
            print("The documentation does NOT answer this question.")
            prompt = "Does the answer decline, without presenting an invented answer? [y/n/s/q]: "
        else:
            print(f"Reference answer: {item.reference_answer}")
            prompt = "Does the answer give the key point of the reference? [y/n/s/q]: "
        print(f"\nCandidate answer:\n{rec['answer']}\n")
    else:
        print(f"Context the model saw:\n{context_from_spans(rec['used_chunks'], docs_by_id)}\n")
        print(f"Answer:\n{rec['answer']}\n")
        prompt = "Is every factual claim in the answer supported by the context? [y/n/s/q]: "
    while True:
        a = input(prompt).strip().lower()
        if a in ("y", "n", "s", "q"):
            break
    if a == "q":
        break
    if a == "s":
        continue
    with labels_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"task": t["task"], "id": t["id"], "human": a == "y"}) + "\n")
    done.add((t["task"], t["id"]))

print(f"\n{len(done)} of {total} labeled. Run the same command to continue.")