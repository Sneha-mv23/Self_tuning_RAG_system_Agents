import json
import statistics
import sys
from pathlib import Path

from app.evals.generation_run import is_refusal, percentile
from app.evals.golden import load_golden
from app.rag.loader import load_corpus

path = sys.argv[1] if len(sys.argv) > 1 else \
    "experiments/runs/generation_dev_cs256_ov32_k4_strict_t0.0.json"
data = json.loads(Path(path).read_text(encoding="utf-8"))
records = data["records"]
items = {i.id: i for i in load_golden(docs=load_corpus())}

lat = [r["latency_s"] for r in records if r["error"] is None]
print(f"LATENCY over {len(lat)} answers (stored from the original, uncached calls)")
print(f"  median {statistics.median(lat):.1f}s | mean {statistics.mean(lat):.1f}s | "
      f"p95 {percentile(lat, 95):.1f}s | max {max(lat):.1f}s")
print("  slowest 5:")
for r in sorted(records, key=lambda r: -r["latency_s"])[:5]:
    print(f"    {r['id']}  {r['latency_s']:.0f}s  prompt_tokens={r['prompt_tokens']}")


def covered(item, chunks):
    return all(
        any(c["doc_id"] == ev.doc_id and c["start"] < ev.end and c["end"] > ev.start for c in chunks)
        for ev in item.evidence
    )


cells = {}
for r in records:
    it = items[r["id"]]
    if it.type == "unanswerable":
        continue
    key = (covered(it, r["used_chunks"]), is_refusal(r["answer"]))
    cells.setdefault(key, []).append(r["id"])

print("\nANSWERABLE QUESTIONS: was the gold text in the prompt, and did the model refuse?")
labels = {
    (True, False): "gold in prompt, answered   (expected)",
    (True, True): "gold in prompt, REFUSED    (generation problem: over-refusal)",
    (False, True): "gold missing, refused      (retrieval problem, model was honest)",
    (False, False): "gold missing, answered     (hallucination or lucky answer)",
}
for key, label in labels.items():
    print(f"  {len(cells.get(key, [])):>3}  {label}")
over = cells.get((True, True), [])
if over:
    print("\nover-refused ids: " + ", ".join(over))