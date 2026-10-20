
import argparse
import json
import random
from pathlib import Path

RUNS = Path("experiments/runs/gemini")
NAME = "cs256_ov32_k4_strict_t0.0"
OUT = Path("data/judge_validation_gemini")

ap = argparse.ArgumentParser()
ap.add_argument("--n-correct", type=int, default=20)
ap.add_argument("--n-faithful", type=int, default=8)
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

sample_path = OUT / "sample.json"
if sample_path.exists() and not args.force:
    raise SystemExit(f"{sample_path} already exists and your labels are tied to it. "
                     "Use --force only to start over (and delete labels.jsonl first).")

judged = json.loads((RUNS / f"judged_dev_{NAME}.json").read_text(encoding="utf-8"))["judgments"]
pool = [j for j in judged if j["error"] is None and not j["pure_refusal"]]
yes_correct = [j["id"] for j in pool if j["correct"] is True]
yes_faithful = [j["id"] for j in pool if j["faithful"] is True]

rng = random.Random(args.seed)
tasks = ([{"task": "correct", "id": i} for i in rng.sample(yes_correct, min(args.n_correct, len(yes_correct)))]
         + [{"task": "faithful", "id": i} for i in rng.sample(yes_faithful, min(args.n_faithful, len(yes_faithful)))])
rng.shuffle(tasks)

OUT.mkdir(parents=True, exist_ok=True)
sample_path.write_text(json.dumps({
    "seed": args.seed,
    "pool": {"correct_yes": len(yes_correct), "faithful_yes": len(yes_faithful),
             "judged_total": len(pool)},
    "tasks": tasks,
}, indent=2), encoding="utf-8")
print(f"{len(tasks)} labeling tasks written to {sample_path}")
print(f"pool: the judge scored {len(pool)} answers with an LLM; "
      f"{len(yes_correct)} correctness 'yes', {len(yes_faithful)} faithfulness 'yes'")