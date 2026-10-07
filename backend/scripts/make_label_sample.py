import argparse
import json
import random
from pathlib import Path

RUNS = Path("experiments/runs")
OUT = Path("data/judge_validation")
NAMES = {"strict": "cs256_ov32_k4_strict_t0.0", "basic": "cs256_ov32_k4_basic_t0.0"}

ap = argparse.ArgumentParser()
ap.add_argument("--n-correct", type=int, default=40)
ap.add_argument("--n-faithful", type=int, default=16)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

sample_path = OUT / "sample.json"
if sample_path.exists() and not args.force:
    raise SystemExit(f"{sample_path} already exists and your labels are tied to it. "
                     "Use --force only to start over (and delete labels.jsonl first).")

pool = {"correct": [], "faithful": []}
for run, name in NAMES.items():
    judged = json.loads((RUNS / f"judged_dev_{name}.json").read_text(encoding="utf-8"))["judgments"]
    for j in judged:
        if j["error"] is not None or j["pure_refusal"]:
            continue  # refusals were scored by rule, so there is no LLM verdict to validate
        pool["correct"].append({"task": "correct", "run": run, "id": j["id"], "judge": bool(j["correct"])})
        if j["faithful"] is not None:
            pool["faithful"].append({"task": "faithful", "run": run, "id": j["id"], "judge": bool(j["faithful"])})

rng = random.Random(args.seed)


def draw(items: list[dict], total: int) -> list[dict]:
    neg = [i for i in items if not i["judge"]]
    pos = [i for i in items if i["judge"]]
    n_neg = min(len(neg), total // 2)
    n_pos = min(len(pos), total - n_neg)
    return rng.sample(neg, n_neg) + rng.sample(pos, n_pos)


tasks = draw(pool["correct"], args.n_correct) + draw(pool["faithful"], args.n_faithful)
rng.shuffle(tasks)
counts = {t: {"true": sum(i["judge"] for i in pool[t]), "false": sum(not i["judge"] for i in pool[t])}
          for t in pool}

OUT.mkdir(parents=True, exist_ok=True)
sample_path.write_text(json.dumps({"seed": args.seed, "pool": counts, "tasks": tasks}, indent=2),
                       encoding="utf-8")
print(f"{len(tasks)} labeling tasks written to {sample_path}")
print(f"pool sizes: {counts}")
print("Do NOT open sample.json before labeling: it contains the judge's verdicts.")