import argparse
import json
import random
from pathlib import Path

from app.db.experiments import connect
from app.evals.failure_classifier import classify, load_outcomes

OUT = Path("data/classifier_validation")
ap = argparse.ArgumentParser()
ap.add_argument("--controls", type=int, default=8)
ap.add_argument("--seed", type=int, default=9)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

sample_path = OUT / "sample.json"
if sample_path.exists() and not args.force:
    raise SystemExit(f"{sample_path} already exists and your labels are tied to it. "
                     "Use --force only to start over (and delete labels.jsonl first).")

conn = connect()
rows = conn.execute(
    "SELECT id, prompt FROM runs WHERE split = 'dev' AND chunk_size = 256 AND overlap = 32 "
    "AND top_k = 4 AND temperature = 0.0 ORDER BY id"
).fetchall()
run_ids: dict[str, int] = {}
for r in rows:
    run_ids.setdefault(r["prompt"], r["id"])
for p in ("strict", "basic"):
    if p not in run_ids:
        raise SystemExit(f"no logged dev run for the {p} prompt. Use log_run.py first.")

tasks: list[dict] = []
seen: set[str] = set()
passes: list[tuple[str, str, str, bool]] = []
for prompt in ("strict", "basic"):
    top_k, outcomes = load_outcomes(conn, run_ids[prompt])
    for o in outcomes:
        if o.type == "unanswerable":
            continue  # nothing in the docs to find, so there is no retrieval-vs-generation question
        cat = classify(o, top_k)
        in_prompt = o.rank_needed is not None and o.rank_needed <= top_k
        if cat in ("pass", "error"):
            if cat == "pass" and in_prompt and prompt == "strict":
                passes.append((prompt, o.question_id, cat, in_prompt))
            continue
        if o.question_id in seen:
            continue  # retrieval is identical across prompts, so one label per question is enough
        seen.add(o.question_id)
        tasks.append({"run": prompt, "id": o.question_id, "category": cat,
                      "predicted_in_prompt": in_prompt, "kind": "failure"})

rng = random.Random(args.seed)
controls = [p for p in passes if p[1] not in seen]
for prompt, qid, cat, in_prompt in rng.sample(controls, min(args.controls, len(controls))):
    tasks.append({"run": prompt, "id": qid, "category": cat,
                  "predicted_in_prompt": in_prompt, "kind": "control"})
rng.shuffle(tasks)

OUT.mkdir(parents=True, exist_ok=True)
sample_path.write_text(json.dumps({"seed": args.seed, "tasks": tasks}, indent=2), encoding="utf-8")
n_fail = sum(1 for t in tasks if t["kind"] == "failure")
print(f"{len(tasks)} labeling tasks ({n_fail} failures, {len(tasks) - n_fail} controls) written to {sample_path}")
print("Do NOT open sample.json before labeling: it contains the classifier's categories.")