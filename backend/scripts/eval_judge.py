import argparse
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.evals.golden import load_golden
from app.evals.judge import Judge, build_jobs, judge_record, summarize_judgments
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default="experiments/runs/generation_dev_cs256_ov32_k4_strict_t0.0.json")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--dry-run", action="store_true", help="only count the judge calls needed")
ap.add_argument("--out", default=None)
args = ap.parse_args()

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = {i.id: i for i in load_golden(docs=docs)}
records = json.loads(Path(args.gen).read_text(encoding="utf-8"))["records"]
if args.limit:
    records = records[: args.limit]

judge = Judge()
print(f"judge: {settings.judge_model} @ {urlparse(settings.judge_base_url).netloc} "
      f"| min interval {judge.min_interval_s}s")

if args.dry_run:
    total = todo = 0
    for r in records:
        if r.get("error"):
            continue
        for kind, prompt in build_jobs(r, items[r["id"]], docs_by_id):
            total += 1
            todo += 0 if judge.is_cached(kind, prompt) else 1
    print(f"{total} judge calls needed, {todo} not cached yet")
    raise SystemExit(0)

judgments = []
for n, r in enumerate(records, start=1):
    j = judge_record(judge, r, items[r["id"]], docs_by_id)
    judgments.append(j)
    tags = []
    if j.pure_refusal:
        tags.append("pure_refusal")
    if j.mixed_refusal:
        tags.append("MIXED_REFUSAL")
    if j.error:
        tags.append(f"ERROR {j.error[:80]}")
    print(f"[{n}/{len(records)}] {j.id} [{j.type}] correct={j.correct} faithful={j.faithful} "
          + " ".join(tags), flush=True)

s = summarize_judgments(judgments)
print(f"\n{s['n']} judged | errors={s['errors']}")
print(f"answer correctness (answerable): {s['answer_correctness']:.2f}")
print(f"faithfulness (over {s['n_faithfulness_scored']} answers with claims): {s['faithfulness']:.2f}")
print(f"proper refusal on unanswerable: {s['proper_refusal_rate_unanswerable']:.2f}")
print(f"pure refusal on answerable: {s['pure_refusal_rate_answerable']:.2f} | "
      f"mixed refusal overall: {s['mixed_refusal_rate']:.2f}")

out = args.out or args.gen.replace("generation_", "judged_")
Path(out).write_text(json.dumps(
    {"judge_model": settings.judge_model, "summary": s, "judgments": [asdict(j) for j in judgments]},
    indent=2), encoding="utf-8")
print(f"saved {out}")