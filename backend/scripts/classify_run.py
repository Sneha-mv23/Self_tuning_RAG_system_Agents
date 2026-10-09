import argparse
from collections import Counter, defaultdict

from app.db.experiments import connect, list_runs
from app.evals.failure_classifier import CATEGORIES, STAGE, classify, load_outcomes

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", type=int, default=None, help="classify one run (default: all runs)")
args = ap.parse_args()

conn = connect()
runs = [r for r in list_runs(conn) if args.run_id in (None, r["id"])]
if not runs:
    raise SystemExit("no matching runs. Use backend/scripts/log_run.py first.")

for r in runs:
    top_k, outcomes = load_outcomes(conn, r["id"])
    by_cat: dict[str, list[str]] = defaultdict(list)
    for o in outcomes:
        by_cat[classify(o, top_k)].append(o.question_id)
    assert set(by_cat) <= set(CATEGORIES)

    n = len(outcomes)
    print(f"RUN {r['id']}  prompt={r['prompt']}  cs/ov/k={r['chunk_size']}/{r['overlap']}/{r['top_k']}  "
          f"{n} questions")
    stages = Counter()
    for cat in CATEGORIES:
        ids = by_cat.get(cat, [])
        if not ids:
            continue
        stages[STAGE[cat]] += len(ids)
        listing = "" if cat == "pass" else ", ".join(ids)
        print(f"  {cat:<19} {len(ids):>3}  [{STAGE[cat]}]  {listing}")
    failures = n - len(by_cat.get("pass", []))
    print(f"  failures: {failures} of {n}  |  retrieval {stages['retrieval']}, "
          f"generation {stages['generation']}, error {stages['error']}\n")