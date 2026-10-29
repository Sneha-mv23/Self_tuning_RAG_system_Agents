import argparse
import json
import time
from pathlib import Path

from app.config import settings
from app.db.experiments import connect
from app.evals.generation_run import GenerationAborted, GenerationCache, count_uncached
from app.evals.golden import load_golden
from app.evals.judge import BudgetReached, Judge
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.llm_limits import QuotaExhausted
from app.optimizer.experiment import pass_map, run_experiment
from app.optimizer.loop import CALLS_PER_NEW_PROMPT
from app.optimizer.test_plan import EVAL_CONFIGS, MAX_REPLICATES, fingerprint, pending, run_order
from app.rag.loader import load_corpus

LOCK_PATH = Path("experiments/test_split_lock.json")

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", choices=list(EVAL_CONFIGS), default=list(EVAL_CONFIGS))
ap.add_argument("--replicates", type=int, default=2, choices=range(1, MAX_REPLICATES + 1))
ap.add_argument("--budget", type=int, default=200, help="max new API calls this session (generation + judging)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

conn = connect()
models = [settings.generator_model, settings.judge_model]
dev = conn.execute("SELECT generator_model, judge_model FROM runs WHERE split = 'dev' "
                   "ORDER BY id DESC LIMIT 1").fetchone()
if dev and [dev["generator_model"], dev["judge_model"]] != models:
    raise SystemExit(f"the dev runs used {dev['generator_model']} / {dev['judge_model']}, but .env points at "
                     f"{models[0]} / {models[1]}. Test results must use the same models.")

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "test")
if not items:
    raise SystemExit("no test questions found. Run make_split.py first.")
fp = fingerprint(items)

lock = json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {"completed": []}
if lock.get("fingerprint") not in (None, fp):
    raise SystemExit("the test questions changed since the first test evaluation. The test results would "
                     "not be comparable, so this run is refused.")
if lock.get("models") not in (None, models):
    raise SystemExit(f"the test split was already evaluated with {lock['models']}. Do not mix models.")

todo = pending(run_order(args.configs, args.replicates), lock["completed"])
print(f"{len(items)} test questions | generator {models[0]} | judge {models[1]}")
print(f"{len(lock['completed'])} runs completed earlier, {len(todo)} pending\n")
if not todo:
    raise SystemExit("nothing to do: every requested (config, replicate) is already in the lock file.")

evaluator = RetrievalEvaluator(docs)
gen_cache = GenerationCache()
judge = Judge()
spent = 0
for name, rep in todo:
    cfg = EVAL_CONFIGS[name]
    est = count_uncached(evaluator, cfg, items, gen_cache, rep)
    est_calls = int(est * CALLS_PER_NEW_PROMPT)
    print(f"{name} replicate {rep}: {est} new generation calls, up to ~{est_calls} API calls with judging")
    if args.dry_run:
        continue
    if est_calls > args.budget - spent:
        print(f"  stopping: ~{est_calls} calls would exceed the {args.budget - spent} left in this session's budget. "
              "Rerun the same command in a later session.")
        break

    judge.max_calls = judge.calls_made + max(0, args.budget - spent - est)
    try:
        result = run_experiment(conn, evaluator, cfg, items, gen_cache, judge, split="test", replicate=rep)
    except (QuotaExhausted, BudgetReached, GenerationAborted) as e:
        print(f"\nSTOPPED: {type(e).__name__}: {str(e)[:250]}")
        print("Finished answers and verdicts are cached. Rerun the same command and it continues.")
        break
    spent += result.new_generation_calls + result.new_judge_calls
    if result.errors:
        shown = "; ".join(f"{q}: {result.detail[q][:80]}" for q in result.errors[:3])
        print(f"  ABORTED: {len(result.errors)} questions had errors ({shown}). Not logged, so rerun to retry.")
        break

    passes = pass_map(conn, result.run_id)
    lock.setdefault("fingerprint", fp)
    lock.setdefault("models", models)
    lock["completed"].append({"config": name, "replicate": rep, "run_id": result.run_id,
                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"  done: run {result.run_id}, {sum(passes.values())}/{len(passes)} pass, "
          f"{result.new_generation_calls + result.new_judge_calls} new API calls ({spent}/{args.budget} used)")

gen_cache.close()
print(f"\nAPI calls used this session: {spent}")