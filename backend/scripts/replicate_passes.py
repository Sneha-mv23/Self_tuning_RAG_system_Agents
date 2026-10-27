import argparse
import json
from dataclasses import asdict

from app.db.experiments import connect
from app.evals.failure_classifier import QuestionOutcome, classify
from app.evals.generation_run import GenerationCache, run_generation
from app.evals.golden import load_golden
from app.evals.judge import BudgetReached, Judge, judge_record
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.llm_limits import QuotaExhausted
from app.optimizer.experiment import pass_map
from app.optimizer.loop import config_of
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", type=int, default=5)
ap.add_argument("--replicate", type=int, default=1)
ap.add_argument("--max-calls", type=int, default=10, help="cap on new judge calls (expected: 0)")
args = ap.parse_args()

conn = connect()
cfg, _ = config_of(conn, args.run_id)
docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = select(load_golden(docs=docs), load_split(), "dev")
by_id = {i.id: i for i in items}

cache = GenerationCache()
records = run_generation(RetrievalEvaluator(docs), cfg, items, cache=cache, verbose=False,
                         replicate=args.replicate)  # all cached from the earlier replicate run
cache.close()
ranks = {r["question_id"]: (r["rank_needed"], json.loads(r["evidence_ranks_json"] or "[]"))
         for r in conn.execute("SELECT question_id, rank_needed, evidence_ranks_json "
                               "FROM question_results WHERE run_id = ?", (args.run_id,))}

judge = Judge(max_calls=args.max_calls)
outcomes = []
try:
    for r in records:
        j = judge_record(judge, asdict(r), by_id[r.id], docs_by_id)
        rank, ev = ranks[r.id]
        outcomes.append(QuestionOutcome(
            question_id=r.id, type=j.type, rank_needed=rank, evidence_ranks=ev,
            answer_correct=j.correct, faithful=j.faithful, pure_refusal=j.pure_refusal,
            mixed_refusal=j.mixed_refusal, error=j.error or r.error))
except (QuotaExhausted, BudgetReached) as e:
    raise SystemExit(f"STOPPED: {str(e)[:200]}")

rep = {o.question_id: classify(o, cfg.top_k) == "pass" for o in outcomes}
orig = pass_map(conn, args.run_id)
print(f"new judge calls made: {judge.calls_made} (0 means everything came from the cache)")
print(f"run {args.run_id} original: {sum(orig.values())}/{len(orig)} pass | "
      f"replicate {args.replicate}: {sum(rep.values())}/{len(rep)} pass")
print("passed originally, fail in the replicate:", sorted(q for q in orig if orig[q] and not rep[q]))
print("failed originally, pass in the replicate:", sorted(q for q in orig if not orig[q] and rep[q]))