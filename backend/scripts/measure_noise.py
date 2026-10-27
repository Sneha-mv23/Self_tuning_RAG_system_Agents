import argparse
from dataclasses import asdict

from app.config import settings
from app.db.experiments import connect
from app.evals.generation_run import GenerationAborted, GenerationCache, run_generation
from app.evals.golden import load_golden
from app.evals.judge import BudgetReached, Judge, is_pure_refusal, judge_record
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.llm_limits import QuotaExhausted
from app.optimizer.loop import config_of
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", type=int, default=5)
ap.add_argument("--replicate", type=int, default=1)
ap.add_argument("--limit", type=int, default=None, help="only the first N dev questions")
ap.add_argument("--judge", action="store_true", help="also judge the answers that changed")
ap.add_argument("--max-calls", type=int, default=40, help="cap on new judge calls")
ap.add_argument("--show", nargs="*", default=[], help="question ids whose two answers to print")
args = ap.parse_args()

conn = connect()
cfg, row = config_of(conn, args.run_id)
if row["generator_model"] != settings.generator_model:
    raise SystemExit(f"run {args.run_id} used {row['generator_model']}, but .env points at {settings.generator_model}")

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = select(load_golden(docs=docs), load_split(), "dev")
if args.limit:
    items = items[: args.limit]
by_id = {i.id: i for i in items}
orig = {r["question_id"]: r for r in conn.execute(
    "SELECT question_id, answer, answer_correct, faithful FROM question_results WHERE run_id = ?", (args.run_id,))}

cache = GenerationCache()
try:
    records = run_generation(RetrievalEvaluator(docs), cfg, items, cache=cache, verbose=False,
                             replicate=args.replicate)
except (QuotaExhausted, GenerationAborted) as e:
    cache.close()
    raise SystemExit(f"STOPPED: {str(e)[:300]}\nFinished answers are cached, so rerunning continues.")
cache.close()


def norm(text: str | None) -> str:
    return " ".join((text or "").split())


ok = [r for r in records if r.error is None]
same = [r for r in ok if norm(r.answer) == norm(orig[r.id]["answer"])]
changed = [r for r in ok if r not in same]
flips = [r.id for r in changed if is_pure_refusal(r.answer) != is_pure_refusal(orig[r.id]["answer"] or "")]
print(f"run {args.run_id} ({cfg.retriever} cs/ov/k={cfg.chunk_size}/{cfg.overlap}/{cfg.top_k}), "
      f"replicate {args.replicate}: {len(ok)} compared, {len(records) - len(ok)} errors skipped")
print(f"identical text: {len(same)} of {len(ok)} | different text: {len(changed)}")
print(f"answered <-> refused flips (rule-based): {len(flips)} {flips}")

for qid in dict.fromkeys(flips + args.show):
    rep = next((r for r in records if r.id == qid), None)
    if rep is None or qid not in orig:
        continue
    print(f"\n{qid} original : {(orig[qid]['answer'] or '')[:300]!r}")
    print(f"{qid} replicate: {rep.answer[:300]!r}")
    
if args.judge and changed:
    judge = Judge(max_calls=args.max_calls)
    c_flips, f_flips, done = [], [], 0
    try:
        for r in changed:
            if is_pure_refusal(r.answer):
                continue  # refusals are scored by rule and counted above
            j = judge_record(judge, asdict(r), by_id[r.id], docs_by_id)
            if j.error:
                continue
            done += 1
            o = orig[r.id]
            if o["answer_correct"] is not None and bool(o["answer_correct"]) != bool(j.correct):
                c_flips.append(r.id)
            if o["faithful"] is not None and j.faithful is not None and bool(o["faithful"]) != bool(j.faithful):
                f_flips.append(r.id)
    except (QuotaExhausted, BudgetReached) as e:
        print(f"judging stopped early: {str(e)[:200]}")
    print(f"judged {done} changed answers | correctness verdict flips: {len(c_flips)} {c_flips} "
          f"| faithfulness verdict flips: {len(f_flips)} {f_flips}")
elif changed:
    print("rerun with --judge to see whether the changed answers change the verdicts")