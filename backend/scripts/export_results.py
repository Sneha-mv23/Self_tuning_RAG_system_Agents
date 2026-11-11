import json
import time
from pathlib import Path

from app.db.experiments import connect
from app.evals.failure_classifier import classify, load_outcomes
from app.evals.golden import load_golden
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.optimizer.experiment import pass_map
from app.optimizer.levers import estimated_prompt_tokens
from app.optimizer.space import pareto, retrieval_space
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig

OUT = Path("experiments/export/results.json")

conn = connect()
docs = load_corpus()
all_items = load_golden(docs=docs)
split = load_split()
dev_items = {i.id: i for i in select(all_items, split, "dev")}
test_ids = {i.id for i in select(all_items, split, "test")}

runs = []
for r in conn.execute("SELECT * FROM runs ORDER BY id"):
    d = dict(r)
    d["metrics"] = json.loads(d.pop("metrics_json"))
    d.pop("config_key")
    pm = pass_map(conn, r["id"])
    d["passes"], d["n_questions"] = sum(pm.values()), len(pm)
    runs.append(d)

dev_questions = []
for r in conn.execute("SELECT id FROM runs WHERE split = 'dev' ORDER BY id"):
    top_k, outcomes = load_outcomes(conn, r["id"])
    answers = {x["question_id"]: x["answer"] for x in conn.execute(
        "SELECT question_id, answer FROM question_results WHERE run_id = ?", (r["id"],))}
    for o in outcomes:
        dev_questions.append({
            "run_id": r["id"], "question_id": o.question_id, "type": o.type,
            "question": dev_items[o.question_id].question, "category": classify(o, top_k),
            "rank_needed": o.rank_needed, "answer": (answers.get(o.question_id) or "")[:600],
        })
assert not any(q["question_id"] in test_ids for q in dev_questions), "a test question leaked into the export"

steps = []
for s in conn.execute("SELECT * FROM optimizer_steps ORDER BY id"):
    d = dict(s)
    d["change"] = json.loads(d.pop("change_json"))
    steps.append(d)

evaluator = RetrievalEvaluator(docs)
dev_list = list(dev_items.values())
ranks, space = {}, []
for c in retrieval_space():
    key = (c.chunk_size, c.overlap, c.retriever)
    if key not in ranks:
        report = evaluator.evaluate(
            RAGConfig(chunk_size=c.chunk_size, overlap=c.overlap, top_k=4, retriever=c.retriever), dev_list)
        ranks[key] = [q.rank_needed for q in report.questions if q.type != "unanswerable"]
    covered = sum(1 for x in ranks[key] if x is not None and x <= c.top_k)
    space.append({"retriever": c.retriever, "chunk_size": c.chunk_size, "overlap": c.overlap,
                  "top_k": c.top_k, "covered": covered, "n_answerable": len(ranks[key]),
                  "prompt_tokens_estimate": estimated_prompt_tokens(c)})
on_front = set(pareto([(s["covered"], s["prompt_tokens_estimate"]) for s in space]))
for i, s in enumerate(space):
    s["pareto"] = i in on_front

lock_path = Path("experiments/test_split_lock.json")
export = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "note": "dev per-question rows only; the test split is exported as aggregates (the runs table).",
    "runs": runs, "dev_questions": dev_questions, "optimizer_steps": steps, "retrieval_space": space,
    "test_lock": json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(export, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB): {len(runs)} runs, {len(dev_questions)} dev question rows, "
      f"{len(steps)} optimizer steps, {len(space)} space points ({len(on_front)} on the frontier)")