import argparse
import json
import random
from pathlib import Path

from app.evals.golden import load_golden
from app.evals.judge import (
    CORRECT_PROMPT, FAITHFUL_PROMPT, BudgetReached, Judge,
    context_from_spans, is_pure_refusal, strip_citations,
)
from app.llm_limits import QuotaExhausted
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--max-calls", type=int, default=20)
ap.add_argument("--gen", default="experiments/runs/gemini/generation_dev_cs256_ov32_k4_strict_t0.0.json")
args = ap.parse_args()

docs = load_corpus()
docs_by_id = {d.doc_id: d for d in docs}
items = {i.id: i for i in load_golden(docs=docs)}
records = {r["id"]: r for r in json.loads(Path(args.gen).read_text(encoding="utf-8"))["records"]}
answered = [qid for qid, r in records.items()
            if r["error"] is None and not is_pure_refusal(r["answer"]) and items[qid].type != "unanswerable"]
print(f"{len(answered)} answered questions available for controls")

rng = random.Random(args.seed)
targets = rng.sample(answered, min(args.n, len(answered)))


def evidence_docs(qid: str) -> set[str]:
    return {e.doc_id for e in items[qid].evidence}


def partner(target: str) -> str:
    others = [q for q in answered if q != target and not (evidence_docs(q) & evidence_docs(target))]
    return rng.choice(others)


judge = Judge(max_calls=args.max_calls)
results = {"control_correct": [], "control_faithful": []}
try:
    for t in targets:
        p = partner(t)
        it, rec_t, rec_p = items[t], records[t], records[p]
        c_prompt = CORRECT_PROMPT.format(question=it.question, reference=it.reference_answer, answer=rec_p["answer"])
        f_prompt = FAITHFUL_PROMPT.format(context=context_from_spans(rec_t["used_chunks"], docs_by_id),
                                          answer=strip_citations(rec_p["answer"]))
        for kind, prompt in (("control_correct", c_prompt), ("control_faithful", f_prompt)):
            v = judge.ask(kind, prompt)
            results[kind].append((t, p, v.value, v.error))
            print(f"{kind:<16} question {t} <- answer of {p}: judge says "
                  f"{'ERROR ' + str(v.error)[:60] if v.error else ('yes' if v.value else 'no')}")
except (QuotaExhausted, BudgetReached) as e:
    print(f"\nSTOPPED: {str(e)[:200]}. Finished verdicts are cached, so rerunning continues.")

print()
for kind, rows in results.items():
    ok = [r for r in rows if r[3] is None]
    yes = [r for r in ok if r[2]]
    print(f"{kind}: judge said YES to {len(yes)} of {len(ok)} mismatched pairs (a working judge says yes to none)")
    for t, p, _, _ in yes:
        print(f"   check by eye: question {t} with the answer of {p}")