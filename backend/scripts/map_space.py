from app.evals.golden import load_golden
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.optimizer.levers import estimated_prompt_tokens
from app.optimizer.space import pareto, retrieval_space
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "dev")
evaluator = RetrievalEvaluator(docs)

space = retrieval_space()
ranks_cache: dict[tuple, list] = {}
rows = []  # (config, covered, tokens)
n_answerable = 0
for c in space:
    key = (c.chunk_size, c.overlap, c.retriever)
    if key not in ranks_cache:  # coverage at every top_k can be read from one retrieval pass
        report = evaluator.evaluate(RAGConfig(chunk_size=c.chunk_size, overlap=c.overlap, top_k=4,
                                              retriever=c.retriever), items)
        ranks_cache[key] = [q.rank_needed for q in report.questions if q.type != "unanswerable"]
    ranks = ranks_cache[key]
    n_answerable = len(ranks)
    covered = sum(1 for r in ranks if r is not None and r <= c.top_k)
    rows.append((c, covered, estimated_prompt_tokens(c)))


def label(c: RAGConfig) -> str:
    return f"{c.retriever:<6} cs/ov/k={c.chunk_size}/{c.overlap}/{c.top_k}"


cover = [r[1] for r in rows]
tokens = [r[2] for r in rows]
print(f"{len(rows)} valid retrieval configs, {n_answerable} answerable dev questions\n")
for name, cfg in (("baseline (run 3)", RAGConfig()), ("optimizer result (run 5)", RAGConfig(top_k=6))):
    cv = next(r[1] for r in rows if r[0] == cfg)
    better = sum(1 for v in cover if v > cv)
    print(f"{name:<24} {label(cfg)}  covers {cv}/{n_answerable}  ~{estimated_prompt_tokens(cfg)} prompt tokens"
          f"  ({better} configs cover more)")
best = max(cover)
print(f"\nbest coverage anywhere in the space: {best}/{n_answerable} "
      f"({cover.count(best)} configs reach it)\n")

print("TOP 10 by coverage, then fewest prompt tokens:")
for i in sorted(range(len(rows)), key=lambda i: (-cover[i], tokens[i]))[:10]:
    print(f"  {label(rows[i][0])}  covers {cover[i]}  ~{tokens[i]} tokens")

print("\nPARETO FRONTIER (no config covers more with fewer prompt tokens):")
for i in sorted(pareto(list(zip(cover, tokens))), key=lambda i: (tokens[i], -cover[i])):
    print(f"  {label(rows[i][0])}  covers {cover[i]}  ~{tokens[i]} tokens")

print("\ncoverage is a free proxy on the dev set: it ignores whether the model answers correctly, and the "
      "dev questions are the ones this search was tuned on.")