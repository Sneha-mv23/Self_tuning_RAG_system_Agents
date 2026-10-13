import argparse

from app.evals.golden import load_golden
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.optimizer.levers import estimated_prompt_tokens
from app.rag.chunker import chunk_corpus
from app.rag.hybrid import HybridIndex
from app.rag.index import VectorIndex
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig

ap = argparse.ArgumentParser()
ap.add_argument("--overlap", type=int, default=32)
ap.add_argument("--bm25-weight", type=float, default=1.0)
args = ap.parse_args()


class SwitchableEvaluator(RetrievalEvaluator):
    def __init__(self, docs, retriever: str):
        super().__init__(docs)
        self.retriever = retriever

    def index_for(self, chunk_size: int, overlap: int):
        key = (chunk_size, overlap)
        if key not in self._indexes:
            dense = VectorIndex.build(chunk_corpus(self.docs, chunk_size, overlap))
            self._indexes[key] = dense if self.retriever == "dense" else HybridIndex(dense, 1.0, args.bm25_weight)
        return self._indexes[key]


docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "dev")
KS = (4, 5, 6, 8)

print(f"{'chunk':>5}  {'retriever':<9} " + " ".join(f"{'r@' + str(k):>5}" for k in KS)
      + f" {'MRR':>6} {'multi@4':>8}")
for size in (128, 192, 256):
    for retriever in ("dense", "hybrid"):
        report = SwitchableEvaluator(docs, retriever).evaluate(
            RAGConfig(chunk_size=size, overlap=args.overlap, top_k=4), items)
        answerable = [q for q in report.questions if q.type != "unanswerable"]
        recall = {k: sum(1 for q in answerable if q.rank_needed is not None and q.rank_needed <= k) / len(answerable)
                  for k in KS}
        print(f"{size:>5}  {retriever:<9} " + " ".join(f"{recall[k]:>5.2f}" for k in KS)
              + f" {report.mrr:>6.3f} {report.recall_by_type.get('multi_hop', 0.0):>8.2f}")
    tokens = {k: estimated_prompt_tokens(RAGConfig(chunk_size=size, top_k=k)) for k in (4, 6, 8)}
    print(f"       approx prompt tokens at top_k 4/6/8: {tokens[4]}/{tokens[6]}/{tokens[8]}")

print(f"\n{len(answerable)} answerable dev questions: one question is {1 / len(answerable):.1%}. "
      f"Overlap {args.overlap}, BM25 weight {args.bm25_weight}.")