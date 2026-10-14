import argparse

from app.evals.golden import load_golden
from app.evals.retrieval_compare import compare_ranks
from app.evals.retrieval_metrics import rank_needed
from app.evals.split import load_split, select
from app.rag.chunker import chunk_corpus
from app.rag.hybrid import HybridIndex
from app.rag.index import VectorIndex
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--chunk-size", type=int, default=256)
ap.add_argument("--overlap", type=int, default=32)
ap.add_argument("--bm25-weight", type=float, default=1.0)
args = ap.parse_args()

docs = load_corpus()
items = [i for i in select(load_golden(docs=docs), load_split(), "dev") if i.evidence]
dense = VectorIndex.build(chunk_corpus(docs, args.chunk_size, args.overlap))
hybrid = HybridIndex(dense, 1.0, args.bm25_weight)


def ranks(index) -> dict[str, int | None]:
    return {it.id: rank_needed(index.search(it.question, 20), it) for it in items}


def fmt(r: int | None) -> str:
    return "-" if r is None else str(r)


rd, rh = ranks(dense), ranks(hybrid)
kind = {it.id: it.type for it in items}

print(f"{len(items)} answerable dev questions, chunk {args.chunk_size}/{args.overlap}. "
      "'rank' = smallest top_k covering ALL gold evidence ('-' = not in the top 20).\n")
for k in (4, 6):
    groups = compare_ranks(rd, rh, k)
    print(f"TOP_K = {k}: " + ", ".join(f"{name} {len(ids)}" for name, ids in groups.items()))
    for name in ("fixed", "broken", "both_miss"):
        for qid in groups[name]:
            print(f"   {name:<9} {qid} [{kind[qid]}]  dense rank {fmt(rd[qid])} -> hybrid rank {fmt(rh[qid])}")
    print()