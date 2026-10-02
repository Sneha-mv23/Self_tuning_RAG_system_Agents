import argparse
import statistics

from app.evals.golden import load_golden
from app.evals.retrieval_metrics import first_hit_rank, rank_needed
from app.evals.split import load_split, select
from app.rag.chunker import chunk_corpus
from app.rag.index import VectorIndex
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--chunk-size", type=int, default=256)
ap.add_argument("--overlap", type=int, default=32)
ap.add_argument("--k", type=int, default=4, help="top_k used to define a failure")
args = ap.parse_args()

SEARCH_K = 20

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "dev")
if not items:
    raise SystemExit("No dev questions found. Run backend/scripts/make_split.py first.")

chunks = chunk_corpus(docs, args.chunk_size, args.overlap)
index = VectorIndex.build(chunks)
print(f"chunk_size={args.chunk_size} overlap={args.overlap} -> {len(chunks)} chunks; "
      f"{len(items)} dev questions\n")

answerable = [i for i in items if i.type != "unanswerable"]
unanswerable = [i for i in items if i.type == "unanswerable"]

rows = []
for it in answerable:
    retrieved = index.search(it.question, SEARCH_K)
    rows.append((it, retrieved, rank_needed(retrieved, it)))

print("ANSWERABLE QUESTIONS (retrieval only, no LLM)")
for k in (1, 3, 4, 5, 10, 20):
    hit = sum(1 for _, _, r in rows if r is not None and r <= k)
    print(f"  recall@{k:<2} = {hit / len(rows):.2f}  ({hit}/{len(rows)})")
mrr = statistics.mean(1 / r if r else 0.0 for _, _, r in rows)
print(f"  MRR       = {mrr:.3f}\n")

print("recall@%d by question type" % args.k)
for qtype in sorted({i.type for i in answerable}):
    group = [r for it, _, r in rows if it.type == qtype]
    hit = sum(1 for r in group if r is not None and r <= args.k)
    print(f"  {qtype:<12} {hit}/{len(group)}")
print()

misses = [(it, ret, r) for it, ret, r in rows if r is None or r > args.k]
print(f"QUESTIONS NOT COVERED BY TOP {args.k}: {len(misses)}")
for it, ret, r in misses:
    status = "not found in top 20" if r is None else f"needs top_k={r}"
    print(f"\n{it.id} [{it.type}] {status}")
    print(f"  Q: {it.question}")
    for ev in it.evidence:
        rk = first_hit_rank(ret, ev)
        print(f"  gold: {ev.doc_id} chars {ev.start}-{ev.end}  -> rank {rk}")
    print("  retrieved: " + ", ".join(
        f"{r.chunk.doc_id.split('/')[-1]}@{r.chunk.start}" for r in ret[:4]))

print("\nTOP-1 SIMILARITY SCORE")
top_ans = [index.search(it.question, 1)[0].score for it in answerable]
top_un = [index.search(it.question, 1)[0].score for it in unanswerable]
print(f"  answerable   mean={statistics.mean(top_ans):.3f} min={min(top_ans):.3f} max={max(top_ans):.3f}")
if top_un:
    print(f"  unanswerable mean={statistics.mean(top_un):.3f} min={min(top_un):.3f} max={max(top_un):.3f}")