import argparse

from app.evals.golden import load_golden
from app.evals.retrieval_eval import KS, RetrievalEvaluator, save_report
from app.evals.split import load_split, select
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig

ap = argparse.ArgumentParser()
ap.add_argument("--chunk-size", type=int, default=256)
ap.add_argument("--overlap", type=int, default=32)
ap.add_argument("--top-k", type=int, default=4)
ap.add_argument("--split", choices=["dev", "test"], default="dev")
ap.add_argument("--final", action="store_true", help="required to touch the test split")
ap.add_argument("--out", default=None)
args = ap.parse_args()

if args.split == "test" and not args.final:
    raise SystemExit("The test split is reserved for the single final evaluation. Use --final only then.")

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), args.split)
config = RAGConfig(chunk_size=args.chunk_size, overlap=args.overlap, top_k=args.top_k)
report = RetrievalEvaluator(docs).evaluate(config, items)

print(f"{args.split}: {report.n_answerable} answerable + {report.n_unanswerable} unanswerable | "
      f"{report.n_chunks} chunks, mean {report.mean_chunk_tokens:.0f} tokens")
print(f"{'k':<4} {'recall':>7} {'evidence_recall':>16}")
for k in KS:
    print(f"{k:<4} {report.recall[k]:>7.2f} {report.evidence_recall[k]:>16.2f}")
print(f"MRR = {report.mrr:.3f}")
print(f"recall@{args.top_k} by type: " + ", ".join(f"{t}={v:.2f}" for t, v in report.recall_by_type.items()))

out = args.out or f"experiments/runs/retrieval_{args.split}_cs{args.chunk_size}_ov{args.overlap}_k{args.top_k}.json"
save_report(report, out)
print(f"saved {out}")