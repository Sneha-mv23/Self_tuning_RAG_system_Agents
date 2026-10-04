import argparse

from app.evals.generation_run import GenerationCache, run_generation, save_generation, summarize_generation
from app.evals.golden import load_golden
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.rag.generator import PROMPTS
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig

ap = argparse.ArgumentParser()
ap.add_argument("--chunk-size", type=int, default=256)
ap.add_argument("--overlap", type=int, default=32)
ap.add_argument("--top-k", type=int, default=4)
ap.add_argument("--prompt", choices=list(PROMPTS), default="strict")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--split", choices=["dev", "test"], default="dev")
ap.add_argument("--final", action="store_true", help="required to touch the test split")
ap.add_argument("--limit", type=int, default=None, help="only the first N questions (for a quick test)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

if args.split == "test" and not args.final:
    raise SystemExit("The test split is reserved for the single final evaluation. Use --final only then.")

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), args.split)
if args.limit:
    items = items[: args.limit]

config = RAGConfig(chunk_size=args.chunk_size, overlap=args.overlap, top_k=args.top_k,
                   prompt=args.prompt, temperature=args.temperature)
cache = GenerationCache()
records = run_generation(RetrievalEvaluator(docs), config, items, cache=cache)
cache.close()

s = summarize_generation(records)
print(f"\n{s['n']} questions | errors={s['errors']} | cache_hits={s['cache_hits']} | over_context={s['over_context']}")
print(f"latency: mean {s['mean_latency_s']:.1f}s, p95 {s['p95_latency_s']:.1f}s | "
      f"mean prompt tokens {s['mean_prompt_tokens']:.0f}")
print(f"refusal on unanswerable (rough): {s['refusal_rate_unanswerable']:.2f} | "
      f"false refusal on answerable (rough): {s['false_refusal_rate_answerable']:.2f}")

out = args.out or (f"experiments/runs/generation_{args.split}_cs{args.chunk_size}_ov{args.overlap}"
                   f"_k{args.top_k}_{args.prompt}_t{args.temperature}.json")
save_generation(records, config, out)
print(f"saved {out}")