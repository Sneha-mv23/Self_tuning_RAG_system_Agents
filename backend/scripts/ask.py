import sys

from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig, RAGPipeline

question = " ".join(sys.argv[1:]) or "How do I declare a query parameter?"

print("Loading corpus and building index (first run downloads the embedding model)...")
pipe = RAGPipeline(load_corpus(), RAGConfig())
print(f"Index ready: {len(pipe.index.chunks)} chunks\n")

result = pipe.answer(question)
print(f"Q: {result.question}\n")
print(f"A: {result.answer}\n")
print(f"({result.latency_s:.1f}s, {result.prompt_tokens} prompt tokens)\n")
print("Sources:")
for r in result.retrieved:
    print(f"  [{r.rank}] score={r.score:.3f}  {r.chunk.doc_id}  chars {r.chunk.start}-{r.chunk.end}")