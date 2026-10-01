import sys

from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig, RAGPipeline

question = " ".join(sys.argv[1:]) or "How do I declare a path parameter?"
pipe = RAGPipeline(load_corpus(), RAGConfig())
for r in pipe.index.search(question, 8):
    c = r.chunk
    preview = c.text.strip().replace("\n", " ")[:150]
    print(f"[{r.rank}] {r.score:.3f} {c.doc_id} {c.start}-{c.end}\n     {preview}\n")