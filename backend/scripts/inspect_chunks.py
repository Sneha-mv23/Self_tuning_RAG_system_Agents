from app.rag.chunker import chunk_corpus
from app.rag.loader import load_corpus

docs = load_corpus()
print(f"documents: {len(docs)}, total chars: {sum(len(d.text) for d in docs):,}")

for size, overlap in [(128, 16), (256, 32), (512, 64), (1024, 128)]:
    chunks = chunk_corpus(docs, size, overlap)
    avg = sum(c.n_tokens for c in chunks) / len(chunks)
    print(f"chunk_size={size:<5} overlap={overlap:<4} chunks={len(chunks):<5} avg_tokens={avg:.0f}")