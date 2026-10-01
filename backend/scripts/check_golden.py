from collections import Counter

from app.evals.golden import load_golden
from app.rag.loader import load_corpus

docs = load_corpus()
texts = {d.doc_id: d.text for d in docs}
items = load_golden(docs=docs)

print(f"{len(items)} questions: {dict(Counter(i.type for i in items))}\n")
for item in items:
    print(f"{item.id} [{item.type}] {item.question}")
    for ev in item.evidence:
        found = texts[ev.doc_id][ev.start : ev.end]
        print(f"    {ev.doc_id}  chars {ev.start}-{ev.end}: {found!r}")