from collections import Counter

from app.evals.golden import load_golden
from app.evals.split import load_split, make_split, save_split, select
from app.rag.loader import load_corpus

items = load_golden(docs=load_corpus())
split = make_split(items, existing=load_split())
save_split(split)

for which in ("dev", "test"):
    chosen = select(items, split, which)
    print(f"{which}: {len(chosen)} questions  {dict(Counter(i.type for i in chosen))}")
    