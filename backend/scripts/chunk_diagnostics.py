import argparse
from collections import Counter

from app.db.experiments import connect
from app.evals.chunk_diagnostics import MAX_SEQ, SPECIAL_TOKENS, chunk_cut_positions, span_status
from app.evals.failure_classifier import STAGE, classify, load_outcomes
from app.evals.golden import load_golden
from app.evals.split import load_split, select
from app.rag.chunker import chunk_corpus
from app.rag.embedder import _model
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", type=int, default=None,
                help="also show gold-span visibility for this run's retrieval failures")
args = ap.parse_args()

tok = _model().tokenizer


def offsets_for(text: str):
    try:
        return tok(text, add_special_tokens=False, return_offsets_mapping=True,
                   verbose=False)["offset_mapping"]
    except NotImplementedError:
        raise SystemExit("This tokenizer cannot report character offsets (it needs a 'fast' tokenizer).")


docs = load_corpus()
all_items = load_golden(docs=docs)
dev = [i for i in select(all_items, load_split(), "dev") if i.evidence]
spans = [(i.id, ev) for i in dev for ev in i.evidence]
print(f"the embedder reads only the first {MAX_SEQ - SPECIAL_TOKENS} word pieces of each chunk\n")

for size, overlap in [(128, 16), (256, 32), (512, 64)]:
    chunks = chunk_corpus(docs, size, overlap)
    lens = [len(offsets_for(c.text)) for c in chunks]
    cut = sum(1 for n in lens if n > MAX_SEQ - SPECIAL_TOKENS)
    cuts = chunk_cut_positions(chunks, offsets_for)
    status = Counter(span_status(chunks, cuts, ev.doc_id, ev.start, ev.end) for _, ev in spans)
    print(f"chunk_size={size:<4} overlap={overlap:<3} {len(chunks)} chunks | "
          f"mean tokens: tiktoken {sum(c.n_tokens for c in chunks) / len(chunks):.0f}, "
          f"word pieces {sum(lens) / len(lens):.0f} | cut by the embedder: {cut} ({cut / len(chunks):.0%})")
    print(f"   {len(spans)} gold spans: "
          + ", ".join(f"{k} {status.get(k, 0)}" for k in ("full", "partial", "none", "split")))

if args.run_id:
    conn = connect()
    run = conn.execute("SELECT chunk_size, overlap FROM runs WHERE id = ?", (args.run_id,)).fetchone()
    if run is None:
        raise SystemExit(f"no run with id {args.run_id}")
    top_k, outcomes = load_outcomes(conn, args.run_id)
    chunks = chunk_corpus(docs, run["chunk_size"], run["overlap"])
    cuts = chunk_cut_positions(chunks, offsets_for)
    items = {i.id: i for i in all_items}
    print(f"\nRETRIEVAL FAILURES IN RUN {args.run_id} "
          f"({run['chunk_size']}/{run['overlap']}): can the embedder see the gold span?")
    for o in outcomes:
        cat = classify(o, top_k)
        if STAGE[cat] != "retrieval":
            continue
        st = [span_status(chunks, cuts, ev.doc_id, ev.start, ev.end) for ev in items[o.question_id].evidence]
        print(f"  {o.question_id:<5} {cat:<18} evidence ranks={o.evidence_ranks}  span status={st}")