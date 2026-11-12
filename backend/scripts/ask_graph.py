import argparse
import json

from app.graphs.rag_graph import RAGService
from app.rag.pipeline import RETRIEVERS

ap = argparse.ArgumentParser()
ap.add_argument("question", nargs="*")
ap.add_argument("--top-k", type=int, default=4)
ap.add_argument("--retriever", choices=list(RETRIEVERS), default="dense")
ap.add_argument("--mermaid", action="store_true", help="print the graph as a Mermaid diagram and exit")
args = ap.parse_args()

service = RAGService()
if args.mermaid:
    print(service.graph.get_graph().draw_mermaid())
    raise SystemExit

question = " ".join(args.question) or "How do I declare a query parameter?"
out = service.ask(question, {"top_k": args.top_k, "retriever": args.retriever})
print(f"Q: {out['question']}\n\nA: {out['answer']}\n")
print(f"prompt tokens: {out['prompt_tokens']} | timings: {json.dumps(out['timings'])}")
for s in out["sources"]:
    print(f"  [{s['rank']}] {s['score']:.3f}  {s['doc_id']}  chars {s['start']}-{s['end']}")