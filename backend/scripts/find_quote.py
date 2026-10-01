import argparse
import json
import re
import sys

from app.rag.loader import load_corpus

sys.stdout.reconfigure(encoding="utf-8")  # avoids Windows console errors on special characters

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def cmd_list(docs):
    for d in docs:
        print(f"{len(d.text):>7} chars  {d.doc_id}")
    print(f"\n{len(docs)} documents")


def cmd_search(docs, terms, limit):
    terms = [t.lower() for t in terms]
    shown = 0
    for d in docs:
        for line in d.text.split("\n"):
            if not line.strip():
                continue
            for sent in _SENT_SPLIT.split(line):
                s = sent.strip()
                if len(s) < 25 or not all(t in s.lower() for t in terms):
                    continue
                n = d.text.count(s)
                status = "unique" if n == 1 else f"APPEARS {n}x, too ambiguous"
                print(f"{d.doc_id}  [{status}]")
                print(f"    {s}")
                if n == 1:
                    print("    " + json.dumps({"doc_id": d.doc_id, "quote": s}))
                print()
                shown += 1
                if shown >= limit:
                    print(f"(stopped at {limit} results; add more terms or use --limit)")
                    return
    if shown == 0:
        print("No matches. Try fewer or different words.")


def cmd_show(docs, doc_id):
    doc = next((d for d in docs if d.doc_id == doc_id), None)
    if doc is None:
        close = [d.doc_id for d in docs if doc_id.lower() in d.doc_id.lower()]
        print(f"Unknown doc_id '{doc_id}'. Close matches: {close or 'none'}")
        return
    for i, line in enumerate(doc.text.split("\n"), start=1):
        print(f"{i:>4}  {line}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Find exact, unique quotes in the corpus.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list all documents")
    s = sub.add_parser("search", help="find sentences containing ALL given terms")
    s.add_argument("terms", nargs="+")
    s.add_argument("--limit", type=int, default=15)
    w = sub.add_parser("show", help="print one document with line numbers")
    w.add_argument("doc_id")
    args = p.parse_args()

    docs = load_corpus()
    if args.cmd == "list":
        cmd_list(docs)
    elif args.cmd == "search":
        cmd_search(docs, args.terms, args.limit)
    else:
        cmd_show(docs, args.doc_id)