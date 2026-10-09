import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.db.experiments import connect, log_run
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
ap.add_argument("--final", action="store_true")
args = ap.parse_args()

if args.split == "test" and not args.final:
    raise SystemExit("The test split is reserved for the single final evaluation. Use --final only then.")

config = RAGConfig(chunk_size=args.chunk_size, overlap=args.overlap, top_k=args.top_k,
                   prompt=args.prompt, temperature=args.temperature)
name = (f"{args.split}_cs{args.chunk_size}_ov{args.overlap}_k{args.top_k}"
        f"_{args.prompt}_t{args.temperature}")
gen_path = Path(f"experiments/runs/generation_{name}.json")
judged_path = Path(f"experiments/runs/judged_{name}.json")
for p in (gen_path, judged_path):
    if not p.exists():
        raise SystemExit(f"missing {p}. Run eval_generation.py and eval_judge.py for this config first.")

generation = json.loads(gen_path.read_text(encoding="utf-8"))
judged = json.loads(judged_path.read_text(encoding="utf-8"))
if generation["config"] != asdict(config):
    raise SystemExit("the generation file's config does not match the arguments you passed")

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), args.split)
report = RetrievalEvaluator(docs).evaluate(config, items)

gen_ids = {r["id"] for r in generation["records"]}
judged_ids = {j["id"] for j in judged["judgments"]}
if gen_ids != {i.id for i in items} or judged_ids != gen_ids:
    raise SystemExit("question IDs differ between the split, the generation file and the judged file. "
                     "Rerun the evaluation steps (a partial --limit run can cause this).")

conn = connect()
run_id = log_run(conn, args.split, asdict(config), report, generation, judged)
print(f"logged run {run_id}: {name}")