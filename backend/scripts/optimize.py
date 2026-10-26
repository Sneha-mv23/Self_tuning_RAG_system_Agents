import argparse

from app.db.experiments import connect
from app.db.optimizer_log import ensure_tables
from app.evals.golden import load_golden
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.optimizer.acceptance import MIN_GAIN
from app.optimizer.loop import optimize
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--start-run", type=int, required=True, help="run id of the baseline to improve")
ap.add_argument("--budget", type=int, default=250, help="max new API calls (generation + judging) this session")
ap.add_argument("--max-experiments", type=int, default=3)
ap.add_argument("--min-gain", type=int, default=MIN_GAIN)
ap.add_argument("--dry-run", action="store_true", help="diagnose, screen and estimate cost, but call no API")
args = ap.parse_args()

conn = connect()
ensure_tables(conn)
docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "dev")
optimize(conn, RetrievalEvaluator(docs), items, args.start_run, budget=args.budget,
         max_experiments=args.max_experiments, min_gain=args.min_gain, dry_run=args.dry_run)