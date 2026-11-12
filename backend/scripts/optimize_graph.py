import argparse

from app.db.experiments import connect
from app.db.optimizer_log import ensure_tables
from app.evals.generation_run import GenerationCache
from app.evals.golden import load_golden
from app.evals.judge import Judge
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.graphs.optimizer_graph import (
    OptimizerContext, build_optimizer_graph, initial_state, make_checkpointer, thread_config)
from app.optimizer.acceptance import MIN_GAIN
from app.optimizer.experiment import pass_map
from app.optimizer.loop import CALLS_PER_NEW_PROMPT, config_of
from app.rag.loader import load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--start-run", type=int, help="run id of the baseline to improve")
ap.add_argument("--thread", help="resume an earlier session by its thread id (needs the SQLite checkpointer)")
ap.add_argument("--budget", type=int, default=250)
ap.add_argument("--max-experiments", type=int, default=3)
ap.add_argument("--min-gain", type=int, default=MIN_GAIN)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--approve", action="store_true", help="pause before each experiment and ask y/n")
ap.add_argument("--mermaid", action="store_true", help="print the graph as a Mermaid diagram and exit")
args = ap.parse_args()

if args.mermaid:
    print(build_optimizer_graph(OptimizerContext(None, None, [], None, None),
                                approve_before_run=True).get_graph().draw_mermaid())
    raise SystemExit
if not args.start_run and not args.thread:
    ap.error("give --start-run for a new session, or --thread to resume one")

conn = connect()
ensure_tables(conn)
docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "dev")
ctx = OptimizerContext(conn, RetrievalEvaluator(docs), items, GenerationCache(), Judge(), dry_run=args.dry_run)
saver, kind = make_checkpointer()
graph = build_optimizer_graph(ctx, saver, approve_before_run=args.approve)

if args.thread:
    thread_id, first_input = args.thread, None
else:
    first_input = initial_state(conn, args.start_run, args.budget, args.max_experiments, args.min_gain)
    thread_id = first_input["session"]
config = thread_config(thread_id, args.max_experiments)
if args.thread and not graph.get_state(config).values:
    raise SystemExit("no checkpoint found for that thread. The memory checkpointer forgets between runs.")
print(f"thread {thread_id} | checkpoints: {kind} | approval pauses: {'on' if args.approve else 'off'}")

graph.invoke(first_input, config)
while args.approve:
    snap = graph.get_state(config)
    if not snap.next:
        break
    cand = snap.values["candidate"]
    print(f"\nPROPOSED: {cand['lever']} {cand['change']} | screen gain {cand['screen_gain']} | "
          f"about {int(cand['est'] * CALLS_PER_NEW_PROMPT)} API calls")
    if input("Run this experiment? [y/n]: ").strip().lower() == "y":
        graph.invoke(None, config)
    else:
        graph.update_state(config, {"rejected": True}, as_node="propose")
        graph.invoke(None, config)

final = graph.get_state(config).values
start_pass, final_pass = pass_map(conn, final["start_run"]), pass_map(conn, final["current_run"])
print("\n" + "=" * 70)
print(f"STOPPED: {final.get('stop') or 'paused before an experiment'}")
print(f"start run {final['start_run']}: {sum(start_pass.values())}/{len(start_pass)} pass  ->  "
      f"final run {final['current_run']}: {sum(final_pass.values())}/{len(final_pass)} pass")
print(f"API calls used this session: {final['spent']} of {final['budget']}")
print(f"final config: {config_of(conn, final['current_run'])[0]}")