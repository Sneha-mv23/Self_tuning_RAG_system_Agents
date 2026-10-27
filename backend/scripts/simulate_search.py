import argparse
import random
from statistics import mean, median

from app.evals.golden import load_golden
from app.evals.retrieval_eval import RetrievalEvaluator
from app.evals.split import load_split, select
from app.optimizer.simulate import Landscape, climb, random_search
from app.optimizer.space import retrieval_space
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=1000)
ap.add_argument("--max-experiments", type=int, default=25)
args = ap.parse_args()
N = args.max_experiments

docs = load_corpus()
items = select(load_golden(docs=docs), load_split(), "dev")
evaluator = RetrievalEvaluator(docs)
space = retrieval_space()

ranks: dict[tuple, list] = {}
for c in space:
    key = (c.chunk_size, c.overlap, c.retriever)
    if key not in ranks:
        report = evaluator.evaluate(
            RAGConfig(chunk_size=c.chunk_size, overlap=c.overlap, top_k=4, retriever=c.retriever), items)
        ranks[key] = [(q.id, q.rank_needed, list(q.evidence_ranks))
                      for q in report.questions if q.type != "unanswerable"]
land = Landscape(ranks)

start = RAGConfig()
target = max(land.coverage(c) for c in space)
n_best = sum(1 for c in space if land.coverage(c) == target)
if (land.coverage(start), land.coverage(RAGConfig(top_k=6)), target) != (36, 41, 41):
    raise SystemExit("landscape does not reproduce the mapped space (36, 41, 41). "
                     "Run map_space.py and compare before trusting this simulation.")
print(f"{len(space)} configs | baseline covers {land.coverage(start)} | best {target} "
      f"({n_best} configs reach it) | {args.seeds} seeds\n")

checkpoints = [n for n in (1, 2, 3, 5, 10, 25) if n <= N]


def first_reach(curve: list[int]):
    for i, v in enumerate(curve, 1):
        if v >= target:
            return i
    return None


def report_row(name: str, curves: list[list[int]], wasted: list[int] | None = None) -> None:
    reached = [r for r in (first_reach(c) for c in curves) if r is not None]
    cells = " ".join(f"{mean(c[n - 1] for c in curves):>6.1f}" for n in checkpoints)
    med = f"{median(reached):>7.1f}" if reached else "    n/a"
    share = f"{len(reached) / len(curves):>6.0%}"
    w = f"{mean(wasted):>7.1f}" if wasted is not None else "      -"
    print(f"{name:<34} {cells}  {med} {share} {w}")


print(f"{'strategy':<34} " + " ".join(f"{'n=' + str(n):>6}" for n in checkpoints)
      + "  median  reach  wasted")
rand = [random_search(land, start, space, random.Random(s), N) for s in range(args.seeds)]
blind = [climb(land, start, random.Random(s), N, guided=False) for s in range(args.seeds)]
guided = climb(land, start, random.Random(0), N, guided=True)
report_row("random search, no screening", rand)
report_row("blind hill-climb, free screen", [b["curve"] for b in blind], [b["wasted"] for b in blind])
report_row("diagnosis-guided, free screen", [guided["curve"]], [guided["wasted"]])
print(f"\ncolumns n=...: mean best coverage after n experiments | median: experiments until the max is run "
      f"(only runs that get there) | reach: share of runs that get there within {N} | wasted: experiments "
      f"that gained nothing")
print(f"random search, exact expectation of experiments to first hit the max: "
      f"{(len(space) + 1) / (n_best + 1):.1f}")

print("\nguided path from the baseline:")
for lever, c, gain in guided["path"]:
    print(f"  {lever}: {c.retriever} cs/ov/k={c.chunk_size}/{c.overlap}/{c.top_k}  (+{gain} covered)")
stuck = sum(1 for b in blind if first_reach(b["curve"]) is None)
print(f"\nblind hill-climb ended below the max in {stuck} of {args.seeds} runs")