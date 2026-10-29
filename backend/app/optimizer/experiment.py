import time
from dataclasses import asdict, dataclass, field

from app.config import settings
from app.db.experiments import log_run
from app.evals.failure_classifier import classify, load_outcomes
from app.evals.generation_run import run_generation, summarize_generation
from app.evals.judge import judge_record, summarize_judgments

RETRY_PAUSE_S = 20


def pass_map(conn, run_id: int) -> dict[str, bool]:
    top_k, outcomes = load_outcomes(conn, run_id)
    return {o.question_id: classify(o, top_k) == "pass" for o in outcomes}


@dataclass
class ExperimentResult:
    run_id: int | None
    errors: list[str]
    new_generation_calls: int
    new_judge_calls: int
    detail: dict[str, str] = field(default_factory=dict)  # question id -> error message


def _attempt(evaluator, config, items, gen_cache, judge, replicate: int = 0):
    records = run_generation(evaluator, config, items, cache=gen_cache, verbose=False, replicate=replicate)
    docs_by_id = {d.doc_id: d for d in evaluator.docs}
    by_id = {i.id: i for i in items}
    gen_dicts = [asdict(r) for r in records]
    judgments = [judge_record(judge, d, by_id[d["id"]], docs_by_id) for d in gen_dicts]
    detail = {r.id: r.error for r in records if r.error}
    detail.update({j.id: j.error for j in judgments if j.error and j.id not in detail})
    new_gen = sum(1 for r in records if not r.cache_hit and r.error is None)
    return records, gen_dicts, judgments, detail, new_gen


def run_experiment(conn, evaluator, config, items, gen_cache, judge, split: str = "dev",
                   replicate: int = 0) -> ExperimentResult:
    """Retrieval, generation and judging for one config, logged like any other run.
    Failed questions get one retry (everything finished is cached, so the retry is cheap).
    A run that still has errors is NOT logged, so a partial result can never be compared."""
    report = evaluator.evaluate(config, items)
    judge_before = judge.calls_made
    new_gen = 0
    for attempt in range(2):
        if attempt:
            time.sleep(RETRY_PAUSE_S)
        records, gen_dicts, judgments, detail, n = _attempt(evaluator, config, items, gen_cache, judge, replicate)
        new_gen += n
        if not detail:
            break
    new_judge = judge.calls_made - judge_before
    if detail:
        return ExperimentResult(None, sorted(detail), new_gen, new_judge, detail)

    generation = {"config": asdict(config), "model": settings.generator_model,
                  "summary": summarize_generation(records), "records": gen_dicts}
    judged = {"judge_model": settings.judge_model, "summary": summarize_judgments(judgments),
              "judgments": [asdict(j) for j in judgments]}
    run_id = log_run(conn, split, asdict(config), report, generation, judged, replicate=replicate)
    return ExperimentResult(run_id, [], new_gen, new_judge)