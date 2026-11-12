import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.db.optimizer_log import log_step
from app.evals.failure_classifier import classify, load_outcomes
from app.evals.generation_run import GenerationAborted, count_uncached
from app.evals.judge import BudgetReached
from app.llm_limits import QuotaExhausted
from app.optimizer.acceptance import MIN_GAIN
from app.optimizer.acceptance import decide as decide_change
from app.optimizer.candidates import config_diff, flatten, rank_candidates, screen
from app.optimizer.experiment import pass_map, run_experiment
from app.optimizer.levers import plan
from app.optimizer.loop import CALLS_PER_NEW_PROMPT, config_of
from app.rag.pipeline import RAGConfig


class OptState(TypedDict, total=False):
    session: str
    start_run: int
    current_run: int
    budget: int
    max_experiments: int
    min_gain: int
    spent: int
    experiments: int
    step: int
    tried: list  # RAGConfig fields as dicts
    candidate: dict | None  # the proposal waiting to run: lever, config, change, screen_gain, est
    candidate_run: int | None
    last_used: int
    rejected: bool
    stop: str  # empty while the loop is running


@dataclass
class OptimizerContext:
    """Things that cannot live in the checkpointed state: database, models, caches."""
    conn: Any
    evaluator: Any
    items: list
    gen_cache: Any
    judge: Any
    dry_run: bool = False
    say: Callable[[str], None] = print


def check_start(conn, run_id: int) -> None:
    _, row = config_of(conn, run_id)
    if row["split"] != "dev":
        raise ValueError("the optimizer only works on the dev split")
    if (row["generator_model"], row["judge_model"]) != (settings.generator_model, settings.judge_model):
        raise ValueError(
            f"run {run_id} used {row['generator_model']} / {row['judge_model']}, but .env now points at "
            f"{settings.generator_model} / {settings.judge_model}. Runs from different models are not comparable.")


def initial_state(conn, start_run_id: int, budget: int = 250, max_experiments: int = 3,
                  min_gain: int = MIN_GAIN) -> dict:
    check_start(conn, start_run_id)
    cfg, _ = config_of(conn, start_run_id)
    return {"session": time.strftime("%Y%m%d-%H%M%S"), "start_run": start_run_id, "current_run": start_run_id,
            "budget": budget, "max_experiments": max_experiments, "min_gain": min_gain, "spent": 0,
            "experiments": 0, "step": 0, "tried": [asdict(cfg)], "candidate": None, "candidate_run": None,
            "last_used": 0, "rejected": False, "stop": ""}


def thread_config(thread_id: str, max_experiments: int) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 3 * max_experiments + 10}


def make_checkpointer(path: str | None = "experiments/optimizer_checkpoints.sqlite"):
    """(saver, kind). SQLite survives a restart; the memory saver only lasts for one process."""
    if path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False)), "sqlite"
        except ImportError:
            pass
    return MemorySaver(), "memory"


def build_optimizer_graph(ctx: OptimizerContext, checkpointer=None, approve_before_run: bool = False):
    def propose_node(state: OptState) -> dict:
        run_id = state["current_run"]
        cfg, _ = config_of(ctx.conn, run_id)
        top_k, outcomes = load_outcomes(ctx.conn, run_id)
        parent_pass = pass_map(ctx.conn, run_id)
        n_parent = sum(parent_pass.values())
        cats = Counter(classify(o, top_k) for o in outcomes)
        ctx.say(f"\nCURRENT RUN {run_id}: {n_parent}/{len(parent_pass)} pass | failures: "
                + ", ".join(f"{k} {v}" for k, v in cats.items() if k != "pass"))

        if state["experiments"] >= state["max_experiments"]:
            return {"candidate": None, "stop": "reached the experiment limit"}

        tried = {RAGConfig(**d) for d in state["tried"]}
        candidates = flatten(plan(outcomes, cfg), tried)
        if not candidates:
            return {"candidate": None, "stop": "no untried candidate matches the current failures"}
        to_try, rejected = rank_candidates(screen(ctx.evaluator, ctx.items, cfg, candidates))

        step = state["step"]
        for c in rejected:
            step += 1
            tried.add(c.config)
            reason = f"free retrieval screen: covers {c.screen_gain:+d} questions, so it is not worth an API run"
            if not ctx.dry_run:
                log_step(ctx.conn, state["session"], step, run_id, c.lever, config_diff(cfg, c.config),
                         "skipped_screen", reason, screen_gain=c.screen_gain, parent_passes=n_parent)
            ctx.say(f"  skip {c.lever} {config_diff(cfg, c.config)}: {reason}")
        if not to_try:
            return {"candidate": None, "stop": "no candidate passes the free screen",
                    "tried": [asdict(t) for t in tried], "step": step}

        rows = []
        for c in to_try:
            est = count_uncached(ctx.evaluator, c.config, ctx.items, ctx.gen_cache)
            rows.append((c, est))
            gain = "n/a" if c.screen_gain is None else f"{c.screen_gain:+d}"
            ctx.say(f"  candidate {c.lever} {config_diff(cfg, c.config)} | screen gain {gain} | "
                    f"{est} new generation calls, up to ~{int(est * CALLS_PER_NEW_PROMPT)} API calls with judging")
        if ctx.dry_run:
            return {"candidate": None, "stop": "dry run, nothing was executed"}

        budget_left = state["budget"] - state["spent"]
        chosen = None
        for c, est in rows:
            if est * CALLS_PER_NEW_PROMPT <= budget_left:
                chosen = (c, est)
                break
            step += 1
            tried.add(c.config)
            reason = f"estimated cost ~{int(est * CALLS_PER_NEW_PROMPT)} exceeds the {budget_left} calls left"
            log_step(ctx.conn, state["session"], step, run_id, c.lever, config_diff(cfg, c.config),
                     "skipped_budget", reason, screen_gain=c.screen_gain, parent_passes=n_parent)
            ctx.say(f"  skip {c.lever}: {reason}")
        if chosen is None:
            return {"candidate": None, "stop": "the budget left is too small for any candidate",
                    "tried": [asdict(t) for t in tried], "step": step}

        c, est = chosen
        tried.add(c.config)
        return {"candidate": {"lever": c.lever, "config": asdict(c.config), "change": config_diff(cfg, c.config),
                              "screen_gain": c.screen_gain, "est": est},
                "tried": [asdict(t) for t in tried], "step": step}

    def run_node(state: OptState) -> dict:
        if state.get("rejected"):
            return {"stop": "the proposed experiment was rejected by the user", "candidate": None}
        cand = state["candidate"]
        step = state["step"] + 1
        experiments = state["experiments"] + 1
        ctx.say(f"\nRUNNING experiment {experiments}: {cand['lever']} {cand['change']}")
        ctx.judge.max_calls = ctx.judge.calls_made + max(0, state["budget"] - state["spent"] - cand["est"])
        try:
            result = run_experiment(ctx.conn, ctx.evaluator, RAGConfig(**cand["config"]), ctx.items,
                                    ctx.gen_cache, ctx.judge)
        except (QuotaExhausted, BudgetReached, GenerationAborted) as e:
            reason = f"stopped early: {type(e).__name__}: {str(e)[:200]}"
            log_step(ctx.conn, state["session"], step, state["current_run"], cand["lever"], cand["change"],
                     "aborted", reason, screen_gain=cand["screen_gain"])
            ctx.say("  Finished answers and verdicts are cached, so rerunning continues without paying again.")
            return {"stop": reason, "step": step, "experiments": experiments, "candidate": None}
        used = result.new_generation_calls + result.new_judge_calls
        spent = state["spent"] + used
        if result.errors:
            shown = "; ".join(f"{q}: {result.detail[q][:90]}" for q in result.errors[:3])
            reason = f"errors on {len(result.errors)} questions ({shown}); run not logged"
            log_step(ctx.conn, state["session"], step, state["current_run"], cand["lever"], cand["change"],
                     "aborted", reason, screen_gain=cand["screen_gain"], api_calls=used)
            ctx.say(f"  ABORTED: {reason}")
            return {"stop": "an experiment had errors, so the loop stopped without judging it", "step": step,
                    "experiments": experiments, "spent": spent, "candidate": None}
        return {"candidate_run": result.run_id, "spent": spent, "step": step, "experiments": experiments,
                "last_used": used}

    def decide_node(state: OptState) -> dict:
        cand = state["candidate"]
        parent_pass = pass_map(ctx.conn, state["current_run"])
        cand_pass = pass_map(ctx.conn, state["candidate_run"])
        d = decide_change(parent_pass, cand_pass, state["min_gain"])
        reason = d.reason + f" | fixed {d.fixed} | broken {d.broken}"
        log_step(ctx.conn, state["session"], state["step"], state["current_run"], cand["lever"], cand["change"],
                 "accepted" if d.accepted else "rejected", reason, candidate_run_id=state["candidate_run"],
                 screen_gain=cand["screen_gain"], parent_passes=sum(parent_pass.values()),
                 candidate_passes=sum(cand_pass.values()), api_calls=state["last_used"])
        ctx.say(f"  {'ACCEPTED' if d.accepted else 'rejected'}: {reason}")
        ctx.say(f"  cost {state['last_used']} API calls ({state['spent']}/{state['budget']} used this session)")
        out = {"candidate": None, "candidate_run": None}
        if d.accepted:
            out["current_run"] = state["candidate_run"]
        return out

    graph = StateGraph(OptState)
    graph.add_node("propose", propose_node)
    graph.add_node("run", run_node)
    graph.add_node("decide", decide_node)
    graph.add_edge(START, "propose")
    graph.add_conditional_edges("propose", lambda s: "run" if s.get("candidate") else "end",
                                {"run": "run", "end": END})
    graph.add_conditional_edges("run", lambda s: "end" if s.get("stop") else "decide",
                                {"decide": "decide", "end": END})
    graph.add_edge("decide", "propose")
    return graph.compile(checkpointer=checkpointer, interrupt_before=["run"] if approve_before_run else None)