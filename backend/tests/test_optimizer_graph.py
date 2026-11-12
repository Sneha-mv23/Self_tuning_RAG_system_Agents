import types
from dataclasses import asdict

import pytest
from langgraph.checkpoint.memory import MemorySaver

import app.graphs.optimizer_graph as og
from app.config import settings
from app.optimizer.candidates import Candidate
from app.optimizer.experiment import ExperimentResult
from app.rag.pipeline import RAGConfig

PASSES = {1: {"a": True, "b": False, "c": False, "d": True}, 2: {"a": True, "b": True, "c": True, "d": True}}
CONFIGS = {1: RAGConfig(top_k=4), 2: RAGConfig(top_k=6)}


@pytest.fixture
def world(monkeypatch):
    log, runs = [], []
    monkeypatch.setattr(og, "config_of", lambda conn, run_id: (CONFIGS[run_id], {}))
    monkeypatch.setattr(og, "load_outcomes", lambda conn, run_id: (CONFIGS[run_id].top_k, []))
    monkeypatch.setattr(og, "pass_map", lambda conn, run_id: PASSES[run_id])
    monkeypatch.setattr(og, "plan", lambda outcomes, cfg: [])
    monkeypatch.setattr(og, "flatten", lambda proposals, tried: [
        x for x in [("top_k", RAGConfig(top_k=6))] if x[1] not in tried])
    monkeypatch.setattr(og, "screen", lambda ev, items, cfg, cands: [
        Candidate(lever, c, i, 3) for i, (lever, c) in enumerate(cands)])
    monkeypatch.setattr(og, "count_uncached", lambda *a, **k: 10)

    def fake_run(conn, evaluator, config, items, cache, judge, split="dev", replicate=0):
        runs.append(config)
        return ExperimentResult(run_id=2, errors=[], new_generation_calls=10, new_judge_calls=20)

    monkeypatch.setattr(og, "run_experiment", fake_run)
    monkeypatch.setattr(og, "log_step", lambda conn, session, step, parent, lever, change, decision, reason, **kw:
                        log.append((step, lever, decision)))
    ctx = og.OptimizerContext(conn=None, evaluator=None, items=[], gen_cache=None,
                              judge=types.SimpleNamespace(calls_made=0, max_calls=None), say=lambda *_: None)
    return ctx, log, runs


def start_state(**over):
    state = {"session": "t", "start_run": 1, "current_run": 1, "budget": 250, "max_experiments": 3,
             "min_gain": 2, "spent": 0, "experiments": 0, "step": 0, "tried": [asdict(CONFIGS[1])],
             "candidate": None, "candidate_run": None, "last_used": 0, "rejected": False, "stop": ""}
    state.update(over)
    return state


def cfg(thread):
    return {"configurable": {"thread_id": thread}, "recursion_limit": 25}


def test_runs_to_convergence(world):
    ctx, log, runs = world
    final = og.build_optimizer_graph(ctx, MemorySaver()).invoke(start_state(), cfg("a"))
    assert final["current_run"] == 2 and final["experiments"] == 1 and final["spent"] == 30
    assert final["stop"] == "no untried candidate matches the current failures"
    assert [d for _, _, d in log] == ["accepted"] and len(runs) == 1


def test_pauses_before_the_experiment_and_resumes(world):
    ctx, log, runs = world
    graph = og.build_optimizer_graph(ctx, MemorySaver(), approve_before_run=True)
    graph.invoke(start_state(), cfg("b"))
    snap = graph.get_state(cfg("b"))
    assert snap.next == ("run",) and runs == []
    assert snap.values["candidate"]["lever"] == "top_k"
    graph.invoke(None, cfg("b"))
    assert len(runs) == 1 and graph.get_state(cfg("b")).values["current_run"] == 2


def test_rejection_skips_the_experiment(world):
    ctx, log, runs = world
    graph = og.build_optimizer_graph(ctx, MemorySaver(), approve_before_run=True)
    graph.invoke(start_state(), cfg("c"))
    graph.update_state(cfg("c"), {"rejected": True}, as_node="propose")
    graph.invoke(None, cfg("c"))
    final = graph.get_state(cfg("c")).values
    assert runs == [] and "rejected by the user" in final["stop"] and final["current_run"] == 1


def test_errors_abort_without_deciding(world, monkeypatch):
    ctx, log, runs = world
    monkeypatch.setattr(og, "run_experiment",
                        lambda *a, **k: ExperimentResult(None, ["q1"], 5, 5, {"q1": "boom"}))
    final = og.build_optimizer_graph(ctx, MemorySaver()).invoke(start_state(), cfg("d"))
    assert [d for _, _, d in log] == ["aborted"]
    assert "errors" in final["stop"] and final["current_run"] == 1 and final["spent"] == 10


def test_budget_too_small_skips_the_candidate(world):
    ctx, log, runs = world
    final = og.build_optimizer_graph(ctx, MemorySaver()).invoke(start_state(budget=10), cfg("e"))
    assert [d for _, _, d in log] == ["skipped_budget"] and runs == []
    assert final["stop"] == "the budget left is too small for any candidate"


def test_dry_run_logs_and_executes_nothing(world):
    ctx, log, runs = world
    ctx.dry_run = True
    final = og.build_optimizer_graph(ctx, MemorySaver()).invoke(start_state(), cfg("f"))
    assert log == [] and runs == [] and final["stop"].startswith("dry run")


def test_check_start(monkeypatch):
    row = {"split": "dev", "generator_model": settings.generator_model, "judge_model": settings.judge_model}
    monkeypatch.setattr(og, "config_of", lambda conn, rid: (RAGConfig(), row))
    og.check_start(None, 1)
    monkeypatch.setattr(og, "config_of", lambda conn, rid: (RAGConfig(), {**row, "generator_model": "other"}))
    with pytest.raises(ValueError):
        og.check_start(None, 1)