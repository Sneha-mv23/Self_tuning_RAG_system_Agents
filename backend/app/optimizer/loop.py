import time
from collections import Counter

from app.config import settings
from app.db.optimizer_log import log_step
from app.evals.failure_classifier import classify, load_outcomes
from app.evals.generation_run import GenerationAborted, GenerationCache, count_uncached
from app.evals.judge import BudgetReached, Judge
from app.llm_limits import QuotaExhausted
from app.optimizer.acceptance import MIN_GAIN, decide
from app.optimizer.candidates import config_diff, flatten, rank_candidates, screen
from app.optimizer.experiment import pass_map, run_experiment
from app.optimizer.levers import plan
from app.rag.pipeline import RAGConfig

CALLS_PER_NEW_PROMPT = 2.5  # one generation call plus 1-2 judge calls; the judge cap below enforces the budget


def config_of(conn, run_id: int):
    r = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if r is None:
        raise ValueError(f"no run with id {run_id}")
    cfg = RAGConfig(chunk_size=r["chunk_size"], overlap=r["overlap"], top_k=r["top_k"],
                    prompt=r["prompt"], temperature=r["temperature"], retriever=r["retriever"])
    return cfg, r


def optimize(conn, evaluator, items, start_run_id: int, budget: int = 250, max_experiments: int = 3,
             min_gain: int = MIN_GAIN, dry_run: bool = False, say=print) -> dict:
    cfg, row = config_of(conn, start_run_id)
    if row["split"] != "dev":
        raise ValueError("the optimizer only works on the dev split")
    if (row["generator_model"], row["judge_model"]) != (settings.generator_model, settings.judge_model):
        raise ValueError(
            f"run {start_run_id} used {row['generator_model']} / {row['judge_model']}, but .env now points at "
            f"{settings.generator_model} / {settings.judge_model}. Runs from different models are not comparable.")

    session = time.strftime("%Y%m%d-%H%M%S")
    gen_cache, judge = GenerationCache(), Judge()
    start_pass = pass_map(conn, start_run_id)
    current_id, tried = start_run_id, {cfg}
    spent = experiments = step = 0
    stop = "reached the experiment limit"
    running = None  # (candidate, parent run id) while an experiment is in flight

    try:
        while experiments < max_experiments:
            top_k, outcomes = load_outcomes(conn, current_id)
            parent_pass = pass_map(conn, current_id)
            n_parent = sum(parent_pass.values())
            cats = Counter(classify(o, top_k) for o in outcomes)
            say(f"\nCURRENT RUN {current_id}: {n_parent}/{len(parent_pass)} pass | failures: "
                + ", ".join(f"{k} {v}" for k, v in cats.items() if k != "pass"))

            candidates = flatten(plan(outcomes, cfg), tried)
            if not candidates:
                stop = "no untried candidate matches the current failures"
                break
            to_try, rejected = rank_candidates(screen(evaluator, items, cfg, candidates))

            for c in rejected:
                step += 1
                tried.add(c.config)
                reason = f"free retrieval screen: covers {c.screen_gain:+d} questions, so it is not worth an API run"
                if not dry_run:
                    log_step(conn, session, step, current_id, c.lever, config_diff(cfg, c.config),
                             "skipped_screen", reason, screen_gain=c.screen_gain, parent_passes=n_parent)
                say(f"  skip {c.lever} {config_diff(cfg, c.config)}: {reason}")
            if not to_try:
                stop = "no candidate passes the free screen"
                break

            plan_rows = []
            for c in to_try:
                est = count_uncached(evaluator, c.config, items, gen_cache)
                plan_rows.append((c, est))
                gain = "n/a" if c.screen_gain is None else f"{c.screen_gain:+d}"
                say(f"  candidate {c.lever} {config_diff(cfg, c.config)} | screen gain {gain} | "
                    f"{est} new generation calls, up to ~{int(est * CALLS_PER_NEW_PROMPT)} API calls with judging")
            if dry_run:
                stop = "dry run, nothing was executed"
                break

            chosen = None
            for c, est in plan_rows:
                if est * CALLS_PER_NEW_PROMPT <= budget - spent:
                    chosen = (c, est)
                    break
                step += 1
                tried.add(c.config)
                reason = f"estimated cost ~{int(est * CALLS_PER_NEW_PROMPT)} exceeds the {budget - spent} calls left"
                log_step(conn, session, step, current_id, c.lever, config_diff(cfg, c.config),
                         "skipped_budget", reason, screen_gain=c.screen_gain, parent_passes=n_parent)
                say(f"  skip {c.lever}: {reason}")
            if chosen is None:
                stop = "the budget left is too small for any candidate"
                break

            c, est = chosen
            experiments += 1
            step += 1
            tried.add(c.config)
            change = config_diff(cfg, c.config)
            say(f"\nRUNNING experiment {experiments}: {c.lever} {change}")
            judge.max_calls = judge.calls_made + max(0, budget - spent - est)
            running = (c, current_id)
            result = run_experiment(conn, evaluator, c.config, items, gen_cache, judge)
            running = None
            used = result.new_generation_calls + result.new_judge_calls
            spent += used

            if result.errors:
                shown = "; ".join(f"{q}: {result.detail[q][:90]}" for q in result.errors[:3])
                reason = f"errors on {len(result.errors)} questions ({shown}); run not logged"
                log_step(conn, session, step, current_id, c.lever, change, "aborted", reason,
                         screen_gain=c.screen_gain, parent_passes=n_parent, api_calls=used)
                say(f"  ABORTED: {reason}")
                say(f"  cost {used} API calls ({spent}/{budget} used this session)")
                stop = "an experiment had errors, so the loop stopped without judging it"
                break

            cand_pass = pass_map(conn, result.run_id)
            d = decide(parent_pass, cand_pass, min_gain)
            reason = d.reason + f" | fixed {d.fixed} | broken {d.broken}"
            log_step(conn, session, step, current_id, c.lever, change,
                     "accepted" if d.accepted else "rejected", reason, candidate_run_id=result.run_id,
                     screen_gain=c.screen_gain, parent_passes=n_parent,
                     candidate_passes=sum(cand_pass.values()), api_calls=used)
            say(f"  {'ACCEPTED' if d.accepted else 'rejected'}: {reason}")
            say(f"  cost {used} API calls ({spent}/{budget} used this session)")
            if d.accepted:
                current_id, cfg = result.run_id, c.config
    except (QuotaExhausted, BudgetReached, GenerationAborted) as e:
        stop = f"stopped early: {type(e).__name__}: {str(e)[:200]}"
        if running is not None:
            c, parent_id = running
            step += 1
            log_step(conn, session, step, parent_id, c.lever, config_diff(config_of(conn, parent_id)[0], c.config),
                     "aborted", stop, screen_gain=c.screen_gain)
        say("\nFinished answers and verdicts are cached, so rerunning continues without paying for them again.")

    final_pass = pass_map(conn, current_id)
    say("\n" + "=" * 70)
    say(f"STOPPED: {stop}")
    say(f"start run {start_run_id}: {sum(start_pass.values())}/{len(start_pass)} pass  ->  "
        f"final run {current_id}: {sum(final_pass.values())}/{len(final_pass)} pass")
    say(f"API calls used this session: {spent} of {budget}")
    final_cfg = config_of(conn, current_id)[0]
    say(f"final config: {final_cfg}")
    return {"session": session, "start_run": start_run_id, "final_run": current_id, "spent": spent, "stop": stop}