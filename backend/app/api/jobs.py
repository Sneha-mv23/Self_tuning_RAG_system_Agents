import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.api.errors import map_error

log = logging.getLogger("rag-api")
ACTIVE = ("queued", "running", "awaiting_approval")


class JobBusy(RuntimeError):
    pass


class NotAwaiting(RuntimeError):
    pass


@dataclass
class Runtime:
    graph: Any
    config: dict
    first_input: dict
    summary: Callable[[], dict]
    close: Callable[[], None]


@dataclass
class Job:
    id: str
    params: dict
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued, running, awaiting_approval, finished, failed
    events: list = field(default_factory=list)
    proposal: dict | None = None
    result: dict | None = None
    error: str | None = None
    decision: str | None = None
    decided: threading.Event = field(default_factory=threading.Event)
    cond: threading.Condition = field(default_factory=threading.Condition)


class JobManager:
    """Runs one optimizer session at a time in a background thread. `build(job, say)` returns a Runtime and
    is called inside the worker thread, because SQLite connections cannot cross threads."""

    def __init__(self, build: Callable[[Job, Callable[[str], None]], Runtime], approval_timeout_s: float = 900.0):
        self.build = build
        self.approval_timeout_s = approval_timeout_s
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, params: dict) -> Job:
        with self._lock:
            if any(j.status in ACTIVE for j in self.jobs.values()):
                raise JobBusy("another optimizer job is still running")
            job = Job(id=uuid.uuid4().hex[:12], params=dict(params))
            self.jobs[job.id] = job
        threading.Thread(target=self._work, args=(job,), daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job:
        return self.jobs[job_id]  # KeyError if unknown

    def all(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def decide(self, job_id: str, approve: bool) -> None:
        job = self.get(job_id)
        with job.cond:
            if job.status != "awaiting_approval" or job.decided.is_set():
                raise NotAwaiting("this job is not waiting for a decision")
            job.decision = "approve" if approve else "reject"
            job.decided.set()

    def _emit(self, job: Job, text: str, kind: str = "log") -> None:
        with job.cond:
            job.events.append({"i": len(job.events), "t": round(time.time(), 2), "kind": kind, "text": text})
            job.cond.notify_all()

    def _status(self, job: Job, status: str) -> None:
        with job.cond:
            job.status = status
            job.events.append({"i": len(job.events), "t": round(time.time(), 2), "kind": "status", "text": status})
            job.cond.notify_all()

    def _work(self, job: Job) -> None:
        rt = None

        def say(text) -> None:
            for line in str(text).splitlines():
                if line.strip():
                    self._emit(job, line.rstrip())

        try:
            self._status(job, "running")
            rt = self.build(job, say)
            rt.graph.invoke(rt.first_input, rt.config)
            while job.params.get("approve"):
                snap = rt.graph.get_state(rt.config)
                if not snap.next:
                    break
                candidate = (snap.values or {}).get("candidate")
                job.proposal = candidate
                job.decision = None
                job.decided.clear()
                self._status(job, "awaiting_approval")
                if not job.decided.wait(self.approval_timeout_s):
                    job.decision = "reject"
                    say("no decision before the timeout, so the experiment was rejected")
                job.proposal = None
                say(f"decision: {job.decision}")
                self._status(job, "running")
                if job.decision != "approve":
                    rt.graph.update_state(rt.config, {"rejected": True}, as_node="propose")
                rt.graph.invoke(None, rt.config)
            job.result = rt.summary()
            self._status(job, "finished")
        except Exception as exc:  # noqa: BLE001 - reported safely below
            log.exception("optimizer job %s failed", job.id)
            job.error = f"{map_error(exc)[1]} ({type(exc).__name__})"
            self._status(job, "failed")
        finally:
            if rt is not None:
                try:
                    rt.close()
                except Exception:  # noqa: BLE001
                    log.exception("closing job resources failed")


def validate_start(run_id: int) -> None:
    """Raises ValueError if the run cannot be the start of an optimizer session."""
    from app.db.experiments import connect
    from app.graphs.optimizer_graph import check_start

    conn = connect()
    try:
        check_start(conn, run_id)
    finally:
        conn.close()


def build_runtime(job: Job, say: Callable[[str], None]) -> Runtime:
    from app.db.experiments import connect
    from app.db.optimizer_log import ensure_tables
    from app.evals.generation_run import GenerationCache
    from app.evals.golden import load_golden
    from app.evals.judge import Judge
    from app.evals.retrieval_eval import RetrievalEvaluator
    from app.evals.split import load_split, select
    from app.graphs.optimizer_graph import (
        OptimizerContext, build_optimizer_graph, initial_state, make_checkpointer, thread_config)
    from app.optimizer.experiment import pass_map
    from app.optimizer.loop import config_of
    from app.rag.loader import load_corpus

    p = job.params
    conn = connect()
    ensure_tables(conn)
    gen_cache, judge = GenerationCache(), Judge()

    def close() -> None:
        for resource in (gen_cache, judge.cache, conn):
            try:
                resource.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        docs = load_corpus()
        items = select(load_golden(docs=docs), load_split(), "dev")
        ctx = OptimizerContext(conn, RetrievalEvaluator(docs), items, gen_cache, judge,
                               dry_run=p["dry_run"], say=say)
        saver, _ = make_checkpointer(None)  # in memory: enough while the server stays up
        graph = build_optimizer_graph(ctx, saver, approve_before_run=p["approve"])
        first = initial_state(conn, p["start_run"], p["budget"], p["max_experiments"])
    except Exception:
        close()
        raise
    config = thread_config(first["session"], p["max_experiments"])

    def summary() -> dict:
        final = graph.get_state(config).values
        start, end = pass_map(conn, final["start_run"]), pass_map(conn, final["current_run"])
        return {"start_run": final["start_run"], "final_run": final["current_run"],
                "start_passes": sum(start.values()), "final_passes": sum(end.values()),
                "n_questions": len(end), "spent": final["spent"], "budget": final["budget"],
                "stop": final.get("stop") or "",
                "final_config": asdict(config_of(conn, final["current_run"])[0])}

    return Runtime(graph, config, first, summary, close)