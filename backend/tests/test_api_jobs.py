import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.jobs import JobBusy, JobManager, NotAwaiting, Runtime
from app.api.limits import RateLimiter
from app.api.main import create_app

CANDIDATE = {"lever": "top_k", "config": {"top_k": 6}, "change": {"top_k": [4, 6]}, "screen_gain": 5, "est": 10}
FIRST = {"first": True}


class FakeGraph:
    """Pauses `pauses` times when approval is on, like the real graph's interrupt before the run node."""

    def __init__(self, say, pauses):
        self.say, self.remaining = say, pauses
        self.invocations, self.updates, self.next = [], [], ()

    def invoke(self, inp, config):
        self.invocations.append(inp)
        self.say("\nstep ran")
        if inp is None:
            self.remaining -= 1
        self.next = ("run",) if self.remaining > 0 else ()

    def get_state(self, config):
        return SimpleNamespace(next=self.next, values={"candidate": CANDIDATE if self.next else None})

    def update_state(self, config, values, as_node=None):
        self.updates.append(values)


def make_manager(pauses=1, fail=None, timeout=5.0):
    graphs, closed = [], []

    def build(job, say):
        if fail:
            raise fail
        g = FakeGraph(say, pauses)
        graphs.append(g)
        return Runtime(graph=g, config={}, first_input=FIRST, summary=lambda: {"ok": True},
                       close=lambda: closed.append(job.id))

    return JobManager(build, approval_timeout_s=timeout), graphs, closed


def wait_for(job, statuses, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if job.status in statuses:
            return
        time.sleep(0.01)
    raise AssertionError(f"job stayed in {job.status!r}, wanted one of {statuses}")


def test_job_without_approval_runs_to_the_end():
    mgr, graphs, closed = make_manager(pauses=0)
    job = mgr.start({"approve": False})
    wait_for(job, {"finished"})
    assert job.result == {"ok": True} and closed == [job.id]
    assert any(e["text"] == "step ran" for e in job.events)
    assert graphs[0].invocations == [FIRST]


def test_approval_flow():
    mgr, graphs, _ = make_manager(pauses=1)
    job = mgr.start({"approve": True})
    wait_for(job, {"awaiting_approval"})
    assert job.proposal == CANDIDATE
    mgr.decide(job.id, True)
    wait_for(job, {"finished"})
    assert graphs[0].invocations == [FIRST, None] and graphs[0].updates == []


def test_rejection():
    mgr, graphs, _ = make_manager(pauses=1)
    job = mgr.start({"approve": True})
    wait_for(job, {"awaiting_approval"})
    mgr.decide(job.id, False)
    wait_for(job, {"finished"})
    assert graphs[0].updates == [{"rejected": True}] and graphs[0].invocations == [FIRST, None]


def test_decisions_when_nothing_is_waiting():
    mgr, _, _ = make_manager(pauses=0)
    job = mgr.start({"approve": False})
    wait_for(job, {"finished"})
    with pytest.raises(NotAwaiting):
        mgr.decide(job.id, True)
    with pytest.raises(KeyError):
        mgr.decide("nope", True)


def test_second_decision_is_refused():
    mgr, _, _ = make_manager(pauses=1)
    job = mgr.start({"approve": True})
    wait_for(job, {"awaiting_approval"})
    mgr.decide(job.id, True)
    with pytest.raises(NotAwaiting):
        mgr.decide(job.id, True)
    wait_for(job, {"finished"})


def test_only_one_active_job():
    mgr, _, _ = make_manager(pauses=1)
    first = mgr.start({"approve": True})
    wait_for(first, {"awaiting_approval"})
    with pytest.raises(JobBusy):
        mgr.start({"approve": True})
    mgr.decide(first.id, False)
    wait_for(first, {"finished"})
    mgr.start({"approve": False})


def test_failure_is_reported_without_details():
    mgr, _, closed = make_manager(fail=RuntimeError("secret boom"))
    job = mgr.start({"approve": False})
    wait_for(job, {"failed"})
    assert job.error and "secret boom" not in job.error and closed == []


def test_approval_timeout_counts_as_rejection():
    mgr, graphs, _ = make_manager(pauses=1, timeout=0.05)
    job = mgr.start({"approve": True})
    wait_for(job, {"finished"})
    assert graphs[0].updates == [{"rejected": True}]


H = {"X-Admin-Token": "secret"}
BODY = {"start_run": 5, "dry_run": True, "approve": False}


def api(tmp_path, mgr=None, token="secret", validate=None):
    mgr = mgr or make_manager(pauses=0)[0]
    app = create_app(object(), results_path=str(tmp_path / "results.json"), docs_dir=str(tmp_path),
                     limiter=RateLimiter(100, 3600, 1000, 86400), jobs=mgr, admin_token=token,
                     validate_start_fn=validate or (lambda run_id: None))
    return TestClient(app), mgr


def test_jobs_are_disabled_without_an_admin_token(tmp_path):
    client, _ = api(tmp_path, token="")
    assert client.post("/api/jobs", json=BODY, headers=H).status_code == 403


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}])
def test_wrong_or_missing_token_is_refused(tmp_path, headers):
    client, _ = api(tmp_path)
    assert client.post("/api/jobs", json=BODY, headers=headers).status_code == 401
    assert client.get("/api/jobs", headers=headers).status_code == 401


@pytest.mark.parametrize("body", [
    {**BODY, "budget": 10_000}, {**BODY, "max_experiments": 99}, {**BODY, "evil": 1}, {"dry_run": True},
])
def test_job_requests_are_validated_and_capped(tmp_path, body):
    client, mgr = api(tmp_path)
    assert client.post("/api/jobs", json=body, headers=H).status_code == 422
    assert mgr.jobs == {}


def test_unusable_start_run_is_rejected(tmp_path):
    def refuse(run_id):
        raise ValueError("run 5 used another model")

    client, mgr = api(tmp_path, validate=refuse)
    r = client.post("/api/jobs", json=BODY, headers=H)
    assert r.status_code == 422 and "another model" in r.text and mgr.jobs == {}


def test_start_view_and_list(tmp_path):
    client, mgr = api(tmp_path)
    r = client.post("/api/jobs", json=BODY, headers=H)
    assert r.status_code == 202
    job_id = r.json()["id"]
    wait_for(mgr.get(job_id), {"finished"})
    view = client.get(f"/api/jobs/{job_id}", headers=H).json()
    assert view["status"] == "finished" and view["result"] == {"ok": True}
    assert [j["id"] for j in client.get("/api/jobs", headers=H).json()] == [job_id]
    assert client.get("/api/jobs/unknown", headers=H).status_code == 404


def test_approval_over_http(tmp_path):
    client, mgr = api(tmp_path, make_manager(pauses=1)[0])
    job_id = client.post("/api/jobs", json={**BODY, "dry_run": False, "approve": True}, headers=H).json()["id"]
    wait_for(mgr.get(job_id), {"awaiting_approval"})
    assert client.get(f"/api/jobs/{job_id}", headers=H).json()["proposal"] == CANDIDATE
    assert client.post("/api/jobs", json=BODY, headers=H).status_code == 409  # one job at a time
    r = client.post(f"/api/jobs/{job_id}/decision", json={"approve": True}, headers=H)
    assert r.status_code == 200 and r.json()["decision"] == "approve"
    wait_for(mgr.get(job_id), {"finished"})
    assert client.post(f"/api/jobs/{job_id}/decision", json={"approve": True}, headers=H).status_code == 409
    assert client.post("/api/jobs/unknown/decision", json={"approve": True}, headers=H).status_code == 404


def test_event_stream_for_a_finished_job(tmp_path):
    client, mgr = api(tmp_path)
    job_id = client.post("/api/jobs", json=BODY, headers=H).json()["id"]
    wait_for(mgr.get(job_id), {"finished"})
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=H) as r:
        text = "".join(r.iter_text())
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "step ran" in text and '"text": "finished"' in text
    with client.stream("GET", f"/api/jobs/{job_id}/events?after=1000", headers=H) as r:
        assert "".join(r.iter_text()) == ""


def test_event_stream_needs_the_token(tmp_path):
    client, mgr = api(tmp_path)
    job_id = client.post("/api/jobs", json=BODY, headers=H).json()["id"]
    wait_for(mgr.get(job_id), {"finished"})
    assert client.get(f"/api/jobs/{job_id}/events").status_code == 401