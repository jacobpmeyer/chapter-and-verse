"""Where indexing jobs run: the Cloud Run Job runner (HTTP faked), runner
selection, and `index.py run-job`, which is what a Cloud Run Job executes.
No network, no database."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import urllib.error

import pytest

import db
import index
import jobs

JOB = "projects/p/locations/us-east1/jobs/cv-index"
EXECUTION = f"{JOB}/executions/cv-index-abc12"


class FakeUrlopen:
    """Replays responses (or raises errors) in order and records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return io.BytesIO(json.dumps(nxt).encode())


TOKEN = {"access_token": "ya29.token", "expires_in": 3599, "token_type": "Bearer"}
OPERATION = {"name": "projects/p/locations/us-east1/operations/op1", "metadata": {"name": EXECUTION}}


def test_cloud_run_runner_starts_an_execution_for_the_job():
    urlopen = FakeUrlopen(TOKEN, OPERATION)
    assert jobs.CloudRunJobRunner(JOB, urlopen=urlopen).start(42) == EXECUTION

    metadata, run = urlopen.requests
    assert metadata.full_url.startswith("http://metadata.google.internal/")
    assert metadata.get_header("Metadata-flavor") == "Google"
    assert run.full_url == f"https://run.googleapis.com/v2/{JOB}:run" and run.get_method() == "POST"
    assert run.get_header("Authorization") == "Bearer ya29.token"
    assert json.loads(run.data) == {"overrides": {"containerOverrides": [{"args": ["index.py", "run-job", "42"]}]}}


def http_error(code, body):
    return urllib.error.HTTPError(f"https://run.googleapis.com/v2/{JOB}:run", code, "Forbidden", {},
                                  io.BytesIO(json.dumps(body).encode()))


def test_a_refusal_reports_googles_reason():
    denied = http_error(403, {"error": {"code": 403, "message": "Permission 'run.jobs.runWithOverrides' denied"}})
    with pytest.raises(jobs.JobStartError, match=r"HTTP 403 Permission 'run.jobs.runWithOverrides' denied"):
        jobs.CloudRunJobRunner(JOB, urlopen=FakeUrlopen(TOKEN, denied)).start(42)


def test_an_unreachable_metadata_server_is_a_start_error():
    unreachable = urllib.error.URLError("Name or service not known")
    with pytest.raises(jobs.JobStartError, match="couldn't reach Cloud Run"):
        jobs.CloudRunJobRunner(JOB, urlopen=FakeUrlopen(unreachable)).start(42)


@pytest.fixture
def configure(monkeypatch):
    def apply(**changes):
        monkeypatch.setattr(jobs, "settings", dataclasses.replace(jobs.settings, **changes))
    monkeypatch.setattr(jobs, "_runner_override", None)
    return apply


def test_the_runner_is_chosen_by_setting(configure):
    configure(job_runner="thread")
    assert isinstance(jobs.get_runner(), jobs.ThreadRunner)
    configure(job_runner="cloud_run", indexing_job=JOB)
    runner = jobs.get_runner()
    assert isinstance(runner, jobs.CloudRunJobRunner) and runner.job_name == JOB


@pytest.mark.parametrize("changes, message", [
    ({"job_runner": "cloud_run", "indexing_job": ""}, "needs INDEXING_JOB"),
    ({"job_runner": "celery"}, "unknown JOB_RUNNER='celery'"),
])
def test_a_misconfigured_runner_is_an_error(configure, changes, message):
    configure(**changes)
    with pytest.raises(RuntimeError, match=message):
        jobs.get_runner()


# --------------------------------------------------------------------------- #
# index.py run-job
# --------------------------------------------------------------------------- #

class FakeConn:
    def close(self):
        pass


@pytest.fixture
def run_job_cli(monkeypatch):
    """Runs `index.py run-job 7` with the job's stored states and run_job faked.
    `states` is what get_job returns: before running, then after."""
    ran = []

    def run(*states, error=None):
        states = list(states)
        monkeypatch.setattr(db, "connect", lambda url=None: FakeConn())
        monkeypatch.setattr(db, "get_job", lambda conn, job_id: states.pop(0) if states else None)

        def run_job(job_id, client=None):
            ran.append(job_id)
            if error:
                raise error
        monkeypatch.setattr(jobs, "run_job", run_job)
        monkeypatch.setattr(sys, "argv", ["index.py", "run-job", "7"])
        index.main()

    run.ran = ran
    return run


def job(status, cost=None, error=None):
    return {"id": 7, "status": status, "cost_usd": cost, "error": error}


def test_run_job_runs_a_queued_job(run_job_cli, capsys):
    run_job_cli(job("queued"), job("done", cost=1.31))
    assert run_job_cli.ran == [7]
    assert "Job 7: done, cost $1.31" in capsys.readouterr().out


def test_run_job_exits_non_zero_when_the_job_did_not_finish(run_job_cli, capsys):
    with pytest.raises(SystemExit) as exit_:
        run_job_cli(job("queued"), job("failed", cost=0, error="estimate rose to $3.00, over the $2.00 limit"))
    assert exit_.value.code == 1
    assert "failed, cost $0.00, estimate rose" in capsys.readouterr().out


def test_run_job_exits_non_zero_when_run_job_raises(run_job_cli):
    with pytest.raises(SystemExit, match="Job 7 failed: RuntimeError: boom"):
        run_job_cli(job("queued"), error=RuntimeError("boom"))


@pytest.mark.parametrize("status", ["running", "done", "failed"])
def test_run_job_never_runs_a_job_twice(run_job_cli, status):
    with pytest.raises(SystemExit, match=f"Job 7 is {status}, not queued"):
        run_job_cli(job(status))
    assert run_job_cli.ran == []


def test_run_job_with_an_unknown_id(run_job_cli):
    with pytest.raises(SystemExit, match="No job 7"):
        run_job_cli()
    assert run_job_cli.ran == []
