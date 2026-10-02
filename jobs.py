"""Indexing one book, shared by the CLI (`index.py book`), the HTTP API, and a
Cloud Run Job (`index.py run-job`).

- estimate_book(): what finishing a book would cost (free: count_tokens only)
- index_book():    summaries, then embeddings, with progress reporting
- scan_library():  extract new or changed EPUBs (free), never discarding paid work
- run_job():       execute a stored job, recording status, progress and cost
- get_runner():    where jobs run (a background thread locally; a Cloud Run Job when deployed)
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import anthropic

import db
import embed
import summarize
from config import settings
from extract import extract_book, file_hash, find_epubs, read_meta
from stage import Progress

log = logging.getLogger("chapter_and_verse.jobs")


# --------------------------------------------------------------------------- #
# Estimate and index one book
# --------------------------------------------------------------------------- #

@dataclass
class Estimate:
    book_id: int
    plans: list  # summarize.BookPlan
    embed_chunks: int
    embed_tokens: int
    summary_cost: float
    embed_cost: float

    @property
    def total(self) -> float:
        return self.summary_cost + self.embed_cost

    @property
    def nothing_to_do(self) -> bool:
        return not self.plans and not self.embed_chunks

    def as_dict(self) -> dict:
        plan = self.plans[0] if self.plans else None
        return {
            "book_id": self.book_id,
            "summaries": {
                "model": settings.summary_model,
                "effort": settings.summary_effort,
                "chapters_to_summarize": len(plan.pending_chapters) if plan else 0,
                "book_summary_needed": bool(plan and plan.needs_book_summary),
                "book_summary_method": plan.method if plan else None,
                "input_tokens": plan.est_input if plan else 0,
                "output_tokens": plan.est_output if plan else 0,
                "cost_usd": round(self.summary_cost, 4),
            },
            "embeddings": {
                "model": settings.embed_model,
                "chunks": self.embed_chunks,
                "tokens": self.embed_tokens,
                "cost_usd": round(self.embed_cost, 4),
            },
            "total_usd": round(self.total, 4),
        }


def estimate_book(conn, client: anthropic.Anthropic, book_id: int) -> Estimate:
    plans = summarize.plan(conn, client, [book_id])  # count_tokens calls are free
    summary_cost = sum(p.est_cost for p in plans)
    future = sum(len(p.pending_chapters) + p.needs_book_summary for p in plans)
    chunks, tokens, embed_cost = embed.estimate(conn, [book_id], future)
    return Estimate(book_id, plans, chunks, tokens, summary_cost, embed_cost)


@dataclass
class IndexResult:
    ok: bool
    cost_usd: float
    error: str | None = None


def index_book(conn, client: anthropic.Anthropic, estimate: Estimate,
               progress: Progress | None = None) -> IndexResult:
    """Summaries, then embeddings. Callers must have confirmed the estimate."""
    cost = 0.0
    if estimate.plans:
        result = summarize.execute(client, estimate.plans, progress)
        cost += result.cost_usd
        if not result:
            return IndexResult(False, cost, "summaries didn't finish; run again to resume")
    if estimate.embed_chunks or estimate.plans:  # new summaries need embedding too
        result = embed.execute(conn, [estimate.book_id], progress)
        cost += result.cost_usd
        if not result:
            return IndexResult(False, cost, "embeddings didn't finish; run again to resume")
    return IndexResult(True, cost)


# --------------------------------------------------------------------------- #
# Library scan (free)
# --------------------------------------------------------------------------- #

def check_file(conn, path: Path, digest: str) -> str:
    """What an EPUB needs, relative to what's stored:

    unchanged  same file, same content: nothing to do
    moved      new location, same Calibre book and content (e.g. renamed in Calibre):
               the stored path is updated here, and summaries are kept
    new        not in the database yet
    changed    content differs, and nothing paid would be lost by re-extracting
    changed_has_paid_work   content differs, and re-extracting would discard
               summaries/embeddings, so it needs a person to decide
    """
    stored = db.stored_hash(conn, path)
    if stored == digest:
        return "unchanged"
    if stored is None:
        moved = db.find_moved_book(conn, read_meta(path).calibre_uuid, digest)
        if moved:
            db.update_source_path(conn, moved["id"], path)
            return "moved"
        return "new"
    return "changed_has_paid_work" if db.has_paid_work(conn, path) else "changed"


def scan_library(conn, library: Path | None = None) -> list[dict]:
    """Extract EPUBs that are new or changed (free). Never discards paid work:
    changed books with summaries/embeddings are reported, not re-extracted."""
    results = []
    for path in find_epubs(library or settings.library_path):
        digest = file_hash(path)
        try:
            status = check_file(conn, path, digest)
            if status in ("unchanged", "moved", "changed_has_paid_work"):
                results.append({"path": path.name, "status": status})
                continue
            book = extract_book(path, digest)
        except Exception as e:  # one bad EPUB shouldn't stop the scan
            results.append({"path": path.name, "status": "failed", "error": f"{type(e).__name__}: {e}"})
            continue
        book_id = db.save_extracted_book(conn, book)
        results.append({"path": path.name, "status": "added" if status == "new" else "updated",
                        "book_id": book_id, "title": book.meta.title, "chapters": len(book.chapters)})
    return results


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

def run_job(job_id: int, client: anthropic.Anthropic | None = None) -> None:
    """Execute a stored job in its own connection (it may run in another thread or process)."""
    conn = db.connect()
    try:
        job = db.get_job(conn, job_id)
        db.update_job(conn, job_id, status="running", progress="estimating")
        client = client or anthropic.Anthropic()
        # Re-estimate: the book may have changed since the job was queued. Never
        # exceed the limit the caller agreed to.
        estimate = estimate_book(conn, client, job["book_id"])
        if estimate.total > float(job["max_cost_usd"]):
            db.update_job(conn, job_id, status="failed", cost_usd=0,
                          error=f"estimate rose to ${estimate.total:.2f}, over the "
                                f"${float(job['max_cost_usd']):.2f} limit; nothing was spent")
            return
        result = index_book(conn, client, estimate,
                            progress=lambda msg: db.update_job(conn, job_id, progress=msg[:500]))
        db.update_job(conn, job_id, status="done" if result.ok else "failed",
                      cost_usd=round(result.cost_usd, 4), error=result.error,
                      progress="finished" if result.ok else "stopped")
    except Exception as e:
        db.update_job(conn, job_id, status="failed", error=f"{type(e).__name__}: {e}"[:1000])
        raise
    finally:
        conn.close()


class ThreadRunner:
    """Runs jobs in a background thread of the current process (local use)."""

    def start(self, job_id: int) -> None:
        threading.Thread(target=self._run, args=(job_id,), daemon=True, name=f"job-{job_id}").start()

    @staticmethod
    def _run(job_id: int) -> None:
        try:
            run_job(job_id)
        except Exception:
            pass  # the failure is recorded on the job


class InlineRunner:
    """Runs jobs synchronously in the caller (tests)."""

    def __init__(self, client=None):
        self.client = client

    def start(self, job_id: int) -> None:
        run_job(job_id, self.client)


class JobStartError(RuntimeError):
    """The job couldn't be handed to its runner; nothing ran and nothing was spent."""


METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
CLOUD_RUN_API = "https://run.googleapis.com/v2"


class CloudRunJobRunner:
    """Starts one Cloud Run Job execution per indexing job (deployed use).

    A thread inside a Cloud Run request isn't reliable for multi-minute work: once
    the response is sent, the instance's CPU is throttled and the instance can be
    shut down. A job execution runs `python index.py run-job <id>` to completion on
    its own. The call is two plain HTTP requests: an access token for the service's
    identity from the metadata server, then the Cloud Run Admin API's `jobs:run`.
    """

    def __init__(self, job_name: str, urlopen=urllib.request.urlopen):
        self.job_name = job_name  # projects/<project>/locations/<region>/jobs/<job>
        self._urlopen = urlopen

    def _token(self) -> str:
        request = urllib.request.Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
        with self._urlopen(request, timeout=5) as r:
            return json.load(r)["access_token"]

    def start(self, job_id: int) -> str:
        """Returns the execution's name."""
        body = {"overrides": {"containerOverrides": [{"args": ["index.py", "run-job", str(job_id)]}]}}
        try:
            request = urllib.request.Request(
                f"{CLOUD_RUN_API}/{self.job_name}:run", data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"})
            with self._urlopen(request, timeout=30) as r:
                operation = json.load(r)  # a long-running operation; its metadata is the execution
        except urllib.error.HTTPError as e:
            raise JobStartError(f"Cloud Run refused to start the job: HTTP {e.code} {_api_error(e)}") from e
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            raise JobStartError(f"couldn't reach Cloud Run to start the job: {type(e).__name__}: {e}") from e
        execution = operation.get("metadata", {}).get("name", operation.get("name", "?"))
        log.info("job %s started as Cloud Run execution %s", job_id, execution)
        return execution


def _api_error(e: urllib.error.HTTPError) -> str:
    """The message from a Google API error body, e.g. which permission was denied."""
    try:
        return json.loads(e.read())["error"]["message"]
    except Exception:
        return e.reason or ""


_runner_override = None


def get_runner():
    if _runner_override is not None:
        return _runner_override
    if settings.job_runner == "thread":
        return ThreadRunner()
    if settings.job_runner == "cloud_run":
        if not settings.cloud_run_job:
            raise RuntimeError("JOB_RUNNER=cloud_run needs CLOUD_RUN_JOB=projects/<project>/locations/<region>/jobs/<job>")
        return CloudRunJobRunner(settings.cloud_run_job)
    raise RuntimeError(f"unknown JOB_RUNNER={settings.job_runner!r} (use 'thread' or 'cloud_run')")
