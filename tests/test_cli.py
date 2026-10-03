"""`index.py book --redo-summaries` against the test database. Regenerating
summaries throws away paid work, so the old summaries must survive a dry run, a
"no" at the prompt, or a failure, and the estimate must price the regeneration.
Claude is never called: token counts and the indexing run are faked."""

from __future__ import annotations

import sys

import pytest

import db
import index
import jobs
import summarize as S
from tests.helpers import make_book

pytestmark = pytest.mark.db


@pytest.fixture
def summarized_book(conn):
    book_id = db.save_extracted_book(conn, make_book("Paid For", chapters=3))
    for i in range(1, 4):
        db.save_chapter_summary(conn, book_id, i, f"Chapter {i}", f"### Summary\nold summary {i}", 5, "m")
    db.save_book_summary(conn, book_id, "### Premise\nold book summary", 9, "m", "full_text")
    return book_id


@pytest.fixture
def run_book(monkeypatch, summarized_book):
    """Runs `index.py book <id> --redo-summaries [flags]`; `answer` is the y/N reply."""
    monkeypatch.setattr(index, "file_hash", lambda path: "unused")
    monkeypatch.setattr(db, "library_file", lambda source_path: type("Missing", (), {"exists": lambda self: False})())
    monkeypatch.setattr("anthropic.Anthropic", lambda: object())
    monkeypatch.setattr(S, "count_tokens", lambda client, text: 5_000)
    indexed = []
    monkeypatch.setattr(jobs, "index_book", lambda conn, client, est, progress=None:
                        indexed.append(est) or jobs.IndexResult(True, 0.0))

    def run(*flags, answer="n"):
        monkeypatch.setattr(S, "confirm", lambda yes: answer == "y")
        monkeypatch.setattr(sys, "argv", ["index.py", "book", str(summarized_book), "--redo-summaries", *flags])
        index.main()

    run.indexed = indexed
    return run


def summaries(conn, book_id):
    return db.get_chapter_summaries(conn, book_id), db.get_book_summary(conn, book_id)


def test_dry_run_prices_the_regeneration_and_deletes_nothing(run_book, conn, summarized_book, capsys):
    before = summaries(conn, summarized_book)
    run_book("--dry-run")
    out = capsys.readouterr().out
    assert "Nothing to do" not in out and "TOTAL for this book" in out  # all 3 chapters + the book, priced
    assert "nothing deleted" in out
    assert summaries(conn, summarized_book) == before


def test_answering_no_keeps_the_paid_summaries(run_book, conn, summarized_book, capsys):
    before = summaries(conn, summarized_book)
    run_book(answer="n")
    assert "existing summaries are kept" in capsys.readouterr().out
    assert summaries(conn, summarized_book) == before
    assert run_book.indexed == []


def test_answering_yes_deletes_then_regenerates_everything(run_book, conn, summarized_book):
    run_book(answer="y")
    assert summaries(conn, summarized_book) == ({}, None)  # the deletion was committed
    (est,) = run_book.indexed
    assert len(est.plans[0].pending_chapters) == 3 and est.plans[0].needs_book_summary
