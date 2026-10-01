"""embed.py: retry/backoff behavior, batching, and the text that gets embedded.
No provider is called: retries use fake errors and sleeping is recorded, not done."""

import dataclasses

import pytest

import embed as E


class Transient(Exception):
    pass


class RateLimited(Exception):
    pass


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(E.time, "sleep", recorded.append)
    monkeypatch.setattr(E.random, "uniform", lambda lo, hi: hi)  # always the max delay
    return recorded


def flaky(failures):
    """A function that raises each exception in `failures` in turn, then returns 'ok'."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if failures:
            raise failures.pop(0)
        return "ok"

    fn.calls = calls
    return fn


# --------------------------------------------------------------------------- #
# Backoff
# --------------------------------------------------------------------------- #

def test_retries_transient_errors_with_exponential_backoff(sleeps):
    fn = flaky([Transient(), Transient(), Transient()])
    assert E.with_backoff(fn, (Transient,), base_delay=2.0, max_delay=60.0) == "ok"
    assert fn.calls["n"] == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_delay_is_capped(sleeps):
    E.with_backoff(flaky([Transient()] * 5), (Transient,), max_attempts=6, base_delay=2.0, max_delay=10.0)
    assert max(sleeps) == 10.0


def test_rate_limits_wait_at_least_ten_seconds(sleeps, monkeypatch):
    monkeypatch.setattr(E.random, "uniform", lambda lo, hi: 0.1)  # jitter picks a tiny delay
    E.with_backoff(flaky([RateLimited(), Transient()]), (RateLimited, Transient), rate_limit=(RateLimited,))
    assert sleeps == [10.0, 0.1]


def test_gives_up_after_max_attempts(sleeps):
    fn = flaky([Transient()] * 10)
    with pytest.raises(Transient):
        E.with_backoff(fn, (Transient,), max_attempts=3)
    assert fn.calls["n"] == 3
    assert len(sleeps) == 2


def test_non_retryable_errors_raise_immediately(sleeps):
    fn = flaky([ValueError("bad input")])
    with pytest.raises(ValueError):
        E.with_backoff(fn, (Transient,))
    assert sleeps == []


def test_missing_payment_method_fails_fast_with_an_explanation(sleeps):
    err = RateLimited("You have not yet added your payment method in the billing page ...")
    with pytest.raises(E.EmbeddingConfigError, match="payment method"):
        E.with_backoff(flaky([err]), (RateLimited,), rate_limit=(RateLimited,))
    assert sleeps == []


# --------------------------------------------------------------------------- #
# What gets embedded
# --------------------------------------------------------------------------- #

def row(level="passage", book_id=1, chapter_index=3, content="Body text.", **kw):
    return dict(id=kw.get("id", 1), book_id=book_id, level=level, chapter_index=chapter_index,
                chapter_title=kw.get("chapter_title", "Chapter 2 (27 WEEKS)"), content=content,
                title=kw.get("title", "Wayward Book"), authors=kw.get("authors", ["Ada Author"]))


def test_passage_header_names_book_author_and_chapter():
    assert E.embedding_text(row()) == "Wayward Book by Ada Author, Chapter 3: Chapter 2 (27 WEEKS)\n\nBody text."


def test_summary_headers_say_which_level_they_are():
    assert E.embedding_text(row(level="chapter_summary")).splitlines()[0].endswith("(chapter summary)")
    assert E.embedding_text(row(level="book_summary", chapter_index=None)).splitlines()[0] == \
        "Wayward Book by Ada Author (book summary)"


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

@pytest.fixture
def batch_limits(monkeypatch):
    def set_limits(size, max_tokens):
        monkeypatch.setattr(E, "settings", dataclasses.replace(E.settings, embed_batch_size=size,
                                                               embed_batch_max_tokens=max_tokens))
    return set_limits


def test_batches_respect_the_count_limit(batch_limits):
    batch_limits(size=3, max_tokens=10_000)
    groups = E.batches([row(id=i) for i in range(7)])
    assert [len(g) for g in groups] == [3, 3, 1]


def test_batches_respect_the_token_limit(batch_limits):
    batch_limits(size=100, max_tokens=300)
    long = "word " * 200  # ~260 tokens with the header
    groups = E.batches([row(id=i, content=long) for i in range(3)])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_batches_never_mix_books(batch_limits):
    batch_limits(size=100, max_tokens=10_000)
    groups = E.batches([row(id=1, book_id=1), row(id=2, book_id=1), row(id=3, book_id=2)])
    assert [[r["id"] for r in g] for g in groups] == [[1, 2], [3]]


def test_every_row_is_batched_once_in_order(batch_limits):
    batch_limits(size=4, max_tokens=500)
    rows = [row(id=i, book_id=i // 5, content="word " * (i * 7)) for i in range(20)]
    assert [r["id"] for g in E.batches(rows) for r in g] == list(range(20))
