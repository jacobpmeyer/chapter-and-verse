"""db.py against a real Postgres + pgvector test database."""

import dataclasses

import pytest
from pgvector import Vector

import db
from tests.helpers import axis, blend, make_book

pytestmark = pytest.mark.db

MODEL = "fake-embed"


def chunk_ids(conn, book_id, level="passage"):
    return [r["id"] for r in conn.execute(
        "SELECT id FROM chunks WHERE book_id = %s AND level = %s ORDER BY id", (book_id, level)).fetchall()]


def summarize_fully(conn, book_id, n_chapters):
    for i in range(1, n_chapters + 1):
        db.save_chapter_summary(conn, book_id, i, f"Chapter {i}", f"summary {i}", 10, "fake-model")
    db.save_book_summary(conn, book_id, "book summary", 20, "fake-model", "full_text")


def embed_all(conn, book_id, model=MODEL):
    rows = db.pending_embeddings(conn, [book_id], model)
    db.save_embeddings(conn, [(r["id"], Vector(axis(i % 50))) for i, r in enumerate(rows)], model)


# --------------------------------------------------------------------------- #
# Schema and saving
# --------------------------------------------------------------------------- #

def test_schema_is_idempotent(conn):
    db.ensure_schema(conn)
    db.ensure_schema(conn)


def test_schema_refuses_a_different_embedding_dimension(conn, monkeypatch):
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, embed_dim=8))
    with pytest.raises(SystemExit, match="vector\\(1024\\) but EMBED_DIM=8"):
        db.ensure_schema(conn)


def test_save_book_stores_chapters_and_paragraph_safe_passages(conn):
    book = make_book(chapters=3)
    book_id = db.save_extracted_book(conn, book)
    assert [c["title"] for c in db.get_chapters(conn, book_id)] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    passages = conn.execute("SELECT chapter_index, content FROM chunks WHERE book_id = %s", (book_id,)).fetchall()
    assert len(passages) > 3  # each chapter is split into several passages
    assert {p["chapter_index"] for p in passages} == {1, 2, 3}
    assert db.stored_hash(conn, str(book.path)) == "hash-1"


def test_re_extracting_keeps_the_id_and_clears_derived_data(conn):
    book_id = db.save_extracted_book(conn, make_book())
    summarize_fully(conn, book_id, 3)
    db.set_exact_token_count(conn, book_id, 12345)
    embed_all(conn, book_id)

    again = db.save_extracted_book(conn, make_book(chapters=2, content_hash="hash-2"))
    assert again == book_id
    row = db.get_book(conn, book_id)
    assert (row["content_hash"], row["chapter_count"], row["exact_token_count"]) == ("hash-2", 2, None)
    assert row["stage"] == "extracted"
    assert row["book_summary_method"] is None and row["embedded_at"] is None
    levels = {r["level"] for r in conn.execute("SELECT level FROM chunks WHERE book_id = %s", (book_id,))}
    assert levels == {"passage"}


def test_a_moved_file_is_matched_by_calibre_uuid(conn):
    first = db.save_extracted_book(conn, make_book(path="/library/old/Book.epub", uuid="uuid-1"))
    moved = db.save_extracted_book(conn, make_book(path="/library/new/Book.epub", uuid="uuid-1"))
    assert moved == first
    assert db.get_book(conn, first)["source_path"] == "/library/new/Book.epub"
    assert conn.execute("SELECT count(*) AS n FROM books").fetchone()["n"] == 1


def test_has_paid_work(conn):
    book = make_book()
    book_id = db.save_extracted_book(conn, book)
    assert not db.has_paid_work(conn, str(book.path))  # passages are free
    db.save_chapter_summary(conn, book_id, 1, "Chapter 1", "s", 1, "m")
    assert db.has_paid_work(conn, str(book.path))


def test_paid_work_includes_embeddings(conn):
    book = make_book()
    book_id = db.save_extracted_book(conn, book)
    first = chunk_ids(conn, book_id)[0]
    db.save_embeddings(conn, [(first, Vector(axis(0)))], MODEL)
    assert db.has_paid_work(conn, str(book.path))


# --------------------------------------------------------------------------- #
# Stages and lookup
# --------------------------------------------------------------------------- #

def test_stage_progression(conn, monkeypatch):
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, embed_model=MODEL))
    book_id = db.save_extracted_book(conn, make_book(chapters=2))
    stage = lambda: db.get_book(conn, book_id)["stage"]
    assert stage() == "extracted"
    db.save_chapter_summary(conn, book_id, 1, "Chapter 1", "s1", 1, "m")
    assert stage() == "summarizing"
    db.save_chapter_summary(conn, book_id, 2, "Chapter 2", "s2", 1, "m")
    assert stage() == "summarizing"  # chapters done, book summary missing
    db.save_book_summary(conn, book_id, "book", 1, "m", "full_text")
    assert stage() == "summarized"
    embed_all(conn, book_id)
    assert stage() == "embedded"


def test_switching_embedding_models_marks_vectors_stale(conn, monkeypatch):
    book_id = db.save_extracted_book(conn, make_book(chapters=1))
    summarize_fully(conn, book_id, 1)
    embed_all(conn, book_id, model="old-model")
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, embed_model="new-model"))
    assert db.get_book(conn, book_id)["stage"] == "summarized"
    assert len(db.pending_embeddings(conn, [book_id], "new-model")) == len(chunk_ids(conn, book_id)) + 2


def test_find_books_by_id_or_words(conn):
    a = db.save_extracted_book(conn, make_book("Witchcraft for Wayward Girls", authors=("Grady Hendrix",)))
    b = db.save_extracted_book(conn, make_book("The Devils", authors=("Joe Abercrombie",), series="The Devils"))
    ids = lambda q: [r["id"] for r in db.find_books(conn, q)]
    assert ids(str(a)) == [a]
    assert ids("wayward") == [a]
    assert ids("hendrix girls") == [a]  # every word must match, across title and author
    assert ids("devils abercrombie") == [b]
    assert ids("wayward abercrombie") == []
    assert ids("the") == [b]
    assert ids("zzz") == []


def test_list_books_filters_and_reports_stage(conn):
    db.save_extracted_book(conn, make_book("Book One"))
    db.save_extracted_book(conn, make_book("Book Two", path="/library/x/Two.epub"))
    assert [b["title"] for b in db.list_books(conn)] == ["Book One", "Book Two"]
    assert [b["title"] for b in db.list_books(conn, "two")] == ["Book Two"]
    assert {b["stage"] for b in db.list_books(conn)} == {"extracted"}


# --------------------------------------------------------------------------- #
# Summaries and usage log
# --------------------------------------------------------------------------- #

def test_book_summary_is_replaced_not_duplicated(conn):
    book_id = db.save_extracted_book(conn, make_book(chapters=1))
    db.save_book_summary(conn, book_id, "first", 1, "m", "full_text")
    db.save_book_summary(conn, book_id, "second", 1, "m", "chapter_summaries")
    assert db.get_book_summary(conn, book_id) == "second"
    assert db.get_book(conn, book_id)["book_summary_method"] == "chapter_summaries"


def test_delete_summaries_keeps_passages(conn):
    book_id = db.save_extracted_book(conn, make_book(chapters=2))
    passages = len(chunk_ids(conn, book_id))
    summarize_fully(conn, book_id, 2)
    db.delete_summaries(conn, book_id)
    assert db.get_chapter_summaries(conn, book_id) == {}
    assert db.get_book_summary(conn, book_id) is None
    assert db.get_book(conn, book_id)["summarized_at"] is None
    assert len(chunk_ids(conn, book_id)) == passages


def test_api_call_stats_average_recent_calls_per_setting(conn):
    for out in (800, 900, 1000):
        db.log_api_call(conn, None, "chapter_summary", "sonnet", "medium", "v4", 4000, out)
    db.log_api_call(conn, None, "chapter_summary", "sonnet", "high", "v4", 4000, 5000)  # other effort
    db.log_api_call(conn, None, "chapter_summary", "sonnet", "medium", "v3", 4000, 5000)  # old prompt
    db.log_api_call(conn, None, "condense", "sonnet", "medium", "v4", 1200, 600)
    stats = db.api_call_stats(conn, "sonnet", "medium", "v4")
    assert stats["chapter_summary"]["n"] == 3
    assert float(stats["chapter_summary"]["avg_out"]) == 900
    assert stats["condense"]["n"] == 1
    assert float(db.api_call_stats(conn, "sonnet", "medium", "v4", recent=2)["chapter_summary"]["avg_out"]) == 950


# --------------------------------------------------------------------------- #
# Embeddings and vector search
# --------------------------------------------------------------------------- #

def test_embedding_bookkeeping(conn):
    book_id = db.save_extracted_book(conn, make_book(chapters=1))
    pending = db.pending_embeddings(conn, [book_id], MODEL)
    assert pending and all(r["title"] == "Wayward Book" for r in pending)
    db.save_embeddings(conn, [(pending[0]["id"], Vector(axis(0)))], MODEL)
    assert not db.mark_embedded(conn, book_id, MODEL)  # some still missing
    embed_all(conn, book_id)
    assert db.pending_embeddings(conn, [book_id], MODEL) == []
    assert db.mark_embedded(conn, book_id, MODEL)
    assert db.get_book(conn, book_id)["embedded_at"] is not None


@pytest.fixture
def two_books(conn):
    """Two books whose chunks have handmade vectors, so search results are predictable."""
    a = db.save_extracted_book(conn, make_book("Book A", chapters=1))
    b = db.save_extracted_book(conn, make_book("Book B", chapters=1, path="/library/b/B.epub"))
    db.save_chapter_summary(conn, a, 1, "Chapter 1", "summary of A", 3, "m")
    pa, pb = chunk_ids(conn, a), chunk_ids(conn, b)
    summary_a = chunk_ids(conn, a, "chapter_summary")[0]
    db.save_embeddings(conn, [
        (pa[0], Vector(axis(0))),                     # exact match for a query along axis 0
        (pa[1], Vector(blend({0: 0.8, 1: 0.6}))),     # close
        (pb[0], Vector(blend({0: 0.6, 1: 0.8}))),     # further, in book B
        (summary_a, Vector(blend({0: 0.95, 2: 0.31}))),
    ], MODEL)
    db.save_embeddings(conn, [(pb[1], Vector(axis(0)))], "other-model")  # identical vector, other model
    return {"a": a, "b": b, "pa": pa, "pb": pb, "summary_a": summary_a}


def search(conn, levels=("passage",), book_id=None, k=8, model=MODEL):
    return db.search_chunks(conn, Vector(axis(0)), model, list(levels), book_id, k)


def test_search_orders_by_cosine_similarity(conn, two_books):
    hits = search(conn)
    assert [h["id"] for h in hits] == [two_books["pa"][0], two_books["pa"][1], two_books["pb"][0]]
    assert hits[0]["similarity"] == pytest.approx(1.0)
    assert hits[1]["similarity"] == pytest.approx(0.8)
    assert hits[0]["book_title"] == "Book A" and hits[0]["chapter_title"] == "Chapter 1"


def test_search_ignores_vectors_from_other_models(conn, two_books):
    assert two_books["pb"][1] not in [h["id"] for h in search(conn)]
    assert [h["id"] for h in search(conn, model="other-model")] == [two_books["pb"][1]]


def test_search_filters_by_book_level_and_k(conn, two_books):
    assert {h["book_id"] for h in search(conn, book_id=two_books["b"])} == {two_books["b"]}
    assert [h["id"] for h in search(conn, levels=["chapter_summary"])] == [two_books["summary_a"]]
    both = search(conn, levels=["passage", "chapter_summary"])
    assert both[0]["id"] == two_books["pa"][0] and both[1]["id"] == two_books["summary_a"]
    assert len(search(conn, k=1)) == 1


def test_searchable_books_are_those_with_current_embeddings(conn, two_books):
    assert [b["title"] for b in db.searchable_books(conn, MODEL)] == ["Book A", "Book B"]
    assert [b["title"] for b in db.searchable_books(conn, "unused-model")] == []
