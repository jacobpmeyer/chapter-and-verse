"""tools.py end to end: the agent's tools against the test database, with a
fake embedder standing in for Voyage."""

import dataclasses

import pytest

import db
import embed
import tools as T
from tests.helpers import FakeEmbedder, make_book

pytestmark = pytest.mark.db

WAYWARD = [
    "Fern arrives at Wellwood House in May and Miss Wellwood renames her. " * 6,
    "The girls cast the Turnabout spell with an egg and lemon peel, and Dr. Vincent vomits. " * 6,
    "Hagar carries a bucket of white eels down the stairs after the birth. " * 6,
]


@pytest.fixture
def library(conn, monkeypatch):
    """Two books: one fully indexed (summaries + fake embeddings), one only extracted."""
    fake = FakeEmbedder()
    for module in (db, embed):
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, embed_model=fake.model_name))
    monkeypatch.setattr(embed, "get_embedder", lambda: fake)

    indexed = db.save_extracted_book(conn, make_book("Witchcraft for Wayward Girls", authors=("Grady Hendrix",),
                                                     texts=WAYWARD))
    for i, theme in enumerate(["arrival and renaming", "the first spell", "eels and the birth"], 1):
        db.save_chapter_summary(conn, indexed, i, f"Chapter {i}", f"### Summary\nThis chapter covers {theme}.", 9, "m")
    db.save_book_summary(conn, indexed, "### Premise\nGirls at a home for unwed mothers find a witchcraft book.",
                         12, "m", "full_text")
    db.set_exact_token_count(conn, indexed, 1_000)
    assert embed.execute(conn, [indexed])

    unindexed = db.save_extracted_book(conn, make_book("The Devils", authors=("Joe Abercrombie",),
                                                       path="/library/joe/Devils.epub"))
    return {"runner": T.ToolRunner(conn=conn, embedder=fake), "indexed": indexed, "unindexed": unindexed,
            "fake": fake}


def test_list_books_reports_stages(library):
    result = library["runner"].run("list_books", {})
    assert "Witchcraft for Wayward Girls by Grady Hendrix" in result.text
    assert "stage: embedded" in result.text and "stage: extracted" in result.text
    assert result.summary == "2 books (1 searchable)"
    assert "The Devils" not in library["runner"].run("list_books", {"query": "hendrix"}).text


def test_passage_search_finds_the_right_chapter(library):
    result = library["runner"].run("search_passages", {"query": "white eels in a bucket"})
    first = result.text.split("\n[1] ")[1]
    assert first.startswith('Witchcraft for Wayward Girls, chapter "Chapter 3"')
    assert "Searched passages in: Witchcraft for Wayward Girls (id" in result.text
    assert "The Devils" not in result.text  # not indexed, so not searched
    assert library["fake"].queries == ["white eels in a bucket"]


def test_summary_search_can_be_limited_to_book_summaries(library):
    result = library["runner"].run("search_summaries", {"query": "witchcraft book", "level": "book_summary"})
    assert "(book summary, similarity" in result.text
    assert "chapter summary" not in result.text


def test_k_is_clamped(library):
    result = library["runner"].run("search_passages", {"query": "girls", "k": 500})
    assert result.text.count("\n[") <= T.MAX_K


def test_an_unrelated_query_is_flagged_as_weak(library):
    result = library["runner"].run("search_passages", {"query": "spaceship asteroid quantum", "k": 2})
    assert "WEAK RETRIEVAL" in result.text


def test_searching_an_unindexed_book_explains_how_to_index_it(library):
    with pytest.raises(T.ToolError, match=f"python index.py book {library['unindexed']}"):
        library["runner"].run("search_passages", {"query": "anything", "book_id": library["unindexed"]})


def test_book_summary_and_its_absence(library):
    ok = library["runner"].run("get_book_summary", {"book_id": library["indexed"]})
    assert "home for unwed mothers" in ok.text
    with pytest.raises(T.ToolError, match="no book summary yet"):
        library["runner"].run("get_book_summary", {"book_id": library["unindexed"]})
    with pytest.raises(T.ToolError, match="No book with id 999"):
        library["runner"].run("get_book_summary", {"book_id": 999})


def test_load_full_book_within_the_limit(library):
    result = library["runner"].run("load_full_book", {"book_id": library["indexed"]})
    assert result.text.count("## Chapter ") >= 3
    assert "Hagar carries a bucket" in result.text


def test_load_full_book_refuses_books_over_the_limit(library, conn):
    db.set_exact_token_count(conn, library["indexed"], 10_000_000)
    with pytest.raises(T.ToolError, match="over the .*-token limit"):
        library["runner"].run("load_full_book", {"book_id": library["indexed"]})


def test_load_full_book_estimates_when_no_exact_count(library, monkeypatch):
    # The Devils has no exact count: the local estimate is scaled up before checking.
    monkeypatch.setattr(T, "settings", dataclasses.replace(T.settings, full_book_token_limit=1))
    with pytest.raises(T.ToolError, match="The Devils"):
        library["runner"].run("load_full_book", {"book_id": library["unindexed"]})


def test_unknown_tool(library):
    with pytest.raises(T.ToolError, match="Unknown tool"):
        library["runner"].run("delete_everything", {})
