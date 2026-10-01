"""The whole pipeline, end to end: EPUB → extract → summarize → embed → agent
tool. Everything is real except the external services: Claude is a scripted
fake client and Voyage is a fake embedder."""

import dataclasses

import pytest

import db
import embed
import summarize as S
from extract import extract_book
from tests.conftest import Doc, EpubSpec, paragraphs
from tests.fakes import FakeClient, msg, text, thinking
from tests.helpers import FakeEmbedder
from tools import ToolRunner

pytestmark = pytest.mark.db


def summary(n):
    return (f"### Summary\nChapter {n} summary.\n\n### Characters\n- Fern\n\n### Motifs and recurring images\n"
            f"- clocks\n\n### Unresolved threads\n- the baby\n\n### Tone\nWry.")


def test_epub_to_searchable_book(conn, make_epub, monkeypatch):
    # 1. Extract a small book with a part divider and front matter.
    spec = EpubSpec(
        title="Wayward Book",
        docs=[Doc("copyright.xhtml", "<p>All rights reserved. ISBN 978-0-00-000000-0.</p>"),
              Doc("part1.xhtml", "<h2>26 WEEKS</h2>"),
              Doc("ch1.xhtml", "<h2>Chapter 1</h2>" + paragraphs(4, seed=1)),
              Doc("ch2.xhtml", "<h2>Chapter 2</h2><p>Hagar carries the bucket of white eels downstairs.</p>"
                               + paragraphs(4, seed=2))],
        toc=[("Copyright", "copyright.xhtml"), ("26 WEEKS", "part1.xhtml"),
             ("Chapter 1", "ch1.xhtml"), ("Chapter 2", "ch2.xhtml")],
    )
    book = extract_book(make_epub(spec))
    assert [c.title for c in book.chapters] == ["Chapter 1 (26 WEEKS)", "Chapter 2 (26 WEEKS)"]
    book_id = db.save_extracted_book(conn, book)
    assert db.get_book(conn, book_id)["stage"] == "extracted"

    # 2. Summarize with a scripted Claude: two chapter summaries, then the book summary.
    client = FakeClient([msg("end_turn", thinking(), text(summary(1))),
                         msg("end_turn", thinking(), text(summary(2))),
                         msg("end_turn", thinking(), text("### Premise\nA home for unwed mothers."))])
    monkeypatch.setattr(S, "count_tokens", lambda client, text: 9_000)  # the free count_tokens call
    plans = S.plan(conn, client, [book_id])
    assert len(plans[0].pending_chapters) == 2 and plans[0].method == "full_text"
    assert plans[0].est_cost > 0
    assert S.execute(client, plans)

    requests = client.messages.requests
    assert len(requests) == 3
    assert "previous_chapter_summary" not in requests[0]["messages"][0]["content"]
    assert summary(1) in requests[1]["messages"][0]["content"]  # chapter 2 sees chapter 1's summary
    assert "Hagar carries the bucket" in requests[2]["messages"][0]["content"]  # full-text book summary
    assert db.get_book(conn, book_id)["stage"] == "summarized"
    assert db.get_book(conn, book_id)["book_summary_method"] == "full_text"
    logged = conn.execute("SELECT purpose, count(*) AS n FROM api_calls GROUP BY 1 ORDER BY 1").fetchall()
    assert [(r["purpose"], r["n"]) for r in logged] == [("book_summary", 1), ("chapter_summary", 2)]

    # 3. Embed with a fake Voyage.
    fake = FakeEmbedder()
    for module in (db, embed):
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, embed_model=fake.model_name))
    monkeypatch.setattr(embed, "get_embedder", lambda: fake)
    assert embed.execute(conn, [book_id])
    assert db.get_book(conn, book_id)["stage"] == "embedded"

    # 4. The agent's search tool finds the passage, cited by chapter title.
    result = ToolRunner(conn=conn, embedder=fake).run("search_passages", {"query": "bucket of white eels", "k": 1})
    assert '[1] Wayward Book, chapter "Chapter 2 (26 WEEKS)"' in result.text
    assert "Hagar carries the bucket of white eels" in result.text
