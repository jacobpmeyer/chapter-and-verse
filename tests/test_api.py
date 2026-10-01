"""The HTTP API, end to end through FastAPI's TestClient, against the test
database, with Claude scripted and Voyage faked. No network, no spend."""

from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import api
import db
import embed
import jobs
import summarize as S
from tests.conftest import Doc, EpubSpec, paragraphs
from tests.fakes import FakeClient, msg, text, thinking, tool
from tests.helpers import FakeEmbedder, make_book

pytestmark = pytest.mark.db

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
TEXTS = ["Fern arrives at Wellwood House and is renamed. " * 6,
         "The girls cast the Turnabout spell with an egg. " * 6,
         "Hagar carries a bucket of white eels downstairs. " * 6]


class FakeDatabase:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connection(self):
        yield self.conn

    def close(self):
        pass


class FakeServices:
    def __init__(self, conn, script=()):
        self.database = FakeDatabase(conn)
        self.client = FakeClient(list(script))
        self.embedder = FakeEmbedder()


@pytest.fixture
def configure(monkeypatch):
    """Patch settings everywhere they're read."""
    def apply(**changes):
        for module in (api, db, embed, jobs, S):
            monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, **changes))
    apply(api_keys=(KEY,), auth_disabled=False, daily_budget_usd=5.0, embed_model=FakeEmbedder.model_name)
    return apply


@pytest.fixture
def make_client(conn, configure, monkeypatch):
    """A TestClient whose services are fakes; `script` is what the fake Claude will reply."""
    monkeypatch.setattr(S, "count_tokens", lambda client, text: 5_000)
    monkeypatch.setattr(embed, "get_embedder", lambda: FakeEmbedder())

    opened = []

    def build(script=()):
        services = FakeServices(conn, script)
        monkeypatch.setattr(api.app.state, "services", services)
        monkeypatch.setattr(jobs, "_runner_override", jobs.InlineRunner(services.client))
        client = TestClient(api.app)
        client.__enter__()  # runs startup (auth config check, schema)
        client.services = services
        opened.append(client)
        return client

    yield build
    for client in opened:
        client.__exit__(None, None, None)


@pytest.fixture
def indexed_book(conn):
    """A fully indexed book, embedded with the fake embedder."""
    book_id = db.save_extracted_book(conn, make_book("Witchcraft for Wayward Girls", authors=("Grady Hendrix",),
                                                     texts=TEXTS))
    for i in range(1, 4):
        db.save_chapter_summary(conn, book_id, i, f"Chapter {i}", f"### Summary\nChapter {i} events.", 5, "m")
    db.save_book_summary(conn, book_id, "### Premise\nGirls at a home find a witchcraft book.", 9, "m", "full_text")
    fake = FakeEmbedder()
    rows = db.pending_embeddings(conn, [book_id], fake.model_name)
    from pgvector import Vector
    db.save_embeddings(conn, [(r["id"], Vector(fake.embed_documents([embed.embedding_text(r)])[0])) for r in rows],
                       fake.model_name)
    return book_id


# --------------------------------------------------------------------------- #
# Service and auth
# --------------------------------------------------------------------------- #

def test_health_needs_no_key(make_client):
    assert make_client().get("/health").json() == {"status": "ok"}


def test_web_page_is_served_without_a_key(make_client):
    r = make_client().get("/")
    assert r.status_code == 200 and "Chapter and Verse" in r.text


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY}])
def test_endpoints_require_a_valid_key(make_client, headers):
    r = make_client().get("/books", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_server_refuses_to_start_without_keys(configure):
    configure(api_keys=(), auth_disabled=False)
    with pytest.raises(RuntimeError, match="API_KEYS"):
        api.check_auth_config()
    configure(api_keys=(), auth_disabled=True)
    api.check_auth_config()  # explicitly disabled for local development


# --------------------------------------------------------------------------- #
# Library and search
# --------------------------------------------------------------------------- #

def test_books_and_summaries(make_client, indexed_book):
    c = make_client()
    books = c.get("/books", headers=AUTH).json()
    assert [(b["id"], b["stage"], b["chapters"]) for b in books] == [(indexed_book, "embedded", 3)]
    detail = c.get(f"/books/{indexed_book}", headers=AUTH).json()
    assert [ch["summarized"] for ch in detail["chapter_list"]] == [True, True, True]
    assert "witchcraft book" in c.get(f"/books/{indexed_book}/summary", headers=AUTH).json()["summary"]
    assert c.get(f"/books/{indexed_book}/chapters/2/summary", headers=AUTH).json()["title"] == "Chapter 2"
    assert c.get("/books/999", headers=AUTH).status_code == 404
    assert c.get(f"/books/{indexed_book}/chapters/9/summary", headers=AUTH).status_code == 404


def test_search_returns_ranked_matches(make_client, indexed_book):
    r = make_client().post("/search", headers=AUTH, json={"query": "bucket of white eels", "level": "passage", "k": 2})
    body = r.json()
    assert r.status_code == 200 and not body["weak"]
    assert body["results"][0]["chapter_title"] == "Chapter 3"
    assert body["results"][0]["similarity"] >= body["results"][1]["similarity"]


def test_search_validates_input(make_client, indexed_book, conn):
    c = make_client()
    assert c.post("/search", headers=AUTH, json={"query": "x", "k": 500}).status_code == 422
    assert c.post("/search", headers=AUTH, json={"query": ""}).status_code == 422
    other = db.save_extracted_book(conn, make_book("Unindexed", path="/library/u/U.epub"))
    assert c.post("/search", headers=AUTH, json={"query": "x", "book_id": other}).status_code == 409


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

def test_ask_answers_and_conversations_continue(make_client, indexed_book):
    c = make_client([
        msg("tool_use", thinking(), tool("t1", "search_passages", query="eels")),
        msg("end_turn", text("Hagar carries the eels (Witchcraft for Wayward Girls, \"Chapter 3\").")),
        msg("end_turn", text("She carries them downstairs.")),
    ])
    first = c.post("/ask", headers=AUTH, json={"question": "Who carries the eels?"}).json()
    assert first["answer"].startswith("Hagar carries the eels")
    assert [t["name"] for t in first["tool_calls"]] == ["search_passages"]
    assert first["usage"]["calls"] == 2 and first["usage"]["cost_usd"] > 0

    second = c.post("/ask", headers=AUTH, json={"question": "Where to?",
                                                 "conversation_id": first["conversation_id"]}).json()
    assert second["conversation_id"] == first["conversation_id"]
    sent = c.services.client.messages.requests[-1]["messages"]
    assert sent[0]["content"] == "Who carries the eels?" and sent[-1]["content"] == "Where to?"

    convo = c.get(f"/conversations/{first['conversation_id']}", headers=AUTH).json()
    assert [t["question"] for t in convo["turns"]] == ["Who carries the eels?", "Where to?"]
    assert convo["turns"][1]["answer"] == "She carries them downstairs."
    assert c.get("/conversations", headers=AUTH).json()[0]["id"] == first["conversation_id"]


def test_agent_spend_is_logged(make_client, indexed_book, conn):
    make_client([msg("end_turn", text("Hi."))]).post("/ask", headers=AUTH, json={"question": "hi"})
    row = conn.execute("SELECT purpose, cost_usd, cache_read_tokens FROM api_calls").fetchone()
    assert row["purpose"] == "agent" and float(row["cost_usd"]) > 0 and row["cache_read_tokens"] == 1000


def test_ask_streams_server_sent_events(make_client, indexed_book):
    c = make_client([msg("tool_use", tool("t1", "list_books")), msg("end_turn", text("Two books."))])
    with c.stream("POST", "/ask", headers=AUTH, json={"question": "What's here?", "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        frames = [f for f in r.read().decode().split("\n\n") if f]
    events = [(f.split("\n")[0][len("event: "):], json.loads(f.split("\n")[1][len("data: "):])) for f in frames]
    kinds = [k for k, _ in events]
    assert kinds[0] == "conversation" and kinds[-1] == "done"
    assert "tool_call" in kinds and "tool_result" in kinds
    assert "".join(e["text"] for k, e in events if k == "text") == "Two books."
    convo_id = events[0][1]["conversation_id"]
    assert c.get(f"/conversations/{convo_id}", headers=AUTH).json()["turns"][0]["answer"] == "Two books."


def test_a_failed_question_is_not_saved(make_client, indexed_book):
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    c = make_client([msg("tool_use", tool("t1", "list_books")), error])
    r = c.post("/ask", headers=AUTH, json={"question": "q"})
    assert r.status_code == 502
    convo = c.get("/conversations", headers=AUTH).json()[0]
    assert c.get(f"/conversations/{convo['id']}", headers=AUTH).json()["turns"] == []


def test_unknown_conversation(make_client):
    c = make_client()
    assert c.post("/ask", headers=AUTH, json={"question": "q", "conversation_id": "nope"}).status_code == 404


def test_daily_budget_blocks_spending(make_client, configure, conn):
    configure(daily_budget_usd=0.01)
    db.log_api_call(conn, None, "agent", "m", "high", "agent", 0, 0, cost_usd=0.02)
    r = make_client().post("/ask", headers=AUTH, json={"question": "q"})
    assert r.status_code == 429 and "daily budget" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #

def summaries_script(n_chapters):
    body = "### Summary\nEvents.\n\n### Characters\n- Fern\n\n### Motifs and recurring images\n- clocks\n\n" \
           "### Unresolved threads\n- the baby\n\n### Tone\nWry."
    return [msg("end_turn", text(body)) for _ in range(n_chapters)] + [msg("end_turn", text("### Premise\nA home."))]


def test_index_refuses_when_the_estimate_exceeds_the_limit(make_client, conn):
    book_id = db.save_extracted_book(conn, make_book(chapters=2))
    c = make_client()
    est = c.get(f"/books/{book_id}/estimate", headers=AUTH).json()
    assert est["summaries"]["chapters_to_summarize"] == 2 and est["total_usd"] > 0
    r = c.post(f"/books/{book_id}/index", headers=AUTH, json={"max_cost_usd": est["total_usd"] / 2})
    assert r.status_code == 409 and "exceeds max_cost_usd" in r.json()["detail"]
    assert c.post(f"/books/{book_id}/index", headers=AUTH, json={"max_cost_usd": 0}).status_code == 422
    assert db.get_book(conn, book_id)["stage"] == "extracted"  # nothing ran


def test_index_job_runs_to_completion(make_client, conn):
    book_id = db.save_extracted_book(conn, make_book(chapters=2))
    c = make_client(summaries_script(2))
    started = c.post(f"/books/{book_id}/index", headers=AUTH, json={"max_cost_usd": 10}).json()
    job = c.get(f"/jobs/{started['job_id']}", headers=AUTH).json()
    assert (job["status"], job["error"]) == ("done", None)
    assert job["cost_usd"] > 0 and job["progress"] == "finished"
    assert db.get_book(conn, book_id)["stage"] == "embedded"
    again = c.post(f"/books/{book_id}/index", headers=AUTH, json={"max_cost_usd": 10}).json()
    assert again["job_id"] is None and "already fully indexed" in again["message"]


def test_only_one_active_job_per_book(make_client, conn):
    book_id = db.save_extracted_book(conn, make_book(chapters=1))
    db.create_job(conn, book_id, {"total_usd": 1}, 5)  # queued, not yet picked up
    r = make_client().post(f"/books/{book_id}/index", headers=AUTH, json={"max_cost_usd": 10})
    assert r.status_code == 409 and "already has an indexing job" in r.json()["detail"]


def test_index_respects_the_daily_budget(make_client, configure, conn):
    configure(daily_budget_usd=0.0001)
    book_id = db.save_extracted_book(conn, make_book(chapters=1))
    assert make_client().post(f"/books/{book_id}/index", headers=AUTH, json={"max_cost_usd": 10}).status_code == 429


def test_scan_adds_new_books_and_protects_paid_work(make_client, configure, make_epub, tmp_path, conn):
    spec = EpubSpec(title="Scanned Book", docs=[Doc("ch1.xhtml", "<h2>One</h2>" + paragraphs(3))],
                    toc=[("One", "ch1.xhtml")])
    path = make_epub(spec)
    configure(library_path=tmp_path)
    c = make_client()
    first = c.post("/library/scan", headers=AUTH).json()
    assert first["counts"] == {"added": 1}
    assert c.post("/library/scan", headers=AUTH).json()["counts"] == {"unchanged": 1}

    book_id = first["results"][0]["book_id"]
    db.save_chapter_summary(conn, book_id, 1, "One", "paid summary", 2, "m")
    spec.docs[0] = Doc("ch1.xhtml", "<h2>One</h2>" + paragraphs(4))  # the EPUB changes
    path.unlink()
    make_epub(spec)
    assert c.post("/library/scan", headers=AUTH).json()["counts"] == {"changed_has_paid_work": 1}
    assert db.get_chapter_summaries(conn, book_id) == {1: "paid summary"}  # not discarded
