"""HTTP API for Chapter and Verse.

    uvicorn api:app --port 8080        # or: docker compose up

Everything except /health and the web page requires `Authorization: Bearer <key>`
with a key from API_KEYS. Spending endpoints (/ask, indexing) are capped by
DAILY_BUDGET_USD, and indexing additionally needs an explicit max_cost_usd that
covers the estimate: the API's version of the CLI's y/N.

When deployed behind Cloudflare Access (CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD
set), every request except /health must also carry a valid Access JWT, so the
service's own URL can't be used to go around the login.

The process is stateless (conversations and jobs live in Postgres), so it can
run as several instances, e.g. on Cloud Run. Interactive docs: /docs.
"""

from __future__ import annotations

import hmac
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import anthropic
import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

import db
import jobs
from agent import Agent
from config import settings
from embed import get_embedder
from tools import MAX_K, ToolRunner

log = logging.getLogger("chapter_and_verse.api")
STATIC_DIR = Path(__file__).parent / "static"
KEEPALIVE_SECONDS = 15  # idle time before a streamed answer sends a keep-alive comment


# --------------------------------------------------------------------------- #
# Resources (replaceable in tests)
# --------------------------------------------------------------------------- #

class Database:
    """Hands out connections from a pool (opened lazily, so importing the app is cheap)."""

    def __init__(self, url: str):
        self.url = url
        self._pool = None
        self._lock = threading.Lock()

    @contextmanager
    def connection(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            with self._lock:
                if self._pool is None:
                    self._pool = ConnectionPool(self.url, min_size=1, max_size=settings.db_pool_max, open=True,
                                                kwargs={"row_factory": dict_row, "autocommit": True},
                                                configure=register_vector)
        with self._pool.connection() as conn:
            yield conn

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()


class Services:
    """External services the API uses. Tests swap these for fakes."""

    def __init__(self):
        self.database = Database(settings.database_url)
        self._client = None
        self._embedder = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder


def check_auth_config() -> None:
    if not settings.api_keys and not settings.auth_disabled:
        raise RuntimeError("Set API_KEYS (comma-separated) or, for local development only, AUTH_DISABLED=1")


class AccessVerifier:
    """Checks the JWT that Cloudflare Access adds to every request it lets through
    (the Cf-Access-Jwt-Assertion header), for a person who logged in or a service
    token. Signed RS256 with the team's rotating keys; `aud` must be this
    application's tag and `iss` the team domain."""

    def __init__(self, team_domain: str, audience: str, jwks_client=None):
        team_domain = team_domain.removeprefix("https://").rstrip("/")
        self.issuer = f"https://{team_domain}"
        self.audience = audience
        # Keys are fetched once and cached; an unknown key id triggers a refetch (rotation).
        self.jwks = jwks_client or jwt.PyJWKClient(f"{self.issuer}/cdn-cgi/access/certs", cache_keys=True,
                                                   lifespan=3600, timeout=5)

    def verify(self, token: str) -> dict:
        key = self.jwks.get_signing_key_from_jwt(token)
        return jwt.decode(token, key.key, algorithms=["RS256"], audience=self.audience, issuer=self.issuer,
                          options={"require": ["exp", "iat", "aud", "iss"]})


def access_verifier() -> AccessVerifier | None:
    """The Access check, if configured. Half a configuration is an error, not "off"."""
    domain, aud = settings.cf_access_team_domain, settings.cf_access_aud
    if bool(domain) != bool(aud):
        raise RuntimeError("Set both CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD (or neither, when not behind Access)")
    return AccessVerifier(domain, aud) if domain else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_auth_config()
    app.state.access = access_verifier()
    jobs.get_runner()  # a misconfigured JOB_RUNNER fails at startup, not at the first indexing request
    log.info("Cloudflare Access check: %s; job runner: %s",
             "on" if app.state.access else "off", settings.job_runner)
    with app.state.services.database.connection() as conn:
        db.ensure_schema(conn)
    yield
    app.state.services.database.close()


app = FastAPI(title="Chapter and Verse", version="1.0",
              description="Ask questions about your ebook library; answers cite the book and chapter.",
              lifespan=lifespan)
app.state.services = Services()
app.state.access = None  # set at startup from CF_ACCESS_*; None = not behind Cloudflare Access
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def cloudflare_access(request: Request, call_next):
    """Behind Cloudflare Access, reject anything that didn't come through it. /health is
    exempt: Cloud Run's probes reach the container directly. API keys are still checked
    afterwards, per endpoint."""
    verifier = request.app.state.access
    if verifier is None or request.url.path == "/health":
        return await call_next(request)
    token = request.headers.get("cf-access-jwt-assertion")
    if not token:
        return JSONResponse({"detail": "missing Cloudflare Access token"}, status.HTTP_403_FORBIDDEN)
    try:
        await run_in_threadpool(verifier.verify, token)  # may fetch keys over the network
    except jwt.PyJWTError as e:
        log.warning("rejected Cloudflare Access token: %s", type(e).__name__)
        return JSONResponse({"detail": "invalid Cloudflare Access token"}, status.HTTP_403_FORBIDDEN)
    return await call_next(request)


# Registered last, so it's the outermost middleware and also logs requests rejected above.
@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    # Method, path, status, and timing only: never headers (keys) or bodies (book text).
    log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


def services(request: Request) -> Services:
    return request.app.state.services


def require_key(request: Request) -> None:
    if settings.auth_disabled:
        return
    header = request.headers.get("authorization", "")
    scheme, _, key = header.partition(" ")
    if scheme.lower() != "bearer" or not any(hmac.compare_digest(key, k) for k in settings.api_keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})


def require_budget(conn, extra: float = 0.0) -> None:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    spent = db.spend_since(conn, today)
    if spent + extra >= settings.daily_budget_usd:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"daily budget reached: ${spent:.2f} spent today (UTC) of "
                            f"${settings.daily_budget_usd:.2f}" + (f"; this would add ${extra:.2f}" if extra else ""))


authed = [Depends(require_key)]


def book_or_404(conn, book_id: int) -> dict:
    book = db.get_book(conn, book_id)
    if book is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no book with id {book_id}")
    return book


def book_json(b: dict) -> dict:
    return {"id": b["id"], "title": b["title"], "authors": b["authors"], "series": b["series"],
            "series_index": float(b["series_index"]) if b["series_index"] is not None else None,
            "stage": b["stage"], "chapters": b["chapter_count"],
            "chapter_summaries": b["chapter_summaries"],
            "tokens": b["exact_token_count"] or b["token_count"], "tokens_exact": b["exact_token_count"] is not None}


# --------------------------------------------------------------------------- #
# Health and web page
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["service"])
def health(svc: Services = Depends(services)):
    try:
        with svc.database.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"database unavailable: {type(e).__name__}")
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def web_page():
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #

@app.get("/books", dependencies=authed, tags=["library"])
def list_books(q: str | None = None, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        return [book_json(b) for b in db.list_books(conn, q)]


@app.get("/books/{book_id}", dependencies=authed, tags=["library"])
def get_book(book_id: int, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        book = book_or_404(conn, book_id)
        chapters = db.get_chapters(conn, book_id)
        summarized = db.get_chapter_summaries(conn, book_id)
    return {**book_json(book), "book_summary_method": book["book_summary_method"],
            "chapter_list": [{"index": c["chapter_index"], "title": c["title"], "tokens": c["token_count"],
                              "summarized": c["chapter_index"] in summarized} for c in chapters]}


@app.get("/books/{book_id}/summary", dependencies=authed, tags=["library"])
def book_summary(book_id: int, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        book = book_or_404(conn, book_id)
        summary = db.get_book_summary(conn, book_id)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{book['title']} has no book summary yet")
    return {"book_id": book_id, "title": book["title"], "method": book["book_summary_method"], "summary": summary}


@app.get("/books/{book_id}/chapters/{chapter}/summary", dependencies=authed, tags=["library"])
def chapter_summary(book_id: int, chapter: int, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        book = book_or_404(conn, book_id)
        summaries = db.get_chapter_summaries(conn, book_id)
        chapters = {c["chapter_index"]: c["title"] for c in db.get_chapters(conn, book_id)}
    if chapter not in summaries:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no summary for chapter {chapter} of {book['title']}")
    return {"book_id": book_id, "chapter": chapter, "title": chapters.get(chapter), "summary": summaries[chapter]}


@app.post("/library/scan", dependencies=authed, tags=["library"])
def scan_library(svc: Services = Depends(services)):
    """Extract new or changed EPUBs from LIBRARY_PATH (free). Never discards paid work."""
    with svc.database.connection() as conn:
        results = jobs.scan_library(conn)
    return {"results": results,
            "counts": {s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}}}


# --------------------------------------------------------------------------- #
# Search (no LLM)
# --------------------------------------------------------------------------- #

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    level: Literal["passage", "chapter_summary", "book_summary"] | None = None
    book_id: int | None = None
    k: int = Field(default=8, ge=1, le=MAX_K)


@app.post("/search", dependencies=authed, tags=["search"])
def search(req: SearchRequest, svc: Services = Depends(services)):
    """Vector search over passages and/or summaries. Returns raw matches, no LLM involved."""
    with svc.database.connection() as conn:
        if req.book_id is not None and book_or_404(conn, req.book_id)["stage"] != "embedded":
            raise HTTPException(status.HTTP_409_CONFLICT, f"book {req.book_id} isn't indexed yet")
        levels = [req.level] if req.level else ["passage", "chapter_summary", "book_summary"]
        vec = Vector(svc.embedder.embed_query(req.query))
        hits = db.search_chunks(conn, vec, svc.embedder.model_name, levels, req.book_id, req.k)
    top = hits[0]["similarity"] if hits else None
    return {
        "query": req.query,
        "weak": top is None or top < settings.weak_match_threshold,
        "results": [{"book_id": h["book_id"], "book_title": h["book_title"], "level": h["level"],
                     "chapter_index": h["chapter_index"], "chapter_title": h["chapter_title"],
                     "similarity": round(float(h["similarity"]), 4), "content": h["content"]} for h in hits],
    }


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    stream: bool = False


def _prepare_conversation(conn, req: AskRequest) -> tuple[str, list[dict]]:
    if req.conversation_id:
        convo = db.get_conversation(conn, req.conversation_id)
        if convo is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such conversation")
        return str(convo["id"]), list(convo["messages"])
    return db.create_conversation(conn, req.question), []


def _run_agent(svc: Services, conversation_id: str, messages: list[dict], question: str, on_event=None):
    """Ask in a dedicated connection; store the conversation only if the question completed."""
    with svc.database.connection() as conn:
        def log_usage(u, cost):
            db.log_api_call(conn, None, "agent", settings.agent_model, settings.agent_effort, "agent",
                            u.input_tokens or 0, u.output_tokens or 0, cost_usd=cost,
                            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)

        agent = Agent(client=svc.client, tools=ToolRunner(conn=conn, embedder=svc.embedder),
                      messages=messages, on_usage=log_usage)
        result = agent.ask(question, on_event=on_event)
        db.save_conversation_messages(conn, conversation_id, agent.messages)
        return result


def _result_json(conversation_id: str, result) -> dict:
    return {"conversation_id": conversation_id, "answer": result.answer, "stop": result.stop,
            "tool_calls": result.tool_calls, "usage": result.usage.as_dict()}


@app.post("/ask", dependencies=authed, tags=["agent"])
def ask(req: AskRequest, svc: Services = Depends(services)):
    """Ask the agent. JSON by default; with stream=true, Server-Sent Events as it works."""
    with svc.database.connection() as conn:
        require_budget(conn)
        conversation_id, messages = _prepare_conversation(conn, req)

    if not req.stream:
        try:
            return _result_json(conversation_id, _run_agent(svc, conversation_id, messages, req.question))
        except anthropic.APIError as e:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"model API error: {type(e).__name__}")

    events: queue.Queue = queue.Queue()
    done = object()

    def worker():
        try:
            _run_agent(svc, conversation_id, messages, req.question, on_event=events.put)
        except Exception as e:
            events.put({"type": "error", "message": f"{type(e).__name__}: question not saved, try again"})
        finally:
            events.put(done)

    threading.Thread(target=worker, daemon=True).start()

    def sse():
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
        while True:
            try:
                event = events.get(timeout=KEEPALIVE_SECONDS)
            except queue.Empty:
                # An SSE comment: clients ignore it, but proxies (Cloudflare) see a live
                # connection while the model thinks for a long time between events.
                yield ": keepalive\n\n"
                continue
            if event is done:
                break
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _turns(messages: list[dict]) -> list[dict]:
    """Questions and final answers from a stored history (tool traffic omitted)."""
    turns = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], str):
            turns.append({"question": m["content"], "answer": ""})
        elif m["role"] == "assistant" and turns:
            text = "\n".join(b.get("text", "") for b in m["content"] if b.get("type") == "text").strip()
            if text:
                turns[-1]["answer"] = text
    return turns


@app.get("/conversations", dependencies=authed, tags=["agent"])
def list_conversations(limit: int = 20, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        rows = conn.execute("SELECT id, title, created_at, updated_at FROM conversations "
                            "ORDER BY updated_at DESC LIMIT %s", (min(max(limit, 1), 100),)).fetchall()
    return [{"id": str(r["id"]), "title": r["title"], "updated_at": r["updated_at"].isoformat()} for r in rows]


@app.get("/conversations/{conversation_id}", dependencies=authed, tags=["agent"])
def get_conversation(conversation_id: str, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        convo = db.get_conversation(conn, conversation_id)
    if convo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such conversation")
    return {"id": str(convo["id"]), "title": convo["title"], "turns": _turns(convo["messages"])}


# --------------------------------------------------------------------------- #
# Indexing (paid)
# --------------------------------------------------------------------------- #

@app.get("/books/{book_id}/estimate", dependencies=authed, tags=["indexing"])
def estimate(book_id: int, svc: Services = Depends(services)):
    """What finishing this book would cost. Free (uses count_tokens)."""
    with svc.database.connection() as conn:
        book_or_404(conn, book_id)
        return jobs.estimate_book(conn, svc.client, book_id).as_dict()


class IndexRequest(BaseModel):
    max_cost_usd: float = Field(gt=0, description="Refuse to start if the estimate is higher than this.")


@app.post("/books/{book_id}/index", status_code=status.HTTP_202_ACCEPTED, dependencies=authed, tags=["indexing"])
def index_book(book_id: int, req: IndexRequest, svc: Services = Depends(services)):
    """Start summarizing + embedding a book, if the estimate is within max_cost_usd."""
    with svc.database.connection() as conn:
        book = book_or_404(conn, book_id)
        est = jobs.estimate_book(conn, svc.client, book_id)
        if est.nothing_to_do:
            return {"job_id": None, "message": f"{book['title']} is already fully indexed", "estimate": est.as_dict()}
        if est.total > req.max_cost_usd:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                f"estimated ${est.total:.2f} exceeds max_cost_usd ${req.max_cost_usd:.2f}")
        require_budget(conn, extra=est.total)
        try:
            job_id = db.create_job(conn, book_id, est.as_dict(), req.max_cost_usd)
        except db.JobAlreadyRunning as e:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e))
        try:
            jobs.get_runner().start(job_id)
        except jobs.JobStartError as e:
            # Don't leave it queued forever (it would also block new jobs for this book).
            db.update_job(conn, job_id, status="failed", error=str(e)[:1000])
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"job {job_id} couldn't be started: {e}")
    return {"job_id": job_id, "estimate": est.as_dict()}


@app.get("/jobs/{job_id}", dependencies=authed, tags=["indexing"])
def get_job(job_id: int, svc: Services = Depends(services)):
    with svc.database.connection() as conn:
        job = db.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}")
    return {"id": job["id"], "book_id": job["book_id"], "status": job["status"], "progress": job["progress"],
            "error": job["error"], "estimate": job["estimate"], "max_cost_usd": float(job["max_cost_usd"]),
            "cost_usd": float(job["cost_usd"]) if job["cost_usd"] is not None else None,
            "created_at": job["created_at"].isoformat(), "updated_at": job["updated_at"].isoformat()}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
