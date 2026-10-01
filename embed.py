"""Phase 3: embeddings.

One small interface (Embedder) with a Voyage and an OpenAI implementation,
chosen by EMBED_PROVIDER. Each chunk's embedding is stored with the model name
that produced it, so changing EMBED_MODEL marks every chunk as stale and the
next run re-embeds them.

What gets embedded is not just the chunk's content: a one-line header (book,
author, chapter, and for summaries which level it is) goes first, so a query
like "the Morrie chapter about death" can match on the book and chapter too.
The stored `content` stays unchanged.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Callable, Protocol, TypeVar

from pgvector import Vector

import db
from config import EMBED_PRICES, settings
from stage import Progress, StageResult, report
from tokens import estimate_tokens

# Voyage's tokenizer vs our local chars/3.8 estimate, measured on Morrie
# (66,162 actual vs ~74,600 estimated). Used only for the cost estimate.
TOKENIZER_RATIO = 0.9
EXPECTED_SUMMARY_TOKENS = 600  # per summary chunk that doesn't exist yet (for `book` estimates)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #

class EmbeddingConfigError(RuntimeError):
    """An error retrying can't fix (e.g. the account's limits are too low for our batches)."""


def with_backoff(fn: Callable[[], T], retryable: tuple[type[BaseException], ...],
                 rate_limit: tuple[type[BaseException], ...] = (),
                 max_attempts: int = 6, base_delay: float = 2.0, max_delay: float = 60.0) -> T:
    """Call fn, retrying retryable errors with exponential backoff and full jitter.

    The delay before retry n is random in [0, min(max_delay, base_delay * 2**n)].
    Jitter spreads retries out, so bursts of rate-limited requests don't all
    retry at the same instant. Rate-limit errors wait at least 10s, because
    provider limits are per minute and an immediate retry just fails again.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable as e:
            if "payment method" in str(e).lower():
                raise EmbeddingConfigError(
                    "the embedding provider has this account on reduced free-tier rate limits "
                    "(no payment method); add one in the provider's dashboard, then re-run"
                ) from e
            if attempt == max_attempts - 1:
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** attempt))
            if isinstance(e, rate_limit):
                delay = max(delay, 10.0)
            print(f"    {type(e).__name__}: retrying in {delay:.1f}s ({attempt + 1}/{max_attempts - 1})",
                  flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

class Embedder(Protocol):
    model_name: str
    dim: int
    tokens_used: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    """Voyage distinguishes documents from queries (input_type), which improves retrieval."""

    def __init__(self, model: str, dim: int):
        import voyageai
        import voyageai.error as verr

        self.client = voyageai.Client(max_retries=0)  # with_backoff handles retries
        self.model_name, self.dim, self.tokens_used = model, dim, 0
        self.retryable = (verr.RateLimitError, verr.ServiceUnavailableError, verr.ServerError,
                          verr.APIConnectionError, verr.Timeout, verr.TryAgain)
        self.rate_limit = verr.RateLimitError

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        result = with_backoff(
            lambda: self.client.embed(texts, model=self.model_name, input_type=input_type,
                                      output_dimension=self.dim),
            self.retryable,
            rate_limit=(self.rate_limit,),
        )
        self.tokens_used += result.total_tokens
        return result.embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class OpenAIEmbedder:
    def __init__(self, model: str, dim: int):
        try:
            import openai
        except ImportError:
            sys.exit("EMBED_PROVIDER=openai needs `pip install openai`")
        self.client = openai.OpenAI(max_retries=0)  # with_backoff handles retries
        self.model_name, self.dim, self.tokens_used = model, dim, 0
        self.retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                          openai.InternalServerError)
        self.rate_limit = openai.RateLimitError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = with_backoff(
            lambda: self.client.embeddings.create(model=self.model_name, input=texts, dimensions=self.dim),
            self.retryable,
            rate_limit=(self.rate_limit,),
        )
        self.tokens_used += result.usage.total_tokens
        return [d.embedding for d in sorted(result.data, key=lambda d: d.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedder() -> Embedder:
    providers = {"voyage": VoyageEmbedder, "openai": OpenAIEmbedder}
    if settings.embed_provider not in providers:
        sys.exit(f"Unknown EMBED_PROVIDER={settings.embed_provider!r}; use one of {sorted(providers)}")
    return providers[settings.embed_provider](settings.embed_model, settings.embed_dim)


# --------------------------------------------------------------------------- #
# What gets embedded
# --------------------------------------------------------------------------- #

def embedding_text(row: dict) -> str:
    book = f"{row['title']} by {', '.join(row['authors'])}"
    if row["level"] == "book_summary":
        header = f"{book} (book summary)"
    else:
        header = f"{book}, Chapter {row['chapter_index']}: {row['chapter_title']}"
        if row["level"] == "chapter_summary":
            header += " (chapter summary)"
    return f"{header}\n\n{row['content']}"


def batches(rows: list[dict]) -> list[list[dict]]:
    """Group rows by book, at most embed_batch_size texts and embed_batch_max_tokens per request."""
    out: list[list[dict]] = []
    current: list[dict] = []
    tokens = 0
    for row in rows:
        t = estimate_tokens(embedding_text(row))
        new_book = current and current[-1]["book_id"] != row["book_id"]
        if current and (new_book or len(current) >= settings.embed_batch_size
                        or tokens + t > settings.embed_batch_max_tokens):
            out.append(current)
            current, tokens = [], 0
        current.append(row)
        tokens += t
    if current:
        out.append(current)
    return out


# --------------------------------------------------------------------------- #
# Estimate / run
# --------------------------------------------------------------------------- #

def estimate(conn, book_ids: list[int] | None, future_summaries: int = 0) -> tuple[int, int, float]:
    """(chunks to embed, estimated tokens, estimated USD). `future_summaries`
    counts summary chunks that will exist after the summary stage runs."""
    rows = db.pending_embeddings(conn, book_ids, settings.embed_model)
    tokens = sum(estimate_tokens(embedding_text(r)) for r in rows) + future_summaries * EXPECTED_SUMMARY_TOKENS
    tokens = round(tokens * TOKENIZER_RATIO)
    cost = tokens * EMBED_PRICES.get(settings.embed_model, 0.0) / 1e6
    return len(rows) + future_summaries, tokens, cost


def print_estimate(n_chunks: int, tokens: int, cost: float) -> None:
    price = EMBED_PRICES.get(settings.embed_model)
    free = " (voyage-4 models include 200M free tokens per account)" if settings.embed_model.startswith("voyage-4") else ""
    print(f"\nEmbedding model: {settings.embed_provider}/{settings.embed_model} ({settings.embed_dim} dims), "
          f"${price}/MTok{free}")
    print(f"  {n_chunks:,} chunks, ~{tokens:,} tokens ≈ ${cost:.3f}")


def execute(conn, book_ids: list[int] | None, progress: Progress | None = None) -> StageResult:
    """Embed every pending chunk for the given books. The result is truthy on success."""
    embedder = get_embedder()
    rows = db.pending_embeddings(conn, book_ids, embedder.model_name)
    if not rows:
        return StageResult(True, 0.0)
    groups = batches(rows)
    done = 0
    try:
        for i, group in enumerate(groups, 1):
            vectors = embedder.embed_documents([embedding_text(r) for r in group])
            if len(vectors) != len(group) or any(len(v) != embedder.dim for v in vectors):
                raise RuntimeError(f"provider returned {len(vectors)} vectors for {len(group)} texts "
                                   f"or the wrong dimension")
            db.save_embeddings(conn, [(r["id"], Vector(v)) for r, v in zip(group, vectors)], embedder.model_name)
            done += len(group)
            report(progress, f"  batch {i}/{len(groups)}: {done}/{len(rows)} chunks embedded")
    except Exception as e:
        # Saved batches stay saved; a re-run picks up the rest.
        report(progress, f"  FAILED after {done}/{len(rows)} chunks: {type(e).__name__}: {e}; re-run to resume")
        return StageResult(False, embedder.tokens_used * EMBED_PRICES.get(embedder.model_name, 0.0) / 1e6)
    cost = embedder.tokens_used * EMBED_PRICES.get(embedder.model_name, 0.0) / 1e6
    print(f"  Actual usage: {embedder.tokens_used:,} tokens ≈ ${cost:.3f}")
    for book_id in sorted({r["book_id"] for r in rows}):
        db.mark_embedded(conn, book_id, embedder.model_name)
    return StageResult(True, cost)


def run(book_ids: list[int] | None, yes: bool, dry_run: bool) -> None:
    from summarize import confirm

    conn = db.connect()
    db.ensure_schema(conn)
    n, tokens, cost = estimate(conn, book_ids)
    if not n:
        print("Nothing to embed: every selected chunk already has a current embedding.")
        return
    print_estimate(n, tokens, cost)
    if dry_run:
        print("\nDRY RUN: nothing embedded.")
        return
    if not confirm(yes):
        print("Aborted.")
        return
    execute(conn, book_ids)
