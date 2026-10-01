"""Postgres/pgvector access: connection, schema, and the queries each stage uses."""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from chunk import chunk_chapter
from config import settings
from extract import ExtractedBook

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(url: str | None = None) -> psycopg.Connection:
    """Connect to DATABASE_URL, or to `url` (the tests use a separate database)."""
    url = url or settings.database_url
    if not url:
        raise SystemExit("DATABASE_URL is not set (see .env.example)")
    # autocommit: each statement commits on its own; multi-statement writes use
    # explicit `with conn.transaction()` blocks.
    conn = psycopg.connect(url, row_factory=dict_row, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    ddl = SCHEMA_PATH.read_text().replace("{EMBED_DIM}", str(settings.embed_dim))
    conn.execute(ddl)
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()
    if row and row["atttypmod"] != settings.embed_dim:
        raise SystemExit(
            f"chunks.embedding is vector({row['atttypmod']}) but EMBED_DIM={settings.embed_dim}. "
            "Changing embedding dimensions needs a migration and a full re-embed."
        )


def stored_hash(conn: psycopg.Connection, source_path: str) -> str | None:
    row = conn.execute("SELECT content_hash FROM books WHERE source_path = %s", (source_path,)).fetchone()
    return row["content_hash"] if row else None


def save_extracted_book(conn: psycopg.Connection, book: ExtractedBook) -> int:
    """Insert a new book, or replace an existing book's content while keeping its id.

    Keeping ids stable means `python index.py book 9` always refers to the same
    book. Replacing content resets everything derived from it (summaries,
    embeddings, exact token count).
    """
    m = book.meta
    values = (m.calibre_uuid, m.calibre_id, m.title, m.authors, m.series, m.series_index,
              m.language, str(book.path), book.content_hash, book.token_count, len(book.chapters))
    with conn.transaction():
        existing = conn.execute(
            "SELECT id FROM books WHERE source_path = %s OR (calibre_uuid IS NOT NULL AND calibre_uuid = %s) "
            "ORDER BY id LIMIT 1",
            (str(book.path), m.calibre_uuid),
        ).fetchone()
        if existing:
            book_id = existing["id"]
            conn.execute("DELETE FROM chapters WHERE book_id = %s", (book_id,))
            conn.execute("DELETE FROM chunks WHERE book_id = %s", (book_id,))
            conn.execute(
                """UPDATE books SET calibre_uuid = %s, calibre_id = %s, title = %s, authors = %s, series = %s,
                       series_index = %s, language = %s, source_path = %s, content_hash = %s, token_count = %s,
                       chapter_count = %s, exact_token_count = NULL, book_summary_method = NULL,
                       indexed_at = now(), summarized_at = NULL, embedded_at = NULL
                   WHERE id = %s""",
                (*values, book_id),
            )
        else:
            book_id = conn.execute(
                """INSERT INTO books (calibre_uuid, calibre_id, title, authors, series, series_index,
                                      language, source_path, content_hash, token_count, chapter_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                values,
            ).fetchone()["id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chapters (book_id, chapter_index, title, content, token_count) VALUES (%s, %s, %s, %s, %s)",
                [(book_id, c.index, c.title, c.content, c.token_count) for c in book.chapters],
            )
            rows = [
                (book_id, "passage", c.index, c.title, p.chunk_index, p.content, p.token_count)
                for c in book.chapters
                for p in chunk_chapter(c.content)
            ]
            cur.executemany(
                """INSERT INTO chunks (book_id, level, chapter_index, chapter_title, chunk_index, content, token_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                rows,
            )
    return book_id


# --------------------------------------------------------------------------- #
# Indexing stage (Phase 1.5)
# --------------------------------------------------------------------------- #

def book_stages(conn: psycopg.Connection, book_ids: list[int] | None = None) -> list[dict]:
    """Every book (or the given ids) with its progress and a `stage` of
    'extracted', 'summarizing', 'summarized', or 'embedded'."""
    rows = conn.execute(
        """SELECT b.*,
                  (SELECT count(*) FROM chunks c WHERE c.book_id = b.id AND c.level = 'chapter_summary')
                      AS chapter_summaries,
                  EXISTS (SELECT 1 FROM chunks c WHERE c.book_id = b.id AND c.level = 'book_summary')
                      AS has_book_summary,
                  (SELECT count(*) FROM chunks c WHERE c.book_id = b.id
                       AND (c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM %s))
                      AS unembedded
           FROM books b
           WHERE %s::int[] IS NULL OR b.id = ANY(%s::int[])
           ORDER BY b.id""",
        (settings.embed_model, book_ids, book_ids),
    ).fetchall()
    for r in rows:
        summarized = r["has_book_summary"] and r["chapter_summaries"] >= r["chapter_count"]
        if summarized and r["unembedded"] == 0:
            r["stage"] = "embedded"
        elif summarized:
            r["stage"] = "summarized"
        elif r["chapter_summaries"] or r["has_book_summary"]:
            r["stage"] = "summarizing"
        else:
            r["stage"] = "extracted"
    return rows


def find_books(conn: psycopg.Connection, selector: str) -> list[dict]:
    """Match a book by numeric id, or by all words appearing in title/authors/series."""
    if selector.strip().isdigit():
        return conn.execute("SELECT * FROM books WHERE id = %s", (int(selector),)).fetchall()
    words = [f"%{w}%" for w in selector.split()]
    return conn.execute(
        """SELECT * FROM books
           WHERE (SELECT bool_and(title || ' ' || array_to_string(authors, ' ') || ' ' || coalesce(series, '')
                                  ILIKE w) FROM unnest(%s::text[]) AS w)
           ORDER BY id""",
        (words,),
    ).fetchall()


# --------------------------------------------------------------------------- #
# Phase 2: summaries
# --------------------------------------------------------------------------- #

def get_chapters(conn: psycopg.Connection, book_id: int) -> list[dict]:
    return conn.execute(
        "SELECT chapter_index, title, content, token_count FROM chapters WHERE book_id = %s ORDER BY chapter_index",
        (book_id,),
    ).fetchall()


def get_chapter_summaries(conn: psycopg.Connection, book_id: int) -> dict[int, str]:
    rows = conn.execute(
        "SELECT chapter_index, content FROM chunks WHERE book_id = %s AND level = 'chapter_summary'",
        (book_id,),
    ).fetchall()
    return {r["chapter_index"]: r["content"] for r in rows}


def set_exact_token_count(conn: psycopg.Connection, book_id: int, tokens: int) -> None:
    conn.execute("UPDATE books SET exact_token_count = %s WHERE id = %s", (tokens, book_id))


def save_chapter_summary(conn, book_id: int, chapter_index: int, title: str, content: str,
                         token_count: int, model: str) -> None:
    conn.execute(
        """INSERT INTO chunks (book_id, level, chapter_index, chapter_title, content, token_count, generated_by)
           VALUES (%s, 'chapter_summary', %s, %s, %s, %s, %s)""",
        (book_id, chapter_index, title, content, token_count, model),
    )


def save_book_summary(conn, book_id: int, content: str, token_count: int, model: str, method: str) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE book_id = %s AND level = 'book_summary'", (book_id,))
        conn.execute(
            """INSERT INTO chunks (book_id, level, content, token_count, generated_by)
               VALUES (%s, 'book_summary', %s, %s, %s)""",
            (book_id, content, token_count, model),
        )
        conn.execute(
            "UPDATE books SET book_summary_method = %s, summarized_at = now() WHERE id = %s",
            (method, book_id),
        )


def delete_summaries(conn: psycopg.Connection, book_id: int) -> None:
    """Drop a book's chapter and book summaries so they can be regenerated."""
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE book_id = %s AND level IN ('chapter_summary', 'book_summary')",
                     (book_id,))
        conn.execute("UPDATE books SET book_summary_method = NULL, summarized_at = NULL WHERE id = %s", (book_id,))


def log_api_call(conn, book_id: int | None, purpose: str, model: str, effort: str, prompt_version: str,
                 input_tokens: int, output_tokens: int) -> None:
    conn.execute(
        """INSERT INTO api_calls (book_id, purpose, model, effort, prompt_version, input_tokens, output_tokens)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (book_id, purpose, model, effort, prompt_version, input_tokens, output_tokens),
    )


def api_call_stats(conn, model: str, effort: str, prompt_version: str, recent: int = 200) -> dict[str, dict]:
    """Per purpose: number of calls and average input/output tokens over the most recent calls."""
    rows = conn.execute(
        """SELECT purpose, count(*) AS n, avg(input_tokens) AS avg_in, avg(output_tokens) AS avg_out
           FROM (SELECT *, row_number() OVER (PARTITION BY purpose ORDER BY id DESC) AS rn
                 FROM api_calls WHERE model = %s AND effort = %s AND prompt_version = %s) t
           WHERE rn <= %s GROUP BY purpose""",
        (model, effort, prompt_version, recent),
    ).fetchall()
    return {r["purpose"]: r for r in rows}


# --------------------------------------------------------------------------- #
# Phase 3: embeddings
# --------------------------------------------------------------------------- #

def pending_embeddings(conn: psycopg.Connection, book_ids: list[int] | None, model: str) -> list[dict]:
    """Chunks with no embedding, or one from a different model, plus the book fields needed for headers."""
    return conn.execute(
        """SELECT c.id, c.book_id, c.level, c.chapter_index, c.chapter_title, c.content, b.title, b.authors
           FROM chunks c JOIN books b ON b.id = c.book_id
           WHERE (c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM %s)
             AND (%s::int[] IS NULL OR c.book_id = ANY(%s::int[]))
           ORDER BY c.book_id, c.id""",
        (model, book_ids, book_ids),
    ).fetchall()


def save_embeddings(conn: psycopg.Connection, rows: list[tuple[int, list[float]]], model: str) -> None:
    """Store one batch atomically, so an interrupted run resumes at the first unsaved batch."""
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
            [(vec, model, chunk_id) for chunk_id, vec in rows],
        )


def mark_embedded(conn: psycopg.Connection, book_id: int, model: str) -> bool:
    """Set books.embedded_at if every chunk of the book has a current embedding."""
    row = conn.execute(
        """UPDATE books SET embedded_at = now()
           WHERE id = %s AND NOT EXISTS (
               SELECT 1 FROM chunks WHERE book_id = %s
                 AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %s))
           RETURNING id""",
        (book_id, book_id, model),
    ).fetchone()
    return row is not None


def search_chunks(conn: psycopg.Connection, query_vec, model: str, levels: list[str],
                  book_id: int | None = None, k: int = 8) -> list[dict]:
    """Cosine-similarity search over chunks of the given levels, using the HNSW index.

    Filters (level, book, model) are applied during the index scan. With
    iterative scans (pgvector >= 0.8) the index keeps searching until it has k
    rows that pass the filters, instead of returning fewer. relaxed_order can
    return rows slightly out of order, so the outer query re-sorts.
    """
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        return conn.execute(
            """WITH hits AS MATERIALIZED (
                   SELECT c.id, c.book_id, b.title AS book_title, c.level, c.chapter_index, c.chapter_title,
                          c.content, c.embedding <=> %s::vector AS distance
                   FROM chunks c JOIN books b ON b.id = c.book_id
                   WHERE c.embedding_model = %s AND c.level = ANY(%s)
                     AND (%s::int IS NULL OR c.book_id = %s)
                   ORDER BY distance
                   LIMIT %s)
               SELECT *, 1 - distance AS similarity FROM hits ORDER BY distance""",
            (query_vec, model, levels, book_id, book_id, k),
        ).fetchall()


def has_paid_work(conn: psycopg.Connection, source_path: str) -> bool:
    """True if the book has summaries or embeddings (which re-extraction would delete)."""
    return conn.execute(
        """SELECT 1 FROM chunks c JOIN books b ON b.id = c.book_id
           WHERE b.source_path = %s AND (c.level <> 'passage' OR c.embedding IS NOT NULL) LIMIT 1""",
        (source_path,),
    ).fetchone() is not None


# --------------------------------------------------------------------------- #
# Phase 4: agent tools
# --------------------------------------------------------------------------- #

def list_books(conn: psycopg.Connection, query: str | None = None) -> list[dict]:
    """Every extracted book (optionally filtered by title/author/series words) with its stage."""
    rows = book_stages(conn)
    if query:
        ids = {b["id"] for b in find_books(conn, query)}
        rows = [r for r in rows if r["id"] in ids]
    return rows


def get_book(conn: psycopg.Connection, book_id: int) -> dict | None:
    rows = book_stages(conn, [book_id])
    return rows[0] if rows else None


def get_book_summary(conn: psycopg.Connection, book_id: int) -> str | None:
    row = conn.execute(
        "SELECT content FROM chunks WHERE book_id = %s AND level = 'book_summary'", (book_id,)
    ).fetchone()
    return row["content"] if row else None


def searchable_books(conn: psycopg.Connection, model: str) -> list[dict]:
    """Books with at least one chunk embedded by `model` (what vector search can reach)."""
    return conn.execute(
        """SELECT DISTINCT b.id, b.title FROM books b JOIN chunks c ON c.book_id = b.id
           WHERE c.embedding_model = %s ORDER BY b.id""",
        (model,),
    ).fetchall()
