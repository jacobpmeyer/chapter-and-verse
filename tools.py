"""The agent's tools: JSON-schema definitions (what the model sees) and the
Python that runs each one (what actually happens).

Every tool returns plain text. The model reads it as the tool_result. A tool
that can't do its job raises ToolError, and the loop sends that message back
with is_error=true so the model can recover (try another tool, tell the user).
"""

from __future__ import annotations

from dataclasses import dataclass

from pgvector import Vector

import db
from config import settings
from embed import Embedder, get_embedder

MAX_K = 20


class ToolError(Exception):
    """A problem the model should see and react to (bad id, book not indexed, too big...)."""


# --------------------------------------------------------------------------- #
# Definitions sent to the model
# --------------------------------------------------------------------------- #
# strict: true makes the API guarantee the model's inputs match the schema.
# Numeric bounds (k between 1 and 20) aren't expressible under strict mode, so
# they're enforced in code below.

TOOLS = [
    {
        "name": "list_books",
        "description": (
            "List books in the user's library with their id, title, authors, series, length in tokens, "
            "and indexing stage. Stages: 'embedded' = fully searchable (passages and summaries); "
            "'summarized' or 'summarizing' = partly processed, not searchable yet; 'extracted' = in the "
            "catalog only (full text exists, but no summaries or search). Use this to find a book's id, "
            "to check whether a book is in the library, or to see what can be searched."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional words to match against title, author, or series. Omit to list everything.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_passages",
        "description": (
            "Semantic search over the actual text of the books, in passages of roughly 500-800 tokens. "
            "Best for specific scenes, events, quotes, descriptions, and details: who said what, what "
            "happens when, how something is described. Only fully indexed books are searchable. Results "
            "include book title, chapter title, and a cosine similarity score. Scores are relative: "
            "a good match is often only 0.3-0.6, so compare results to each other rather than to 1.0."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, phrased like the text you hope to find (names and concrete details help).",
                },
                "book_id": {"type": "integer", "description": "Optional: restrict to one book (ids from list_books)."},
                "k": {"type": "integer", "description": "Number of passages to return, 1-20. Default 8."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_summaries",
        "description": (
            "Semantic search over chapter summaries and book summaries. Each chapter summary covers what "
            "happens, characters (or key concepts), motifs and recurring images, unresolved threads, and "
            "tone. Best for themes, character arcs, tone, motifs, how something develops across a book, "
            "finding which chapter something happens in, and comparisons across books. Use level to pick "
            "chapter summaries or book summaries only. Only fully indexed books are searchable."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The theme, idea, character, or event to look for."},
                "level": {
                    "type": "string",
                    "enum": ["chapter_summary", "book_summary"],
                    "description": "Optional: search only chapter summaries or only book summaries. Omit for both.",
                },
                "book_id": {"type": "integer", "description": "Optional: restrict to one book."},
                "k": {"type": "integer", "description": "Number of summaries to return, 1-20. Default 8."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_book_summary",
        "description": (
            "Get a book's stored book-level summary: premise, arc (including the ending), major themes, "
            "tone, motifs, and key characters or concepts. A cheap first step for any question about a "
            "whole book."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"book_id": {"type": "integer", "description": "Book id from list_books."}},
            "required": ["book_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "load_full_book",
        "description": (
            "Load a book's ENTIRE text into the conversation. This is very large (often 50K-400K tokens) "
            "and stays in the conversation for the rest of the session, which makes every later step "
            "slower and more expensive. Use it only for deep analysis of a single book that summaries and "
            "passage search can't support, such as close reading across the whole text or tracing "
            "something through every chapter. Books over the size limit are refused; use the summaries "
            "and passage search for those."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"book_id": {"type": "integer", "description": "Book id from list_books."}},
            "required": ["book_id"],
            "additionalProperties": False,
        },
    },
]


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

@dataclass
class ToolResult:
    text: str  # what the model sees
    summary: str  # one line for the terminal


class ToolRunner:
    def __init__(self, conn=None, embedder: Embedder | None = None):
        self.conn = conn or db.connect()
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:  # created lazily: list_books etc. don't need it
            self._embedder = get_embedder()
        return self._embedder

    def run(self, name: str, args: dict) -> ToolResult:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool {name!r}.")
        return handler(**args)

    # -- helpers ------------------------------------------------------------ #

    def _book(self, book_id: int) -> dict:
        book = db.get_book(self.conn, book_id)
        if book is None:
            raise ToolError(f"No book with id {book_id}. Use list_books to find ids.")
        return book

    def _require_indexed(self, book: dict) -> None:
        if book["stage"] != "embedded":
            raise ToolError(
                f'"{book["title"]}" (id {book["id"]}) is in the library but not searchable yet '
                f'(stage: {book["stage"]}). Tell the user they can index it with: python index.py book {book["id"]}'
            )

    @staticmethod
    def _k(k: int | None) -> int:
        return max(1, min(MAX_K, k or 8))

    def _search(self, query: str, levels: list[str], book_id: int | None, k: int | None, what: str) -> ToolResult:
        if book_id is not None:
            self._require_indexed(self._book(book_id))
        searchable = db.searchable_books(self.conn, self.embedder.model_name)
        if not searchable:
            raise ToolError("No books are indexed yet, so there is nothing to search. "
                            "Tell the user to index a book with: python index.py book <id>")
        hits = db.search_chunks(self.conn, Vector(self.embedder.embed_query(query)), self.embedder.model_name,
                                levels, book_id, self._k(k))
        scope = next((b["title"] for b in searchable if b["id"] == book_id), None) if book_id else \
            "; ".join(f'{b["title"]} (id {b["id"]})' for b in searchable)
        lines = [f'Searched {what} in: {scope}.']
        if not hits:
            return ToolResult("\n".join(lines + ["No results."]), "0 results")
        top = hits[0]["similarity"]
        if top < settings.weak_match_threshold:
            # Absolute scores depend a lot on query wording (short queries score
            # lower), so this is a prompt to check relevance, not a verdict.
            lines.append(f"WEAK RETRIEVAL: the best match scores only {top:.2f} (below {settings.weak_match_threshold}). "
                         "Check whether these results actually address the question before relying on them. "
                         "If they don't, try a differently worded search, or say you couldn't find it.")
        for i, h in enumerate(hits, 1):
            where = "book summary" if h["level"] == "book_summary" else f'chapter "{h["chapter_title"]}"'
            kind = {"passage": "passage", "chapter_summary": "chapter summary", "book_summary": "book summary"}[h["level"]]
            lines.append(f'\n[{i}] {h["book_title"]}, {where} ({kind}, similarity {h["similarity"]:.2f})\n{h["content"]}')
        best = hits[0]
        best_where = "book summary" if best["level"] == "book_summary" else f'"{best["chapter_title"]}"'
        return ToolResult("\n".join(lines), f"{len(hits)} results, top {top:.2f}: {best['book_title'][:40]}, {best_where}")

    # -- tools -------------------------------------------------------------- #

    def tool_list_books(self, query: str | None = None) -> ToolResult:
        books = db.list_books(self.conn, query)
        if not books:
            msg = f'No books match "{query}".' if query else "The library catalog is empty."
            return ToolResult(msg, "0 books")
        lines = []
        for b in books:
            tokens = b["exact_token_count"] or b["token_count"]
            series = f', series: {b["series"]} #{b["series_index"]:g}' if b["series"] else ""
            lines.append(f'id {b["id"]}: {b["title"]} by {", ".join(b["authors"])}{series}; '
                         f'{b["chapter_count"]} chapters, ~{tokens:,} tokens; stage: {b["stage"]}')
        indexed = sum(b["stage"] == "embedded" for b in books)
        return ToolResult("\n".join(lines), f"{len(books)} books ({indexed} searchable)")

    def tool_search_passages(self, query: str, book_id: int | None = None, k: int | None = None) -> ToolResult:
        return self._search(query, ["passage"], book_id, k, "passages")

    def tool_search_summaries(self, query: str, level: str | None = None, book_id: int | None = None,
                              k: int | None = None) -> ToolResult:
        levels = [level] if level else ["chapter_summary", "book_summary"]
        return self._search(query, levels, book_id, k, "summaries")

    def tool_get_book_summary(self, book_id: int) -> ToolResult:
        book = self._book(book_id)
        summary = db.get_book_summary(self.conn, book_id)
        if summary is None:
            raise ToolError(f'"{book["title"]}" has no book summary yet (stage: {book["stage"]}). '
                            f'Tell the user they can create one with: python index.py book {book_id}')
        return ToolResult(f'Book summary of {book["title"]} by {", ".join(book["authors"])}:\n\n{summary}',
                          f'{book["title"][:40]} ({len(summary.split())} words)')

    def tool_load_full_book(self, book_id: int) -> ToolResult:
        book = self._book(book_id)
        # Exact count once the book has been through count_tokens; otherwise the
        # local estimate scaled by the typical undercount (~1.35x).
        tokens = book["exact_token_count"] or round(book["token_count"] * 1.35)
        limit = settings.full_book_token_limit
        if tokens > limit:
            raise ToolError(
                f'"{book["title"]}" is ~{tokens:,} tokens, over the {limit:,}-token limit for loading a full '
                "book. Use get_book_summary, search_summaries (chapter summaries trace arcs and themes), and "
                "search_passages for specific scenes instead."
            )
        chapters = db.get_chapters(self.conn, book_id)
        text = "\n\n".join(f'## {c["title"]}\n\n{c["content"]}' for c in chapters)
        return ToolResult(f'Full text of {book["title"]} by {", ".join(book["authors"])} '
                          f'({len(chapters)} chapters, ~{tokens:,} tokens):\n\n{text}',
                          f'{book["title"][:40]}: {len(chapters)} chapters, ~{tokens:,} tokens')
