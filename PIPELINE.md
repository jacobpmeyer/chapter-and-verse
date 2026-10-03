# Pipeline: from a new book to the agent

How each file is used, in order, from adding an EPUB to asking the agent about it. Time runs
top to bottom, and each column is a file or service. `config.py` (settings from `.env`) is used at
every step, and every database read or write goes through `db.py`, which creates the tables from
`schema.sql` on first use.

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant Cal as Calibre files<br/>.epub + .opf
    participant IDX as index.py
    participant EXT as extract.py
    participant CHK as chunk.py
    participant SUM as summarize.py
    participant EMB as embed.py
    participant DB as db.py<br/>Postgres + pgvector
    participant ANT as Anthropic API
    participant VOY as Voyage AI
    participant AG as agent.py
    participant TL as tools.py

    Note over You,Cal: 1 · Add the book
    You->>Cal: add the EPUB in Calibre (writes .epub + .opf)

    Note over You,VOY: 2 · python index.py extract (free, no API calls)
    You->>IDX: python index.py extract
    IDX->>DB: have we seen this file? (sha256 of the EPUB)
    DB-->>IDX: no, it's new
    IDX->>EXT: extract_book(path)
    EXT->>Cal: read .epub spine + TOC, and .opf title/authors/series
    EXT->>EXT: TOC → chapters → Markdown, drop front/back matter, label parts
    EXT-->>IDX: chapters (token estimates via tokens.py)
    IDX->>DB: save_extracted_book()
    DB->>CHK: chunk_chapter() for each chapter
    CHK-->>DB: 500–800-token passages, whole paragraphs, ~10% overlap
    DB->>DB: INSERT books, chapters, chunks (passages, no embeddings yet)

    Note over You,VOY: 3 · python index.py book "title" (paid, asks first)
    You->>IDX: python index.py book "title"
    IDX->>SUM: plan()
    SUM->>ANT: count_tokens (free)
    SUM->>DB: averages from past api_calls rows
    IDX->>EMB: estimate()
    IDX-->>You: estimated cost. Proceed? [y/N]
    You->>IDX: y
    loop every chapter, in order
        SUM->>DB: chapter text + previous chapter's summary
        SUM->>ANT: Sonnet 5.5: summarize this chapter
        ANT-->>SUM: summary (plus a condense call if it's too long)
        SUM->>DB: INSERT chunk (chapter_summary) + api_calls row
    end
    SUM->>ANT: Sonnet 5.5: book summary (full text, or from chapter summaries)
    SUM->>DB: INSERT chunk (book_summary), books.book_summary_method
    loop batches of up to 64 chunks
        EMB->>DB: chunks without a current embedding
        EMB->>VOY: embed, input_type=document
        VOY-->>EMB: 1024-dim vectors
        EMB->>DB: UPDATE chunks.embedding + embedding_model
    end
    EMB->>DB: books.embedded_at (stage: embedded)

    Note over You,TL: 4 · python agent.py
    You->>AG: question
    loop until it answers (at most AGENT_MAX_TURNS calls)
        AG->>ANT: Opus 5.5: system prompt + tools + whole history
        ANT-->>AG: reasoning summary + tool calls (or the final answer)
        AG->>TL: run every tool call in the reply
        TL->>EMB: embed the search query
        EMB->>VOY: embed, input_type=query
        TL->>DB: cosine search over chunks (or book list, summary, full text)
        DB-->>TL: matching passages / summaries
        TL-->>AG: tool results (flagged if the match is weak)
        AG->>AG: append reply + results to history
    end
    AG-->>You: answer, citing book and chapter titles
```

The HTTP API (`api.py`) runs this same flow:
- **Adding books:** `POST /library/scan` runs step 2 over `LIBRARY_PATH`. It never re-extracts a
  book that has paid work.
- **Indexing:** `POST /books/{id}/index` runs step 3 through `jobs.py`, the same code as
  `index.py book`, after checking the estimate against the caller's `max_cost_usd`. Locally the
  job runs in a background thread. When deployed, the API starts a Cloud Run Job execution of
  `python index.py run-job <id>`, which re-estimates, refuses to exceed the limit, and records
  progress and the actual cost on the job.
- **Questions:** `POST /ask` runs step 4's agent loop, storing the conversation in Postgres
  instead of keeping it in memory.

Deployed, step 1 also uploads the book: `deploy/sync-library.sh` copies the Calibre library to a
Cloud Storage bucket, which the API reads as a read-only folder at `LIBRARY_PATH`. Every other
step is the same code, with Cloud SQL in place of the local Postgres. The
[README](README.md#deployment) shows where each piece runs.

## The same flow in words

| Step | Command | Files involved | External calls | Writes |
|---|---|---|---|---|
| 1 | add the book in Calibre | `.epub`, `.opf` | – | – |
| 2 | `python index.py extract` | `index.py` → `extract.py` → `chunk.py` (both use `tokens.py`) → `db.py` | none | `books`, `chapters`, `chunks` (passages) |
| 3a | `python index.py status` / `book --dry-run` | `index.py`, `summarize.py` (plan), `embed.py` (estimate) | `count_tokens` (free) | `books.exact_token_count` |
| 3b | `python index.py book <id>` → `y` | `summarize.py` | Sonnet 5.5, one call per chapter + one for the book (+ condense if needed) | `chunks` (summaries), `books`, `api_calls` |
| 3c | (same command, continued) | `embed.py` | Voyage, batched `document` embeddings | `chunks.embedding`, `books.embedded_at` |
| 4 | `python agent.py` | `agent.py` ⇄ Opus 5.5; `tools.py` → `embed.py` (query) + `db.py` | Opus 5.5 each step; Voyage once per search | nothing (read-only) |

Read-only helpers along the way: `python index.py summaries <book>` (reads summary chunks) and
`python index.py search "…"` (embeds a query with `embed.py`, then runs the same cosine search the agent uses).
