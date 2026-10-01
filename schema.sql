-- Idempotent schema. {EMBED_DIM} is substituted from config by db.py.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS books (
    id                  serial PRIMARY KEY,
    calibre_uuid        text UNIQUE,
    calibre_id          integer,
    title               text NOT NULL,
    authors             text[] NOT NULL DEFAULT '{}',
    series              text,
    series_index        numeric,
    language            text,
    source_path         text NOT NULL UNIQUE,
    content_hash        text NOT NULL,
    token_count         integer NOT NULL,          -- estimated, sum of kept chapters
    chapter_count       integer NOT NULL,
    book_summary_method text CHECK (book_summary_method IN ('full_text', 'chapter_summaries')),
    indexed_at          timestamptz NOT NULL DEFAULT now(),
    summarized_at       timestamptz,
    embedded_at         timestamptz
);

-- Added in Phase 2: exact token count from the Anthropic count_tokens API.
ALTER TABLE books ADD COLUMN IF NOT EXISTS exact_token_count integer;

-- Full chapter text. load_full_book reads from here, because passages overlap
-- and would duplicate text if concatenated.
CREATE TABLE IF NOT EXISTS chapters (
    book_id       integer NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_index integer NOT NULL,
    title         text NOT NULL,
    content       text NOT NULL,
    token_count   integer NOT NULL,
    PRIMARY KEY (book_id, chapter_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    book_id         integer NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    level           text NOT NULL CHECK (level IN ('passage', 'chapter_summary', 'book_summary')),
    chapter_index   integer,          -- NULL for book_summary
    chapter_title   text,
    chunk_index     integer NOT NULL DEFAULT 0,
    content         text NOT NULL,
    token_count     integer NOT NULL,
    generated_by    text,             -- summary model; NULL for passages
    embedding       vector({EMBED_DIM}),
    embedding_model text
);

CREATE INDEX IF NOT EXISTS chunks_book_level_idx ON chunks (book_id, level, chapter_index);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Added in Phase 2: one row per summary API call. The cost estimate uses these
-- averages (per model, effort, and prompt version) instead of fixed guesses.
CREATE TABLE IF NOT EXISTS api_calls (
    id             bigserial PRIMARY KEY,
    book_id        integer REFERENCES books(id) ON DELETE SET NULL,
    purpose        text NOT NULL,   -- chapter_summary | book_summary | condense
    model          text NOT NULL,   -- requested model
    effort         text NOT NULL,
    prompt_version text NOT NULL,
    input_tokens   integer NOT NULL,
    output_tokens  integer NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);
