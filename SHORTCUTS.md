# Shortcuts

Run everything from the project folder. Activate the virtual environment once per
terminal session, then `python` means the project's Python:

```sh
cd ~/projects/chapter-and-verse
source .venv/bin/activate        # or prefix each command with .venv/bin/python
```

A book can be named by id or by words from its title/author (`10`, `wayward`, `"devils abercrombie"`).
`python index.py <command> --help` lists every option.

## The agent

```sh
python agent.py            # ask questions about your library (~$0.10-0.20 per question)
```

Inside it: `/usage` shows tokens and cost so far, `/reset` starts a new conversation, and `/quit`
(or Ctrl-D) exits. Ctrl-C during an answer drops just that question.
`AGENT_THINKING_DISPLAY=omitted python agent.py` hides the reasoning summaries for one session.

## HTTP API and web page

```sh
docker compose up -d --build       # Postgres + API (rebuild after code changes)
docker compose logs -f api         # follow the API's log (one line per request, no secrets)
docker compose stop api            # stop just the API
```

- **Web page:** http://localhost:8080. From your phone on the same Wi-Fi, use
  http://<your Mac's address>:8080 (`ipconfig getifaddr en0` shows the address).
- **API key:** the page asks for it once. It's the `API_KEYS` value in `.env`.
- **API docs:** http://localhost:8080/docs

```sh
KEY=$(grep '^API_KEYS=' .env | cut -d= -f2 | cut -d, -f1)
curl -H "Authorization: Bearer $KEY" localhost:8080/books
curl -X POST localhost:8080/ask -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"question": "Who delivers the eels?"}'
curl -H "Authorization: Bearer $KEY" localhost:8080/books/7/estimate      # free
```

## Reading summaries (free, no API calls)

```sh
python index.py summaries wayward                 # book summary, then every chapter, in a pager (q quits)
python index.py summaries wayward --book-only     # just the book summary
python index.py summaries wayward --chapter 11    # one chapter (our number; `status`/`search` show it)
python index.py summaries 10 --out hendrix.md     # write Markdown to a file to open in an editor
```

In the pager: arrow keys or space to scroll, `/word` to search, `n` for the next match, `q` to quit.

## Library status and indexing

```sh
python index.py status                 # every book: stage, exact tokens, summaries done, cost to finish
python index.py extract                # add new/changed EPUBs to the catalog (free, no API calls)
python index.py extract --dry-run -v   # preview what would be kept/skipped and why, write nothing
python index.py extract --force        # re-extract unchanged books (skips ones with summaries/embeddings)

python index.py book devils --dry-run  # show the cost estimate for fully indexing one book
python index.py book devils            # summarize + embed one book (asks y/N first)
python index.py book 9 --redo-summaries  # throw away a book's summaries and regenerate them
```

`summarize` and `embed` also exist on their own, but need `--book-id ID` or `--all`
so nothing runs on the whole library by accident. `book` is usually what you want.

## Backups

```sh
python index.py backup                       # dated, verified dump to ~/Backups/chapter-and-verse
python index.py backup --label before-redo   # label it (do this before --redo-summaries or extract --force)
python index.py backup --keep 10             # then delete all but the newest 10
python index.py backup --list                # what's there
```

Restoring replaces the current database contents. The backup command prints the exact command:

```sh
docker compose exec -T db pg_restore -U library_rag -d library_rag --clean --if-exists \
  --single-transaction < ~/Backups/chapter-and-verse/<file>.dump
```

The database lives in the Docker volume `chapter-and-verse_pgdata`. `docker compose down -v`,
`docker system prune --volumes`, or a Docker Desktop factory reset deletes it, so back up first.

## Searching (debugging retrieval, no LLM)

```sh
python index.py search "the girls learn witchcraft from a book"
python index.py search "institutional control" --level chapter_summary -k 5
python index.py search "Dr. Vincent" --book-id 10
```

Levels: `passage`, `chapter_summary`, `book_summary`. Scores are cosine similarity. Correct
hits have ranged from ~0.26 to ~0.66 depending on wording, and nonsense scores ~0.1.

## Tests

```sh
pytest -m "not db"                 # unit tests (no database, no API calls, under a second)
pytest                             # everything, incl. database tests (needs `docker compose up -d`)
pytest -m db                       # only the database tests (throwaway chapter_and_verse_test DB)
pytest tests/test_extract.py -v    # one file, listing each test
pytest -k printer                  # only tests whose name matches
```

## Database (Docker)

```sh
docker compose up -d        # start Postgres (it restarts on its own after a reboot)
docker compose ps           # is it running / healthy?
docker compose stop         # stop it (data is kept)
```

Open a `psql` prompt inside the container (no password needed):

```sh
docker exec -it chapter-and-verse-db psql -U library_rag -d library_rag
```

Useful inside `psql` (`\q` quits, `\x auto` makes wide rows readable):

```sql
\x auto
-- books and their stage-related fields
select id, title, chapter_count, exact_token_count, book_summary_method, summarized_at, embedded_at from books order by id;
-- a book summary
select content from chunks where book_id = 10 and level = 'book_summary';
-- one chapter summary (our chapter number)
select chapter_title, content from chunks where book_id = 10 and level = 'chapter_summary' and chapter_index = 11;
-- the full source text of that chapter, to compare against its summary
select content from chapters where book_id = 10 and chapter_index = 11;
-- tokens used by summary API calls, per book (logging began after Morrie was summarized,
-- so book 9 only shows 6 test calls)
select book_id, purpose, count(*), sum(input_tokens) as input, sum(output_tokens) as output
from api_calls group by 1, 2 order by 1, 2;
```

All of a book's summaries as one scrollable document, without entering `psql`:

```sh
docker exec chapter-and-verse-db psql -U library_rag -d library_rag -At -c "
  select case when level = 'book_summary' then E'# Book summary\n\n' || content
              else E'## ' || chapter_index || '. ' || chapter_title || E'\n\n' || content end
  from chunks where book_id = 10 and level in ('book_summary', 'chapter_summary')
  order by chapter_index nulls first" | less
```

Replace `| less` with `> hendrix-summaries.md` to save it to a file, and change `book_id = 10` for other books.
