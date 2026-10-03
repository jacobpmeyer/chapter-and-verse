# Shortcuts

Run everything from the project folder. Activate the virtual environment once per
terminal session, then `python` means the project's Python:

```sh
cd ~/projects/chapter-and-verse
source .venv/bin/activate        # or prefix each command with .venv/bin/python
```

A book can be named by id or by words from its title/author (`10`, `wayward`, `"devils abercrombie"`).
`python index.py <command> --help` lists every option.

**Two databases:** the deployed one (Cloud SQL) is the real library. Local Docker Compose is for
development and tests. Every `index.py` and `agent.py` command below uses whichever database
`DATABASE_URL` points at: the local one by default, or the cloud one after
`source deploy/cloud-shell.sh` in that terminal.

## The deployed service

- **Web page:** https://books.jacobpm.com. Log in with your email; Cloudflare emails you a PIN,
  and the login lasts a week. The page then asks once for the API key:
  `gcloud secrets versions access latest --secret=api-keys`.
- **API docs:** https://books.jacobpm.com/docs

From a terminal:

```sh
source deploy/cloud-shell.sh       # cloud database for this terminal + the `cv` command
cv /books                          # the deployed API, through Cloudflare Access (cli service token)
cv /books/7/estimate               # free
cv /books/7/index '{"max_cost_usd": 3}'   # index a book: a Cloud Run Job (refuses above the limit)
cv /jobs/1                         # job progress and actual cost
python index.py status             # stage and cost to finish, from the cloud database
python agent.py                    # the REPL, answering from the cloud database
```

A deployed job is a Cloud Run Job execution of `python index.py run-job <id>`. That command
refuses a job that isn't `queued`, so a job never runs (or bills) twice.

### Adding books

```sh
deploy/sync-library.sh             # upload new/changed files from LIBRARY_PATH (never deletes)
cv /library/scan '{}'              # extract them (free); then index each with cv /books/<id>/index
```

### Releasing and operating

```sh
git push                           # to main: tests, then GitHub Actions deploys (tagged with the commit)
deploy/deploy.sh                   # deploy from this Mac (uncommitted work is tagged -dirty)
gcloud run services logs read cv-api --region us-east1 --limit 50    # API log (no secrets)
gcloud run jobs executions list --job cv-index --region us-east1     # indexing runs
gcloud sql backups list --instance cv-db                             # daily backups (7 kept)
gcloud sql backups create --instance cv-db                           # an extra one, e.g. before --redo-summaries
python3 deploy/cloudflare.py --proxied   # re-apply DNS + Access settings (needs CLOUDFLARE_API_TOKEN)
```

[deploy/README.md](deploy/README.md) explains what runs where and how to get at the database
directly.

## The agent

```sh
python agent.py            # ask questions about your library (~$0.10-0.20 per question)
```

Inside it: `/usage` shows tokens and cost so far, `/reset` starts a new conversation, and `/quit`
(or Ctrl-D) exits. Ctrl-C during an answer drops just that question.
`AGENT_THINKING_DISPLAY=omitted python agent.py` hides the reasoning summaries for one session.

## Reading summaries (free, no API calls)

```sh
python index.py summaries wayward                 # book summary, then every chapter, in a pager (q quits)
python index.py summaries wayward --book-only     # just the book summary
python index.py summaries wayward --chapter 11    # one chapter (our number; `status`/`search` show it)
python index.py summaries 10 --out hendrix.md     # write Markdown to a file to open in an editor
```

In the pager: arrow keys or space to scroll, `/word` to search, `n` for the next match, `q` to quit.

## Library status and indexing from the CLI

```sh
python index.py status                 # every book: stage, exact tokens, summaries done, cost to finish
python index.py extract                # add new/changed EPUBs to the catalog (free, no API calls)
python index.py extract --dry-run -v   # preview what would be kept/skipped and why, write nothing
python index.py extract --force        # re-extract unchanged books (skips ones with summaries/embeddings)

python index.py book devils --dry-run  # show the cost estimate for fully indexing one book
python index.py book devils            # summarize + embed one book on this Mac (asks y/N first)
python index.py book 9 --redo-summaries  # throw away a book's summaries and regenerate them
```

`extract` reads `LIBRARY_PATH` on this Mac. For the cloud library, use `deploy/sync-library.sh` and
`cv /library/scan '{}'` instead. `summarize` and `embed` also exist on their own, but need
`--book-id ID` or `--all`, so nothing runs on the whole library by accident.

## Searching (debugging retrieval, no LLM)

```sh
python index.py search "the girls learn witchcraft from a book"
python index.py search "institutional control" --level chapter_summary -k 5
python index.py search "Dr. Vincent" --book-id 10
```

Levels: `passage`, `chapter_summary`, `book_summary`. Scores are cosine similarity. Correct
hits have ranged from ~0.26 to ~0.69 depending on wording, and nonsense scores ~0.1.

## Local development (Docker Compose)

```sh
docker compose up -d --build       # Postgres + API on http://localhost:8080 (rebuild after code changes)
docker compose logs -f api         # follow the API's log (one line per request, no secrets)
docker compose ps                  # running / healthy?
docker compose stop                # stop both (data is kept)
```

The local API uses the `API_KEYS` value in `.env`, runs indexing jobs in a background thread, and
has no Cloudflare Access check:

```sh
KEY=$(grep '^API_KEYS=' .env | cut -d= -f2 | cut -d, -f1)
curl -H "Authorization: Bearer $KEY" localhost:8080/books
```

### Local backups

```sh
python index.py backup                       # dated, verified dump to ~/Backups/chapter-and-verse
python index.py backup --label before-redo   # label it
python index.py backup --keep 10             # then delete all but the newest 10
python index.py backup --list                # what's there
```

These back up the local Docker database. The cloud one has Cloud SQL's daily backups (see above).
Restoring replaces the database contents, and the backup command prints the exact command. The
local database lives in the Docker volume `chapter-and-verse_pgdata`. `docker compose down -v`,
`docker system prune --volumes`, or a Docker Desktop factory reset deletes it.

### SQL

```sh
docker exec -it chapter-and-verse-db psql -U library_rag -d library_rag   # local
source deploy/cloud-shell.sh && /opt/homebrew/opt/libpq/bin/psql "$DATABASE_URL"   # cloud
```

Useful inside `psql` (`\q` quits, `\x auto` makes wide rows readable):

```sql
\x auto
-- books and their stage-related fields
select id, title, chapter_count, exact_token_count, book_summary_method, summarized_at, embedded_at from books order by id;
-- one chapter summary next to the chapter's source text
select chapter_title, content from chunks where book_id = 10 and level = 'chapter_summary' and chapter_index = 11;
select content from chapters where book_id = 10 and chapter_index = 11;
-- spending per book and purpose (summaries, agent); logging began after Morrie was summarized
select book_id, purpose, count(*), round(sum(cost_usd), 2) as usd from api_calls group by 1, 2 order by 1, 2;
```

## Tests

```sh
pytest -m "not db"                 # unit tests (no database, no API calls, under a second)
pytest                             # everything, incl. database tests (needs `docker compose up -d`)
pytest -m db                       # only the database tests (throwaway chapter_and_verse_test DB)
pytest tests/test_extract.py -v    # one file, listing each test
pytest -k printer                  # only tests whose name matches
```
