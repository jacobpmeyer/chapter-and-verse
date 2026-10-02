# Deployment (GCP)

Scripts that create and update the cloud deployment. Every resource name lives in
[`config.sh`](config.sh). The non-secret runtime settings are in [`env.yaml`](env.yaml).

| Script | When |
|---|---|
| [`setup.sh`](setup.sh) | once, before the first deploy. It's safe to re-run: it only creates what's missing and never replaces a secret. |
| [`sync-library.sh`](sync-library.sh) | after adding books in Calibre, then `POST /library/scan` |
| [`deploy.sh`](deploy.sh) | every release: build, push, update the service and the job |
| [`cloudflare.py`](cloudflare.py) | DNS and Cloudflare Access for books.jacobpm.com (`--proxied` once Google has issued the certificate) |
| [`cloud-shell.sh`](cloud-shell.sh) | `source` it to point this terminal at the cloud database and get the `cv` command |

## What runs where

| Piece | GCP resource | Notes |
|---|---|---|
| Front door | Cloudflare (DNS, proxy, Access) | `books.jacobpm.com`: email one-time PIN for people, service tokens for programs |
| API + web page | Cloud Run service `cv-api` | request-based billing, scales to zero, at most 2 instances; custom domain via a domain mapping |
| Indexing | Cloud Run Job `cv-index` | one execution per job: `python index.py run-job <id>`, no retries |
| Database | Cloud SQL `cv-db`, PostgreSQL 18 + pgvector | `db-f1-micro`, 7 daily backups, reached only through the Cloud SQL connector |
| EPUB library | bucket `chapter-and-verse-510420-library` | mounted read-only at `/library` (Cloud Storage FUSE) |
| Keys and the database URL | Secret Manager | `anthropic-api-key`, `voyage-api-key`, `api-keys`, `database-url`, and each Access service token's id and secret |
| Images | Artifact Registry `cv` | a cleanup policy keeps the 5 newest |

Each workload has its own service account:
- **`cv-api`:** connects to Cloud SQL, reads its 4 secrets and the library bucket, and starts
  executions of `cv-index`. That last permission comes from
  `roles/run.jobsExecutorWithOverrides`, granted on that one job only.
- **`cv-indexer`:** connects to Cloud SQL and reads 3 secrets.

## How a request gets in

1. **Cloudflare Access** answers first. A browser without a session is sent to the email PIN login,
   and only the owner's address is allowed. A program sends a service token instead
   (`CF-Access-Client-Id` / `-Secret`). `deploy/cloudflare.py` creates one token per client
   (`shelfspace`, `cli`), so each can be revoked alone.
2. Access forwards the request with a signed JWT. **The app verifies it** (`CF_ACCESS_*` in
   `env.yaml`), so the public `*.run.app` URL is useless without going through Access. `deploy.sh`
   makes the service public only when that check is configured.
3. **The API key** is still required on every endpoint except `/health` and the page itself.

Google issues and renews the `books.jacobpm.com` certificate with an HTTP challenge. Access has a
Bypass for `/.well-known/acme-challenge/` so renewals get through. The first renewal is due
around early December 2026 (see TODO.md).

## Common tasks

```sh
# The deployed API's key (to paste into the web page)
gcloud secrets versions access latest --secret=api-keys

# The cloud database and API from this terminal
source deploy/cloud-shell.sh
python index.py status          # uses the cloud database via cloud-sql-proxy on 127.0.0.1:6543
cv /books                       # the API through Cloudflare, with the cli service token

# Logs
gcloud run services logs read cv-api --region us-east1 --limit 50
gcloud run jobs executions list --job cv-index --region us-east1
```
