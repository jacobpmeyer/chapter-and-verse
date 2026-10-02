# Deployment (GCP)

Scripts that create and update the cloud deployment. Every resource name lives in
[`config.sh`](config.sh). The non-secret runtime settings are in [`env.yaml`](env.yaml).

| Script | When |
|---|---|
| [`setup.sh`](setup.sh) | once, before the first deploy. It's safe to re-run: it only creates what's missing and never replaces a secret. |
| [`sync-library.sh`](sync-library.sh) | after adding books in Calibre, then `POST /library/scan` |
| [`deploy.sh`](deploy.sh) | every release: build, push, update the service and the job |

## What runs where

| Piece | GCP resource | Notes |
|---|---|---|
| API + web page | Cloud Run service `cv-api` | request-based billing, scales to zero, at most 2 instances |
| Indexing | Cloud Run Job `cv-index` | one execution per job: `python index.py run-job <id>`, no retries |
| Database | Cloud SQL `cv-db`, PostgreSQL 18 + pgvector | `db-f1-micro`, 7 daily backups, reached only through the Cloud SQL connector |
| EPUB library | bucket `chapter-and-verse-510420-library` | mounted read-only at `/library` (Cloud Storage FUSE) |
| Keys and the database URL | Secret Manager | `anthropic-api-key`, `voyage-api-key`, `api-keys`, `database-url` |
| Images | Artifact Registry `cv` | a cleanup policy keeps the 5 newest |

Each workload has its own service account:
- **`cv-api`:** connects to Cloud SQL, reads its 4 secrets and the library bucket, and starts
  executions of `cv-index`. That last permission comes from
  `roles/run.jobsExecutorWithOverrides`, granted on that one job only.
- **`cv-indexer`:** connects to Cloud SQL and reads 3 secrets.

## Common tasks

```sh
# The deployed API's key (to paste into the web page)
gcloud secrets versions access latest --secret=api-keys

# Reach the private service from this Mac, with your Google login (until it's public)
gcloud run services proxy cv-api --region us-east1 --port 8081   # then http://localhost:8081

# The cloud database from this Mac (e.g. index.py status), through the IAM-checked proxy
cloud-sql-proxy --port 6543 chapter-and-verse-510420:us-east1:cv-db &
export DATABASE_URL=$(gcloud secrets versions access latest --secret=database-url |
  sed -E 's|@/([^?]+)\?host=.*|@127.0.0.1:6543/\1|')
python index.py status

# Logs
gcloud run services logs read cv-api --region us-east1 --limit 50
gcloud run jobs executions list --job cv-index --region us-east1
```
