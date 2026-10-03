# Names shared by setup.sh and deploy.sh. Sourced, not run.

PROJECT=chapter-and-verse-510420
REGION=us-east1

SQL_INSTANCE=cv-db
SQL_CONNECTION="$PROJECT:$REGION:$SQL_INSTANCE"
DB_NAME=library_rag
DB_USER=library_rag

REPO=cv                                     # Artifact Registry (Docker)
IMAGE_BASE="$REGION-docker.pkg.dev/$PROJECT/$REPO/chapter-and-verse"
BUCKET="$PROJECT-library"                   # the EPUB library, mounted read-only at /library

SERVICE=cv-api                              # Cloud Run service: the API and web page
JOB=cv-index                                # Cloud Run Job: one execution per indexing job
API_SA="cv-api@$PROJECT.iam.gserviceaccount.com"
INDEXER_SA="cv-indexer@$PROJECT.iam.gserviceaccount.com"
DEPLOYER_SA="cv-deployer@$PROJECT.iam.gserviceaccount.com"   # GitHub Actions releases

# GitHub Actions deploys with Workload Identity Federation: no service account keys.
# Numeric ids, not just names, so a renamed or re-created repo can't inherit access.
GITHUB_REPO=jacobpmeyer/chapter-and-verse
GITHUB_REPO_ID=1398895925
GITHUB_OWNER_ID=49496782
WIF_POOL=github
WIF_PROVIDER=chapter-and-verse

# Secret Manager secret -> environment variable
API_SECRETS="ANTHROPIC_API_KEY=anthropic-api-key:latest,VOYAGE_API_KEY=voyage-api-key:latest,API_KEYS=api-keys:latest,DATABASE_URL=database-url:latest"
INDEXER_SECRETS="ANTHROPIC_API_KEY=anthropic-api-key:latest,VOYAGE_API_KEY=voyage-api-key:latest,DATABASE_URL=database-url:latest"

gcloud() { command gcloud --project "$PROJECT" --quiet "$@"; }
step() { printf '\n== %s\n' "$*"; }
