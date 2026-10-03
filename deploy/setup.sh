#!/usr/bin/env bash
# One-time GCP setup for Chapter and Verse. Safe to re-run: every step checks
# what already exists and only creates what's missing. Secrets are read from
# .env or generated, and never printed.
#
#   deploy/setup.sh          # then deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

exists() { "$@" >/dev/null 2>&1; }
env_value() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }

step "APIs"
gcloud services enable run.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iam.googleapis.com iamcredentials.googleapis.com storage.googleapis.com

step "Service accounts (one per workload, each with only what it needs)"
for sa in cv-api:"Chapter and Verse API" cv-indexer:"Chapter and Verse indexing jobs"; do
  name=${sa%%:*}
  exists gcloud iam service-accounts describe "$name@$PROJECT.iam.gserviceaccount.com" ||
    gcloud iam service-accounts create "$name" --display-name="${sa#*:}"
done
for sa in "$API_SA" "$INDEXER_SA"; do  # connect through the Cloud SQL connector
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$sa" \
    --role=roles/cloudsql.client --condition=None >/dev/null
done

step "Artifact Registry repository (keeps the 5 newest images)"
exists gcloud artifacts repositories describe "$REPO" --location="$REGION" ||
  gcloud artifacts repositories create "$REPO" --location="$REGION" --repository-format=docker \
    --description="Chapter and Verse images"
gcloud artifacts repositories set-cleanup-policies "$REPO" --location="$REGION" \
  --policy=deploy/cleanup-policy.json --no-dry-run >/dev/null

step "Library bucket (private; the API reads it through a read-only mount)"
exists gcloud storage buckets describe "gs://$BUCKET" ||
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access \
    --public-access-prevention
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$API_SA" \
  --role=roles/storage.objectViewer >/dev/null

step "Cloud SQL: PostgreSQL 18 (pgvector), smallest tier, daily backups"
# Shared-core db-f1-micro has no SLA: the cost/availability trade-off for a single-user service.
# Enterprise edition must be explicit (PostgreSQL 16+ defaults to Enterprise Plus). No authorized
# networks: connections only arrive through the IAM-checked Cloud SQL connector/proxy.
exists gcloud sql instances describe "$SQL_INSTANCE" ||
  gcloud sql instances create "$SQL_INSTANCE" --database-version=POSTGRES_18 --edition=ENTERPRISE \
    --tier=db-f1-micro --region="$REGION" --availability-type=ZONAL \
    --storage-type=SSD --storage-size=10 --storage-auto-increase \
    --backup-start-time=08:00 --retained-backups-count=7 \
    --ssl-mode=ENCRYPTED_ONLY --deletion-protection
exists gcloud sql databases describe "$DB_NAME" --instance="$SQL_INSTANCE" ||
  gcloud sql databases create "$DB_NAME" --instance="$SQL_INSTANCE"

step "Secrets"
secret() {  # secret <name>: create it and add a first version from stdin, unless it has one
  local name=$1 value
  value=$(cat)
  [ -n "$value" ] || { echo "no value for $name" >&2; exit 1; }
  exists gcloud secrets describe "$name" || gcloud secrets create "$name" --replication-policy=automatic >/dev/null
  if [ -z "$(gcloud secrets versions list "$name" --filter=state:ENABLED --format='value(name)' --limit=1)" ]; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- >/dev/null
    echo "  $name: added"
  else
    echo "  $name: already set (kept)"
  fi
}
env_value ANTHROPIC_API_KEY | secret anthropic-api-key
env_value VOYAGE_API_KEY | secret voyage-api-key
# A key just for the deployed service: the local one never leaves this machine.
python3 -c "import secrets; print(secrets.token_urlsafe(32))" | secret api-keys

# The database user's password is generated once and lives only inside the database-url secret.
if ! exists gcloud secrets describe database-url ||
   [ -z "$(gcloud secrets versions list database-url --filter=state:ENABLED --format='value(name)' --limit=1)" ]; then
  password=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
  if exists gcloud sql users describe "$DB_USER" --instance="$SQL_INSTANCE"; then
    gcloud sql users set-password "$DB_USER" --instance="$SQL_INSTANCE" --password="$password" >/dev/null
  else
    gcloud sql users create "$DB_USER" --instance="$SQL_INSTANCE" --password="$password" >/dev/null
  fi
  printf 'postgresql://%s:%s@/%s?host=/cloudsql/%s' "$DB_USER" "$password" "$DB_NAME" "$SQL_CONNECTION" |
    secret database-url
  unset password
else
  echo "  database-url: already set (kept)"
fi

for s in anthropic-api-key voyage-api-key database-url; do
  for sa in "$API_SA" "$INDEXER_SA"; do
    gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:$sa" \
      --role=roles/secretmanager.secretAccessor >/dev/null
  done
done
gcloud secrets add-iam-policy-binding api-keys --member="serviceAccount:$API_SA" \
  --role=roles/secretmanager.secretAccessor >/dev/null

step "GitHub Actions deploys (Workload Identity Federation, no keys)"
gcloud services enable sts.googleapis.com
number=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
exists gcloud iam workload-identity-pools describe "$WIF_POOL" --location=global ||
  gcloud iam workload-identity-pools create "$WIF_POOL" --location=global --display-name="GitHub Actions"
# Only pushes to main of this repository (by numeric id) can get a token at all.
condition="assertion.repository_id == '$GITHUB_REPO_ID' && assertion.repository_owner_id == '$GITHUB_OWNER_ID' && assertion.ref == 'refs/heads/main'"
mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_id=assertion.repository_id,attribute.ref=assertion.ref"
if exists gcloud iam workload-identity-pools providers describe "$WIF_PROVIDER" --location=global --workload-identity-pool="$WIF_POOL"; then
  gcloud iam workload-identity-pools providers update-oidc "$WIF_PROVIDER" --location=global \
    --workload-identity-pool="$WIF_POOL" --attribute-mapping="$mapping" --attribute-condition="$condition" >/dev/null
else
  gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" --location=global \
    --workload-identity-pool="$WIF_POOL" --display-name="$GITHUB_REPO" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="$mapping" --attribute-condition="$condition"
fi

exists gcloud iam service-accounts describe "$DEPLOYER_SA" ||
  gcloud iam service-accounts create cv-deployer --display-name="Chapter and Verse deploys (GitHub Actions)"
# The repository's workflows may act as the deployer...
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$number/locations/global/workloadIdentityPools/$WIF_POOL/attribute.repository_id/$GITHUB_REPO_ID" >/dev/null
# ...which can push images to this one repository, deploy Cloud Run revisions (not change who may
# call them: that's run.admin), and run them as the two runtime service accounts.
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$DEPLOYER_SA" \
  --role=roles/run.developer --condition=None >/dev/null
gcloud artifacts repositories add-iam-policy-binding "$REPO" --location="$REGION" \
  --member="serviceAccount:$DEPLOYER_SA" --role=roles/artifactregistry.writer >/dev/null
for sa in "$API_SA" "$INDEXER_SA"; do
  gcloud iam service-accounts add-iam-policy-binding "$sa" --member="serviceAccount:$DEPLOYER_SA" \
    --role=roles/iam.serviceAccountUser >/dev/null
done
echo "  workload_identity_provider: projects/$number/locations/global/workloadIdentityPools/$WIF_POOL/providers/$WIF_PROVIDER"
echo "  service_account: $DEPLOYER_SA"

step "Done. Next: deploy/sync-library.sh, restore the database, then deploy/deploy.sh"
