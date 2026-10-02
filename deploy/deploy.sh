#!/usr/bin/env bash
# Build the image, push it to Artifact Registry, and roll it out to the Cloud
# Run service (API + web page) and job (indexing). Run deploy/setup.sh first.
#
#   deploy/deploy.sh             # tagged with the current commit
#
# Access: the service is public only when env.yaml turns on the app's Cloudflare
# Access check (CF_ACCESS_AUD). Otherwise it requires Google IAM, reachable only
# through `gcloud run services proxy`.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

tag=$(git rev-parse --short HEAD)
[ -z "$(git status --porcelain)" ] || tag="$tag-dirty"
image="$IMAGE_BASE:$tag"

step "Build and push $image"
if [ -z "${SKIP_BUILD:-}" ]; then  # CI builds on its own runner and sets SKIP_BUILD
  docker buildx build --platform linux/amd64 --tag "$image" --push .
fi

common=(
  --image="$image" --region="$REGION"
  --env-vars-file=deploy/env.yaml
  --set-cloudsql-instances="$SQL_CONNECTION"
  --cpu=1 --memory=1Gi
)

step "Service $SERVICE"
if grep -q '^CF_ACCESS_AUD: ..' deploy/env.yaml; then
  access=--no-invoker-iam-check  # public: Cloudflare Access + the app's JWT check + API keys
else
  access=--invoker-iam-check     # private: Google IAM only
fi
gcloud run deploy "$SERVICE" "${common[@]}" \
  --service-account="$API_SA" --set-secrets="$API_SECRETS" \
  --execution-environment=gen2 \
  --clear-volumes --clear-volume-mounts \
  --add-volume=name=library,type=cloud-storage,bucket="$BUCKET",readonly=true \
  --add-volume-mount=volume=library,mount-path=/library \
  --min-instances=0 --max-instances=2 --concurrency=10 --timeout=3600 \
  "$access"

step "Job $JOB"
# One task, no retries (a failed job is never silently re-run and re-billed), and
# a deliberately invalid default command: executions must name a job id.
gcloud run jobs deploy "$JOB" "${common[@]}" \
  --service-account="$INDEXER_SA" --set-secrets="$INDEXER_SECRETS" \
  --command=python --args=index.py,run-job \
  --tasks=1 --max-retries=0 --task-timeout=3h

# The API starts executions (with the job id as an argument) and may do nothing else with the job.
gcloud run jobs add-iam-policy-binding "$JOB" --region="$REGION" --member="serviceAccount:$API_SA" \
  --role=roles/run.jobsExecutorWithOverrides >/dev/null

step "Deployed $tag"
gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)'
