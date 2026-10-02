#!/usr/bin/env bash
# Copy the local Calibre library (LIBRARY_PATH in .env) to the library bucket.
# Only changed files are uploaded. Nothing in the bucket is deleted unless you
# pass --delete (then it mirrors the folder exactly).
#
#   deploy/sync-library.sh [--delete]
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

library=$(grep -E '^LIBRARY_PATH=' .env | cut -d= -f2-)
library=${library/#\~/$HOME}
[ -d "$library" ] || { echo "LIBRARY_PATH ($library) isn't a folder" >&2; exit 1; }

extra=()
[ "${1:-}" = "--delete" ] && extra+=(--delete-unmatched-destination-objects)
# Skip Finder files and Calibre's database (the .epub and .opf files are what's read).
gcloud storage rsync -r "$library" "gs://$BUCKET" ${extra[@]+"${extra[@]}"} \
  --exclude='(^|.*/)\.DS_Store$|^metadata\.db.*|^\.cal.*'
gcloud storage ls -r "gs://$BUCKET/**" | grep -c '\.epub$' | xargs echo "EPUBs in the bucket:"
