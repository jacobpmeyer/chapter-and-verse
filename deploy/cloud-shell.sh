# Work with the deployed service from this terminal. Source it, don't run it:
#
#   source deploy/cloud-shell.sh
#
# - Starts cloud-sql-proxy on 127.0.0.1:6543 (unless it's already running) and
#   points DATABASE_URL at it, so index.py and agent.py in this terminal use the
#   cloud database instead of the local one.
# - Defines `cv`, which calls the deployed API through Cloudflare Access with the
#   `cli` service token and the cloud API key:
#       cv /books
#       cv /books/7/estimate
#       cv /books/7/index '{"max_cost_usd": 3}'      # a JSON body makes it a POST
# Credentials come from Secret Manager into this shell's environment only.

_cv_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)
_cv_project=chapter-and-verse-510420
_cv_secret() { gcloud secrets versions access latest --secret="$1" --project "$_cv_project"; }

if ! nc -z 127.0.0.1 6543 2>/dev/null; then
  # Braces matter: in zsh, "$_cv_project:us-east1" would apply the :u (uppercase) modifier.
  cloud-sql-proxy --port 6543 "${_cv_project}:us-east1:cv-db" > /tmp/cloud-sql-proxy.log 2>&1 &
  disown 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do nc -z 127.0.0.1 6543 2>/dev/null && break; sleep 1; done
fi
export DATABASE_URL=$(_cv_secret database-url | sed -E 's|@/([^?]+)\?host=.*|@127.0.0.1:6543/\1|')

CV_URL=https://books.jacobpm.com
_CV_KEY=$(_cv_secret api-keys)
_CV_ID=$(_cv_secret cli-access-client-id)
_CV_SECRET=$(_cv_secret cli-access-client-secret)

cv() {  # cv <path> [json body]
  local route=$1; shift  # not `path`: in zsh that is tied to $PATH
  curl -sS "$CV_URL$route" -H "Authorization: Bearer $_CV_KEY" \
    -H "CF-Access-Client-Id: $_CV_ID" -H "CF-Access-Client-Secret: $_CV_SECRET" \
    ${1:+-X POST -H "Content-Type: application/json" -d "$1"}
  echo
}

echo "Cloud database on 127.0.0.1:6543 (DATABASE_URL set for this terminal); \`cv /books\` calls $CV_URL"
