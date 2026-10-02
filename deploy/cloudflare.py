#!/usr/bin/env python3
"""Cloudflare in front of the Cloud Run service: DNS and the Access login wall.

    python3 deploy/cloudflare.py              # DNS only: Google can issue the certificate
    python3 deploy/cloudflare.py --proxied    # through Cloudflare, behind Access

Safe to re-run: everything is looked up by name and created or updated to match.
It needs CLOUDFLARE_API_TOKEN in .env (Zone DNS Edit; Account Access: Apps and
Policies / Service Tokens / Organizations, Identity Providers, and Groups Edit).

What it sets up:
- books.jacobpm.com: a CNAME to Cloud Run's domain mapping (ghs.googlehosted.com)
- login by email one-time PIN, allowed only for the owner's address
  (ACCESS_EMAIL, defaulting to the active gcloud account, so it isn't committed)
- service tokens with a Service Auth policy, for calls without a browser: one
  per client, so each can be revoked alone (`shelfspace`, and `cli` for
  deploy/cloud-shell.sh). Each id and secret goes straight into Secret Manager
  (<name>-access-client-id/-secret), because Cloudflare shows the secret only once.
- a Bypass for /.well-known/acme-challenge/, so Google can renew the certificate
  through Cloudflare

At the end it prints the values the API needs: CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ZONE_NAME = "jacobpm.com"
HOSTNAME = "books.jacobpm.com"
CLOUD_RUN_TARGET = "ghs.googlehosted.com"
APP_NAME = "Chapter and Verse"
ACME_APP_NAME = "Chapter and Verse: certificate renewal"
SERVICE_TOKENS = ("shelfspace", "cli")
SESSION = "168h"  # a week between logins on the phone
GCP_PROJECT = "chapter-and-verse-510420"
API = "https://api.cloudflare.com/client/v4"
ROOT = Path(__file__).resolve().parent.parent


def env_value(name: str) -> str:
    if os.getenv(name):
        return os.environ[name]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


TOKEN = env_value("CLOUDFLARE_API_TOKEN")


def cf(method: str, path: str, body: dict | None = None):
    request = urllib.request.Request(f"{API}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            return json.load(r)["result"]
    except urllib.error.HTTPError as e:
        errors = json.loads(e.read() or b"{}").get("errors", [])
        sys.exit(f"Cloudflare API {method} {path}: HTTP {e.code} {errors}")


def find(items: list[dict], **match) -> dict | None:
    return next((i for i in items if all(i.get(k) == v for k, v in match.items())), None)


def upsert(collection: str, existing: dict | None, body: dict, label: str) -> dict:
    if existing:
        result = cf("PUT", f"{collection}/{existing['id']}", body)
        print(f"  {label}: updated")
    else:
        result = cf("POST", collection, body)
        print(f"  {label}: created")
    return result


def store_secret(name: str, value: str) -> None:
    """Create the secret if needed and add `value` as a new version (via stdin, never argv)."""
    gcloud = ["gcloud", "--project", GCP_PROJECT, "--quiet"]
    if subprocess.run([*gcloud, "secrets", "describe", name], capture_output=True).returncode != 0:
        subprocess.run([*gcloud, "secrets", "create", name, "--replication-policy=automatic"],
                       check=True, capture_output=True)
    subprocess.run([*gcloud, "secrets", "versions", "add", name, "--data-file=-"],
                   input=value.encode(), check=True, capture_output=True)


def main() -> None:
    if not TOKEN:
        sys.exit("Set CLOUDFLARE_API_TOKEN in .env")
    proxied = "--proxied" in sys.argv[1:]
    email = env_value("ACCESS_EMAIL") or subprocess.run(
        ["gcloud", "config", "get", "account"], capture_output=True, text=True).stdout.strip()
    if "@" not in email:
        sys.exit("Couldn't tell which email may log in: set ACCESS_EMAIL")

    zone = cf("GET", f"/zones?name={ZONE_NAME}")[0]
    account = zone["account"]["id"]
    access = f"/accounts/{account}/access"

    print(f"DNS: {HOSTNAME} → {CLOUD_RUN_TARGET} ({'proxied, behind Access' if proxied else 'DNS only'})")
    records = cf("GET", f"/zones/{zone['id']}/dns_records?name={HOSTNAME}")
    upsert(f"/zones/{zone['id']}/dns_records", find(records, type="CNAME"),
           {"type": "CNAME", "name": HOSTNAME, "content": CLOUD_RUN_TARGET, "proxied": proxied, "ttl": 1,
            "comment": "Chapter and Verse (Cloud Run domain mapping)"}, "CNAME")

    print("Login method: email one-time PIN")
    otp = find(cf("GET", f"{access}/identity_providers"), type="onetimepin")
    if not otp:
        otp = cf("POST", f"{access}/identity_providers", {"name": "One-time PIN", "type": "onetimepin", "config": {}})
        print("  created")
    else:
        print("  already on")

    print("Service tokens")
    existing = cf("GET", f"{access}/service_tokens")
    tokens = []
    for name in SERVICE_TOKENS:
        token = find(existing, name=name)
        if not token:
            token = cf("POST", f"{access}/service_tokens", {"name": name, "duration": "8760h"})
            store_secret(f"{name}-access-client-id", token["client_id"])
            store_secret(f"{name}-access-client-secret", token["client_secret"])
            print(f"  {name}: created; id and secret stored in Secret Manager ({name}-access-client-*)")
        else:
            print(f"  {name}: exists (expires {token.get('expires_at', '?')})")
        tokens.append(token)

    print("Policies")
    policies = cf("GET", f"{access}/policies")
    owner = upsert(f"{access}/policies", find(policies, name="Chapter and Verse: owner"),
                   {"name": "Chapter and Verse: owner", "decision": "allow",
                    "include": [{"email": {"email": email}}], "session_duration": SESSION}, "owner (email PIN)")
    service = upsert(f"{access}/policies", find(policies, name="Chapter and Verse: services"),
                     {"name": "Chapter and Verse: services", "decision": "non_identity",
                      "include": [{"service_token": {"token_id": t["id"]}} for t in tokens]},
                     f"{', '.join(SERVICE_TOKENS)} (Service Auth)")
    everyone = upsert(f"{access}/policies", find(policies, name="Chapter and Verse: certificate renewal"),
                      {"name": "Chapter and Verse: certificate renewal", "decision": "bypass",
                       "include": [{"everyone": {}}]}, "certificate renewal (Bypass)")

    print("Applications")
    apps = cf("GET", f"{access}/apps")
    app = upsert(f"{access}/apps", find(apps, name=APP_NAME), {
        "name": APP_NAME, "type": "self_hosted", "domain": HOSTNAME, "session_duration": SESSION,
        "allowed_idps": [otp["id"]], "auto_redirect_to_identity": True, "app_launcher_visible": False,
        "policies": [{"id": owner["id"], "precedence": 1}, {"id": service["id"], "precedence": 2}],
    }, HOSTNAME)
    # A more specific path takes precedence over the app above.
    upsert(f"{access}/apps", find(apps, name=ACME_APP_NAME), {
        "name": ACME_APP_NAME, "type": "self_hosted", "domain": f"{HOSTNAME}/.well-known/acme-challenge",
        "app_launcher_visible": False, "policies": [{"id": everyone["id"], "precedence": 1}],
    }, f"{HOSTNAME}/.well-known/acme-challenge")

    team = cf("GET", f"{access}/organizations")["auth_domain"]
    print(f"\nFor deploy/env.yaml:\n  CF_ACCESS_TEAM_DOMAIN: {team}\n  CF_ACCESS_AUD: {app['aud']}")


if __name__ == "__main__":
    main()
