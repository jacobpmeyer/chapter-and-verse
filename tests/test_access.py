"""The Cloudflare Access check in api.py. Tokens are signed here with RSA keys
generated for the test, and the key set is a fake, so nothing touches the
network or a database."""

from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager
from types import SimpleNamespace as NS

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import api

TEAM = "example.cloudflareaccess.com"
AUD = "a1b2c3-application-aud-tag"
KEY = "test-key"


def new_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


TEAM_KEY = new_key()
OTHER_KEY = new_key()


class FakeJWKS:
    """Stands in for PyJWKClient: always hands out the team's public key."""

    def get_signing_key_from_jwt(self, token):
        return NS(key=TEAM_KEY.public_key())


def token(key=TEAM_KEY, alg="RS256", drop=(), **overrides):
    now = int(time.time())
    claims = {"aud": [AUD], "iss": f"https://{TEAM}", "iat": now, "exp": now + 600,
              "email": "reader@example.com", "sub": "user-id", "type": "app", **overrides}
    for name in drop:
        claims.pop(name)
    return jwt.encode(claims, key, algorithm=alg)


def service_token():
    """What Access sends for a request authenticated with a service token: no email,
    `common_name` is the token's client id."""
    return token(drop=("email",), sub="", common_name="0123abcd.access")


class FakeDatabase:
    @contextmanager
    def connection(self):
        yield NS(execute=lambda sql: None)


@pytest.fixture
def client(monkeypatch):
    """The app with the Access check on (startup isn't run, so no database is needed)."""
    monkeypatch.setattr(api, "settings", dataclasses.replace(api.settings, api_keys=(KEY,), auth_disabled=False))
    monkeypatch.setattr(api.app.state, "services", NS(database=FakeDatabase()))
    monkeypatch.setattr(api.app.state, "access", api.AccessVerifier(TEAM, AUD, jwks_client=FakeJWKS()))
    return TestClient(api.app)


def jwt_header(t):
    return {"Cf-Access-Jwt-Assertion": t}


@pytest.mark.parametrize("path", ["/", "/books", "/docs", "/openapi.json"])
def test_requests_without_an_access_token_are_rejected(client, path):
    r = client.get(path, headers={"Authorization": f"Bearer {KEY}"})  # a valid API key isn't enough
    assert r.status_code == 403 and r.json()["detail"] == "missing Cloudflare Access token"


def test_a_logged_in_user_gets_through_and_the_api_key_is_still_required(client):
    assert client.get("/", headers=jwt_header(token())).status_code == 200
    r = client.get("/books", headers=jwt_header(token()))  # Access passed; no API key
    assert r.status_code == 401


def test_a_service_token_gets_through(client):
    assert client.get("/", headers=jwt_header(service_token())).status_code == 200


@pytest.mark.parametrize("bad", [
    pytest.param(lambda: token(aud=["another-application"]), id="wrong audience"),
    pytest.param(lambda: token(iss="https://evil.cloudflareaccess.com"), id="wrong issuer"),
    pytest.param(lambda: token(exp=int(time.time()) - 60), id="expired"),
    pytest.param(lambda: token(key=OTHER_KEY), id="signed by another key"),
    pytest.param(lambda: token(drop=("exp",)), id="no expiry"),
    pytest.param(lambda: token(key="shared-secret-of-at-least-32-bytes!", alg="HS256"), id="HS256 instead of RS256"),
    pytest.param(lambda: "not-a-jwt", id="garbage"),
])
def test_invalid_tokens_are_rejected(client, bad):
    r = client.get("/", headers=jwt_header(bad()))
    assert r.status_code == 403 and r.json()["detail"] == "invalid Cloudflare Access token"


def test_health_is_exempt_so_cloud_run_probes_work(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_the_check_is_off_when_not_behind_access(client, monkeypatch):
    monkeypatch.setattr(api.app.state, "access", None)
    assert client.get("/").status_code == 200


def configure(monkeypatch, domain, aud):
    monkeypatch.setattr(api, "settings", dataclasses.replace(api.settings, cf_access_team_domain=domain,
                                                             cf_access_aud=aud))


def test_half_a_configuration_refuses_to_start(monkeypatch):
    for domain, aud in [(TEAM, ""), ("", AUD)]:
        configure(monkeypatch, domain, aud)
        with pytest.raises(RuntimeError, match="both CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD"):
            api.access_verifier()
    configure(monkeypatch, "", "")
    assert api.access_verifier() is None


def test_configuration_builds_the_issuer_and_key_url(monkeypatch):
    configure(monkeypatch, f"https://{TEAM}/", AUD)  # pasted with a scheme and slash: still works
    verifier = api.access_verifier()
    assert verifier.issuer == f"https://{TEAM}" and verifier.audience == AUD
    assert verifier.jwks.uri == f"https://{TEAM}/cdn-cgi/access/certs"
