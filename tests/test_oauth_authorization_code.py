import base64
from urllib.parse import urlsplit, parse_qs

from starlette.testclient import TestClient

import roamresearch_client_py.server as server


def test_oauth_authorize_and_exchange_code_with_pkce(monkeypatch):
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_AUDIENCE", "test-aud")

    monkeypatch.setattr(
        server,
        "load_config",
        lambda: {
            "oauth": {
                "scopes_supported": ["mcp"],
                "clients": [
                    {
                        "id": "local-dev",
                        "secret": "",  # public client (PKCE)
                        "scopes": ["mcp"],
                        "redirect_uris": ["http://localhost:6274/oauth/callback"],
                    }
                ],
            }
        },
    )

    client = TestClient(server.create_app())

    code_verifier = "verifier-123"
    code_challenge = server._sha256_b64url(code_verifier)

    auth_resp = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": "local-dev",
            "redirect_uri": "http://localhost:6274/oauth/callback",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": "abc",
            "scope": "mcp",
        },
        follow_redirects=False,
    )
    assert auth_resp.status_code == 302
    location = auth_resp.headers["location"]
    q = parse_qs(urlsplit(location).query)
    code = q["code"][0]
    assert q["state"][0] == "abc"

    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": "local-dev",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": "http://localhost:6274/oauth/callback",
        },
    )
    assert token_resp.status_code == 200
    body = token_resp.json()
    assert body["token_type"] == "Bearer"
    assert body["scope"] == "mcp"


def test_oauth_client_credentials_requires_secret(monkeypatch):
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: {
            "oauth": {
                "clients": [
                    {"id": "public", "secret": "", "scopes": ["mcp"], "redirect_uris": ["http://localhost/cb"]},
                ]
            }
        },
    )
    client = TestClient(server.create_app())

    resp = client.post("/oauth/token", data="grant_type=client_credentials&client_id=public")
    assert resp.status_code == 401
