"""Tests for RFC 7591 Dynamic Client Registration."""

from urllib.parse import urlsplit, parse_qs

from starlette.testclient import TestClient

import roamresearch_client_py.server as server


def test_oauth_metadata_includes_registration_endpoint(monkeypatch):
    """OAuth metadata should include registration_endpoint when dynamic registration is enabled."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {}})

    client = TestClient(server.create_app())
    resp = client.get("/.well-known/oauth-authorization-server")
    assert resp.status_code == 200
    data = resp.json()
    assert "registration_endpoint" in data
    assert data["registration_endpoint"] == "http://testserver/oauth/register"


def test_oauth_metadata_excludes_registration_endpoint_when_disabled(monkeypatch):
    """OAuth metadata should NOT include registration_endpoint when dynamic registration is disabled."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "0")
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: {
            "oauth": {
                "clients": [{"id": "static", "secret": "s", "scopes": ["mcp"], "redirect_uris": ["http://localhost/cb"]}]
            }
        },
    )

    client = TestClient(server.create_app())
    resp = client.get("/.well-known/oauth-authorization-server")
    assert resp.status_code == 200
    data = resp.json()
    assert "registration_endpoint" not in data


def test_dynamic_client_registration_success(monkeypatch):
    """Successfully register a new client dynamically."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {"scopes_supported": ["mcp", "read"]}})

    client = TestClient(server.create_app())
    resp = client.post(
        "/oauth/register",
        json={
            "redirect_uris": ["http://localhost:8080/callback"],
            "client_name": "Test Client",
            "grant_types": ["authorization_code"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "client_id" in data
    assert data["redirect_uris"] == ["http://localhost:8080/callback"]
    assert data["client_name"] == "Test Client"
    assert data["token_endpoint_auth_method"] == "none"
    assert "client_secret" not in data  # public client
    assert data["scope"] == "mcp read"


def test_dynamic_client_registration_with_secret(monkeypatch):
    """Register a confidential client with client_secret."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {}})

    client = TestClient(server.create_app())
    resp = client.post(
        "/oauth/register",
        json={
            "redirect_uris": ["http://localhost:8080/callback"],
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "client_id" in data
    assert "client_secret" in data
    assert len(data["client_secret"]) == 64  # two uuid4 hex strings
    assert data["client_secret_expires_at"] == 0


def test_dynamic_client_registration_missing_redirect_uris(monkeypatch):
    """Registration should fail without redirect_uris."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {}})

    client = TestClient(server.create_app())
    resp = client.post("/oauth/register", json={"client_name": "No URIs"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_redirect_uri"


def test_dynamic_client_registration_invalid_redirect_uri(monkeypatch):
    """Registration should fail with invalid redirect_uri format."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {}})

    client = TestClient(server.create_app())
    resp = client.post("/oauth/register", json={"redirect_uris": ["not-a-url"]})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_redirect_uri"


def test_dynamic_client_registration_invalid_scope(monkeypatch):
    """Registration should fail with unsupported scope."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {"scopes_supported": ["mcp"]}})

    client = TestClient(server.create_app())
    resp = client.post(
        "/oauth/register",
        json={
            "redirect_uris": ["http://localhost/cb"],
            "scope": "mcp admin",  # admin is not supported
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_client_metadata"


def test_dynamic_client_registration_disabled(monkeypatch):
    """Registration should fail when dynamic registration is disabled."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "0")
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: {
            "oauth": {
                "clients": [{"id": "static", "secret": "s", "scopes": ["mcp"], "redirect_uris": ["http://localhost/cb"]}]
            }
        },
    )

    client = TestClient(server.create_app())
    # The route should not even be registered
    resp = client.post("/oauth/register", json={"redirect_uris": ["http://localhost/cb"]})
    assert resp.status_code == 404  # route not found


def test_dynamic_client_full_flow(monkeypatch):
    """End-to-end: register client dynamically, then use it for authorization_code flow."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {"scopes_supported": ["mcp"]}})

    app = server.create_app()
    client = TestClient(app)

    # Step 1: Register client dynamically
    reg_resp = client.post(
        "/oauth/register",
        json={
            "redirect_uris": ["http://localhost:9999/callback"],
            "grant_types": ["authorization_code"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    client_id = reg_data["client_id"]

    # Step 2: Start authorization flow with PKCE
    code_verifier = "test-verifier-12345678901234567890"
    code_challenge = server._sha256_b64url(code_verifier)

    auth_resp = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": "http://localhost:9999/callback",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": "state123",
            "scope": "mcp",
        },
        follow_redirects=False,
    )
    assert auth_resp.status_code == 302
    location = auth_resp.headers["location"]
    q = parse_qs(urlsplit(location).query)
    code = q["code"][0]
    assert q["state"][0] == "state123"

    # Step 3: Exchange code for token
    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": "http://localhost:9999/callback",
        },
    )
    assert token_resp.status_code == 200
    token_data = token_resp.json()
    assert token_data["token_type"] == "Bearer"
    assert token_data["scope"] == "mcp"
    assert "access_token" in token_data


def test_dynamic_client_registration_wrong_content_type(monkeypatch):
    """Registration should fail with wrong Content-Type."""
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "1")
    monkeypatch.setattr(server, "load_config", lambda: {"oauth": {}})

    client = TestClient(server.create_app())
    resp = client.post(
        "/oauth/register",
        content="redirect_uris=http://localhost/cb",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_request"
