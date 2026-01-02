import base64

import pytest
from starlette.testclient import TestClient

import roamresearch_client_py.server as server


def _make_client(monkeypatch, *, require_auth: bool, allow_query: bool) -> TestClient:
    monkeypatch.setenv("OAUTH_ENABLED", "1")
    monkeypatch.setenv("OAUTH_REQUIRE_AUTH", "1" if require_auth else "0")
    monkeypatch.setenv("OAUTH_ALLOW_ACCESS_TOKEN_QUERY", "1" if allow_query else "0")
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("OAUTH_AUDIENCE", "test-aud")
    monkeypatch.setenv("OAUTH_ACCESS_TOKEN_TTL_SECONDS", "3600")

    monkeypatch.setattr(
        server,
        "load_config",
        lambda: {
            "oauth": {
                "scopes_supported": ["mcp"],
                "clients": [
                    {"id": "c1", "secret": "s1", "scopes": ["mcp"]},
                ],
            }
        },
    )
    return TestClient(server.create_app())


def _basic(client_id: str, client_secret: str) -> str:
    token = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def test_oauth_discovery_uses_host_header(monkeypatch):
    client = _make_client(monkeypatch, require_auth=False, allow_query=False)
    resp = client.get("/.well-known/oauth-authorization-server", headers={"host": "example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["issuer"] == "http://example.com"
    assert body["token_endpoint"] == "http://example.com/oauth/token"
    assert "client_credentials" in body["grant_types_supported"]


def test_oauth_token_issues_jwt_and_defaults_scope_to_all(monkeypatch):
    client = _make_client(monkeypatch, require_auth=False, allow_query=False)
    resp = client.post(
        "/oauth/token",
        headers={"authorization": _basic("c1", "s1"), "content-type": "application/x-www-form-urlencoded"},
        data="grant_type=client_credentials",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "Bearer"
    assert body["scope"] == "mcp"
    assert isinstance(body["access_token"], str) and body["access_token"].count(".") == 2


def test_require_auth_blocks_mcp_without_token(monkeypatch):
    client = _make_client(monkeypatch, require_auth=True, allow_query=False)
    resp = client.get("/mcp")
    assert resp.status_code == 401


def _settings_for_middleware_tests() -> server.OAuthSettings:
    return server.OAuthSettings(
        enabled=True,
        require_auth=True,
        allow_access_token_query=False,
        allow_dynamic_registration=False,
        audience="test-aud",
        signing_secret="test-secret",
        access_token_ttl_seconds=3600,
        scopes_supported=["mcp"],
        clients_by_id={
            "c1": server.OAuthClientConfig(client_id="c1", client_secret="s1", scopes=["mcp"], redirect_uris=[]),
        },
    )


def _request_for_middleware(
    *,
    path: str,
    host: str = "example.com",
    scheme: str = "http",
    headers: list[tuple[bytes, bytes]] | None = None,
    query_string: bytes = b"",
) -> server.Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": scheme,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string,
        "headers": [(b"host", host.encode("ascii"))] + (headers or []),
        "client": ("testclient", 12345),
        "server": (host, 80),
    }
    return server.Request(scope, receive)


@pytest.mark.anyio
async def test_middleware_allows_valid_bearer_token():
    settings = _settings_for_middleware_tests()
    now = int(server.time.time())
    token = server._jwt_encode(
        {
            "iss": "http://example.com",
            "sub": "c1",
            "aud": settings.audience,
            "iat": now,
            "exp": now + 3600,
            "scope": "mcp",
        },
        settings.signing_secret,
    )

    called = {"ok": False}
    response_started = {}

    async def inner_app(scope, receive, send):
        called["ok"] = True
        response = server.PlainTextResponse("ok")
        await response(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started["status"] = message["status"]

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [(b"host", b"example.com"), (b"authorization", f"Bearer {token}".encode("ascii"))],
        "client": ("testclient", 12345),
        "server": ("example.com", 80),
    }

    middleware = server.OAuthAuthMiddleware(inner_app, settings=settings)
    await middleware(scope, receive, send)

    assert response_started["status"] == 200
    assert called["ok"] is True


@pytest.mark.anyio
async def test_middleware_allows_access_token_query_when_enabled():
    settings = _settings_for_middleware_tests()
    settings = server.OAuthSettings(**{**settings.__dict__, "allow_access_token_query": True})
    now = int(server.time.time())
    token = server._jwt_encode(
        {
            "iss": "http://example.com",
            "sub": "c1",
            "aud": settings.audience,
            "iat": now,
            "exp": now + 3600,
            "scope": "mcp",
        },
        settings.signing_secret,
    )

    response_started = {}

    async def inner_app(scope, receive, send):
        response = server.PlainTextResponse("ok")
        await response(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started["status"] = message["status"]

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": f"access_token={token}".encode("ascii"),
        "headers": [(b"host", b"example.com")],
        "client": ("testclient", 12345),
        "server": ("example.com", 80),
    }

    middleware = server.OAuthAuthMiddleware(inner_app, settings=settings)
    await middleware(scope, receive, send)

    assert response_started["status"] == 200


@pytest.mark.anyio
async def test_middleware_does_not_require_token_when_require_auth_false():
    settings = _settings_for_middleware_tests()
    settings = server.OAuthSettings(**{**settings.__dict__, "require_auth": False})

    response_started = {}

    async def inner_app(scope, receive, send):
        response = server.PlainTextResponse("ok")
        await response(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started["status"] = message["status"]

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [(b"host", b"example.com")],
        "client": ("testclient", 12345),
        "server": ("example.com", 80),
    }

    middleware = server.OAuthAuthMiddleware(inner_app, settings=settings)
    await middleware(scope, receive, send)

    assert response_started["status"] == 200


@pytest.mark.anyio
async def test_middleware_returns_generic_error_for_malformed_token():
    """Malformed tokens should return generic error, not leak exception details."""
    settings = _settings_for_middleware_tests()

    # Token with invalid base64 in payload
    malformed_token = "eyJhbGciOiJIUzI1NiJ9.!!!invalid-base64!!!.sig"

    response_started = {}
    response_body = []

    async def inner_app(scope, receive, send):
        response = server.PlainTextResponse("ok")
        await response(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started["status"] = message["status"]
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [(b"host", b"example.com"), (b"authorization", f"Bearer {malformed_token}".encode("ascii"))],
        "client": ("testclient", 12345),
        "server": ("example.com", 80),
    }

    middleware = server.OAuthAuthMiddleware(inner_app, settings=settings)
    await middleware(scope, receive, send)

    assert response_started["status"] == 401
    # Should return generic message, not leak binascii.Error or JSONDecodeError details
    body = b"".join(response_body).decode("utf-8")
    assert "invalid_token: malformed" in body
    assert "binascii" not in body.lower()
    assert "json" not in body.lower()