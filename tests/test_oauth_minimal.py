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
        audience="test-aud",
        signing_secret="test-secret",
        access_token_ttl_seconds=3600,
        scopes_supported=["mcp"],
        clients_by_id={
            "c1": server.OAuthClientConfig(client_id="c1", client_secret="s1", scopes=["mcp"]),
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

    req = _request_for_middleware(headers=[(b"authorization", f"Bearer {token}".encode("ascii"))], path="/mcp")
    middleware = server.OAuthAuthMiddleware(lambda scope, receive, send: None, settings=settings)

    called = {"ok": False}

    async def call_next(_request):
        called["ok"] = True
        return server.PlainTextResponse("ok")

    resp = await middleware.dispatch(req, call_next)
    assert resp.status_code == 200
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

    req = _request_for_middleware(path="/mcp", query_string=f"access_token={token}".encode("ascii"))
    middleware = server.OAuthAuthMiddleware(lambda scope, receive, send: None, settings=settings)

    async def call_next(_request):
        return server.PlainTextResponse("ok")

    resp = await middleware.dispatch(req, call_next)
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_middleware_does_not_require_token_when_require_auth_false():
    settings = _settings_for_middleware_tests()
    settings = server.OAuthSettings(**{**settings.__dict__, "require_auth": False})
    req = _request_for_middleware(path="/mcp")
    middleware = server.OAuthAuthMiddleware(lambda scope, receive, send: None, settings=settings)

    async def call_next(_request):
        return server.PlainTextResponse("ok")

    resp = await middleware.dispatch(req, call_next)
    assert resp.status_code == 200
