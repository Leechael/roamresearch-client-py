from starlette.testclient import TestClient

import roamresearch_client_py.server as server


def test_options_sse_preflight_returns_204_not_405(monkeypatch):
    # Ensure oauth is disabled so create_app doesn't require oauth config.
    monkeypatch.delenv("OAUTH_ENABLED", raising=False)
    monkeypatch.delenv("ROAM_CONFIG_FILE", raising=False)
    client = TestClient(server.create_app())

    resp = client.options(
        "/sse",
        headers={
            "origin": "http://example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 204


def test_cors_allows_origin_when_configured(monkeypatch):
    monkeypatch.setenv("MCP_CORS_ALLOW_ORIGIN_REGEX", r"^http://example\.com$")
    client = TestClient(server.create_app())

    resp = client.options(
        "/sse",
        headers={
            "origin": "http://example.com",
            "access-control-request-method": "GET",
        },
    )
    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "http://example.com"


def test_cors_auto_allows_origin_from_host_and_forwarded_proto(monkeypatch):
    monkeypatch.setenv("MCP_CORS_AUTO_ALLOW_ORIGIN_FROM_HOST", "1")
    client = TestClient(server.create_app())

    resp = client.options(
        "/sse",
        headers={
            "host": "example.com",
            "x-forwarded-proto": "https",
            "origin": "https://example.com",
            "access-control-request-method": "GET",
        },
    )
    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "https://example.com"
