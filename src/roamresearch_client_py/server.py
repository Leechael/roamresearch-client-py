from typing import Any, List, cast
from dataclasses import dataclass
import argparse
import pprint
from itertools import chain
import logging
import signal
import os
import asyncio
import uuid
from pathlib import Path
import traceback
import sqlite3
from datetime import datetime
import json
import re
import base64
import hashlib
import hmac
import time
from urllib.parse import parse_qs
from urllib.parse import urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import pendulum

from .client import RoamClient, create_page, create_block
from .config import get_env_or_config, get_config_dir, load_config
from .formatter import format_block, format_block_as_markdown, extract_ref, expand_refs_in_text
from .gfm_to_roam import gfm_to_batch_actions, gfm_to_blocks, normalize_task_marker
from .diff import parse_existing_blocks, diff_block_trees, generate_batch_actions


class CancelledErrorFilter(logging.Filter):
    def filter(self, record):
        return "asyncio.exceptions.CancelledError" not in record.getMessage()

for logger_name in ("uvicorn.error", "uvicorn.access", "uvicorn", "starlette"):
    logging.getLogger(logger_name).addFilter(CancelledErrorFilter())


def _get_transport_security() -> TransportSecuritySettings | None:
    """Get transport security settings from config."""
    allowed_hosts_str = get_env_or_config("ALLOWED_HOSTS", "mcp.allowed_hosts")
    if not allowed_hosts_str:
        # Default: disable DNS rebinding protection for remote MCP servers
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    allowed_hosts = [h.strip() for h in str(allowed_hosts_str).split(",") if h.strip()]
    # Also allow localhost variants
    allowed_hosts.extend(["127.0.0.1:*", "localhost:*", "[::1]:*"])
    allowed_origins = [f"https://{h.split(':')[0]}" for h in allowed_hosts if not h.startswith(("127.", "localhost", "[::1]"))]
    allowed_origins.extend(["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )

mcp = FastMCP(name="RoamResearch", stateless_http=True, transport_security=_get_transport_security())
logger = logging.getLogger(__name__)

# Heuristic: when updating a single block, treat large/multiline content as
# markdown that should be expanded into children blocks.
BLOCK_UPDATE_MARKDOWN_NEWLINE_THRESHOLD = 5

# OAuth / Auth (minimal, config-only; no DB/users)
OAUTH_REQUIRED_SCOPE = "mcp"

# Background task management
background_tasks: set[asyncio.Task] = set()


def create_background_task(coro):
    """Create a background task and track it for graceful shutdown."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


async def shutdown_background_tasks(timeout=30):
    """Wait for all background tasks to complete with timeout."""
    if background_tasks:
        logger.info(f"Waiting for {len(background_tasks)} background tasks to complete...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*background_tasks, return_exceptions=True),
                timeout=timeout
            )
            logger.info("All background tasks completed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Background tasks timeout after {timeout}s, some tasks may be incomplete")
            # Cancel remaining tasks
            for task in background_tasks:
                task.cancel()


# Database management
def init_db():
    """Initialize SQLite database for task tracking."""
    db_path = get_config_dir() / 'tasks.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging for crash safety
    conn.execute('PRAGMA synchronous=NORMAL')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            page_uid TEXT NOT NULL,
            title TEXT NOT NULL,
            markdown TEXT NOT NULL,
            status TEXT NOT NULL,
            total_blocks INTEGER,
            processed_blocks INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_db_connection():
    """Get a database connection."""
    db_path = get_config_dir() / 'tasks.db'
    return sqlite3.connect(str(db_path))


def save_task(task_id: str, page_uid: str, title: str, markdown: str, status: str, total_blocks: int = 0):
    """Save a new task to database."""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO tasks (task_id, page_uid, title, markdown, status, total_blocks, processed_blocks)
        VALUES (?, ?, ?, ?, ?, ?, 0)
    ''', (task_id, page_uid, title, markdown, status, total_blocks))
    conn.commit()
    conn.close()


def update_task(task_id: str, status: str = None, processed_blocks: int = None, error_message: str = None):
    """Update task status and progress."""
    conn = get_db_connection()
    updates = []
    params = []

    if status is not None:
        updates.append('status = ?')
        params.append(status)
    if processed_blocks is not None:
        updates.append('processed_blocks = ?')
        params.append(processed_blocks)
    if error_message is not None:
        updates.append('error_message = ?')
        params.append(error_message)

    updates.append('updated_at = ?')
    params.append(datetime.now().isoformat())

    if status in ('completed', 'failed', 'completed_with_warnings'):
        updates.append('completed_at = ?')
        params.append(datetime.now().isoformat())

    params.append(task_id)

    conn.execute(f'''
        UPDATE tasks
        SET {', '.join(updates)}
        WHERE task_id = ?
    ''', params)
    conn.commit()
    conn.close()


def get_task(task_id: str):
    """Get task information."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def _task_debug_log_path(task_id: str) -> Path | None:
    storage_dir = get_env_or_config("ROAM_STORAGE_DIR", "storage.dir")
    if not storage_dir:
        return None
    # Keep task JSONL logs in a dedicated subdirectory to avoid mixing with
    # markdown debug dumps and other artifacts.
    directory = Path(str(storage_dir)) / "task_logs"
    directory.mkdir(parents=True, exist_ok=True)
    dt = pendulum.now().format("YYYYMMDD")
    return directory / f"{dt}_task_{task_id}.jsonl"


def _append_task_event(task_id: str, event: str, payload: dict):
    """
    Append a structured event record to a per-task debug log file.

    This is intentionally file-based (not SQLite) to avoid migrations and to keep
    large payloads (actions / responses) out of the task table.
    """
    path = _task_debug_log_path(task_id)
    if not path:
        return
    record = {
        "ts": pendulum.now().to_iso8601_string(),
        "task_id": task_id,
        "event": event,
        "payload": payload,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_when(when = None):
    if not when:
        date_obj = pendulum.now()
    else:
        try:
            date_obj = cast(pendulum.DateTime, pendulum.parse(when.strip()))
        except Exception as e:
            raise ValueError(f"Unrecognized date format: {when}")
    return date_obj


def parse_uid(s: str) -> str | None:
    """
    Parse uid from ((uid)) or uid format.
    Returns None if identifier looks like a page title.
    """
    s = s.strip()
    if s.startswith('((') and s.endswith('))'):
        return s[2:-2]
    if ' ' in s or any('\u4e00' <= c <= '\u9fff' for c in s):
        return None
    if len(s) <= 40 and all(c.isalnum() or c in '-_' for c in s):
        return s
    return None


_MD_HEADING_RE = re.compile(r"^(#{1,6})(?:\s+(.*))?\s*$")


def _should_parse_block_update_as_markdown(text: str) -> bool:
    """
    Heuristic: treat block updates as structured markdown when the payload is
    clearly multi-line content or starts with a markdown heading.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.count("\n") > BLOCK_UPDATE_MARKDOWN_NEWLINE_THRESHOLD:
        return True
    return re.match(r"^#{1,6}\s", stripped) is not None


def _split_root_and_children_markdown(markdown: str) -> tuple[str, int | None, str]:
    """
    Split markdown intended to update a single block into:
      - root block text (string)
      - root block heading (1..3) or None
      - children markdown (remaining content)

    Rule:
      - If the first non-empty line is a markdown heading, use it for root text/heading.
      - Otherwise, use that line as root text (no heading).
      - The remaining lines (if any) are treated as markdown for children.
    """
    stripped = markdown.strip()
    if not stripped:
        return ("", None, "")

    lines = stripped.splitlines()
    first_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_idx is None:
        return ("", None, "")

    first_line = lines[first_idx].rstrip()
    rest_lines = lines[first_idx + 1 :]

    heading_match = _MD_HEADING_RE.match(first_line.strip())
    if heading_match:
        level = len(heading_match.group(1))
        root_heading = min(level, 3)
        root_text = (heading_match.group(2) or "").strip()
    else:
        root_heading = None
        root_text = first_line.strip()

    children_markdown = "\n".join(rest_lines).strip("\n")
    return (root_text, root_heading, children_markdown)


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class OAuthClientConfig:
    client_id: str
    client_secret: str | None
    scopes: list[str]
    redirect_uris: list[str]


@dataclass(frozen=True)
class OAuthSettings:
    enabled: bool
    require_auth: bool
    allow_access_token_query: bool
    allow_dynamic_registration: bool
    audience: str
    signing_secret: str
    access_token_ttl_seconds: int
    scopes_supported: list[str]
    clients_by_id: dict[str, OAuthClientConfig]


def _load_oauth_settings() -> OAuthSettings:
    enabled = _parse_bool(get_env_or_config("OAUTH_ENABLED", "oauth.enabled", False), False)
    require_auth = _parse_bool(get_env_or_config("OAUTH_REQUIRE_AUTH", "oauth.require_auth", False), False)
    allow_access_token_query = _parse_bool(
        get_env_or_config("OAUTH_ALLOW_ACCESS_TOKEN_QUERY", "oauth.allow_access_token_query", False),
        False,
    )
    allow_dynamic_registration = _parse_bool(
        get_env_or_config("OAUTH_ALLOW_DYNAMIC_REGISTRATION", "oauth.allow_dynamic_registration", True),
        True,
    )
    audience = str(get_env_or_config("OAUTH_AUDIENCE", "oauth.audience", "roamresearch-mcp"))
    signing_secret = str(get_env_or_config("OAUTH_SIGNING_SECRET", "oauth.signing_secret", "") or "")
    access_token_ttl_seconds = int(get_env_or_config(
        "OAUTH_ACCESS_TOKEN_TTL_SECONDS",
        "oauth.access_token_ttl_seconds",
        -1,  # -1 means never expires
    ))

    cfg = load_config()
    oauth_cfg = cfg.get("oauth", {}) if isinstance(cfg, dict) else {}
    scopes_supported = oauth_cfg.get("scopes_supported") if isinstance(oauth_cfg, dict) else None
    if not scopes_supported:
        scopes_supported = [OAUTH_REQUIRED_SCOPE]

    raw_clients = oauth_cfg.get("clients", []) if isinstance(oauth_cfg, dict) else []
    clients_by_id: dict[str, OAuthClientConfig] = {}
    if isinstance(raw_clients, list):
        for item in raw_clients:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "")
            raw_secret = item.get("secret")
            csecret = str(raw_secret) if raw_secret is not None and str(raw_secret) != "" else None
            scopes = item.get("scopes") or []
            redirect_uris = item.get("redirect_uris") or []
            if not cid:
                continue
            if not isinstance(scopes, list):
                scopes = [str(scopes)]
            scopes = [str(s).strip() for s in scopes if str(s).strip()]
            if not scopes:
                scopes = [OAUTH_REQUIRED_SCOPE]
            if not isinstance(redirect_uris, list):
                redirect_uris = [str(redirect_uris)]
            redirect_uris = [str(u).strip() for u in redirect_uris if str(u).strip()]
            clients_by_id[cid] = OAuthClientConfig(
                client_id=cid,
                client_secret=csecret,
                scopes=scopes,
                redirect_uris=redirect_uris,
            )

    if enabled and not signing_secret:
        raise ValueError("oauth.enabled=true but oauth.signing_secret is missing")
    if enabled and not clients_by_id and not allow_dynamic_registration:
        raise ValueError("oauth.enabled=true but oauth.clients is empty and dynamic registration is disabled")

    return OAuthSettings(
        enabled=enabled,
        require_auth=require_auth,
        allow_access_token_query=allow_access_token_query,
        allow_dynamic_registration=allow_dynamic_registration,
        audience=audience,
        signing_secret=signing_secret,
        access_token_ttl_seconds=access_token_ttl_seconds,
        scopes_supported=list(scopes_supported),
        clients_by_id=clients_by_id,
    )


def _request_issuer(request: Request) -> str:
    """
    Infer issuer from the incoming request.

    Prefer Host header (user can control via nginx). Scheme defaults to request.url.scheme,
    with optional support for X-Forwarded-Proto.
    """
    host = request.headers.get("host") or request.url.netloc
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    return f"{scheme}://{host}"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + padding).encode("ascii"))


def _jwt_encode(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, digestmod="sha256").digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _jwt_decode(token: str, secret: str) -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        raise ValueError("invalid_token: bad format")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, digestmod="sha256").digest()
    actual_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("invalid_token: bad signature")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_token: bad payload")
    return payload


def _sha256_b64url(raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return _b64url_encode(digest)


def _append_query(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    existing = parse_qs(parts.query, keep_blank_values=True)
    for k, v in params.items():
        existing[k] = [v]
    query = urlencode(existing, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _extract_bearer_token(request: Request, *, allow_query: bool) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token or None
    if allow_query:
        token = request.query_params.get("access_token")
        return token or None
    return None


def _unauthorized(detail: str = "Unauthorized", *, request: Request | None = None) -> Response:
    """Return 401 response with WWW-Authenticate header per RFC 9728."""
    if request is not None:
        issuer = _request_issuer(request)
        resource_metadata_url = f"{issuer}/.well-known/oauth-protected-resource"
        www_auth = f'Bearer realm="{resource_metadata_url}"'
    else:
        www_auth = 'Bearer realm="mcp"'
    logger.debug(f"OAuth 401: {detail} | WWW-Authenticate: {www_auth}")
    return PlainTextResponse(
        detail,
        status_code=401,
        headers={"WWW-Authenticate": www_auth},
    )


class OAuthAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: OAuthSettings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        if not self.settings.enabled or not self.settings.require_auth:
            return await call_next(request)

        # Always allow CORS preflight (handled elsewhere); browsers can't attach Authorization reliably.
        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        # Always allow OAuth discovery, protected resource metadata, and token endpoints without auth.
        if path.startswith("/.well-known/oauth-"):
            return await call_next(request)
        if path.startswith("/mcp/.well-known/oauth-"):
            return await call_next(request)
        if path in ("/oauth/token", "/oauth/register"):
            return await call_next(request)
        if path in ("/authorize", "/oauth/authorize"):
            return await call_next(request)

        protected = path == "/mcp" or path.startswith("/sse") or path.startswith("/messages")
        if not protected:
            return await call_next(request)

        token = _extract_bearer_token(request, allow_query=self.settings.allow_access_token_query)
        if not token:
            logger.info(f"OAuth: No token in request to {path}")
            return _unauthorized("Missing Bearer token", request=request)

        try:
            payload = _jwt_decode(token, self.settings.signing_secret)
        except Exception as e:
            logger.warning(f"OAuth: Token decode failed for {path}: {e}")
            return _unauthorized("invalid_token: malformed", request=request)

        now = int(time.time())
        exp = payload.get("exp")
        # exp is optional: if present, must be valid future timestamp; if absent, token never expires
        if exp is not None:
            if not isinstance(exp, int) or exp <= now:
                logger.warning(f"OAuth: Token expired for {path}")
                return _unauthorized("invalid_token: expired", request=request)

        if payload.get("aud") != self.settings.audience:
            logger.warning(f"OAuth: Bad audience for {path}: got {payload.get('aud')}, expected {self.settings.audience}")
            return _unauthorized("invalid_token: bad audience", request=request)

        # issuer is request-derived (host-based) so tokens are bound to the served host.
        expected_issuer = _request_issuer(request)
        if payload.get("iss") != expected_issuer:
            logger.warning(f"OAuth: Bad issuer for {path}: got {payload.get('iss')}, expected {expected_issuer}")
            return _unauthorized("invalid_token: bad issuer", request=request)

        scope = payload.get("scope") or ""
        scopes = set(str(scope).split())
        if OAUTH_REQUIRED_SCOPE not in scopes:
            logger.warning(f"OAuth: Insufficient scope for {path}: got {scopes}, need {OAUTH_REQUIRED_SCOPE}")
            return _unauthorized("insufficient_scope", request=request)

        return await call_next(request)


class McpCorsMiddleware(BaseHTTPMiddleware):
    """
    Minimal CORS support for browser-based MCP clients.

    We handle OPTIONS preflight explicitly for /mcp and the SSE transport endpoints
    (/sse, /messages) because the underlying MCP routes may not register OPTIONS.
    """

    def __init__(self, app):
        super().__init__(app)
        self.cors = _load_mcp_cors_settings()

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        is_options = request.method.upper() == "OPTIONS"

        # Only treat as CORS preflight if browser sets these headers.
        is_preflight = is_options and bool(request.headers.get("access-control-request-method"))

        path = request.url.path
        is_mcp_transport = path == "/mcp" or path == "/sse" or path == "/messages"

        if is_preflight and is_mcp_transport:
            headers = _cors_headers_for_request(request, cors=self.cors)
            return Response(status_code=204, headers=headers)

        response = await call_next(request)

        # Add CORS headers to actual responses when Origin is present and allowed.
        if origin:
            headers = _cors_headers_for_request(request, cors=self.cors)
            allow_origin = headers.get("Access-Control-Allow-Origin")
            if allow_origin:
                response.headers["Access-Control-Allow-Origin"] = allow_origin
            if "Access-Control-Allow-Credentials" in headers:
                response.headers["Access-Control-Allow-Credentials"] = headers["Access-Control-Allow-Credentials"]
            if "Vary" in headers:
                response.headers["Vary"] = headers["Vary"]

        return response


def _load_mcp_cors_settings() -> dict:
    cfg = load_config()
    mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

    allow_origins_raw = get_env_or_config("MCP_CORS_ALLOW_ORIGINS", "mcp.cors_allow_origins", "")
    allow_origin_regex = get_env_or_config("MCP_CORS_ALLOW_ORIGIN_REGEX", "mcp.cors_allow_origin_regex", "")
    allow_headers_raw = get_env_or_config(
        "MCP_CORS_ALLOW_HEADERS",
        "mcp.cors_allow_headers",
        "authorization,content-type",
    )
    allow_methods_raw = get_env_or_config(
        "MCP_CORS_ALLOW_METHODS",
        "mcp.cors_allow_methods",
        "GET,POST,OPTIONS",
    )
    allow_credentials = _parse_bool(get_env_or_config(
        "MCP_CORS_ALLOW_CREDENTIALS",
        "mcp.cors_allow_credentials",
        False,
    ), False)
    auto_allow_origin_from_host = _parse_bool(get_env_or_config(
        "MCP_CORS_AUTO_ALLOW_ORIGIN_FROM_HOST",
        "mcp.cors_auto_allow_origin_from_host",
        True,
    ), False)
    max_age = int(get_env_or_config("MCP_CORS_MAX_AGE", "mcp.cors_max_age", 600))
    allow_private_network = _parse_bool(get_env_or_config(
        "MCP_CORS_ALLOW_PRIVATE_NETWORK",
        "mcp.cors_allow_private_network",
        False,
    ), False)

    allow_origins: list[str] = []
    if allow_origins_raw:
        allow_origins = [o.strip() for o in str(allow_origins_raw).split(",") if o.strip()]

    return {
        "allow_origins": allow_origins,
        "allow_origin_regex": str(allow_origin_regex or ""),
        "allow_headers": [h.strip().lower() for h in str(allow_headers_raw).split(",") if h.strip()],
        "allow_methods": [m.strip().upper() for m in str(allow_methods_raw).split(",") if m.strip()],
        "allow_credentials": allow_credentials,
        "auto_allow_origin_from_host": auto_allow_origin_from_host,
        "max_age": max_age,
        "allow_private_network": allow_private_network,
    }


def _cors_headers_for_request(request: Request, *, cors: dict) -> dict[str, str]:
    """
    Minimal CORS preflight support for browser-based MCP clients (SSE).
    """
    origin = request.headers.get("origin")
    headers: dict[str, str] = {
        "Access-Control-Allow-Methods": ", ".join(cors["allow_methods"]),
        "Access-Control-Max-Age": str(cors["max_age"]),
    }

    requested_headers = request.headers.get("access-control-request-headers")
    if requested_headers:
        # Validate requested headers against configured allow_headers whitelist
        allowed_headers_set = set(cors["allow_headers"])
        requested_list = [h.strip().lower() for h in requested_headers.split(",") if h.strip()]
        validated = [h for h in requested_list if h in allowed_headers_set]
        headers["Access-Control-Allow-Headers"] = ", ".join(validated) if validated else ", ".join(cors["allow_headers"])
    else:
        headers["Access-Control-Allow-Headers"] = ", ".join(cors["allow_headers"])

    if cors["allow_private_network"] and request.headers.get("access-control-request-private-network") == "true":
        headers["Access-Control-Allow-Private-Network"] = "true"

    if not origin:
        return headers

    allowed = False
    allow_origins: list[str] = cors["allow_origins"]
    allow_origin_regex: str = cors["allow_origin_regex"]
    if "*" in allow_origins:
        allowed = True
    elif origin in allow_origins:
        allowed = True
    elif allow_origin_regex:
        try:
            if re.match(allow_origin_regex, origin):
                allowed = True
        except re.error:
            # Bad regex config: treat as disallowed rather than failing requests.
            allowed = False

    # Optional: allow same-origin by comparing Origin against Host (as set by nginx)
    # and inferred scheme (x-forwarded-proto preferred).
    if not allowed and cors.get("auto_allow_origin_from_host"):
        try:
            host = request.headers.get("host") or request.url.netloc
            scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
            expected_origin = f"{scheme}://{host}"
            if origin == expected_origin:
                allowed = True
        except Exception:
            allowed = False

    if allowed:
        # If credentials are allowed, we must echo origin (not "*").
        if cors["allow_credentials"]:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
            headers["Access-Control-Allow-Credentials"] = "true"
        else:
            headers["Access-Control-Allow-Origin"] = "*" if "*" in allow_origins else origin
            if "*" not in allow_origins:
                headers["Vary"] = "Origin"

    return headers


async def _process_content_blocks_background(task_id: str, page_uid: str, actions: list):
    """Process content blocks in batches in the background."""
    batch_size = int(get_env_or_config('BATCH_SIZE', 'batch.size', '100'))
    max_retries = int(get_env_or_config('MAX_RETRIES', 'batch.max_retries', '3'))

    total_blocks = len(actions)
    processed = 0

    logger.info(f"Task {task_id}: Processing {total_blocks} blocks in batches of {batch_size}")
    update_task(task_id, status='processing')

    try:
        async with RoamClient() as client:
            # Process in batches
            for i in range(0, total_blocks, batch_size):
                batch = actions[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_blocks + batch_size - 1) // batch_size

                logger.info(f"Task {task_id}: Processing batch {batch_num}/{total_batches} ({len(batch)} blocks)")

                # Retry logic for this batch
                retry_count = 0
                last_error = None

                while retry_count <= max_retries:
                    try:
                        await client.batch_actions(batch)
                        processed += len(batch)
                        update_task(task_id, processed_blocks=processed)
                        logger.info(f"Task {task_id}: Batch {batch_num}/{total_batches} completed. Progress: {processed}/{total_blocks}")
                        break  # Success, move to next batch
                    except Exception as e:
                        retry_count += 1
                        last_error = str(e)
                        if retry_count <= max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                            logger.warning(f"Task {task_id}: Batch {batch_num} failed (attempt {retry_count}/{max_retries}), retrying in {wait_time}s: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            # Max retries exceeded
                            error_msg = f"Batch {batch_num} failed after {max_retries} retries: {last_error}"
                            logger.error(f"Task {task_id}: {error_msg}")
                            update_task(task_id, status='failed', error_message=error_msg, processed_blocks=processed)
                            return

        # All batches completed successfully
        update_task(task_id, status='completed', processed_blocks=processed)
        logger.info(f"Task {task_id}: All {total_blocks} blocks processed successfully")

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Task {task_id}: {error_msg}")
        update_task(task_id, status='failed', error_message=error_msg, processed_blocks=processed)


async def _process_update_actions_background(task_id: str, actions: list):
    """Process update actions in batches in the background."""
    batch_size = int(get_env_or_config('BATCH_SIZE', 'batch.size', '100'))
    max_retries = int(get_env_or_config('MAX_RETRIES', 'batch.max_retries', '3'))

    total_actions = len(actions)
    processed = 0

    logger.info(f"Task {task_id}: Processing {total_actions} update actions in batches of {batch_size}")
    update_task(task_id, status='processing')
    _append_task_event(
        task_id,
        "update_started",
        {"total_actions": total_actions, "batch_size": batch_size, "max_retries": max_retries},
    )

    async with RoamClient() as client:
        for i in range(0, total_actions, batch_size):
            batch = actions[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_actions + batch_size - 1) // batch_size

            logger.info(f"Task {task_id}: Processing batch {batch_num}/{total_batches} ({len(batch)} actions)")
            _append_task_event(
                task_id,
                "batch_start",
                {
                    "batch_num": batch_num,
                    "total_batches": total_batches,
                    "actions_in_batch": len(batch),
                },
            )

            retry_count = 0
            last_error = None

            while retry_count <= max_retries:
                try:
                    resp = await client.batch_actions(batch)
                    logger.info(f"Task {task_id}: Batch {batch_num}/{total_batches} response: {resp}")
                    _append_task_event(
                        task_id,
                        "batch_ok",
                        {
                            "batch_num": batch_num,
                            "total_batches": total_batches,
                            "response": resp if isinstance(resp, dict) else {"_raw": str(resp)},
                        },
                    )
                    processed += len(batch)
                    update_task(task_id, processed_blocks=processed)
                    logger.info(f"Task {task_id}: Batch {batch_num}/{total_batches} completed. Progress: {processed}/{total_actions}")
                    break
                except Exception as e:
                    retry_count += 1
                    last_error = str(e)
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"Task {task_id}: Batch {batch_num} failed (attempt {retry_count}/{max_retries}), retrying in {wait_time}s: {e}")
                        _append_task_event(
                            task_id,
                            "batch_retry",
                            {
                                "batch_num": batch_num,
                                "attempt": retry_count,
                                "max_retries": max_retries,
                                "error": last_error,
                            },
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        error_msg = f"Batch {batch_num} failed after {max_retries} retries: {last_error}"
                        logger.error(f"Task {task_id}: {error_msg}")
                        _append_task_event(
                            task_id,
                            "batch_failed",
                            {
                                "batch_num": batch_num,
                                "total_batches": total_batches,
                                "error": error_msg,
                                "note": "Roam batch-actions is not transactional; earlier actions in this batch may already be applied.",
                                "actions": batch,
                            },
                        )
                        update_task(task_id, status='failed', error_message=error_msg, processed_blocks=processed)
                        return

        # Verification phase: fetch page and confirm diff is empty.
        task = get_task(task_id)
        if not task:
            msg = "Task record not found for verification"
            logger.warning(f"Task {task_id}: {msg}")
            _append_task_event(task_id, "verify_skipped", {"reason": msg})
            update_task(task_id, status='completed_with_warnings', processed_blocks=processed, error_message=msg)
            return

        page_uid = task.get("page_uid")
        markdown = task.get("markdown") or ""
        if not page_uid:
            msg = "Missing page_uid for verification"
            logger.warning(f"Task {task_id}: {msg}")
            _append_task_event(task_id, "verify_skipped", {"reason": msg})
            update_task(task_id, status='completed_with_warnings', processed_blocks=processed, error_message=msg)
            return

        update_task(task_id, status="verifying", processed_blocks=processed)
        _append_task_event(task_id, "verify_start", {"page_uid": page_uid})

        try:
            page = await client.get_block_by_uid(page_uid)
            if not page:
                raise ValueError(f"Page UID not found after update: {page_uid}")

            from .verify import diff_page_against_markdown

            verify_diff = diff_page_against_markdown(page, markdown)
            verify_stats = verify_diff.stats()
            _append_task_event(task_id, "verify_result", {"stats": verify_stats})

            if verify_diff.is_empty():
                update_task(task_id, status='completed', processed_blocks=processed)
                logger.info(f"Task {task_id}: All {total_actions} actions processed successfully; verification OK")
                return

            warn_msg = (
                "Verification diff is not empty after update: "
                f"{verify_stats['creates']} creates, {verify_stats['updates']} updates, "
                f"{verify_stats['moves']} moves, {verify_stats['deletes']} deletes"
            )
            _append_task_event(
                task_id,
                "verify_warning",
                {
                    "message": warn_msg,
                    "stats": verify_stats,
                    "suggested_actions_preview": generate_batch_actions(verify_diff)[:50],
                    "note": "Non-empty diff may indicate partial application or unexpected Roam behavior; consider re-running update or inspecting task log.",
                },
            )
            update_task(task_id, status="completed_with_warnings", processed_blocks=processed, error_message=warn_msg)
            logger.warning(f"Task {task_id}: {warn_msg}")
            return

        except Exception as e:
            err = f"Verification failed: {e}"
            _append_task_event(task_id, "verify_error", {"error": err})
            update_task(task_id, status="completed_with_warnings", processed_blocks=processed, error_message=err)
            logger.warning(f"Task {task_id}: {err}")
            return


async def get_or_create_topic_uid(client, topic: str, when: pendulum.DateTime) -> str:
    """Get topic node UID under daily page, creating it if not exists."""
    daily_uid = when.format('MM-DD-YYYY')

    # Query existing topic node
    block = await client.q(f"""
        [:find (pull ?id [:block/uid :node/title :block/string])
         :where [?id :block/string "{topic}"]
                [?id :block/parents ?pid]
                [?pid :block/uid "{daily_uid}"]
        ]
    """)

    if block:
        return block[0][0][':block/uid']

    # Create topic node
    topic_uid = uuid.uuid4().hex
    await client.batch_actions([create_block(topic, daily_uid, topic_uid)])
    logger.info(f"Created topic node '{topic}' under {daily_uid} with UID {topic_uid}")
    return topic_uid

#
#
#


@mcp.tool(
    name="save_markdown",
    description="""Create a new page in Roam Research.

Use when: saving notes, creating pages, storing content to the graph.

- title: Page title (plain text)
- markdown: Content in GFM. Omit title as H1.

Returns task ID and page link. Auto-links to today's Daily Notes.
"""
)
async def save_markdown(title: str, markdown: str) -> str:
    # Generate unique IDs
    task_id = uuid.uuid4().hex
    link_block_uid = uuid.uuid4().hex  # UID for the link block (this is the deterministic return value)

    try:
        # Create page and generate content actions
        page = create_page(title)
        page_uid = page['page']['uid']
        content_actions = gfm_to_batch_actions(markdown, page_uid)

        logger.info(f"Task {task_id}: Page UID: {page_uid}, Content blocks: {len(content_actions)}")

        # Save task to database
        save_task(task_id, page_uid, title, markdown, 'pending', len(content_actions))

        # Phase 1 (Synchronous): Create page + link block
        when = get_when()
        topic_node = get_env_or_config("TOPIC_NODE", "mcp.topic_node")

        async with RoamClient() as client:
            if topic_node:
                topic_uid = await get_or_create_topic_uid(client, topic_node, when)
                logger.info(f"Task {task_id}: Topic UID: {topic_uid}")
                link_action = create_block(f"[[{title}]]", topic_uid, link_block_uid)
            else:
                link_action = create_block(f"[[{title}]]", when.format('MM-DD-YYYY'), link_block_uid)

            # Submit page + link block synchronously
            await client.batch_actions([page, link_action])
            logger.info(f"Task {task_id}: Page and link block created successfully")

        # Update task status
        update_task(task_id, status='link_created')

        # Phase 2 (Background): Process content blocks
        if content_actions:
            create_background_task(
                _process_content_blocks_background(task_id, page_uid, content_actions)
            )
            logger.info(f"Task {task_id}: Background processing started for {len(content_actions)} blocks")

        # Return immediately
        return f"Task {task_id} started. Page [[{title}]] created with link block {link_block_uid}. Processing {len(content_actions)} content blocks in background."

    except Exception as e:
        logger.error(f"Task {task_id}: Error during initial setup: {e}\n{traceback.format_exc()}")
        error_msg = f"Error: {e}"
        if type(e) == httpx.HTTPStatusError:
            error_msg = f"Error: {e.response.text}\n\n{e.response.status_code}"
        update_task(task_id, status='failed', error_message=error_msg)
        return error_msg
    finally:
        # Save debug file (always, for recovery purposes)
        storage_dir = get_env_or_config("ROAM_STORAGE_DIR", "storage.dir")
        if storage_dir:
            try:
                directory = Path(storage_dir)
                directory.mkdir(parents=True, exist_ok=True)
                dt = pendulum.now().format('YYYYMMDD')
                debug_file = directory / f"{dt}_{link_block_uid}.md"
                with open(debug_file, 'w') as f:
                    f.write(f"{title}\n\n{markdown}")
                logger.info(f"Task {task_id}: Debug file saved: {debug_file}")
            except Exception as storage_error:
                logger.warning(f"Task {task_id}: Failed to write debug file: {storage_error}")
        else:
            logger.info(f"Task {task_id}: ROAM_STORAGE_DIR not set; skipped saving debug file.")


@mcp.tool(
    name="query",
    description="""Run raw Datalog query on the Roam graph.

Use when: advanced queries, custom filters, specific attributes. Prefer search/get for common cases.

- q: Datalog query string

Returns JSON.
"""
)
async def handle_query_roam(q: str) -> str:
    async with RoamClient() as client:
        try:
            result = await client.q(q)
            if result:
                return result
            return pprint.pformat(result)
        except httpx.HTTPStatusError as e:
            print(e.response.text)
            return f"Error: {e}"
        except Exception as e:
            return f"{type(e)}: {e}"


@mcp.tool(
    name="get",
    description="""Read a page or block from Roam, including all children.

Use when: reading notes, viewing page content, retrieving specific blocks.

- identifier: Page title or block UID. Accepts list for batch retrieval.
- raw: Return JSON instead of markdown. Default: false.
- expand_refs: Expand ((uid)) references inline. Default: true.

Returns markdown with children. Block refs shown as quoted blocks.
"""
)
async def handle_get(identifier: str | List[str], raw: bool = False, expand_refs: bool = True) -> str:
    # Normalize to list for batch processing
    identifiers = [identifier] if isinstance(identifier, str) else identifier

    if not identifiers:
        return "Error: No identifier provided"

    def collect_all_refs(block: dict, collected: set) -> set:
        """Recursively collect all ((uid)) refs from a block and its children."""
        text = block.get(':block/string', '')
        refs = extract_ref(text)
        collected.update(refs)
        for child in block.get(':block/children', []):
            collect_all_refs(child, collected)
        return collected

    def collect_existing_uids(block: dict, uids: set) -> set:
        """Collect all UIDs already present in the result."""
        uid = block.get(':block/uid')
        if uid:
            uids.add(uid)
        for child in block.get(':block/children', []):
            collect_existing_uids(child, uids)
        return uids

    async def get_single(ident: str, client) -> tuple[str, str | None, bool]:
        """Get a single identifier, returns (identifier, result_text, is_error)"""
        uid = parse_uid(ident)
        result = None
        is_page = False

        if uid:
            result = await client.get_block_by_uid(uid)

        if not result:
            result = await client.get_page_by_title(ident)
            is_page = True

        if not result:
            return (ident, f"Error: '{ident}' not found.", True)

        if raw:
            return (ident, json.dumps(result, indent=2, ensure_ascii=False), False)

        children = result.get(':block/children', [])
        if children:
            children = sorted(children, key=lambda x: x.get(':block/order', 0))

        # Format content
        if is_page:
            content = format_block_as_markdown(children)
        else:
            content = format_block_as_markdown([result])

        # Expand refs if enabled
        if expand_refs:
            # Collect all refs in the content
            all_refs = set()
            if is_page:
                for child in children:
                    collect_all_refs(child, all_refs)
            else:
                collect_all_refs(result, all_refs)

            # Get existing UIDs to avoid re-fetching
            existing_uids = set()
            if is_page:
                for child in children:
                    collect_existing_uids(child, existing_uids)
            else:
                collect_existing_uids(result, existing_uids)

            # Fetch external refs (not in current result)
            external_refs = all_refs - existing_uids
            ref_blocks = {}
            for ref_uid in external_refs:
                ref_block = await client.get_block_by_uid(ref_uid)
                if ref_block:
                    ref_blocks[ref_uid] = ref_block

            # Expand refs in the content
            if ref_blocks:
                content = expand_refs_in_text(content, ref_blocks)

        return (ident, content, False)

    try:
        async with RoamClient() as client:
            results = []
            for ident in identifiers:
                ident_name, content, is_error = await get_single(ident, client)
                if len(identifiers) > 1:
                    # Add header for batch results
                    header = f"## {ident_name}\n\n" if not is_error else ""
                    results.append(f"{header}{content}")
                else:
                    results.append(content)

        return "\n\n---\n\n".join(results) if len(results) > 1 else results[0]

    except Exception as e:
        return f"Error: {e}"


@mcp.tool(
    name="search",
    description="""Search blocks in Roam.

Use when: finding content, locating keywords, searching by tag.

Syntax (Google-style):
  python async        → OR (either term)
  +python +async      → AND (both required)
  -javascript         → exclude term
  "exact phrase"      → phrase match

- query: Search string with +/- operators
- tag: Filter by tag. Optional.
- page: Limit to page. Optional.
- limit: Max results. Default: 20.

Returns blocks grouped by page with UID.
"""
)
async def handle_search(
    query: str = "",
    tag: str | None = None,
    page: str | None = None,
    case_sensitive: bool = True,
    limit: int = 20
) -> str:
    try:
        async with RoamClient() as client:
            # If only tag is specified (no query), use search_by_tag
            if not query and tag:
                results = await client.search_by_tag(
                    tag,
                    limit,
                    page_title=page
                )
            else:
                results = await client.search_blocks_query(
                    query,
                    limit,
                    case_sensitive=case_sensitive,
                    page_title=page,
                    tag=tag
                )

        if not results:
            return "No results found."

        # Group results by page
        by_page: dict[str, list[tuple[str, str]]] = {}
        page_order: list[str] = []
        for item in results:
            uid, text, page_title = item[0], item[1], item[2]
            if page_title not in by_page:
                by_page[page_title] = []
                page_order.append(page_title)
            by_page[page_title].append((uid, text))

        # Format output
        lines = []
        for page_title in page_order:
            blocks = by_page[page_title]
            lines.append(f"[[{page_title}]]")
            for uid, text in blocks:
                display_text = text.replace('\n', ' ')
                if len(display_text) > 80:
                    display_text = display_text[:77] + "..."
                lines.append(f"  {uid}   {display_text}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


@mcp.tool(
    name="find_references",
    description="""Find backlinks to a page or block.

Use when: finding what links to a page, discovering connections, tracing references.

- identifier: Page title or block UID
  - Page: finds [[page]] and #page refs
  - Block: finds ((uid)) refs

Returns referencing blocks grouped by page.
"""
)
async def handle_find_references(identifier: str, limit: int = 50) -> str:
    uid = parse_uid(identifier)

    try:
        async with RoamClient() as client:
            if uid:
                # It's a block UID - find block references
                results = await client.find_references(uid, limit)
                ref_type = f"(({uid}))"
            else:
                # It's a page title - find page references
                results = await client.find_page_references(identifier, limit)
                ref_type = f"[[{identifier}]]"

        if not results:
            return f"No references found for {ref_type}"

        # Group results by page
        by_page: dict[str, list[tuple[str, str]]] = {}
        page_order: list[str] = []
        for item in results:
            block_uid, text, page_title = item[0], item[1], item[2]
            if page_title not in by_page:
                by_page[page_title] = []
                page_order.append(page_title)
            by_page[page_title].append((block_uid, text))

        # Format output
        lines = [f"References to {ref_type}: {len(results)} found\n"]
        for page_title in page_order:
            blocks = by_page[page_title]
            lines.append(f"[[{page_title}]]")
            for block_uid, text in blocks:
                display_text = text.replace('\n', ' ')
                if len(display_text) > 80:
                    display_text = display_text[:77] + "..."
                lines.append(f"  {block_uid}   {display_text}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


@mcp.tool(
    name="search_todos",
    description="""Find TODO or DONE items in Roam.

Use when: listing tasks, checking pending/completed items.

- status: "TODO" or "DONE". Default: "TODO".
- page: Limit to page. Optional.
- limit: Max results. Default: 50.

Returns items grouped by page.
"""
)
async def handle_search_todos(
    status: str = "TODO",
    page: str | None = None,
    limit: int = 50
) -> str:
    try:
        async with RoamClient() as client:
            results = await client.search_todos(
                status=status,
                limit=limit,
                page_title=page
            )

        if not results:
            return f"No {status} items found."

        # Group results by page
        by_page: dict[str, list[tuple[str, str]]] = {}
        page_order: list[str] = []
        for item in results:
            uid, text, page_title = item[0], item[1], item[2]
            if page_title not in by_page:
                by_page[page_title] = []
                page_order.append(page_title)
            by_page[page_title].append((uid, text))

        # Format output
        lines = [f"{status} items: {len(results)} found\n"]
        for page_title in page_order:
            blocks = by_page[page_title]
            lines.append(f"[[{page_title}]]")
            for uid, text in blocks:
                display_text = text.replace('\n', ' ')
                if len(display_text) > 80:
                    display_text = display_text[:77] + "..."
                lines.append(f"  {uid}   {display_text}")
            lines.append("")

        return "\n".join(lines)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(
    name="update_markdown",
    description="""Update an existing page or block in Roam.

Use when: editing pages, modifying content, revising notes.

- identifier: Page title or block UID
- markdown: New content in GFM
- dry_run: Preview changes only. Default: false.

Returns change summary. Preserves block UIDs where possible.

Note: when identifier is a block UID, multi-line markdown (or content starting
with a markdown heading) is treated as structured content: the first line/heading
updates the target block itself, and the remaining markdown is converted into
children blocks.
"""
)
async def update_markdown(identifier: str, markdown: str, dry_run: bool = False) -> str:
    uid = parse_uid(identifier)

    async with RoamClient() as client:
        if uid:
            text = markdown.strip()

            if not _should_parse_block_update_as_markdown(text):
                # Single block update (plain text)
                result = await client.update_block_text(uid, normalize_task_marker(text), dry_run=dry_run)
                stats = result["stats"]
                prefix = "Dry run - would make" if dry_run else "Updated"
                return f"{prefix}: {stats['updates']} updates"

            # Structured update: first line/heading updates the block itself,
            # remaining markdown becomes children.
            existing_block = await client.get_block_by_uid(uid)
            if not existing_block:
                return f"Error: Block not found: {uid}"

            root_text, root_heading, children_markdown = _split_root_and_children_markdown(text)
            root_text = normalize_task_marker(root_text)

            existing_text = existing_block.get(":block/string", "")
            existing_heading = existing_block.get(":block/heading")

            root_update_action: dict = {"action": "update-block", "block": {"uid": uid}}
            needs_root_update = False

            if root_text != existing_text:
                root_update_action["block"]["string"] = root_text
                needs_root_update = True

            if root_heading is not None:
                if root_heading != existing_heading:
                    root_update_action["block"]["heading"] = root_heading
                    needs_root_update = True
            elif existing_heading is not None:
                # Clear heading if the new root is not a heading.
                root_update_action["block"]["heading"] = 0
                needs_root_update = True

            actions: list[dict] = []
            updates_root = 0
            if needs_root_update:
                actions.append(root_update_action)
                updates_root = 1

            creates = moves = deletes = 0
            updates_children = 0

            if children_markdown.strip():
                existing_children = parse_existing_blocks(existing_block)
                new_children = gfm_to_blocks(children_markdown, uid, skip_h1=False)
                diff = diff_block_trees(existing_children, new_children, uid)
                actions.extend(generate_batch_actions(diff))
                stats = diff.stats()
                creates = stats["creates"]
                updates_children = stats["updates"]
                moves = stats["moves"]
                deletes = stats["deletes"]

            updates = updates_root + updates_children
            prefix = "Dry run - would make" if dry_run else "Updated"

            if dry_run:
                return f"{prefix}: {creates} creates, {updates} updates, {moves} moves, {deletes} deletes"

            if not actions:
                return "No changes needed"

            await client.batch_actions(actions)
            return f"{prefix}: {creates} creates, {updates} updates, {moves} moves, {deletes} deletes"

        # Page update - compute diff first
        page = await client.get_page_by_title(identifier)
        if not page:
            return f"Error: Page not found: {identifier}"

        page_uid = page.get(':block/uid')
        if not page_uid:
            return f"Error: Page has no UID: {identifier}"

        existing_blocks = parse_existing_blocks(page)
        new_blocks = gfm_to_blocks(markdown, page_uid)
        diff = diff_block_trees(existing_blocks, new_blocks, page_uid)
        actions = generate_batch_actions(diff)
        stats = diff.stats()
        preserved = list(diff.preserved_uids)

        if dry_run:
            summary = f"Dry run - would make: {stats['creates']} creates, " \
                      f"{stats['updates']} updates, {stats['moves']} moves, " \
                      f"{stats['deletes']} deletes"
            if preserved:
                summary += f"\nWould preserve {len(preserved)} block UID(s)"
            return summary

        if not actions:
            return "No changes needed"

        # Save task and start background processing
        task_id = uuid.uuid4().hex
        save_task(task_id, page_uid, f"[UPDATE] {identifier}", markdown, 'pending', len(actions))

        create_background_task(
            _process_update_actions_background(task_id, actions)
        )

        summary = f"Task {task_id} started. Updating [[{identifier}]]: " \
                  f"{stats['creates']} creates, {stats['updates']} updates, " \
                  f"{stats['moves']} moves, {stats['deletes']} deletes. " \
                  f"Processing {len(actions)} actions in background."
        if preserved:
            summary += f"\nPreserving {len(preserved)} block UID(s)"
        return summary


@mcp.tool(
    name="get_journaling_by_date",
    description="""Get Daily Notes journal for a date.

Use when: reading daily notes, checking journal entries.

- when: Date string (ISO8601). Default: today.

Returns journal content as text.
"""
)
async def get_journaling_by_date(when=None):
    if not when:
        date_obj = pendulum.now()
    else:
        try:
            date_obj = cast(pendulum.DateTime, pendulum.parse(when.strip()))
        except Exception as e:
            return f"Unrecognized date format: {when}"
    logger.info('get_journaling_by_date: %s', date_obj)
    topic_node = get_env_or_config("TOPIC_NODE", "mcp.topic_node")
    logger.info('topic_node: %s', topic_node)
    if topic_node:
        query = f"""
[:find (pull ?e [*
                 {{:block/children [*]}}
                 {{:block/refs [*]}}
                ])
 :where
    [?id :block/string "{topic_node}"]
    [?id :block/parents ?pid]
    [?pid :block/uid "{date_obj.format('MM-DD-YYYY')}"]
    [?e :block/parents ?id]
    [?e :block/parents ?pid]
]
        """
    else:
        query = f"""
[:find (pull ?e [*
                 {{:block/children [*]}}
                 {{:block/refs [*]}}
                ])
 :where
    [?pid :block/uid "{date_obj.format('MM-DD-YYYY')}"]
    [?e :block/parents ?id]
    [?e :block/parents ?pid]
]
        """
    async with RoamClient() as client:
        resp = await client.q(query)
    if resp is None:
        return ''
    logger.info(f'get_journaling_by_date: found {len(resp)} blocks')
    nodes = list(chain(*(i for i in resp)))
    if topic_node:
        root = list(sorted([i for i in nodes if len(i.get(':block/parents', [])) == 2], key=lambda i: i[':block/order']))
    else:
        root = list(sorted([i for i in nodes if len(i.get(':block/parents', [])) == 1], key=lambda i: i[':block/order']))
    blocks = []
    for i in root:
        blocks.append(format_block(i, nodes))
    if not blocks:
        return ''
    return "\n\n".join(blocks).strip()


#
#
#


def create_app():
    app = mcp.streamable_http_app()
    app.routes.extend(mcp.sse_app().routes)

    settings = _load_oauth_settings()
    if settings.enabled:
        app.add_middleware(OAuthAuthMiddleware, settings=settings)

        def _oauth_protected_resource_metadata(request: Request) -> Response:
            """OAuth 2.0 Protected Resource Metadata (RFC 9728)."""
            issuer = _request_issuer(request)
            logger.debug(f"OAuth: Serving protected resource metadata for issuer={issuer}")
            return JSONResponse({
                "resource": issuer,
                "authorization_servers": [issuer],
                "scopes_supported": settings.scopes_supported,
                "bearer_methods_supported": ["header"],
            })

        def _oauth_metadata(request: Request) -> Response:
            issuer = _request_issuer(request)
            metadata: dict = {
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/oauth/token",
                "grant_types_supported": ["client_credentials", "authorization_code"],
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": [
                    "client_secret_basic",
                    "client_secret_post",
                    "none",
                ],
                "scopes_supported": settings.scopes_supported,
            }
            if settings.allow_dynamic_registration:
                metadata["registration_endpoint"] = f"{issuer}/oauth/register"
            return JSONResponse(metadata)

        # NOTE: In-memory store for used authorization codes. This only works correctly
        # in single-worker deployments. For multi-worker setups, a shared store (Redis,
        # database) would be needed to prevent auth code replay across workers.
        used_auth_codes: dict[str, int] = {}

        # In-memory store for dynamically registered clients (RFC 7591).
        # Same single-worker caveat as above.
        dynamic_clients: dict[str, OAuthClientConfig] = {}

        def _get_client(client_id: str) -> OAuthClientConfig | None:
            """Look up client from static config or dynamic registration."""
            return settings.clients_by_id.get(client_id) or dynamic_clients.get(client_id)

        def _purge_used_codes(now: int):
            # Keep dict small; auth codes are very short-lived.
            expired = [k for k, exp in used_auth_codes.items() if exp <= now]
            for k in expired:
                used_auth_codes.pop(k, None)

        def _mark_code_used(jti: str, exp: int, now: int) -> bool:
            _purge_used_codes(now)
            if jti in used_auth_codes:
                return False
            used_auth_codes[jti] = exp
            return True

        def _oauth_error_redirect(redirect_uri: str, *, error: str, desc: str | None, state: str | None) -> Response:
            params = {"error": error}
            if desc:
                params["error_description"] = desc
            if state:
                params["state"] = state
            return RedirectResponse(_append_query(redirect_uri, params), status_code=302)

        async def _oauth_authorize(request: Request) -> Response:
            qp = request.query_params
            response_type = (qp.get("response_type") or "").strip()
            client_id = (qp.get("client_id") or "").strip()
            redirect_uri = (qp.get("redirect_uri") or "").strip()
            state = (qp.get("state") or "").strip() or None
            requested_scope = (qp.get("scope") or "").strip()
            code_challenge = (qp.get("code_challenge") or "").strip()
            code_challenge_method = (qp.get("code_challenge_method") or "").strip()

            logger.info(f"OAuth /authorize: client_id={client_id}, response_type={response_type}, redirect_uri={redirect_uri}, scope={requested_scope}")

            if response_type != "code":
                logger.warning(f"OAuth /authorize: unsupported response_type={response_type}")
                return PlainTextResponse("unsupported_response_type", status_code=400)
            if not client_id:
                logger.warning("OAuth /authorize: missing client_id")
                return PlainTextResponse("invalid_request: missing client_id", status_code=400)
            client = _get_client(client_id)
            if not client:
                logger.warning(f"OAuth /authorize: unknown client_id={client_id}")
                return PlainTextResponse("unauthorized_client", status_code=400)
            if not redirect_uri:
                logger.warning("OAuth /authorize: missing redirect_uri")
                return PlainTextResponse("invalid_request: missing redirect_uri", status_code=400)
            if not client.redirect_uris or redirect_uri not in client.redirect_uris:
                # If redirect is not allowed, don't redirect (avoid open redirect).
                logger.warning(f"OAuth /authorize: redirect_uri not allowed: {redirect_uri}, allowed: {client.redirect_uris}")
                return PlainTextResponse("invalid_request: redirect_uri not allowed", status_code=400)

            if code_challenge_method != "S256" or not code_challenge:
                return _oauth_error_redirect(
                    redirect_uri,
                    error="invalid_request",
                    desc="PKCE S256 required (code_challenge + code_challenge_method=S256)",
                    state=state,
                )

            if requested_scope:
                requested_scopes = [s for s in requested_scope.split() if s]
                if any(s not in client.scopes for s in requested_scopes):
                    return _oauth_error_redirect(
                        redirect_uri,
                        error="invalid_scope",
                        desc="Requested scope not allowed for client",
                        state=state,
                    )
                scope = " ".join(requested_scopes)
            else:
                scope = " ".join(client.scopes)

            issuer = _request_issuer(request)
            now = int(time.time())
            exp = now + 120
            jti = uuid.uuid4().hex
            code = _jwt_encode(
                {
                    "iss": issuer,
                    "aud": "oauth-code",
                    "iat": now,
                    "exp": exp,
                    "jti": jti,
                    "client_id": client.client_id,
                    "redirect_uri": redirect_uri,
                    "scope": scope,
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                },
                settings.signing_secret,
            )

            params = {"code": code}
            if state:
                params["state"] = state
            logger.info(f"OAuth /authorize: issuing code for client_id={client_id}, scope={scope}, redirect_uri={redirect_uri}")
            return RedirectResponse(_append_query(redirect_uri, params), status_code=302)

        async def _oauth_token(request: Request) -> Response:
            # RFC 6749: x-www-form-urlencoded
            params: dict[str, str] = {}
            content_type = (request.headers.get("content-type") or "").lower()
            raw = await request.body()
            if "application/json" in content_type:
                try:
                    payload = json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    params = {str(k): str(v) for k, v in payload.items() if v is not None}
            else:
                parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
                params = {k: (v[-1] if v else "") for k, v in parsed.items()}

            grant_type = (params.get("grant_type") or "").strip()
            logger.info(f"OAuth /token: grant_type={grant_type}, client_id={params.get('client_id', '(from auth header)')}")
            if grant_type not in {"client_credentials", "authorization_code"}:
                logger.warning(f"OAuth /token: unsupported grant_type={grant_type}")
                return JSONResponse(
                    {"error": "unsupported_grant_type", "error_description": "Supported: client_credentials, authorization_code"},
                    status_code=400,
                )

            # Client authentication: Basic or client_secret_post or none (public client for auth code)
            client_id = ""
            client_secret: str | None = None
            auth = request.headers.get("authorization") or ""
            if auth.lower().startswith("basic "):
                try:
                    decoded = base64.b64decode(auth.split(" ", 1)[1].strip()).decode("utf-8")
                    client_id, client_secret = decoded.split(":", 1)
                except Exception:
                    return JSONResponse(
                        {"error": "invalid_client", "error_description": "Bad basic auth"},
                        status_code=401,
                    )
            else:
                client_id = (params.get("client_id") or "").strip()
                raw_secret = (params.get("client_secret") or "").strip()
                client_secret = raw_secret or None

            client = _get_client(client_id)
            if not client:
                logger.warning(f"OAuth /token: unknown client_id={client_id}")
                return JSONResponse(
                    {"error": "invalid_client", "error_description": "Unknown client"},
                    status_code=401,
                )

            if grant_type == "client_credentials":
                if not client.client_secret:
                    return JSONResponse(
                        {"error": "invalid_client", "error_description": "Client has no secret; client_credentials not allowed"},
                        status_code=401,
                    )
                if not client_secret or not hmac.compare_digest(client.client_secret, client_secret):
                    return JSONResponse(
                        {"error": "invalid_client", "error_description": "Bad secret"},
                        status_code=401,
                    )

                requested_scope = (params.get("scope") or "").strip()
                if requested_scope:
                    requested_scopes = [s for s in requested_scope.split() if s]
                    if any(s not in client.scopes for s in requested_scopes):
                        return JSONResponse(
                            {"error": "invalid_scope", "error_description": "Requested scope not allowed for client"},
                            status_code=400,
                        )
                    scope = " ".join(requested_scopes)
                else:
                    scope = " ".join(client.scopes)

            else:
                # authorization_code + PKCE (S256)
                code = (params.get("code") or "").strip()
                redirect_uri = (params.get("redirect_uri") or "").strip()
                code_verifier = (params.get("code_verifier") or "").strip()

                if not code or not redirect_uri or not code_verifier:
                    return JSONResponse(
                        {"error": "invalid_request", "error_description": "Missing code, redirect_uri, or code_verifier"},
                        status_code=400,
                    )

                # If the client has a secret configured, require it (confidential client).
                if client.client_secret:
                    if not client_secret or not hmac.compare_digest(client.client_secret, client_secret):
                        return JSONResponse(
                            {"error": "invalid_client", "error_description": "Bad secret"},
                            status_code=401,
                        )

                try:
                    code_payload = _jwt_decode(code, settings.signing_secret)
                except Exception:
                    return JSONResponse(
                        {"error": "invalid_grant", "error_description": "Invalid code"},
                        status_code=400,
                    )

                now = int(time.time())
                if code_payload.get("aud") != "oauth-code":
                    return JSONResponse({"error": "invalid_grant", "error_description": "Bad code audience"}, status_code=400)
                exp = code_payload.get("exp")
                if not isinstance(exp, int) or exp <= now:
                    return JSONResponse({"error": "invalid_grant", "error_description": "Code expired"}, status_code=400)

                issuer = _request_issuer(request)
                if code_payload.get("iss") != issuer:
                    return JSONResponse({"error": "invalid_grant", "error_description": "Bad code issuer"}, status_code=400)

                if code_payload.get("client_id") != client.client_id:
                    return JSONResponse({"error": "invalid_grant", "error_description": "Code client mismatch"}, status_code=400)
                if code_payload.get("redirect_uri") != redirect_uri:
                    return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, status_code=400)

                if code_payload.get("code_challenge_method") != "S256":
                    return JSONResponse({"error": "invalid_grant", "error_description": "PKCE S256 required"}, status_code=400)
                expected = code_payload.get("code_challenge") or ""
                if not isinstance(expected, str) or not expected:
                    return JSONResponse({"error": "invalid_grant", "error_description": "Missing code_challenge"}, status_code=400)
                actual = _sha256_b64url(code_verifier)
                if not hmac.compare_digest(expected, actual):
                    return JSONResponse({"error": "invalid_grant", "error_description": "Bad code_verifier"}, status_code=400)

                jti = code_payload.get("jti") or ""
                if not isinstance(jti, str) or not jti:
                    return JSONResponse({"error": "invalid_grant", "error_description": "Bad code jti"}, status_code=400)
                if not _mark_code_used(jti, exp, now):
                    return JSONResponse({"error": "invalid_grant", "error_description": "Code already used"}, status_code=400)

                scope = str(code_payload.get("scope") or "").strip() or " ".join(client.scopes)

            issuer = _request_issuer(request)
            now = int(time.time())
            ttl = int(settings.access_token_ttl_seconds)
            claims: dict[str, Any] = {
                "iss": issuer,
                "sub": client.client_id,
                "aud": settings.audience,
                "iat": now,
                "scope": scope,
            }
            # -1 means never expires; otherwise set exp claim
            if ttl >= 0:
                claims["exp"] = now + max(1, ttl)
            token = _jwt_encode(claims, settings.signing_secret)

            expires_log = "never" if ttl < 0 else f"{max(1, ttl)}s"
            logger.info(f"OAuth /token: issued access_token for client_id={client.client_id}, scope={scope}, expires_in={expires_log}")
            response_body: dict[str, Any] = {
                "access_token": token,
                "token_type": "Bearer",
                "scope": scope,
            }
            if ttl >= 0:
                response_body["expires_in"] = max(1, ttl)
            return JSONResponse(response_body)

        async def _oauth_register(request: Request) -> Response:
            """RFC 7591 Dynamic Client Registration endpoint."""
            if not settings.allow_dynamic_registration:
                return JSONResponse(
                    {"error": "registration_not_supported", "error_description": "Dynamic client registration is disabled"},
                    status_code=400,
                )

            content_type = (request.headers.get("content-type") or "").lower()
            if "application/json" not in content_type:
                return JSONResponse(
                    {"error": "invalid_request", "error_description": "Content-Type must be application/json"},
                    status_code=400,
                )

            try:
                body = await request.body()
                payload = json.loads(body.decode("utf-8") or "{}")
            except Exception:
                return JSONResponse(
                    {"error": "invalid_request", "error_description": "Invalid JSON body"},
                    status_code=400,
                )

            if not isinstance(payload, dict):
                return JSONResponse(
                    {"error": "invalid_request", "error_description": "Request body must be a JSON object"},
                    status_code=400,
                )

            # Extract client metadata
            redirect_uris = payload.get("redirect_uris") or []
            if not isinstance(redirect_uris, list):
                redirect_uris = [str(redirect_uris)]
            redirect_uris = [str(u).strip() for u in redirect_uris if str(u).strip()]

            # Validate redirect_uris (required for authorization_code flow)
            if not redirect_uris:
                return JSONResponse(
                    {"error": "invalid_redirect_uri", "error_description": "At least one redirect_uri is required"},
                    status_code=400,
                )

            # Validate redirect_uri format (must be valid URLs)
            for uri in redirect_uris:
                if not uri.startswith(("http://", "https://")):
                    return JSONResponse(
                        {"error": "invalid_redirect_uri", "error_description": f"Invalid redirect_uri: {uri}"},
                        status_code=400,
                    )

            client_name = payload.get("client_name") or ""
            grant_types = payload.get("grant_types") or ["authorization_code"]
            if not isinstance(grant_types, list):
                grant_types = [str(grant_types)]

            token_endpoint_auth_method = payload.get("token_endpoint_auth_method") or "none"

            # Generate client credentials
            client_id = uuid.uuid4().hex

            # For public clients (typical for Claude Code), no secret is generated.
            # For confidential clients, generate a secret.
            client_secret: str | None = None
            if token_endpoint_auth_method in ("client_secret_basic", "client_secret_post"):
                client_secret = uuid.uuid4().hex + uuid.uuid4().hex

            # Determine allowed scopes (default to all supported scopes)
            requested_scope = payload.get("scope") or ""
            if requested_scope:
                scopes = [s.strip() for s in str(requested_scope).split() if s.strip()]
                # Validate scopes against supported scopes
                invalid = [s for s in scopes if s not in settings.scopes_supported]
                if invalid:
                    return JSONResponse(
                        {"error": "invalid_client_metadata", "error_description": f"Unsupported scopes: {invalid}"},
                        status_code=400,
                    )
            else:
                scopes = list(settings.scopes_supported)

            # Create and store the client
            client_config = OAuthClientConfig(
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
                redirect_uris=redirect_uris,
            )
            dynamic_clients[client_id] = client_config

            logger.info(f"OAuth /register: registered new client client_id={client_id}, redirect_uris={redirect_uris}, scopes={scopes}")

            # Build response per RFC 7591
            response_payload: dict = {
                "client_id": client_id,
                "client_id_issued_at": int(time.time()),
                "redirect_uris": redirect_uris,
                "grant_types": grant_types,
                "token_endpoint_auth_method": token_endpoint_auth_method,
                "scope": " ".join(scopes),
            }
            if client_name:
                response_payload["client_name"] = client_name
            if client_secret:
                response_payload["client_secret"] = client_secret
                # Secrets don't expire in this implementation
                response_payload["client_secret_expires_at"] = 0

            return JSONResponse(response_payload, status_code=201)

        # OAuth 2.0 Protected Resource Metadata (RFC 9728)
        # Clients discover authorization by fetching this endpoint
        for path in (
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
            "/mcp/.well-known/oauth-protected-resource",
        ):
            app.add_route(path, _oauth_protected_resource_metadata, methods=["GET"])

        # OAuth Authorization Server Metadata (RFC 8414)
        for path in (
            "/.well-known/oauth-authorization-server",
            "/.well-known/oauth-authorization-server/mcp",
            "/mcp/.well-known/oauth-authorization-server",
        ):
            app.add_route(path, _oauth_metadata, methods=["GET"])

        for path in ("/authorize", "/oauth/authorize"):
            app.add_route(path, _oauth_authorize, methods=["GET"])

        app.add_route("/oauth/token", _oauth_token, methods=["POST"])

        if settings.allow_dynamic_registration:
            app.add_route("/oauth/register", _oauth_register, methods=["POST"])

    # Add CORS last so it can decorate even error responses from inner middleware.
    app.add_middleware(McpCorsMiddleware)

    return app


async def serve(host: str | None = None, port: int | None = None):
    load_dotenv()

    # Initialize database
    init_db()
    logger.info("Database initialized")

    import uvicorn

    app = create_app()

    config_host = get_env_or_config("HOST", "mcp.host")
    config_port = get_env_or_config("PORT", "mcp.port")
    host = host or (str(config_host) if config_host else mcp.settings.host)
    default_port = 9000
    port = port or (int(config_port) if config_port else default_port)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received exit signal, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    # Also handle SIGHUP if possible
    try:
        loop.add_signal_handler(signal.SIGHUP, _signal_handler)
    except (NotImplementedError, AttributeError):
        # SIGHUP may not be available on all platforms
        pass

    try:
        server_task = asyncio.create_task(server.serve())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            [server_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            await server.shutdown()
        await server_task
    except KeyboardInterrupt:
        logger.info("Server interrupted by KeyboardInterrupt")
    except asyncio.CancelledError:
        logger.info("Server cancelled and exited gracefully.")
    finally:
        # Wait for background tasks to complete
        await shutdown_background_tasks()
        logger.info("All resources cleaned up, exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m roamresearch_client_py.server")
    parser.add_argument("--host", help="Host to bind (overrides config/env)")
    parser.add_argument("--port", type=int, help="Port to bind (overrides config/env)")
    parser.add_argument(
        "--config",
        help="Path to config.toml (sets ROAM_CONFIG_FILE)",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["ROAM_CONFIG_FILE"] = args.config

    asyncio.run(serve(host=args.host, port=args.port))
