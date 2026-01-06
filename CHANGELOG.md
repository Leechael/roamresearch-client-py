# Changelog

## 0.4.1 - 2026-01-06

- chore(release): v0.3.5
- feat: improve OAuth logging and return large expires_in for never-expiring tokens
- ci: fix broken tests
- fix: responses that legitimately need duplicate header names
- fix: convert middlewares to pure ASGI for SSE compatibility
- fix: typing
- feat: support never-expiring OAuth access tokens with ttl=-1

## 0.3.5 - 2026-01-06

- feat: improve OAuth logging and return large expires_in for never-expiring tokens
- ci: fix broken tests
- fix: responses that legitimately need duplicate header names
- fix: convert middlewares to pure ASGI for SSE compatibility
- fix: typing
- feat: support never-expiring OAuth access tokens with ttl=-1

## 0.4.0 - 2026-01-02

- feat: add OAuth Dynamic Client Registration (RFC 7591)
- fix: prevent info leak in OAuth error and CORS header reflection
- feat: add OAuth Protected Resource Metadata endpoint (RFC 9728)
- feat: add OAuth authorization_code + PKCE /authorize endpoint
- feat: add default CORS auto-allow Origin from Host for MCP SSE
- feat: add configurable config.toml path for pdm run start
- feat: Add optional OAuth2 client_credentials auth for MCP endpoints
- Normalize GFM task checkboxes to Roam TODO/DONE
- Treat block update_markdown as structured markdown when multiline/heading


## 0.3.4 - 2025-12-29

- fix(mcp): handle empty identifier list in handle_get
- feat(mcp): enhance search syntax and batch operations

## 0.3.3 - 2025-12-29

- fix(cli): exit code and refs lookup improvements
- fix(mcp): auto-create topic node if not exists
- docs: rewrite README
- feat(cli): unify save/update, add search enhancements

## 0.3.2 - 2025-12-29

- chore: update pdm scripts for server start and add test command
- fix: improve tag matching to handle end-of-string and prevent false positives
- ci: added compileall check
- test: add unit tests for search functions
- feat: add tag search, block references, and TODO search

