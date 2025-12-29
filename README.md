# roamresearch-client-py

Roam Research client. Programmable. CLI. SDK. MCP.

For developers who automate. For LLMs that need graph access. Smart diff keeps UIDs intact. References survive. Minimal API calls.

## Install

```bash
pip install roamresearch-client-py

# standalone
uv tool install roamresearch-client-py
```

## Setup

```bash
export ROAM_API_TOKEN="your-token"
export ROAM_API_GRAPH="your-graph"
```

Or `rr init` creates `~/.config/roamresearch-client-py/config.toml`:

```toml
[roam]
api_token = "your-token"
api_graph = "your-graph"

[mcp]
host = "127.0.0.1"
port = 9000
topic_node = ""

[batch]
size = 100
max_retries = 3

[logging]
level = "WARNING"
httpx_level = "WARNING"
```

Env vars take precedence.

## CLI

### Get

```bash
rr get "Page Title"
rr get "((block-uid))"
rr get "Page Title" --debug
```

### Search

```bash
rr search "keyword"
rr search "term1" "term2"
rr search "term" --tag "#TODO"
rr search --tag "[[Project]]"
rr search "term" --page "Page" -i -n 50
```

### Save

Create or update. Preserves UIDs.

```bash
rr save -t "Page" -f content.md
echo "# Hello" | rr save -t "Page"
```

### Refs

```bash
rr refs "Page Title"
rr refs "block-uid"
```

### Todos

```bash
rr todos
rr todos --done
rr todos --page "Work" -n 100
```

### Query

```bash
rr q '[:find ?title :where [?e :node/title ?title]]'
rr q '[:find ?t :in $ ?p :where ...]' --args "Page"
```

### MCP

```bash
rr mcp
rr mcp --port 9100
rr mcp --token <T> --graph <G>
```

## SDK

### Connect

```python
from roamresearch_client_py import RoamClient

async with RoamClient() as client:
    pass

async with RoamClient(api_token="...", graph="...") as client:
    pass
```

### Write

```python
async with client.create_block("Root") as blk:
    blk.write("Child 1")
    blk.write("Child 2")
    with blk:
        blk.write("Grandchild")
    blk.write("Child 3")
```

### Read

```python
page = await client.get_page_by_title("Page")
block = await client.get_block_by_uid("uid")
daily = await client.get_daily_page()
```

### Search

```python
results = await client.search_blocks(["python", "async"], limit=50)
todos = await client.search_by_tag("#TODO", limit=50)
refs = await client.find_references("block-uid")
refs = await client.find_page_references("Page Title")
todos = await client.search_todos(status="TODO", page_title="Work")
```

### Update

```python
await client.update_block_text("uid", "New text")

result = await client.update_page_markdown("Page", "## New\n- Item", dry_run=False)
# result['stats'] = {'creates': 0, 'updates': 2, 'moves': 0, 'deletes': 0}
# result['preserved_uids'] = ['uid1', 'uid2']
```

### Query

```python
result = await client.q('[:find ?title :where [?e :node/title ?title]]')
```

### Batch

Atomic operations.

```python
from roamresearch_client_py.client import (
    create_page, create_block, update_block, remove_block, move_block
)

actions = [
    create_page("New Page"),
    create_block("Text", parent_uid="page-uid"),
    update_block("uid", "Updated"),
    move_block("uid", parent_uid="new-parent", order=0),
    remove_block("old-uid"),
]
await client.batch_actions(actions)
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `save_markdown` | Create/update page |
| `get` | Fetch as markdown |
| `search` | Text + tag search |
| `query` | Raw Datalog |
| `find_references` | Block/page refs |
| `search_todos` | TODO/DONE items |
| `update_markdown` | Smart diff update |

```json
{"title": "Notes", "markdown": "## Topic\n- Point"}
{"terms": ["python"], "tag": "TODO", "limit": 20}
{"identifier": "Page", "markdown": "## New", "dry_run": true}
```

## Internals

**Smart Diff** — Match by content. Preserve UIDs. Detect moves. Minimize calls.

**Markdown ↔ Roam** — Bidirectional. Headings, lists, tables, code, inline.

**Task Queue** — SQLite. Background. Retry. JSONL logs.

## Requirements

- Python 3.10+
- Roam API token

## License

MIT
