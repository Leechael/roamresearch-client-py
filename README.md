# roamresearch-client-py

This is another Roam Research Python Client with opinionated design.

Hightlight:
- Built-in CLI & MCP supports which should be LLM & Agent friendly.
- Pythonic coding API for building templates.

## Installation

```bash
# Install from PyPI
pip install roamresearch-client-py

# Or with uv as a CLI tool
uv tool install roamresearch-client-py
```

## Configuration

### Environment Variables

```bash
export ROAM_API_TOKEN="your-api-token"
export ROAM_API_GRAPH="your-graph-name"
```

### Configuration File

Run `rr init` to create a config file at `~/.config/roamresearch-client-py/config.toml`:

```toml
[roam]
api_token = "your-api-token"
api_graph = "your-graph-name"

[logging]
level = "WARNING"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
httpx_level = "WARNING"  # Control httpx library logging
```

## CLI Usage

```bash
# Initialize config file
rr init

# Get a page by title
rr get "Page Title"

# Get a block by uid
rr get "((block-uid))"

# Search blocks
rr search "keyword"
rr search "term1" "term2" --ignore-case --limit 50

# Save markdown to Roam as a new page
rr save --title "New Page" --file content.md
echo "# Hello" | rr save --title "New Page"

# Execute raw datalog query
rr q '[:find ?title :where [?e :node/title ?title]]'

# Start MCP server
rr mcp
rr mcp --port 9100
```

## Python API

### Create Blocks

```python
from roamresearch_client_py import RoamClient

async with RoamClient() as client:
    async with client.create_block("This is title") as blk:
        blk.write("Line 1")
        blk.write("Line 2")
        with blk:
            blk.write("Indented Line 3")
        blk.write("Back to normal")
# Everything saves in batch when exiting
```

### Query and Search

```python
async with RoamClient() as client:
    # Get page by title
    page = await client.get_page_by_title("My Page")

    # Get block by uid
    block = await client.get_block_by_uid("block-uid")

    # Search blocks
    results = await client.search_blocks(["keyword"], limit=20)

    # Raw datalog query
    result = await client.q('[:find ?title :where [?e :node/title ?title]]')
```

## MCP Server

The built-in MCP server exposes Roam Research tools for AI assistants.

```bash
# Start SSE MCP server (default port 9000)
rr mcp

# With custom settings
rr mcp --port 9100 --token <TOKEN> --graph <GRAPH>
```

## License

MIT
