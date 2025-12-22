# roamresearch-client-py

This is another Roam Research Python Client with opinionated design.

Highlight:
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

# Update existing page or block (smart diff, preserves UIDs)
rr update "Page Title" --file updated.md
rr update "Page Title" --file updated.md --dry-run  # Preview changes
rr update "((block-uid))" --file content.md
echo "new content" | rr update "Page Title" --force

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

### Update Existing Content

```python
async with RoamClient() as client:
    # Update a single block's text
    result = await client.update_block_text(
        uid="block-uid",
        text="Updated content",
        dry_run=False  # Set True to preview without executing
    )

    # Update entire page with smart diff (preserves block UIDs)
    result = await client.update_page_markdown(
        title="Page Title",
        markdown="## New content\n- Item 1\n- Item 2",
        dry_run=False
    )

    # Result contains stats and preserved UIDs
    print(result['stats'])  # {'creates': 0, 'updates': 2, 'moves': 0, 'deletes': 0}
    print(result['preserved_uids'])  # List of reused block UIDs
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
