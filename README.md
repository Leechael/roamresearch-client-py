# roamresearch-client-py

This is another Roam Research Python Client with opinionated design.

## Highlights

### Create Blocks Like Code

Quick example:

```python
async with RoamClient() as c:
    async c.create_block("This is title") as blk:
        blk.write('Line 1')
        blk.write('Line 2')
        with blk:
            blk.write('Indent & Line 3')
        blk.write('Dedent')
# Everything saves in batch when exiting
```

### Built-in MCP Server

```bash
# Install (either from PyPI or the current directory)
uv tool install roamresearch-client-py
# or
uv tool install .

# Starts an SSE MCP server
rr mcp
# Listen on a custom port (defaults to 9000)
rr mcp --port 9100
# Provide credentials directly
rr mcp --token <ROAM_API_TOKEN> --graph <ROAM_API_GRAPH>
# Write failed payloads somewhere specific
rr mcp --debug-storage /tmp/rr-debug
```

## Prerequisites

Set `ROAM_API_GRAPH` and `ROAM_API_TOKEN` via environment variables. Or specify them when initializing `RoamClient`. To capture failed payloads, set `ROAM_STORAGE_DIR` or pass `--debug-storage`.
