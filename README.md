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
# Starts an SSE MCP server by default
python -m roamresearch_client_py.mcp
```

## Prerequisites

Set `ROAM_API_GRAPH` and `ROAM_API_TOKEN` via environment variables. Or specify them when initializing `RoamClient`.