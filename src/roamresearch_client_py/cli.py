import argparse
import asyncio
import json
import os
import sys
from typing import Sequence

from . import __version__
from .config import init_config_file, CONFIG_FILE
from .server import serve
from .client import RoamClient, create_page
from .gfm_to_roam import gfm_to_batch_actions
from .formatter import format_block_as_markdown


def _run_async(coro):
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        pass


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rr",
        description="Roam Research helper utilities.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    mcp_cmd = subcommands.add_parser(
        "mcp",
        help="Run the RoamResearch MCP server.",
        description="Run the RoamResearch MCP server.",
    )
    mcp_cmd.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port to listen on (default 9000; overrides PORT env var).",
    )
    mcp_cmd.add_argument(
        "--token",
        help="Roam Research API token (overrides ROAM_API_TOKEN env var).",
    )
    mcp_cmd.add_argument(
        "--graph",
        help="Roam Research graph name (overrides ROAM_API_GRAPH env var).",
    )
    mcp_cmd.add_argument(
        "--debug-storage",
        help="Directory to write debug payloads (overrides ROAM_STORAGE_DIR env var).",
    )

    init_cmd = subcommands.add_parser(
        "init",
        help="Initialize configuration file.",
        description="Create a default configuration file at ~/.config/roamresearch-client-py/config.toml",
    )
    init_cmd.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing configuration file.",
    )

    # save command
    save_cmd = subcommands.add_parser(
        "save",
        help="Save markdown to Roam Research.",
        description="Save a markdown file or stdin content to Roam Research as a new page.",
    )
    save_cmd.add_argument(
        "--title",
        "-t",
        required=True,
        help="Title of the page to create.",
    )
    save_cmd.add_argument(
        "--file",
        "-f",
        help="Markdown file to save. If not provided, reads from stdin.",
    )

    # page command
    page_cmd = subcommands.add_parser(
        "page",
        help="Read a page and output as markdown.",
        description="Fetch a page by title and output its content as GFM markdown.",
    )
    page_cmd.add_argument(
        "title",
        help="Title of the page to read.",
    )
    page_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Output raw JSON data instead of markdown.",
    )

    # uid command
    uid_cmd = subcommands.add_parser(
        "uid",
        help="Read a block by uid and output as markdown.",
        description="Fetch a block by uid and output its content as GFM markdown.",
    )
    uid_cmd.add_argument(
        "uid",
        help="Block uid (accepts ((uid)) or uid format).",
    )
    uid_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Output raw JSON data instead of markdown.",
    )

    # search command
    search_cmd = subcommands.add_parser(
        "search",
        help="Search blocks containing text.",
        description="Search for blocks containing all given terms.",
    )
    search_cmd.add_argument(
        "terms",
        nargs='+',
        help="Search terms (all must match).",
    )
    search_cmd.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum number of results (default: 20).",
    )

    return parser


def main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        if CONFIG_FILE.exists() and not args.force:
            print(f"Configuration file already exists: {CONFIG_FILE}")
            print("Use --force to overwrite.")
            return
        if args.force and CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        config_path = init_config_file()
        print(f"Configuration file created: {config_path}")
        return

    if args.command == "mcp":
        if args.token:
            os.environ["ROAM_API_TOKEN"] = args.token
        if args.graph:
            os.environ["ROAM_API_GRAPH"] = args.graph
        if args.debug_storage:
            os.environ["ROAM_STORAGE_DIR"] = args.debug_storage
        _run_async(serve(port=args.port))
        return

    if args.command == "save":
        _run_async(_save_markdown(args.title, args.file))
        return

    if args.command == "page":
        _run_async(_read_page(args.title, args.debug))
        return

    if args.command == "uid":
        _run_async(_read_uid(args.uid, args.debug))
        return

    if args.command == "search":
        _run_async(_search_blocks(args.terms, args.limit))
        return


async def _save_markdown(title: str, file_path: str | None):
    """Save markdown content to Roam as a new page."""
    if file_path:
        with open(file_path, 'r') as f:
            markdown = f.read()
    else:
        markdown = sys.stdin.read()

    if not markdown.strip():
        print("Error: No content provided.", file=sys.stderr)
        return

    page = create_page(title)
    page_uid = page['page']['uid']
    actions = [page] + gfm_to_batch_actions(markdown, page_uid)

    async with RoamClient() as client:
        await client.batch_actions(actions)

    print(f"Saved page: {title}")


async def _read_page(title: str, debug: bool = False):
    """Read a page and output as markdown."""
    async with RoamClient() as client:
        page = await client.get_page_by_title(title)

    if not page:
        print(f"Error: Page '{title}' not found.", file=sys.stderr)
        return

    if debug:
        print(json.dumps(page, indent=2, ensure_ascii=False))
        return

    # Get top-level children
    children = page.get(':block/children', [])
    if children:
        children = sorted(children, key=lambda x: x.get(':block/order', 0))

    output = format_block_as_markdown(children)
    print(output)


def _parse_uid(uid: str) -> str:
    """Parse uid from ((uid)) or uid format."""
    uid = uid.strip()
    if uid.startswith('((') and uid.endswith('))'):
        return uid[2:-2]
    return uid


async def _read_uid(uid: str, debug: bool = False):
    """Read a block by uid and output as markdown."""
    uid = _parse_uid(uid)

    async with RoamClient() as client:
        block = await client.get_block_by_uid(uid)

    if not block:
        print(f"Error: Block '{uid}' not found.", file=sys.stderr)
        return

    if debug:
        print(json.dumps(block, indent=2, ensure_ascii=False))
        return

    # Format the block itself and its children
    output = format_block_as_markdown([block])
    print(output)


async def _search_blocks(terms: list[str], limit: int):
    """Search blocks and output results."""
    async with RoamClient() as client:
        results = await client.search_blocks(terms, limit)

    if not results:
        print("No results found.")
        return

    for item in results:
        block = item[0]
        uid = block.get(':block/uid', '')
        text = block.get(':block/string', '')
        # Truncate long text
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"(({uid})) {text}")


if __name__ == "__main__":
    main()
