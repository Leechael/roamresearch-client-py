import argparse
import asyncio
import json
import os
import sys
from typing import Sequence

from . import __version__
from .config import init_config_file, get_config_file, configure_logging
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
        help="Save markdown to Roam Research (create or update).",
        description="Save markdown to Roam Research. Creates a new page or updates existing.",
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

    # get command (supports both page title and uid)
    get_cmd = subcommands.add_parser(
        "get",
        help="Read a page or block and output as markdown.",
        description="Fetch a page by title or block by uid and output its content as GFM markdown.",
    )
    get_cmd.add_argument(
        "identifier",
        help="Page title or block uid (accepts ((uid)) format).",
    )
    get_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Output raw JSON data instead of markdown.",
    )

    # search command
    search_cmd = subcommands.add_parser(
        "search",
        help="Search blocks containing text or tags.",
        description="Search for blocks containing all given terms, optionally filtered by tag.",
    )
    search_cmd.add_argument(
        "terms",
        nargs='*',
        help="Search terms (all must match). Optional if --tag is provided.",
    )
    search_cmd.add_argument(
        "--tag",
        "-t",
        help="Filter by tag (e.g., #TODO, [[Project]]).",
    )
    search_cmd.add_argument(
        "--page",
        "-p",
        help="Limit search to a specific page title.",
    )
    search_cmd.add_argument(
        "--ignore-case",
        "-i",
        action="store_true",
        help="Case-insensitive search.",
    )
    search_cmd.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum number of results (default: 20).",
    )

    # q command - raw datalog query
    q_cmd = subcommands.add_parser(
        "q",
        help="Execute a raw datalog query.",
        description="Execute a raw datalog query for debugging. Query can be provided as argument or via stdin.",
    )
    q_cmd.add_argument(
        "query",
        nargs='?',
        help="Datalog query string. If not provided, reads from stdin.",
    )
    q_cmd.add_argument(
        "--args",
        "-a",
        nargs='*',
        help="Query arguments (optional).",
    )

    # refs command
    refs_cmd = subcommands.add_parser(
        "refs",
        help="Find blocks referencing a page or block.",
        description="Find all blocks that reference a given page or block.",
    )
    refs_cmd.add_argument(
        "identifier",
        help="Page title or block UID to find references to.",
    )
    refs_cmd.add_argument(
        "--limit",
        "-n",
        type=int,
        default=50,
        help="Maximum number of results (default: 50).",
    )

    # todos command
    todos_cmd = subcommands.add_parser(
        "todos",
        help="Search TODO/DONE items.",
        description="Find TODO or DONE items in Roam.",
    )
    todos_cmd.add_argument(
        "--done",
        action="store_true",
        help="Search for DONE items instead of TODO.",
    )
    todos_cmd.add_argument(
        "--page",
        "-p",
        help="Limit search to a specific page title.",
    )
    todos_cmd.add_argument(
        "--limit",
        "-n",
        type=int,
        default=50,
        help="Maximum number of results (default: 50).",
    )

    return parser


def main(argv: Sequence[str] | None = None):
    # Configure logging early to suppress httpx INFO logs
    configure_logging()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        config_file = get_config_file()
        if config_file.exists() and not args.force:
            print(f"Configuration file already exists: {config_file}")
            print("Use --force to overwrite.")
            return
        if args.force and config_file.exists():
            config_file.unlink()
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

    if args.command == "get":
        _run_async(_get(args.identifier, args.debug))
        return

    if args.command == "search":
        if not args.terms and not args.tag:
            parser.error("search requires terms or --tag")
        _run_async(_search_blocks(
            args.terms or [],
            args.limit,
            case_sensitive=not args.ignore_case,
            page_title=args.page,
            tag=args.tag
        ))
        return

    if args.command == "q":
        _run_async(_query(args.query, args.args))
        return

    if args.command == "refs":
        _run_async(_refs(args.identifier, args.limit))
        return

    if args.command == "todos":
        status = "DONE" if args.done else "TODO"
        _run_async(_todos(status, args.limit, args.page))
        return


async def _save_markdown(title: str, file_path: str | None):
    """Save markdown content to Roam, creating or updating as needed."""
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown = f.read()
    else:
        markdown = sys.stdin.read()

    if not markdown.strip():
        print("Error: No content provided.", file=sys.stderr)
        return

    async with RoamClient() as client:
        # Check if page exists
        existing = await client.get_page_by_title(title)

        if existing:
            # Update existing page
            result = await client.update_page_markdown(title, markdown)
            stats = result['stats']
            print(f"Updated page: {title}")
            print(f"  {stats.get('creates', 0)} creates, "
                  f"{stats.get('updates', 0)} updates, "
                  f"{stats.get('moves', 0)} moves, "
                  f"{stats.get('deletes', 0)} deletes")
        else:
            # Create new page
            page = create_page(title)
            page_uid = page['page']['uid']
            actions = [page] + gfm_to_batch_actions(markdown, page_uid)
            await client.batch_actions(actions)
            print(f"Created page: {title}")


def _parse_uid(identifier: str) -> str | None:
    """
    Parse uid from ((uid)) or uid format.
    Returns None if identifier looks like a page title.
    """
    identifier = identifier.strip()

    # Wrapped in (()) - definitely a uid
    if identifier.startswith('((') and identifier.endswith('))'):
        return identifier[2:-2]

    # Contains spaces or CJK characters - likely a page title
    if ' ' in identifier or any('\u4e00' <= c <= '\u9fff' for c in identifier):
        return None

    # Short alphanumeric with possible - and _ could be uid
    # Roam uids are typically 9 chars or 32 hex chars
    if len(identifier) <= 40 and all(c.isalnum() or c in '-_' for c in identifier):
        return identifier

    return None


async def _get(identifier: str, debug: bool = False):
    """Read a page or block and output as markdown."""
    uid = _parse_uid(identifier)

    async with RoamClient() as client:
        result = None
        is_page = False

        # Try as uid first if it looks like one
        if uid:
            result = await client.get_block_by_uid(uid)

        # If not found or doesn't look like uid, try as page title
        if not result:
            result = await client.get_page_by_title(identifier)
            is_page = True

    if not result:
        print(f"Error: '{identifier}' not found.", file=sys.stderr)
        return

    if debug:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Get children
    children = result.get(':block/children', [])
    if children:
        children = sorted(children, key=lambda x: x.get(':block/order', 0))

    if is_page:
        # For pages, format children directly
        output = format_block_as_markdown(children)
    else:
        # For blocks, include the block itself
        output = format_block_as_markdown([result])

    print(output)


def _print_results_grouped(results: list):
    """Print search results grouped by page."""
    by_page: dict[str, list[tuple[str, str]]] = {}
    page_order: list[str] = []
    for item in results:
        uid, text, page = item[0], item[1], item[2]
        if page not in by_page:
            by_page[page] = []
            page_order.append(page)
        by_page[page].append((uid, text))

    for page in page_order:
        blocks = by_page[page]
        print(f"[[{page}]]")
        for uid, text in blocks:
            display_text = text.replace('\n', ' ')
            if len(display_text) > 60:
                display_text = display_text[:57] + "..."
            print(f"  {uid}   {display_text}")
        print()


async def _search_blocks(
    terms: list[str],
    limit: int,
    case_sensitive: bool = True,
    page_title: str | None = None,
    tag: str | None = None
):
    """Search blocks and output results grouped by page."""
    async with RoamClient() as client:
        if terms:
            # Text search (optionally with tag filter)
            results = await client.search_blocks(
                terms,
                limit,
                case_sensitive=case_sensitive,
                page_title=page_title,
                tag=tag
            )
        else:
            # Tag-only search
            results = await client.search_by_tag(
                tag,
                limit=limit,
                page_title=page_title
            )

    if not results:
        print("No results found.")
        return

    _print_results_grouped(results)


async def _query(query: str | None, args: list[str] | None):
    """Execute a raw datalog query."""
    if query:
        q = query
    else:
        q = sys.stdin.read()

    if not q.strip():
        print("Error: No query provided.", file=sys.stderr)
        return

    async with RoamClient() as client:
        result = await client.q(q, args)

    print(json.dumps(result, indent=2, ensure_ascii=False))


async def _refs(identifier: str, limit: int):
    """Find blocks referencing a page or block."""
    identifier = identifier.strip()

    # Explicit ((uid)) format - definitely block reference
    if identifier.startswith('((') and identifier.endswith('))'):
        uid = identifier[2:-2]
        async with RoamClient() as client:
            results = await client.find_references(uid, limit=limit)
        if not results:
            print("No references found.")
            return
        _print_results_grouped(results)
        return

    # Otherwise: try page refs first, fallback to block refs
    async with RoamClient() as client:
        results = await client.find_page_references(identifier, limit=limit)
        if not results:
            # Fallback: maybe it's a block UID
            uid = _parse_uid(identifier)
            if uid:
                results = await client.find_references(uid, limit=limit)

    if not results:
        print("No references found.")
        return

    _print_results_grouped(results)


async def _todos(status: str, limit: int, page_title: str | None = None):
    """Search TODO/DONE items."""
    async with RoamClient() as client:
        results = await client.search_todos(
            status=status,
            limit=limit,
            page_title=page_title
        )

    if not results:
        print(f"No {status} items found.")
        return

    _print_results_grouped(results)


if __name__ == "__main__":
    main()
