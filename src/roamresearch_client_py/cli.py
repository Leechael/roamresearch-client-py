import argparse
import asyncio
import os
from typing import Sequence

from . import __version__
from .mcp import serve


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
    return parser


def main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "mcp":
        if args.token:
            os.environ["ROAM_API_TOKEN"] = args.token
        if args.graph:
            os.environ["ROAM_API_GRAPH"] = args.graph
        if args.debug_storage:
            os.environ["ROAM_STORAGE_DIR"] = args.debug_storage
        _run_async(serve(port=args.port))
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
