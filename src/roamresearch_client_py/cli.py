import argparse
import asyncio
import os
from typing import Sequence

from . import __version__
from .config import init_config_file, CONFIG_FILE
from .server import serve


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


if __name__ == "__main__":
    main()
