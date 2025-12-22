import logging
import os
from pathlib import Path
from typing import Any

import toml


CONFIG_DIR = Path.home() / ".config" / "roamresearch-client-py"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def get_config_dir() -> Path:
    """Get the configuration directory, creating it if necessary."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict[str, Any]:
    """Load configuration from the config file."""
    if not CONFIG_FILE.exists():
        return {}
    return toml.load(CONFIG_FILE)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value by key (supports nested keys with dots)."""
    config = load_config()
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def get_env_or_config(env_key: str, config_key: str | None = None, default: Any = None) -> Any:
    """Get a value from environment variable or config file.
    
    Environment variables take precedence over config file values.
    """
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    if config_key is None:
        config_key = env_key.lower()
    return get_config_value(config_key, default)


def init_config_file() -> Path:
    """Create a default config file if it doesn't exist."""
    get_config_dir()
    if not CONFIG_FILE.exists():
        default_config = """\
# Roam Research Client Configuration
# https://github.com/user/roamresearch-client-py

[roam]
# api_token = "your-api-token"
# api_graph = "your-graph-name"

[mcp]
# host = "127.0.0.1"
# port = 9000
# topic_node = ""
# allowed_hosts = ""  # Comma-separated list of allowed hosts for remote MCP

[storage]
# dir = ""  # Directory for debug files

[batch]
# size = 100
# max_retries = 3

[logging]
# level = "WARNING"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
# httpx_level = "WARNING"  # Control httpx library logging separately
"""
        with open(CONFIG_FILE, "w") as f:
            f.write(default_config)
    return CONFIG_FILE


def configure_logging(
    level: str | None = None,
    httpx_level: str | None = None,
) -> None:
    """Configure logging levels for the library and httpx.

    Args:
        level: Log level for roamresearch_client_py (default from config or WARNING)
        httpx_level: Log level for httpx library (default from config or WARNING)
    """
    # Get levels from config if not provided
    if level is None:
        level = get_env_or_config("ROAM_LOG_LEVEL", "logging.level", "WARNING")
    if httpx_level is None:
        httpx_level = get_env_or_config("ROAM_HTTPX_LOG_LEVEL", "logging.httpx_level", "WARNING")

    # Convert string to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_map.get(level.upper(), logging.WARNING)
    httpx_log_level = level_map.get(httpx_level.upper(), logging.WARNING)

    # Configure roamresearch_client_py logger
    logger = logging.getLogger("roamresearch_client_py")
    logger.setLevel(log_level)

    # Configure httpx and httpcore loggers
    logging.getLogger("httpx").setLevel(httpx_log_level)
    logging.getLogger("httpcore").setLevel(httpx_log_level)
