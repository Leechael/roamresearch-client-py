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
    with open(CONFIG_FILE, "r") as f:
        return toml.loads(f.read())


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

[storage]
# dir = ""  # Directory for debug files

[batch]
# size = 100
# max_retries = 3
"""
        with open(CONFIG_FILE, "w") as f:
            f.write(default_config)
    return CONFIG_FILE
