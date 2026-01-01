import os

from roamresearch_client_py.config import get_config_file, get_env_or_config


def test_get_env_or_config_respects_roam_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.toml"
    cfg.write_text(
        """
[mcp]
host = "0.0.0.0"
port = 9999
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ROAM_CONFIG_FILE", str(cfg))
    assert get_config_file() == cfg
    assert get_env_or_config("HOST", "mcp.host") == "0.0.0.0"
    assert str(get_env_or_config("PORT", "mcp.port")) == "9999"

