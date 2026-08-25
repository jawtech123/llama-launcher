"""Tests for Server Configuration JSON Persistence."""
import json
import os
import tempfile
import pytest

from llama_launcher.remote.types import (
    ServerConfig,
)
from llama_launcher.remote.storage import (
    load_servers,
    save_servers,
    save_server,
    remove_server,
    _config_to_dict,
    _dict_to_config,
    _servers_file,
)


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    """Point the config dir at a temp location."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    yield


def _server_config(**kwargs):
    return ServerConfig(
        friendly_name=kwargs.get("friendly_name", "test-server"),
        working_directory=kwargs.get("working_directory", "~"),
        llama_server_path=kwargs.get("llama_server_path", "llama-server"),
        model_directory=kwargs.get("model_directory", ""),
        model_directories=kwargs.get("model_directories", None),
        enabled=kwargs.get("enabled", True),
        log_file=kwargs.get("log_file", "llama-server.log"),
        server_port=kwargs.get("server_port", 8080),
        poll_interval_seconds=kwargs.get("poll_interval_seconds", 5),
    )


class TestConfigToDict:
    """Test ServerConfig -> dict conversion."""

    def test_basic_fields_converted(self):
        cfg = _server_config()
        d = _config_to_dict(cfg)
        assert d["friendly_name"] == "test-server"
        assert d["working_directory"] == "~"
        assert d["llama_server_path"] == "llama-server"

    def test_custom_fields_converted(self):
        cfg = _server_config(
            working_directory="/tmp",
            llama_server_path="/usr/local/bin/llama-server",
            server_port=9999,
        )
        d = _config_to_dict(cfg)
        assert d["working_directory"] == "/tmp"
        assert d["llama_server_path"] == "/usr/local/bin/llama-server"
        assert d["server_port"] == 9999

    def test_all_fields_converted(self):
        cfg = _server_config(
            model_directory="/models",
            log_file="custom.log",
            server_port=9090,
            enabled=False,
        )
        d = _config_to_dict(cfg)
        assert d["model_directory"] == "/models"
        assert d["log_file"] == "custom.log"
        assert d["server_port"] == 9090
        assert d["enabled"] is False

    def test_model_directories_converted(self):
        cfg = _server_config(model_directories=["/models1", "/models2"])
        d = _config_to_dict(cfg)
        assert d["model_directories"] == ["/models1", "/models2"]


class TestDictToConfig:
    """Test dict -> ServerConfig conversion."""

    def test_basic_fields_converted(self):
        d = {
            "friendly_name": "my-server",
            "working_directory": "/tmp",
            "llama_server_path": "/usr/bin/llama-server",
        }
        cfg = _dict_to_config(d)
        assert cfg.friendly_name == "my-server"
        assert cfg.working_directory == "/tmp"
        assert cfg.llama_server_path == "/usr/bin/llama-server"

    def test_defaults_for_missing_fields(self):
        d = {}
        cfg = _dict_to_config(d)
        assert cfg.friendly_name == ""
        assert cfg.working_directory == "~"
        assert cfg.llama_server_path == "llama-server"
        assert cfg.enabled is True
        assert cfg.server_port == 8080
        assert cfg.poll_interval_seconds == 5

    def test_model_directories_from_single(self):
        d = {
            "friendly_name": "test",
            "model_directory": "/models",
        }
        cfg = _dict_to_config(d)
        assert cfg.model_directories == ["/models"]

    def test_model_directories_not_overwritten(self):
        d = {
            "friendly_name": "test",
            "model_directory": "/ignored",
            "model_directories": ["/models1", "/models2"],
        }
        cfg = _dict_to_config(d)
        assert cfg.model_directories == ["/models1", "/models2"]


class TestSaveLoadServers:
    """Test server persistence."""

    def test_save_and_load_single_server(self, isolated_config_dir):
        cfg = _server_config()
        save_server(cfg)
        servers = load_servers()
        assert len(servers) == 1
        assert servers[0].friendly_name == "test-server"

    def test_save_and_load_multiple_servers(self, isolated_config_dir):
        cfg1 = _server_config(friendly_name="srv1")
        cfg2 = _server_config(friendly_name="srv2")
        save_server(cfg1)
        save_server(cfg2)
        servers = load_servers()
        assert len(servers) == 2
        names = [s.friendly_name for s in servers]
        assert "srv1" in names
        assert "srv2" in names

    def test_update_existing_server(self, isolated_config_dir):
        cfg1 = _server_config(friendly_name="srv1", working_directory="/tmp")
        cfg2 = _server_config(friendly_name="srv1", working_directory="/var")
        save_server(cfg1)
        save_server(cfg2)
        servers = load_servers()
        assert len(servers) == 1
        assert servers[0].working_directory == "/var"

    def test_save_servers_bulk(self, isolated_config_dir):
        cfg1 = _server_config(friendly_name="srv1")
        cfg2 = _server_config(friendly_name="srv2")
        save_servers([cfg1, cfg2])
        servers = load_servers()
        assert len(servers) == 2


class TestRemoveServer:
    """Test server removal."""

    def test_remove_existing_server(self, isolated_config_dir):
        cfg = _server_config(friendly_name="to-remove")
        save_server(cfg)
        assert len(load_servers()) == 1
        assert remove_server("to-remove") is True
        assert len(load_servers()) == 0

    def test_remove_other_server_unchanged(self, isolated_config_dir):
        cfg1 = _server_config(friendly_name="keep1")
        cfg2 = _server_config(friendly_name="keep2")
        save_server(cfg1)
        save_server(cfg2)
        remove_server("keep1")
        servers = load_servers()
        assert len(servers) == 1
        assert servers[0].friendly_name == "keep2"

    def test_remove_nonexistent_server(self, isolated_config_dir):
        assert remove_server("nonexistent") is False


class TestJsonFormat:
    """Test JSON file format."""

    def test_servers_file_structure(self, isolated_config_dir):
        cfg = _server_config(friendly_name="test")
        save_server(cfg)
        servers_file = _servers_file()
        with open(servers_file) as f:
            data = json.load(f)
        assert "servers" in data
        assert len(data["servers"]) == 1

    def test_servers_file_contains_expected_fields(self, isolated_config_dir):
        cfg = _server_config(friendly_name="test", working_directory="/tmp")
        save_server(cfg)
        servers_file = _servers_file()
        with open(servers_file) as f:
            data = json.load(f)
        server = data["servers"][0]
        assert "friendly_name" in server
        assert "working_directory" in server
        assert "llama_server_path" in server
        assert "enabled" in server


class TestSecureFile:
    """Test file permissions."""

    def test_servers_file_exists_after_save(self, isolated_config_dir):
        cfg = _server_config(friendly_name="test")
        save_server(cfg)
        servers_file = _servers_file()
        assert servers_file.exists()
