"""Server Configuration JSON Persistence."""
import json
import logging
from pathlib import Path
from typing import Dict, List

from llama_launcher.utils.paths import config_file, secure_file
from .types import LOCAL_SERVER_NAME, ServerConfig

logger = logging.getLogger(__name__)


def _servers_file() -> Path:
    """Path to the servers store."""
    return config_file("servers.json")


def _config_to_dict(config: ServerConfig) -> Dict:
    d = {}
    d["friendly_name"] = config.friendly_name
    d["working_directory"] = config.working_directory
    d["llama_server_path"] = config.llama_server_path
    d["model_directory"] = config.model_directory
    d["model_directories"] = config.model_directories
    d["enabled"] = config.enabled
    d["log_file"] = config.log_file
    d["server_port"] = config.server_port
    d["poll_interval_seconds"] = config.poll_interval_seconds
    # Remote / SSH fields
    d["hostname"] = config.hostname
    d["ssh_port"] = config.ssh_port
    d["username"] = config.username
    d["auth_method"] = config.auth_method
    d["key_path"] = config.key_path
    # servers.json is plaintext (0600 only), so a password is stored only when
    # the user asked for it. Key auth is the supported way to avoid this.
    d["remember_password"] = config.remember_password
    d["password"] = config.password if config.remember_password else ""
    d["environment"] = config.environment
    return d


def _dict_to_config(d: Dict) -> ServerConfig:
    model_directories = d.get("model_directories", None)
    if model_directories is None and d.get("model_directory"):
        model_directories = [d["model_directory"]]
    # Entries written before the opt-in existed already hold a password; keep
    # honouring that rather than silently dropping it on the next save.
    stored_password = d.get("password", "")
    remember_password = d.get("remember_password", bool(stored_password))
    return ServerConfig(
        friendly_name=d.get("friendly_name", ""),
        working_directory=d.get("working_directory", "~"),
        llama_server_path=d.get("llama_server_path", "llama-server"),
        model_directory=d.get("model_directory", ""),
        model_directories=model_directories,
        enabled=d.get("enabled", True),
        log_file=d.get("log_file", "llama-server.log"),
        server_port=d.get("server_port", 8080),
        poll_interval_seconds=d.get("poll_interval_seconds", 5),
        hostname=d.get("hostname", ""),
        ssh_port=d.get("ssh_port", 22),
        username=d.get("username", ""),
        auth_method=d.get("auth_method", "key"),
        key_path=d.get("key_path", ""),
        password=stored_password,
        remember_password=remember_password,
        environment=d.get("environment", "") or "",
    )


def load_servers() -> List[ServerConfig]:
    """Load all saved server configurations."""
    servers_file = _servers_file()
    if not servers_file.exists():
        return []
    try:
        with open(servers_file, "r") as f:
            data = json.load(f)
        servers = []
        for entry in data.get("servers", []):
            servers.append(_dict_to_config(entry))
        return servers
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Could not read servers file %s: %s", servers_file, e)
        return []


def save_servers(servers: List[ServerConfig]) -> None:
    """Persist all server configurations."""
    data = {
        "servers": [_config_to_dict(s) for s in servers],
    }
    servers_file = _servers_file()
    with open(servers_file, "w") as f:
        json.dump(data, f, indent=2)
    secure_file(servers_file)


def save_server(config: ServerConfig) -> None:
    """Add or update a single server configuration."""
    servers = load_servers()
    for i, existing in enumerate(servers):
        if existing.friendly_name == config.friendly_name:
            servers[i] = config
            save_servers(servers)
            return
    servers.append(config)
    save_servers(servers)


def remove_server(friendly_name: str) -> bool:
    """Remove a server configuration by friendly name.

    The local machine entry is permanent and is never removed, whichever code
    path asks -- the GUI blocks it too, this is the backstop.
    """
    if friendly_name == LOCAL_SERVER_NAME:
        logger.debug("Refusing to remove the permanent %r entry", LOCAL_SERVER_NAME)
        return False
    servers = load_servers()
    original_len = len(servers)
    servers = [s for s in servers if s.friendly_name != friendly_name]
    if len(servers) < original_len:
        save_servers(servers)
        return True
    return False
