"""Multi-Server Manager."""
from typing import Dict, List, Optional
from llama_launcher.models.types import ModelInfo
from .types import (
    ServerConfig,
    ServerStatus,
    ConnectionState,
    ServerProcessState,
    ConnectionType,
    FileEntry,
)
from .connection import IServerConnection
from .local import LocalServerConnection
from .ssh import SSHServerConnection
from .storage import load_servers, save_server, remove_server as storage_remove


class ServerManager:
    """Manages multiple server connections of heterogeneous types.

    The UI interacts only through this manager, never directly with
    individual connection implementations.
    """

    def __init__(self):
        self._connections: Dict[str, IServerConnection] = {}

    def add_server(self, config: ServerConfig, persist: bool = True) -> bool:
        """Register a server configuration and create its connection."""
        if persist:
            save_server(config)
        conn = self._create_connection(config)
        self._connections[config.friendly_name] = conn
        return True

    def remove_server(self, friendly_name: str) -> bool:
        """Disconnect and remove a server."""
        if friendly_name in self._connections:
            self._connections[friendly_name].disconnect()
            del self._connections[friendly_name]
        return storage_remove(friendly_name)

    def get_connection(self, friendly_name: str) -> Optional[IServerConnection]:
        """Return the connection for a server by name."""
        return self._connections.get(friendly_name)

    def get_or_create_connection(self, config: ServerConfig) -> IServerConnection:
        """Get existing connection or create a new one."""
        name = config.friendly_name
        if name not in self._connections:
            self.add_server(config)
        return self._connections[name]

    def load_saved_servers(self) -> List[str]:
        """Load all saved server configs and return their names."""
        configs = load_servers()
        for config in configs:
            if config.friendly_name and config.enabled:
                if config.friendly_name not in self._connections:
                    self.add_server(config, persist=False)
        return list(self._connections.keys())

    def get_all_configs(self) -> List[ServerConfig]:
        """Return configs for all registered servers."""
        return [
            conn.config for conn in self._connections.values()
        ]

    def connect(self, friendly_name: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.connect()

    def disconnect(self, friendly_name: str) -> None:
        conn = self._connections.get(friendly_name)
        if conn:
            conn.disconnect()

    def disconnect_all(self) -> None:
        for conn in self._connections.values():
            conn.disconnect()

    def get_status(self, friendly_name: str) -> Optional[ServerStatus]:
        conn = self._connections.get(friendly_name)
        if not conn:
            return None
        return conn.get_status()

    def get_connection_state(self, friendly_name: str) -> ConnectionState:
        conn = self._connections.get(friendly_name)
        if not conn:
            return ConnectionState.DISCONNECTED
        return conn.connection_state

    def start_server(self, friendly_name: str, command: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.start_server(command)

    def stop_server(self, friendly_name: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.stop_server()

    def restart_server(self, friendly_name: str, command: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.restart_server(command)

    def execute(self, friendly_name: str, command: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.execute(command)

    def execute_and_capture(self, friendly_name: str, command: str) -> str:
        conn = self._connections.get(friendly_name)
        if not conn:
            raise ConnectionError(f"No connection: {friendly_name}")
        return conn.execute_and_capture(command)

    def upload_file(self, friendly_name: str, local_path: str, remote_path: str,
                    progress_cb=None) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.upload_file(local_path, remote_path, progress_cb=progress_cb)

    def download_file(self, friendly_name: str, remote_path: str, local_path: str,
                      progress_cb=None) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.download_file(remote_path, local_path, progress_cb=progress_cb)

    def list_directory(self, friendly_name: str, remote_path: str) -> List[FileEntry]:
        conn = self._connections.get(friendly_name)
        if not conn:
            return []
        return conn.list_directory(remote_path)

    def create_directory(self, friendly_name: str, remote_path: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.create_directory(remote_path)

    def delete_file(self, friendly_name: str, remote_path: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.delete_file(remote_path)

    def rename_file(self, friendly_name: str, old_path: str, new_path: str) -> bool:
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.rename_file(old_path, new_path)

    def tail_log(self, friendly_name: str, lines: int = 100) -> str:
        conn = self._connections.get(friendly_name)
        if not conn:
            return ""
        return conn.tail_log(lines)

    def get_server_pid(self, friendly_name: str) -> Optional[int]:
        conn = self._connections.get(friendly_name)
        if not conn:
            return None
        return conn.get_server_pid()

    def get_all_names(self) -> List[str]:
        return list(self._connections.keys())

    def scan_server_models(self, friendly_name: str, directory: str) -> List[ModelInfo]:
        conn = self._connections.get(friendly_name)
        if not conn:
            return []
        return conn.scan_model_directory(directory)

    def get_server_model_directories(self, friendly_name: str) -> List[str]:
        conn = self._connections.get(friendly_name)
        if not conn:
            return []
        cfg = conn.config
        dirs = cfg.model_directories or []
        if cfg.model_directory and cfg.model_directory not in dirs:
            dirs = [cfg.model_directory] + dirs
        return [d for d in dirs if d]

    @staticmethod
    def _create_connection(config: ServerConfig) -> IServerConnection:
        if config.connection_type == ConnectionType.LOCAL:
            return LocalServerConnection(config)
        return SSHServerConnection(config)
