"""Multi-Server Manager."""
from typing import Dict, List, Optional
from llama_launcher.models.types import ModelInfo
from .types import (
    ServerConfig,
    ServerStatus,
    ConnectionState,
    ServerProcessState,
    FileEntry,
)
from .connection import IServerConnection
from .local import LocalServerConnection
from .storage import load_servers, save_server, remove_server as storage_remove


def is_remote_config(config: ServerConfig) -> bool:
    """True if the config describes a remote (SSH) server."""
    return bool(config.hostname and config.hostname.strip())


def create_connection(config: ServerConfig) -> IServerConnection:
    """Build a connection for a config without registering or persisting it.

    The single place transport selection happens. Callers that need to talk to a
    server the user is still editing (the Add Server dialog, connection tests)
    use this directly so nothing is written to disk until Save.
    """
    if is_remote_config(config):
        from .ssh import SSHServerConnection
        return SSHServerConnection(config)
    return LocalServerConnection(config)


class ServerManager:
    """Manages multiple server connections.

    The UI interacts only through this manager, never directly with
    individual connection implementations.
    """

    def __init__(self):
        self._connections: Dict[str, IServerConnection] = {}

    _is_remote = staticmethod(is_remote_config)

    def is_remote(self, friendly_name: str) -> bool:
        """True if the named server uses SSH/remote connection."""
        conn = self._connections.get(friendly_name)
        if conn:
            return self._is_remote(conn.config)
        return False

    def add_server(self, config: ServerConfig, persist: bool = True) -> bool:
        """Register a server configuration and create its connection."""
        if persist:
            save_server(config)
        self._connections[config.friendly_name] = create_connection(config)
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
        """Load all saved server configs and return their names.

        Disabled servers are included (greyed in the UI) so they can be
        edited/re-enabled; they simply show as disconnected.
        """
        configs = load_servers()
        for config in configs:
            if config.friendly_name:
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
            shutdown = getattr(conn, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
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

    def test_connection(self, friendly_name: str) -> tuple:
        """Return ``(ok, message)`` for a connectivity check."""
        conn = self._connections.get(friendly_name)
        if not conn:
            return (False, f"No such server: {friendly_name}")
        return conn.test_connection()

    def resolve_executable(self, friendly_name: str, path_or_name: str) -> str:
        conn = self._connections.get(friendly_name)
        return conn.resolve_executable(path_or_name) if conn else ""

    def find_server_binary(self, friendly_name: str) -> str:
        conn = self._connections.get(friendly_name)
        return conn.find_server_binary() if conn else ""

    def detect_gpu(self, friendly_name: str):
        conn = self._connections.get(friendly_name)
        return conn.detect_gpu() if conn else None

    def total_ram_mb(self, friendly_name: str) -> float:
        conn = self._connections.get(friendly_name)
        return conn.total_ram_mb() if conn else -1.0

    def sample_usage(self, friendly_name: str):
        conn = self._connections.get(friendly_name)
        return conn.sample_usage() if conn else None

    def port_listener_pid(self, friendly_name: str, port: int) -> Optional[int]:
        conn = self._connections.get(friendly_name)
        if not conn:
            return None
        return conn.port_listener_pid(port)

    def check_port_conflict(self, friendly_name: str) -> Optional[str]:
        """Error message if starting this server would collide with a live
        listener already on its configured port, else None.

        A listener that *is* this server's own tracked process is not a
        conflict (e.g. Restart) -- only a foreign one blocks the start.
        """
        conn = self._connections.get(friendly_name)
        port = conn.config.server_port if conn else 0
        if not port:
            return None
        listener = self.port_listener_pid(friendly_name, port)
        if listener is not None and not self.tracked_pid_alive(friendly_name):
            return (f"Port {port} is already in use on this host by PID {listener}.\n\n"
                    "Give this server its own port (Edit Server > Server Port) to run "
                    "it alongside the existing one.")
        return None

    def find_port_conflicts(self, config: ServerConfig) -> List[str]:
        """Names of other configured servers that would fight over the same port.

        Running several llama-servers on one host is supported, but only if each
        listens on its own port -- the second one to start otherwise dies at bind
        time. Compares configuration only, so it costs nothing and works offline.
        """
        host_key = (config.hostname or "").strip().lower()
        conflicts = []
        for other in self.get_all_configs():
            if other.friendly_name == config.friendly_name:
                continue
            if (other.hostname or "").strip().lower() != host_key:
                continue
            if other.server_port == config.server_port:
                conflicts.append(other.friendly_name)
        return conflicts

    def home_directory(self, friendly_name: str) -> str:
        conn = self._connections.get(friendly_name)
        if not conn:
            return ""
        return conn.home_directory()

    def path_kind(self, friendly_name: str, path: str) -> str:
        conn = self._connections.get(friendly_name)
        if not conn:
            return "missing"
        return conn.path_kind(path)

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

    def tracked_pid_alive(self, friendly_name: str) -> bool:
        """True if the launcher-started process for this server is alive."""
        conn = self._connections.get(friendly_name)
        if not conn:
            return False
        return conn.tracked_pid_alive()

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
