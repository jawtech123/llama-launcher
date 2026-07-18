"""Abstract Server Connection Interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from llama_launcher.models.types import ModelInfo
from .types import (
    ConnectionState,
    ServerConfig,
    ServerStatus,
    FileEntry,
)


class IServerConnection(ABC):
    """Transport-agnostic interface for managing a llama-server instance.

    UI code depends only on this interface, never on a concrete implementation.
    """

    @property
    @abstractmethod
    def config(self) -> ServerConfig:
        """Return the server configuration."""

    @property
    @abstractmethod
    def connection_state(self) -> ConnectionState:
        """Return the current connection state."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the target host."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    def execute(self, command: str) -> bool:
        """Execute a command on the target host.

        Returns True if the command completed successfully (exit code 0).
        """

    @abstractmethod
    def execute_and_capture(self, command: str) -> str:
        """Execute a command and return its stdout.

        Raises on non-zero exit.
        """

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str, progress_cb=None) -> bool:
        """Upload a file to the target host.

        Args:
            progress_cb: Optional callable(bytes_done, total_bytes) invoked
                during the transfer. Raising inside it aborts the transfer.
        """

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str, progress_cb=None) -> bool:
        """Download a file from the target host.

        Args:
            progress_cb: Optional callable(bytes_done, total_bytes) invoked
                during the transfer. Raising inside it aborts the transfer.
        """

    @abstractmethod
    def list_directory(self, remote_path: str) -> List[FileEntry]:
        """List contents of a remote directory."""

    @abstractmethod
    def create_directory(self, remote_path: str) -> bool:
        """Create a directory on the target host."""

    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """Delete a file on the target host."""

    @abstractmethod
    def rename_file(self, old_path: str, new_path: str) -> bool:
        """Rename or move a file on the target host."""

    @abstractmethod
    def get_status(self) -> ServerStatus:
        """Collect and return the current server status."""

    @abstractmethod
    def start_server(self, command: str) -> bool:
        """Start llama-server as a detached background process.

        Args:
            command: Full command string to launch the server.
        """

    @abstractmethod
    def stop_server(self) -> bool:
        """Stop a running llama-server instance."""

    @abstractmethod
    def restart_server(self, command: str) -> bool:
        """Restart the llama-server instance."""

    @abstractmethod
    def tail_log(self, lines: int = 100) -> str:
        """Return the last N lines of the server log file."""

    @abstractmethod
    def get_server_pid(self) -> Optional[int]:
        """Return the PID of the running llama-server process, or None."""

    @abstractmethod
    def tracked_pid_alive(self) -> bool:
        """True if a launcher-started process (stored/PID-file) is still alive.

        Unlike get_server_pid, this never falls back to system-wide pgrep,
        so it can't be fooled by unrelated llama-server instances.
        """

    @abstractmethod
    def scan_model_directory(self, directory: str) -> List[ModelInfo]:
        """Scan a directory on the target host for model files.

        Args:
            directory: Path on the target host to scan.
        """
