"""Abstract Server Connection Interface."""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from llama_launcher.gpu.types import GPUInfo
from llama_launcher.models.types import ModelInfo
from .procinfo import ResourceSample
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
    def test_connection(self) -> Tuple[bool, str]:
        """Probe the host and return ``(ok, message)``.

        Unlike :meth:`connect`, the message explains *why* a probe failed
        (authentication, DNS, no route) so the UI can show something actionable
        instead of a bare failure.
        """

    @abstractmethod
    def home_directory(self) -> str:
        """Return the login user's home directory on the target host.

        Used as the fallback starting point when browsing the filesystem.
        """

    @abstractmethod
    def path_kind(self, path: str) -> str:
        """Classify ``path`` on the target host as ``dir``, ``file``, or ``missing``."""

    @abstractmethod
    def resolve_executable(self, path_or_name: str) -> str:
        """Return the runnable absolute path for a command, or "" if it is not.

        A bare name is looked up on the target's PATH; anything containing a
        slash is tested directly. Note the launcher's commands run
        non-interactively, so a name only present via a shell rc file will
        correctly resolve to "".
        """

    @abstractmethod
    def find_server_binary(self) -> str:
        """Search the usual build locations for llama-server on this host."""

    @abstractmethod
    def detect_gpu(self) -> "GPUInfo":
        """Describe the GPU of the machine this connection targets.

        The launcher runs on one machine and the model runs on another, so the
        GPU that matters for sizing a model is this one, not the local card.
        """

    @abstractmethod
    def total_ram_mb(self) -> float:
        """Total system RAM of the target machine, or -1 if unknown."""

    @abstractmethod
    def sample_usage(self) -> ResourceSample:
        """Take a live CPU/RAM/GPU utilization reading from the target machine."""

    @abstractmethod
    def port_listener_pid(self, port: int) -> Optional[int]:
        """PID listening on ``port``, or None if the port is free.

        The port is what tells co-hosted llama-server instances apart, so this
        both identifies a server and detects a conflict before starting one.
        Returns None for a listener owned by another user, which is
        indistinguishable from free at this level.
        """

    @abstractmethod
    def log_follow_argv(self, lines: int = 0) -> List[str]:
        """Return an argv that streams this server's log until killed.

        The caller owns the process. Returning an argv rather than a file handle
        is what lets one worker follow both a local file and a remote one.
        """

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
