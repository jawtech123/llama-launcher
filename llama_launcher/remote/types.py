"""Remote Server Management Types."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


#: Where a llama-server build usually lands, searched in order when the
#: configured path turns out not to be runnable on a host.
LLAMA_SERVER_CANDIDATES = (
    "~/llama.cpp/build/bin/llama-server",
    "~/llama.cpp/llama-server",
    "/workspace/llama.cpp/build/bin/llama-server",
    "/opt/llama.cpp/build/bin/llama-server",
    "/usr/local/bin/llama-server",
    "/usr/bin/llama-server",
)


def pid_file_name(friendly_name: str) -> str:
    """PID file name for a server entry.

    Per-entry rather than per-directory: several servers can share a working
    directory (they usually all default to ``~``), and a single shared PID file
    would make starting one server clobber another's tracked process.
    """
    slug = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in (friendly_name or "server")).strip("-")
    return f".llama_launcher_{slug or 'server'}.pid"


#: PID file used before per-server names; still read so a server started by an
#: older build can still be tracked and stopped.
LEGACY_PID_FILE = ".llama_launcher_server.pid"


#: Name of the always-present entry representing the machine the launcher runs on.
#: It is created on first start, cannot be renamed, removed, or turned into a
#: remote server, and is the storage key for its own persisted config and model
#: cache -- hence a constant rather than a repeated literal.
LOCAL_SERVER_NAME = "Local Machine"


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ServerProcessState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass
class ServerConfig:
    """Configuration for a managed server (local or remote)."""
    friendly_name: str = ""
    working_directory: str = "~"
    llama_server_path: str = "llama-server"
    model_directory: str = ""
    model_directories: Optional[List[str]] = None
    enabled: bool = True
    log_file: str = "llama-server.log"
    server_port: int = 8080
    poll_interval_seconds: int = 5
    # Remote / SSH fields
    hostname: str = ""
    ssh_port: int = 22
    username: str = ""
    auth_method: str = "key"  # "key" or "password"
    key_path: str = ""
    #: Held in memory for the session. Only written to disk when
    #: ``remember_password`` is set, since servers.json is plaintext.
    password: str = ""
    remember_password: bool = False


@dataclass
class ServerStatus:
    """Real-time status of a managed llama-server instance."""
    state: ServerProcessState = ServerProcessState.STOPPED
    pid: Optional[int] = None
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0
    listening_port: Optional[int] = None
    loaded_model: str = ""
    uptime_seconds: float = 0.0
    error_message: str = ""


@dataclass
class FileEntry:
    """Represents a file or directory entry."""
    name: str = ""
    path: str = ""
    is_directory: bool = False
    size_bytes: int = 0
    modified_time: float = 0.0
