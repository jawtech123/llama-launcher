"""Remote Server Management Types."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AuthMethod(Enum):
    SSH_KEY = "ssh_key"
    PASSWORD = "password"


class ConnectionType(Enum):
    LOCAL = "local"
    SSH = "ssh"


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
    connection_type: ConnectionType = ConnectionType.LOCAL
    hostname: str = "localhost"
    port: int = 22
    username: str = ""
    auth_method: AuthMethod = AuthMethod.SSH_KEY
    password: str = ""
    private_key_path: str = ""
    working_directory: str = "~"
    llama_server_path: str = "llama-server"
    model_directory: str = ""
    model_directories: Optional[List[str]] = None
    enabled: bool = True
    log_file: str = "llama-server.log"
    server_port: int = 8080
    poll_interval_seconds: int = 5


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
    """Represents a file or directory entry for remote browsing."""
    name: str = ""
    path: str = ""
    is_directory: bool = False
    size_bytes: int = 0
    modified_time: float = 0.0
