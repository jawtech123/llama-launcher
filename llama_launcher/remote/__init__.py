"""Remote Server Management Module."""
from .types import (
    LOCAL_SERVER_NAME,
    ServerConfig,
    ServerStatus,
    FileEntry,
    ConnectionState,
    ServerProcessState,
)
from .connection import IServerConnection
from .local import LocalServerConnection
from .manager import ServerManager, create_connection
from .storage import load_servers, save_servers, save_server, remove_server

__all__ = [
    "LOCAL_SERVER_NAME",
    "create_connection",
    "ServerConfig",
    "ServerStatus",
    "FileEntry",
    "ConnectionState",
    "ServerProcessState",
    "IServerConnection",
    "LocalServerConnection",
    "ServerManager",
    "load_servers",
    "save_servers",
    "save_server",
    "remove_server",
]
