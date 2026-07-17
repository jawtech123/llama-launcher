"""Live Console for streaming server logs."""
import threading
import time
from typing import Callable, Optional
from PySide6.QtCore import QObject, Signal, QThread


class LogStreamWorker(QThread):
    """Background thread that reads from a log source and emits lines."""

    line_received = Signal(str)
    error_occurred = Signal(str)
    disconnected = Signal()

    def __init__(self, connection, friendly_name: str):
        super().__init__()
        self._connection = connection
        self._friendly_name = friendly_name
        self._running = False

    def run(self):
        self._running = True
        log_path = f"{self._connection.config.working_directory}/{self._connection.config.log_file}"
        tail_cmd = f"tail -f {log_path}"

        try:
            if hasattr(self._connection, 'execute_and_capture'):
                import paramiko
                config = self._connection.config
                transport = paramiko.Transport(
                    (config.hostname, config.port)
                )
                connect_kwargs = {"username": config.username}
                if config.private_key_path:
                    key = paramiko.RSAKey.from_private_key_file(
                        str(__import__("pathlib").Path(config.private_key_path).expanduser())
                    )
                    connect_kwargs["pkey"] = key
                if config.password:
                    connect_kwargs["password"] = config.password
                transport.connect(**connect_kwargs)
                channel = transport.open_session()
                channel.get_pty()
                channel.exec_command(tail_cmd)

                while self._running and channel.recv_exit_status() == -1:
                    if channel.recv_ready():
                        data = channel.recv(8192).decode("utf-8", errors="ignore")
                        for line in data.split("\n"):
                            if line:
                                self.line_received.emit(line)
                    time.sleep(0.05)

                channel.close()
                transport.close()

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.disconnected.emit()

    def stop(self):
        self._running = False


class LocalLogWorker(QThread):
    """Background thread that tails a local log file."""

    line_received = Signal(str)
    error_occurred = Signal(str)
    disconnected = Signal()

    def __init__(self, log_path: str):
        super().__init__()
        self._log_path = str(__import__("pathlib").Path(log_path).expanduser())
        self._running = False

    def run(self):
        self._running = True
        try:
            import subprocess
            proc = subprocess.Popen(
                f"tail -f {self._log_path}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while self._running:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                stripped = line.rstrip("\n")
                if stripped:
                    self.line_received.emit(stripped)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.disconnected.emit()

    def stop(self):
        self._running = False


class LiveConsole(QObject):
    """Manages live log streaming for a server connection."""

    line_received = Signal(str)
    error_occurred = Signal(str)
    disconnected = Signal()
    connected = Signal()

    def __init__(self, manager, friendly_name: str):
        super().__init__()
        self._manager = manager
        self._friendly_name = friendly_name
        self._worker: Optional[QThread] = None

    def start(self) -> bool:
        conn = self._manager.get_connection(self._friendly_name)
        if not conn:
            return False

        if self._worker and self._worker.isRunning():
            self.stop()

        from llama_launcher.remote.local import LocalServerConnection
        if isinstance(conn, LocalServerConnection):
            import pathlib
            log_path = str(
                pathlib.Path(conn.config.working_directory).expanduser()
                / conn.config.log_file
            )
            self._worker = LocalLogWorker(log_path)
        else:
            self._worker = LogStreamWorker(conn, self._friendly_name)

        self._worker.line_received.connect(self.line_received.emit)
        self._worker.error_occurred.connect(self.error_occurred.emit)
        self._worker.disconnected.connect(self.disconnected.emit)
        self._worker.start()
        self.connected.emit()
        return True

    def stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
