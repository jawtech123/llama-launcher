"""Local Server Connection Implementation."""
import logging
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import List, Optional
from llama_launcher.models.types import ModelInfo
from llama_launcher.models.scanner import scan_models as _scan_filesystem
from .connection import IServerConnection
from .types import (
    ConnectionState,
    ServerConfig,
    ServerStatus,
    ServerProcessState,
    FileEntry,
)

logger = logging.getLogger(__name__)


class LocalServerConnection(IServerConnection):
    """Manages a local llama-server instance via subprocess."""

    def __init__(self, config: ServerConfig):
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._stored_pid: Optional[int] = None
        self._log_process: Optional[subprocess.Popen] = None

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def connection_state(self) -> ConnectionState:
        if self._state == ConnectionState.DISCONNECTED:
            try:
                subprocess.run(
                    ["which", self._config.llama_server_path.split()[0]],
                    capture_output=True,
                    check=True,
                    timeout=1,
                )
                self._state = ConnectionState.CONNECTED
            except Exception:
                self._state = ConnectionState.ERROR
        return self._state

    def connect(self) -> bool:
        try:
            subprocess.run(
                ["which", self._config.llama_server_path.split()[0]],
                capture_output=True,
                check=True,
                timeout=1,
            )
            self._state = ConnectionState.CONNECTED
            return True
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    def disconnect(self) -> None:
        # Don't reset to DISCONNECTED — keep cached CONNECTED/ERROR state
        # so subsequent connection_state checks don't spawn subprocesses.
        if self._log_process:
            self._log_process.terminate()
            self._log_process = None

    def _run_command(self, command: str, capture: bool = False) -> tuple:
        expanded = command
        try:
            result = subprocess.run(
                expanded,
                shell=True,
                capture_output=capture,
                text=True,
                timeout=30,
                cwd=str(Path(self._config.working_directory).expanduser()),
            )
            return (result.returncode == 0, result.stdout)
        except subprocess.TimeoutExpired:
            return (False, "")
        except Exception:
            return (False, "")

    def execute(self, command: str) -> bool:
        success, _ = self._run_command(command)
        return success

    def execute_and_capture(self, command: str) -> str:
        success, output = self._run_command(command, capture=True)
        if not success:
            raise RuntimeError(f"Command failed: {command}")
        return output

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            target = str(Path(remote_path).expanduser())
            parent = Path(target).parent
            parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(local_path, target)
            return True
        except Exception:
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            source = str(Path(remote_path).expanduser())
            parent = Path(local_path).parent
            parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source, local_path)
            return True
        except Exception:
            return False

    def list_directory(self, remote_path: str) -> List[FileEntry]:
        target = Path(remote_path).expanduser()
        entries = []
        if not target.exists():
            return entries
        for item in target.iterdir():
            stat = item.stat()
            entries.append(FileEntry(
                name=item.name,
                path=str(item),
                is_directory=item.is_dir(),
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
            ))
        return entries

    def create_directory(self, remote_path: str) -> bool:
        try:
            Path(remote_path).expanduser().mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def delete_file(self, remote_path: str) -> bool:
        try:
            p = Path(remote_path).expanduser()
            if p.is_file():
                p.unlink()
                return True
            return False
        except Exception:
            return False

    def rename_file(self, old_path: str, new_path: str) -> bool:
        try:
            old = Path(old_path).expanduser()
            new = Path(new_path).expanduser()
            old.rename(new)
            return True
        except Exception:
            return False

    def get_server_pid(self) -> Optional[int]:
        pid = self._stored_pid
        if pid:
            try:
                os.kill(pid, 0)
                return pid
            except OSError:
                self._stored_pid = None
        pids = self._find_llama_pids()
        if pids:
            return pids[0]
        return None

    def _find_llama_pids(self) -> List[int]:
        try:
            result = subprocess.run(
                "pgrep -f 'llama-server'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return [int(p) for p in result.stdout.strip().split("\n")]
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)
        return []

    def get_status(self) -> ServerStatus:
        pid = self.get_server_pid()
        if not pid:
            return ServerStatus(state=ServerProcessState.STOPPED)

        status = ServerStatus(
            state=ServerProcessState.RUNNING,
            pid=pid,
        )

        try:
            ps_out = subprocess.run(
                f"ps -p {pid} -o %cpu,rss,etimes,comm --no-headers",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ps_out.returncode == 0 and ps_out.stdout.strip():
                parts = ps_out.stdout.strip().split()
                if len(parts) >= 3:
                    status.cpu_percent = float(parts[0])
                    status.ram_mb = float(parts[1]) / 1024.0
                    status.uptime_seconds = float(parts[2])
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)

        port = self._config.server_port or 8080
        try:
            ss_out = subprocess.run(
                f"ss -tlnp | grep -E ':{int(port)} '",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ss_out.returncode == 0:
                status.listening_port = port
        except Exception:
            logger.debug("ss port check failed", exc_info=True)

        try:
            nvidia_out = subprocess.run(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if nvidia_out.returncode == 0 and nvidia_out.stdout.strip():
                lines = nvidia_out.stdout.strip().split("\n")
                for line in lines:
                    vals = [v.strip() for v in line.split(",")]
                    if len(vals) >= 2:
                        status.gpu_utilization = float(vals[0])
                        status.gpu_memory_mb = float(vals[1])
                        break
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)

        return status

    def start_server(self, command: str) -> bool:
        log_path = str(Path(self._config.working_directory).expanduser() / self._config.log_file)
        start_cmd = f"nohup {command} > {shlex.quote(log_path)} 2>&1 & echo $!"

        try:
            result = subprocess.run(
                start_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(Path(self._config.working_directory).expanduser()),
            )
            if result.returncode == 0 and result.stdout.strip():
                self._stored_pid = int(result.stdout.strip())
                return True
            logger.warning("start_server failed (rc=%s): %s", result.returncode, result.stderr.strip())
            return False
        except Exception:
            logger.exception("start_server raised")
            return False

    def stop_server(self) -> bool:
        pid = self.get_server_pid()
        if not pid:
            return True
        try:
            os.kill(pid, signal.SIGTERM)
            self._stored_pid = None
            return True
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
                self._stored_pid = None
                return True
            except Exception:
                return False

    def restart_server(self, command: str) -> bool:
        self.stop_server()
        import time
        time.sleep(1)
        return self.start_server(command)

    def tail_log(self, lines: int = 100) -> str:
        log_path = str(Path(self._config.working_directory).expanduser() / self._config.log_file)
        try:
            result = subprocess.run(
                f"tail -n {int(lines)} {shlex.quote(log_path)}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout
        except Exception:
            logger.debug("tail_log failed", exc_info=True)
            return ""

    def scan_model_directory(self, directory: str) -> List[ModelInfo]:
        target = str(Path(directory).expanduser())
        return _scan_filesystem(target)
