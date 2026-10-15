"""Local Server Connection Implementation."""
import logging
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple
from llama_launcher.models.types import ModelInfo
from llama_launcher.models.scanner import scan_models as _scan_filesystem
from llama_launcher.gpu.detector import detect_gpu
from llama_launcher.gpu.types import GPUInfo

from .connection import IServerConnection
from .procinfo import (
    ResourceSample, build_status_probe, build_usage_probe, parse_gpu_stats,
    parse_listener_pid, parse_meminfo, parse_ps_stats, parse_usage_probe,
    split_status_probe,
)
from .types import (
    LEGACY_PID_FILE,
    LLAMA_SERVER_CANDIDATES,
    environment_prefix,
    pid_file_name,
    ConnectionState,
    ServerConfig,
    ServerStatus,
    ServerProcessState,
    FileEntry,
)

logger = logging.getLogger(__name__)

#: How long to wait for llama-server to exit after SIGTERM before SIGKILL.
_STOP_GRACE_SECONDS = 10


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
        """Last known state; connecting is an explicit user action."""
        return self._state

    def connect(self) -> bool:
        ok, _ = self.test_connection()
        return ok

    def test_connection(self) -> Tuple[bool, str]:
        """Check the configured binary is actually runnable on this machine."""
        binary = (self._config.llama_server_path or "llama-server").split()[0]
        resolved = shutil.which(binary)
        if resolved is None and Path(binary).expanduser().is_file():
            resolved = str(Path(binary).expanduser())
        if resolved is None:
            self._state = ConnectionState.ERROR
            return (False, f"llama-server binary not found: {binary}")
        self._state = ConnectionState.CONNECTED
        return (True, f"Local machine — using {resolved}")

    def home_directory(self) -> str:
        return str(Path.home())

    def log_follow_argv(self, lines: int = 0) -> List[str]:
        # -F (not -f) keeps following across the truncation/rotation that a
        # server restart causes, and waits for a log that does not exist yet.
        return ["tail", "-n", str(int(lines)), "-F", self._log_path()]

    def path_kind(self, path: str) -> str:
        target = Path(path).expanduser()
        if target.is_dir():
            return "dir"
        if target.exists():
            return "file"
        return "missing"

    def disconnect(self) -> None:
        """Mark this server disconnected and release anything it was holding.

        The state really is cleared: connection_state no longer re-probes, so
        leaving it CONNECTED would mean Disconnect appeared to do nothing.
        """
        self._state = ConnectionState.DISCONNECTED
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

    def upload_file(self, local_path: str, remote_path: str, progress_cb=None) -> bool:
        try:
            target = str(Path(remote_path).expanduser())
            parent = Path(target).parent
            parent.mkdir(parents=True, exist_ok=True)
            self._copy_with_progress(local_path, target, progress_cb)
            return True
        except Exception:
            logger.debug("upload_file failed", exc_info=True)
            return False

    def download_file(self, remote_path: str, local_path: str, progress_cb=None) -> bool:
        try:
            source = str(Path(remote_path).expanduser())
            parent = Path(local_path).parent
            parent.mkdir(parents=True, exist_ok=True)
            self._copy_with_progress(source, local_path, progress_cb)
            return True
        except Exception:
            logger.debug("download_file failed", exc_info=True)
            return False

    @staticmethod
    def _copy_with_progress(source: str, target: str, progress_cb=None) -> None:
        total = os.path.getsize(source)
        copied = 0
        with open(source, "rb") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                if progress_cb:
                    progress_cb(copied, total)

    def list_directory(self, remote_path: str) -> List[FileEntry]:
        target = Path(remote_path).expanduser()
        entries = []
        if not target.exists():
            return entries
        for item in target.iterdir():
            # A broken symlink or an unreadable entry must not abort the whole
            # listing -- report what we can and leave its metadata at zero.
            try:
                stat = item.stat()
                size_bytes, modified_time = stat.st_size, stat.st_mtime
                is_directory = item.is_dir()
            except OSError:
                size_bytes, modified_time, is_directory = 0, 0.0, False
            entries.append(FileEntry(
                name=item.name,
                path=str(item),
                is_directory=is_directory,
                size_bytes=size_bytes,
                modified_time=modified_time,
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
        """PID of the llama-server process for this connection.

        Resolution order: in-memory PID -> PID file (written at start) -> the
        process listening on this server's configured port.

        Deliberately no process-name search: any llama-server on the machine
        would match, so a stopped entry would report an unrelated instance's PID,
        CPU and RAM as its own. The port is what identifies *this* server, and it
        still finds one started outside the launcher.
        """
        pid = self._stored_pid
        if pid and self._pid_alive(pid):
            return pid
        if pid:
            self._stored_pid = None

        pid = self._read_pid_file()
        if pid and self._pid_alive(pid):
            self._stored_pid = pid
            return pid

        pid = self._pid_on_configured_port()
        if pid and self._pid_alive(pid):
            return pid
        return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _working_dir(self) -> Path:
        return Path(self._config.working_directory).expanduser()

    def _log_path(self) -> str:
        return str(self._working_dir() / self._config.log_file)

    def _pid_file_path(self) -> Path:
        return self._working_dir() / pid_file_name(self._config.friendly_name)

    def _read_pid_file(self) -> Optional[int]:
        candidates = [self._pid_file_path(),
                      self._working_dir() / LEGACY_PID_FILE]
        for path in candidates:
            try:
                text = path.read_text().strip()
                if text:
                    return int(text)
            except (OSError, ValueError):
                continue
        return None

    def _write_pid_file(self, pid: int) -> None:
        try:
            self._pid_file_path().write_text(f"{pid}\n")
        except OSError:
            logger.debug("Could not write PID file", exc_info=True)

    def _remove_pid_file(self) -> None:
        try:
            self._pid_file_path().unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove PID file", exc_info=True)

    def resolve_executable(self, path_or_name: str) -> str:
        name = (path_or_name or "").strip()
        if not name:
            return ""
        if "/" in name:
            candidate = Path(name).expanduser()
            return str(candidate) if os.access(candidate, os.X_OK) else ""
        return shutil.which(name) or ""

    def find_server_binary(self) -> str:
        for candidate in LLAMA_SERVER_CANDIDATES:
            resolved = self.resolve_executable(candidate)
            if resolved:
                return resolved
        return self.resolve_executable("llama-server")

    def detect_gpu(self) -> GPUInfo:
        return detect_gpu()

    def total_ram_mb(self) -> float:
        try:
            with open("/proc/meminfo", "r") as f:
                _, total_mb = parse_meminfo(f.read())
            return total_mb
        except OSError:
            logger.debug("Could not read /proc/meminfo", exc_info=True)
            return -1.0

    def sample_usage(self) -> ResourceSample:
        """Sample this machine through the same probe used for remote hosts.

        Sharing the probe keeps local and remote readings directly comparable
        rather than subtly different measurements of the same quantities.
        """
        try:
            own = self._own_pids()
            result = subprocess.run(
                ["sh", "-c", build_usage_probe(own)],
                capture_output=True, text=True, timeout=20,
            )
            return parse_usage_probe(result.stdout, own)
        except Exception:
            logger.debug("usage sample failed", exc_info=True)
            return ResourceSample()

    def _own_pids(self) -> List[int]:
        pid = self._tracked_pid()
        return [pid] if pid else []

    def port_listener_pid(self, port: int) -> Optional[int]:
        try:
            result = subprocess.run(
                ["ss", "-tlnpH", f"sport = :{int(port)}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return parse_listener_pid(result.stdout)
        except Exception:
            logger.debug("port -> pid lookup failed", exc_info=True)
        return None

    def _pid_on_configured_port(self) -> Optional[int]:
        """PID listening on this server's own port."""
        return self.port_listener_pid(self._config.server_port or 8080)

    def get_status(self) -> ServerStatus:
        pid = self.get_server_pid()
        if not pid:
            return ServerStatus(state=ServerProcessState.STOPPED)

        status = ServerStatus(state=ServerProcessState.RUNNING, pid=pid)
        port = self._config.server_port or 8080

        # One probe for every field instead of three separate spawns.
        try:
            result = subprocess.run(
                ["sh", "-c", build_status_probe(pid, port)],
                capture_output=True, text=True, timeout=15,
            )
            ps_out, ss_out, gpu_out = split_status_probe(result.stdout)
        except Exception:
            logger.debug("status probe failed", exc_info=True)
            return status

        status.cpu_percent, status.ram_mb, status.uptime_seconds = parse_ps_stats(ps_out)
        # Only claim the port when *this* process owns it.
        listener_pid = parse_listener_pid(ss_out)
        if ss_out and (listener_pid is None or listener_pid == pid):
            status.listening_port = port
        status.gpu_utilization, status.gpu_memory_mb = parse_gpu_stats(gpu_out)
        return status

    def start_server(self, command: str) -> bool:
        log_path = self._log_path()
        # "(cd ... && exec ...)" makes $! the PID of llama-server itself
        # (otherwise $! tracks a wrapper subshell). < /dev/null detaches it
        # so subprocess.run sees pipe EOF immediately.
        start_cmd = (
            f"(cd {shlex.quote(str(self._working_dir()))} && "
            f"exec nohup {environment_prefix(self._config)}{command} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null) & echo $!"
        )

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
                pid = int(result.stdout.strip())
                # Verify this PID is actually alive and matches what we expect
                # (in case the shell reused a PID or something went wrong)
                try:
                    os.kill(pid, 0)
                    self._stored_pid = pid
                    self._write_pid_file(pid)
                    return True
                except OSError:
                    logger.warning("start_server: reported PID %s exited immediately", pid)
            logger.warning("start_server failed (rc=%s): %s", result.returncode, result.stderr.strip())
            return False
        except Exception:
            logger.exception("start_server raised")
            return False

    def _tracked_pid(self) -> Optional[int]:
        """PID of a process started by the launcher (stored or PID file)."""
        if self._stored_pid and self._pid_alive(self._stored_pid):
            return self._stored_pid
        pid = self._read_pid_file()
        if pid and self._pid_alive(pid):
            return pid
        return None

    def tracked_pid_alive(self) -> bool:
        return self._tracked_pid() is not None

    def stop_server(self) -> bool:
        # Only ever signal processes we started (stored PID / PID file) --
        # never a random llama-server found via machine-wide pgrep.
        pid = self._tracked_pid()
        if not pid:
            logger.info("stop_server: no tracked llama-server process to stop")
            return True
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            logger.exception("stop_server: SIGTERM failed for pid %s", pid)
            return False
        # llama-server needs a moment to release the GPU and its port; a
        # Restart issued straight after an unfinished TERM used to fail to bind.
        deadline = time.monotonic() + _STOP_GRACE_SECONDS
        while self._pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if self._pid_alive(pid):
            logger.warning("llama-server %s ignored SIGTERM; sending SIGKILL", pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                logger.exception("stop_server: SIGKILL failed for pid %s", pid)
                return False
        self._stored_pid = None
        self._remove_pid_file()
        return True

    def restart_server(self, command: str) -> bool:
        self.stop_server()
        return self.start_server(command)

    def tail_log(self, lines: int = 100) -> str:
        log_path = self._log_path()
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
