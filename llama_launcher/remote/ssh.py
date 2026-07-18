"""SSH Server Connection Implementation."""
import logging
import os
import time
from pathlib import Path
from typing import List, Optional
from llama_launcher.models.types import ModelInfo
from .connection import IServerConnection
from .types import (
    ConnectionState,
    ServerConfig,
    ServerStatus,
    ServerProcessState,
    FileEntry,
    AuthMethod,
)

try:
    import paramiko
except ImportError:
    paramiko = None

logger = logging.getLogger(__name__)


class SSHServerConnection(IServerConnection):
    """Manages a remote llama-server instance via SSH/SFTP."""

    SSH_TIMEOUT = 15
    COMMAND_TIMEOUT = 30

    def __init__(self, config: ServerConfig):
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp = None
        self._stored_pid: Optional[int] = None

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def connection_state(self) -> ConnectionState:
        return self._state

    def _get_client(self) -> paramiko.SSHClient:
        if self._client is None or not self._client.get_transport().is_active():
            self.connect()
        if self._client is None:
            raise ConnectionError("Not connected")
        return self._client

    def _get_sftp(self):
        """Return a live SFTP client, recreating it if the channel died."""
        if self._sftp is not None:
            try:
                chan = self._sftp.get_channel()
                if chan is None or chan.closed:
                    self._sftp = None
            except Exception:
                self._sftp = None
        if self._sftp is None:
            self._sftp = self._get_client().open_sftp()
        return self._sftp

    def connect(self) -> bool:
        if paramiko is None:
            self._state = ConnectionState.ERROR
            return False

        try:
            self._state = ConnectionState.CONNECTING
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            kwargs = {
                "hostname": self._config.hostname,
                "port": self._config.port,
                "username": self._config.username,
                "timeout": self.SSH_TIMEOUT,
                "allow_agent": False,
                "look_for_keys": False,
            }

            if self._config.auth_method == AuthMethod.SSH_KEY:
                if self._config.private_key_path:
                    kwargs["key_filename"] = str(
                        Path(self._config.private_key_path).expanduser()
                    )

            if self._config.password:
                kwargs["password"] = self._config.password

            self._client.connect(**kwargs)
            self._state = ConnectionState.CONNECTED
            return True

        except Exception:
            self._state = ConnectionState.ERROR
            if self._client:
                self._client.close()
                self._client = None
            return False

    def disconnect(self) -> None:
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)
            self._client = None
        self._state = ConnectionState.DISCONNECTED

    def execute(self, command: str) -> bool:
        try:
            client = self._get_client()
            _, stdout, stderr = client.exec_command(
                command, timeout=self.COMMAND_TIMEOUT
            )
            exit_status = stdout.channel.recv_exit_status()
            return exit_status == 0
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    def execute_and_capture(self, command: str) -> str:
        try:
            client = self._get_client()
            _, stdout, stderr = client.exec_command(
                command, timeout=self.COMMAND_TIMEOUT
            )
            output = stdout.read().decode("utf-8", errors="ignore")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                err = stderr.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Command failed: {err or command}")
            return output
        except Exception:
            self._state = ConnectionState.ERROR
            raise

    def upload_file(self, local_path: str, remote_path: str, progress_cb=None) -> bool:
        try:
            sftp = self._get_sftp()
            remote_dir = str(Path(remote_path).parent)
            try:
                sftp.stat(remote_dir)
            except FileNotFoundError:
                self._mkdirs(sftp, remote_dir)
            sftp.put(str(Path(local_path).expanduser()), remote_path,
                     callback=progress_cb)
            return True
        except Exception:
            logger.debug("upload_file failed", exc_info=True)
            return False

    def download_file(self, remote_path: str, local_path: str, progress_cb=None) -> bool:
        try:
            sftp = self._get_sftp()
            local_dir = Path(local_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_dir / Path(local_path).name),
                     callback=progress_cb)
            return True
        except Exception:
            logger.debug("download_file failed", exc_info=True)
            return False

    def _mkdirs(self, sftp, path: str) -> None:
        parts = Path(path).parts
        current = ""
        for part in parts:
            if not part:
                current = "/"
                continue
            current = current.rstrip("/") + "/" + part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current, 0o755)

    def list_directory(self, remote_path: str) -> List[FileEntry]:
        entries = []
        try:
            sftp = self._get_sftp()
            for item in sftp.listdir_attr(remote_path):
                entries.append(FileEntry(
                    name=item.filename,
                    path=f"{remote_path}/{item.filename}",
                    is_directory=item.st_mode & 0o40000 != 0,
                    size_bytes=item.st_size,
                    modified_time=float(item.st_mtime),
                ))
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)
        return entries

    def create_directory(self, remote_path: str) -> bool:
        try:
            sftp = self._get_sftp()
            try:
                # Idempotent: already-existing directory is success.
                if sftp.stat(remote_path).st_mode & 0o40000:
                    return True
            except FileNotFoundError:
                pass
            parent = str(Path(remote_path).parent)
            try:
                sftp.stat(parent)
            except FileNotFoundError:
                self._mkdirs(sftp, parent)
            sftp.mkdir(remote_path, 0o755)
            return True
        except Exception:
            logger.debug("create_directory failed", exc_info=True)
            return False

    def delete_file(self, remote_path: str) -> bool:
        try:
            sftp = self._get_sftp()
            sftp.remove(remote_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def rename_file(self, old_path: str, new_path: str) -> bool:
        try:
            sftp = self._get_sftp()
            sftp.rename(old_path, new_path)
            return True
        except Exception:
            return False

    def get_server_pid(self) -> Optional[int]:
        """PID of the remote llama-server (stored -> PID file -> pgrep)."""
        pid = self._stored_pid
        if pid:
            try:
                result = self.execute_and_capture(
                    f"kill -0 {pid} 2>/dev/null && echo alive"
                )
                if "alive" in result:
                    return pid
            except Exception:
                self._stored_pid = None

        pid = self._read_pid_file()
        if pid and self._remote_pid_alive(pid):
            self._stored_pid = pid
            return pid

        try:
            # -x: exact process name only; -f would match any cmdline
            # containing "llama-server" (shells, editors, pgrep itself).
            output = self.execute_and_capture(
                "pgrep -x llama-server"
            )
            if output.strip():
                pids = [int(p.strip()) for p in output.strip().split("\n")]
                if pids:
                    return pids[0]
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)
        return None

    def _remote_pid_alive(self, pid: int) -> bool:
        try:
            result = self.execute_and_capture(
                f"kill -0 {pid} 2>/dev/null && echo alive"
            )
            return "alive" in result
        except Exception:
            return False

    def _pid_file_path(self) -> str:
        return f"{self._config.working_directory}/.llama_launcher_server.pid"

    def _read_pid_file(self) -> Optional[int]:
        try:
            out = self.execute_and_capture(f"cat {self._pid_file_path()} 2>/dev/null")
            text = out.strip()
            return int(text) if text else None
        except Exception:
            return None

    def _write_pid_file(self, pid: int) -> None:
        try:
            self.execute(f"echo {pid} > {self._pid_file_path()}")
        except Exception:
            logger.debug("Could not write remote PID file", exc_info=True)

    def _remove_pid_file(self) -> None:
        try:
            self.execute(f"rm -f {self._pid_file_path()}")
        except Exception:
            logger.debug("Could not remove remote PID file", exc_info=True)

    def get_status(self) -> ServerStatus:
        pid = self.get_server_pid()
        if not pid:
            return ServerStatus(state=ServerProcessState.STOPPED)

        status = ServerStatus(
            state=ServerProcessState.RUNNING,
            pid=pid,
        )

        try:
            ps_out = self.execute_and_capture(
                f"ps -p {pid} -o %cpu,rss,etimes --no-headers"
            )
            if ps_out.strip():
                parts = ps_out.strip().split()
                if len(parts) >= 3:
                    status.cpu_percent = float(parts[0])
                    status.ram_mb = float(parts[1]) / 1024.0
                    status.uptime_seconds = float(parts[2])
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)

        port = self._config.server_port or 8080
        try:
            ss_out = self.execute_and_capture(
                f"ss -tlnp | grep :{port}"
            )
            if ss_out.strip():
                status.listening_port = port
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)

        try:
            nvidia_out = self.execute_and_capture(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                "--format=csv,noheader,nounits 2>/dev/null"
            )
            if nvidia_out.strip():
                lines = nvidia_out.strip().split("\n")
                for line in lines:
                    vals = [v.strip() for v in line.split(",")]
                    if len(vals) >= 2:
                        status.gpu_utilization = float(vals[0])
                        status.gpu_memory_mb = float(vals[1])
                        break
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)

        return status

    def _exec_quick(self, command: str, timeout: float = 8.0) -> str:
        """Execute a command and return stdout without waiting for channel EOF.

        ``execute_and_capture`` waits for EOF, which hangs when a backgrounded
        child (e.g. nohup'd llama-server) inherits channel descriptors. This
        waits only for the remote shell to exit, then drains and closes.
        """
        client = self._get_client()
        chan = client.get_transport().open_session()
        try:
            chan.exec_command(command)
            end = time.time() + timeout
            data = b""
            while time.time() < end:
                if chan.recv_ready():
                    data += chan.recv(4096)
                    continue
                if chan.exit_status_ready():
                    break
                time.sleep(0.05)
            while chan.recv_ready():
                data += chan.recv(4096)
            return data.decode("utf-8", errors="ignore")
        finally:
            chan.close()

    def start_server(self, command: str) -> bool:
        log_path = f"{self._config.working_directory}/{self._config.log_file}"
        # < /dev/null detaches the child from the SSH channel's stdin so the
        # channel can close immediately (otherwise stdout.read() hangs until
        # the timeout waiting for channel EOF).
        start_cmd = (
            f"cd {self._config.working_directory} && "
            f"nohup {command} > {log_path} 2>&1 < /dev/null & echo $!"
        )

        try:
            output = self._exec_quick(start_cmd)
            if output.strip():
                self._stored_pid = int(output.strip())
                self._write_pid_file(self._stored_pid)
                return True
            logger.warning("start_server: no PID returned for: %s", command[:80])
            return False
        except Exception:
            logger.exception("start_server failed")
            return False

    def stop_server(self) -> bool:
        # Only signal processes started by the launcher (tracked PID) --
        # never a random llama-server found via machine-wide pgrep.
        pid = self._stored_pid if self._stored_pid and self._remote_pid_alive(self._stored_pid) else None
        if pid is None:
            pid = self._read_pid_file()
            if pid and not self._remote_pid_alive(pid):
                pid = None
        if not pid:
            logger.info("stop_server: no tracked llama-server process to stop")
            return True
        try:
            self.execute(f"kill {pid}")
            time.sleep(0.5)
            if self._remote_pid_alive(pid):
                self.execute(f"kill -9 {pid}")
            self._stored_pid = None
            self._remove_pid_file()
            return True
        except Exception:
            logger.exception("stop_server failed for remote pid %s", pid)
            return False

    def restart_server(self, command: str) -> bool:
        self.stop_server()
        time.sleep(1)
        return self.start_server(command)

    def tail_log(self, lines: int = 100) -> str:
        log_path = f"{self._config.working_directory}/{self._config.log_file}"
        try:
            return self.execute_and_capture(f"tail -n {lines} {log_path}")
        except Exception:
            return ""

    def scan_model_directory(self, directory: str) -> List[ModelInfo]:
        exts = ["*.gguf", "*.bin", "*.safetensors"]
        models: List[ModelInfo] = []
        try:
            for ext in exts:
                find_cmd = f"find '{directory}' -type f -name '{ext}' 2>/dev/null"
                output = self.execute_and_capture(find_cmd)
                for line in output.strip().split("\n"):
                    fpath = line.strip()
                    if not fpath:
                        continue
                    info = self._parse_remote_model(fpath)
                    if info:
                        models.append(info)
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)
        return models

    def _parse_remote_model(self, file_path: str) -> Optional[ModelInfo]:
        try:
            stat_out = self.execute_and_capture(f"stat --format='%s' '{file_path}' 2>/dev/null")
            size_bytes = int(stat_out.strip())
            size_mb = round(size_bytes / (1024 * 1024), 2)
        except Exception:
            size_mb = 0.0

        file_name = Path(file_path).name
        suffix = Path(file_path).suffix.lower()
        if suffix == ".gguf":
            file_format = "gguf"
        elif suffix == ".bin":
            file_format = "bin"
        elif suffix == ".safetensors":
            file_format = "safetensors"
        else:
            file_format = "unknown"

        name_lower = file_name.lower()
        family = "unknown"
        if "llama" in name_lower or "llm" in name_lower:
            family = "llama"
        elif "mistral" in name_lower:
            family = "mistral"
        elif "qwen" in name_lower or "qwen2" in name_lower or "qwq" in name_lower:
            family = "qwen"
        elif "codellama" in name_lower:
            family = "codellama"
        elif "phi" in name_lower:
            family = "phi"
        elif "gemma" in name_lower:
            family = "gemma"

        quant_type = None
        quant_patterns = [
            ("q8_0", "Q8_0"), ("q6_k", "Q6_K"), ("q5_k_m", "Q5_K_M"),
            ("q5_0", "Q5_0"), ("q5_1", "Q5_1"), ("q4_k_m", "Q4_K_M"),
            ("q4_0", "Q4_0"), ("q4_1", "Q4_1"), ("q3_k_m", "Q3_K_M"),
            ("q3_k_s", "Q3_K_S"), ("q2_k", "Q2_K"),
        ]
        for pattern, quant_name in quant_patterns:
            if pattern in name_lower:
                quant_type = quant_name
                break

        return ModelInfo(
            file_path=file_path,
            file_name=file_name,
            file_size_mb=size_mb,
            format=file_format,
            family=family,
            quant_type=quant_type,
        )
