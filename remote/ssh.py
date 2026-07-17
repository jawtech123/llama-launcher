"""SSH Server Connection Implementation."""
import io
import os
import re
import time
from pathlib import Path
from typing import List, Optional
from models.types import ModelInfo
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
        self._log_channel: Optional[paramiko.Channel] = None
        self._log_transport: Optional[paramiko.Transport] = None

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
        if self._sftp is None or self._sftp.closed:
            client = self._get_client()
            self._sftp = client.open_sftp()
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
        self._stop_live_log()
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
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

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            sftp = self._get_sftp()
            remote_dir = str(Path(remote_path).parent)
            try:
                sftp.stat(remote_dir)
            except FileNotFoundError:
                self._mkdirs(sftp, remote_dir)
            sftp.put(str(Path(local_path).expanduser()), remote_path)
            return True
        except Exception:
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            sftp = self._get_sftp()
            local_dir = Path(local_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_dir / Path(local_path).name))
            return True
        except Exception:
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
            pass
        return entries

    def create_directory(self, remote_path: str) -> bool:
        try:
            sftp = self._get_sftp()
            parent = str(Path(remote_path).parent)
            try:
                sftp.stat(parent)
            except FileNotFoundError:
                self._mkdirs(sftp, parent)
            sftp.mkdir(remote_path, 0o755)
            return True
        except Exception:
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
        try:
            output = self.execute_and_capture(
                "pgrep -f 'llama-server'"
            )
            if output.strip():
                pids = [int(p.strip()) for p in output.strip().split("\n")]
                if pids:
                    return pids[0]
        except Exception:
            pass
        return None

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
            pass

        port = self._config.server_port or 8080
        try:
            ss_out = self.execute_and_capture(
                f"ss -tlnp | grep :{port}"
            )
            if ss_out.strip():
                status.listening_port = port
        except Exception:
            pass

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
            pass

        return status

    def start_server(self, command: str) -> bool:
        log_path = f"{self._config.working_directory}/{self._config.log_file}"
        start_cmd = (
            f"cd {self._config.working_directory} && "
            f"nohup {command} > {log_path} 2>&1 & echo $!"
        )

        try:
            output = self.execute_and_capture(start_cmd)
            if output.strip():
                self._stored_pid = int(output.strip())
                return True
            return False
        except Exception:
            return False

    def stop_server(self) -> bool:
        pid = self.get_server_pid()
        if not pid:
            return True
        try:
            self.execute(f"kill {pid}")
            time.sleep(0.5)
            if self.get_server_pid() == pid:
                self.execute(f"kill -9 {pid}")
            self._stored_pid = None
            return True
        except Exception:
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

    def _stop_live_log(self) -> None:
        if self._log_channel:
            try:
                self._log_channel.close()
            except Exception:
                pass
            self._log_channel = None
        if self._log_transport:
            try:
                self._log_transport.close()
            except Exception:
                pass
            self._log_transport = None

    def start_live_log(self, callback) -> bool:
        """Start streaming the log file to a callback.

        Args:
            callback: Function called with each new line of output.
        """
        self._stop_live_log()

        if paramiko is None:
            return False

        log_path = f"{self._config.working_directory}/{self._config.log_file}"
        tail_cmd = f"tail -f {log_path}"

        try:
            self._log_transport = paramiko.Transport(
                (self._config.hostname, self._config.port)
            )

            connect_kwargs = {
                "username": self._config.username,
            }
            if self._config.auth_method == AuthMethod.SSH_KEY:
                if self._config.private_key_path:
                    key = paramiko.RSAKey.from_private_key_file(
                        str(Path(self._config.private_key_path).expanduser())
                    )
                    connect_kwargs["pkey"] = key
            if self._config.password:
                connect_kwargs["password"] = self._config.password

            self._log_transport.connect(**connect_kwargs)
            self._log_channel = self._log_transport.open_session()
            self._log_channel.get_pty()
            self._log_channel.exec_command(tail_cmd)

            def _reader():
                try:
                    while self._log_channel and self._log_channel.recv_exit_status() == -1:
                        if self._log_channel.recv_ready():
                            line = self._log_channel.recv(4096).decode(
                                "utf-8", errors="ignore"
                            )
                            if line:
                                callback(line)
                        time.sleep(0.1)
                except Exception:
                    pass

            import threading
            self._log_thread = threading.Thread(target=_reader, daemon=True)
            self._log_thread.start()
            return True

        except Exception:
            self._stop_live_log()
            return False

    def stop_live_log(self) -> None:
        self._stop_live_log()

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
            pass
        return models

    def _parse_remote_model(self, file_path: str) -> Optional[ModelInfo]:
        try:
            stat_out = self.execute_and_capture(f"stat --format='%s' '{file_path}' 2>/dev/null")
            size_bytes = int(stat_out.strip())
            size_mb = round(size_bytes / (1024 * 1024), 2)
        except Exception:
            size_mb = 0.0

        file_name = Path(file_path).name
        suffix = file_path[file_path.rfind(":") + 1:]
        if suffix.lower() == ".gguf":
            file_format = "gguf"
        elif suffix.lower() == ".bin":
            file_format = "bin"
        elif suffix.lower() == ".safetensors":
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
