"""SSH Server Connection Implementation.

Commands run through the ``ssh`` client rather than an in-process SSH library.
Two consequences worth knowing:

* A ``ControlMaster`` socket is shared by every command to this host, so only the
  first call pays for the TCP handshake and authentication. That is what makes
  interactive directory browsing usable, and it means a password is supplied once
  per ``ControlPersist`` window instead of once per command.
* Each command is an independent process, so several GUI worker threads can talk
  to the same server concurrently without any locking on our side.
"""
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from llama_launcher.gpu.probe import build_full_probe, parse_gpu_probe
from llama_launcher.gpu.types import GPUInfo
from llama_launcher.models.scanner import build_model_info
from llama_launcher.models.types import ModelInfo
from llama_launcher.utils.paths import ssh_control_dir

#: Glob patterns for the remote ``find``; mirrors scanner.MODEL_EXTENSIONS.
MODEL_EXTENSIONS = ("*.gguf", "*.bin", "*.safetensors")

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

#: How long the shared master connection lingers after the last command.
_CONTROL_PERSIST_SECONDS = 120

#: Model files are huge; a transfer gets hours, not the usual command timeout.
_SCP_TIMEOUT = 6 * 60 * 60

#: How often to re-measure the destination while a transfer runs.
_PROGRESS_POLL_SECONDS = 0.5

#: How long to wait for llama-server to exit after SIGTERM before SIGKILL.
_STOP_GRACE_SECONDS = 10

#: A ControlPath is a unix socket. ssh refuses one whose expanded length reaches
#: ~108 bytes, and expands %C to 64 hex characters.
_CONTROL_PATH_TOKEN_LEN = 64
_CONTROL_PATH_MAX = 100


def _control_path() -> Optional[str]:
    """ControlPath template for connection reuse, or None if it would be too long.

    Losing connection reuse only costs speed; an over-long path would make every
    ssh command fail outright, so we drop the option instead.
    """
    template = f"{ssh_control_dir()}/%C"
    expanded_len = len(template) - len("%C") + _CONTROL_PATH_TOKEN_LEN
    if expanded_len >= _CONTROL_PATH_MAX:
        logger.warning(
            "SSH connection reuse disabled: control path would be %d bytes (%s)",
            expanded_len, template)
        return None
    return template

#: Helper that hands ssh a password from the environment. ``SSH_ASKPASS_REQUIRE``
#: (OpenSSH 8.4+) makes ssh use it even with a terminal attached.
_ASKPASS_SCRIPT = '#!/bin/sh\nprintf \'%s\\n\' "$LL_SSH_PASSWORD"\n'


class SSHServerConnection(IServerConnection):
    """Manages a remote llama-server instance via SSH."""

    def __init__(self, config: ServerConfig, allow_multiplexing: bool = True):
        """
        Args:
            allow_multiplexing: Share one authenticated connection across
                commands. Turn it off to force a real authentication -- the
                control socket is keyed on host/port/user but *not* on the
                credentials, so an existing master would otherwise let a probe
                succeed without the credentials being exercised at all.
        """
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._stored_pid: Optional[int] = None
        self._last_stderr = ""
        self._remote_home: Optional[str] = None
        self._allow_multiplexing = allow_multiplexing
        self._ssh_base_cmd = self._build_ssh_base()

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _uses_password(self) -> bool:
        return bool(self._config.auth_method == "password" and self._config.password)

    def _key_file(self) -> Optional[str]:
        key = (self._config.key_path or "").strip()
        if not key:
            return None
        key = os.path.expanduser(key)
        return key if os.path.isfile(key) else None

    def _shared_options(self) -> List[str]:
        """Options common to ssh and scp: host checking, timeout, auth, reuse."""
        opts = [
            # Trust a host the first time, but refuse one whose key changed.
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
        ]
        control_path = _control_path() if self._allow_multiplexing else None
        if not self._allow_multiplexing:
            # Explicitly refuse an existing master, not merely decline to open one.
            opts += ["-o", "ControlPath=none"]
        if control_path:
            # Reuse one authenticated connection for every subsequent command.
            opts += [
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={control_path}",
                "-o", f"ControlPersist={_CONTROL_PERSIST_SECONDS}",
            ]
        if self._uses_password():
            # BatchMode would make ssh fail rather than ask; the askpass helper
            # supplies the password non-interactively instead.
            opts += [
                "-o", "BatchMode=no",
                "-o", "NumberOfPasswordPrompts=1",
                "-o", "PubkeyAuthentication=no",
                "-o", "PreferredAuthentications=password,keyboard-interactive",
            ]
        else:
            opts += ["-o", "BatchMode=yes"]
        key = self._key_file()
        if key:
            # Must precede the destination: ssh stops parsing options at user@host.
            opts += ["-i", key]
        return opts

    def _build_ssh_base(self) -> List[str]:
        """Build the ssh argv prefix (everything up to the remote command)."""
        return [
            "ssh",
            *self._shared_options(),
            "-p", str(self._config.ssh_port),
            f"{self._config.username}@{self._config.hostname}",
        ]

    def _askpass_env(self) -> Optional[dict]:
        """Environment that lets ssh read the password from our helper script.

        The password travels in the child's environment rather than argv, so it
        is not visible in the process list.
        """
        if not self._uses_password():
            return None
        helper = ssh_control_dir() / "askpass.sh"
        try:
            if not helper.exists() or helper.read_text() != _ASKPASS_SCRIPT:
                helper.write_text(_ASKPASS_SCRIPT)
            os.chmod(helper, 0o700)
        except OSError:
            logger.warning("Could not create SSH askpass helper at %s", helper)
            return None
        env = dict(os.environ)
        env.update({
            "SSH_ASKPASS": str(helper),
            "SSH_ASKPASS_REQUIRE": "force",
            "LL_SSH_PASSWORD": self._config.password,
            # Older ssh only consults SSH_ASKPASS when DISPLAY is set.
            "DISPLAY": env.get("DISPLAY", ":0"),
        })
        return env

    def _run_remote(self, cmd: str, capture: bool = False, timeout: int = 30) -> tuple:
        """Run a command on the remote host. Returns ``(ok, stdout)``.

        Stderr is always captured and kept in ``self._last_stderr`` so callers
        can report why something failed.
        """
        argv = [*self._ssh_base_cmd, cmd]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._askpass_env(),
            )
            self._last_stderr = (result.stderr or "").strip()
            return (result.returncode == 0, result.stdout if capture else "")
        except subprocess.TimeoutExpired:
            self._last_stderr = f"Timed out after {timeout}s"
            return (False, "")
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug("SSH command failed: %s", e)
            self._last_stderr = str(e)
            return (False, "")

    def _scp_remote(self, local_path: str, remote_path: str,
                    direction: str = "upload", progress_cb=None) -> bool:
        """Copy a file to (``upload``) or from (``download``) the remote host."""
        remote_spec = f"{self._config.username}@{self._config.hostname}:{remote_path}"
        argv = ["scp", *self._shared_options(), "-P", str(self._config.ssh_port)]
        local_is_dir = os.path.isdir(local_path)
        if direction == "upload":
            if local_is_dir:
                argv.append("-r")
            argv += [local_path, remote_spec]
        else:
            # Recursion is decided by the remote side, which we cannot cheaply
            # stat here; -r is harmless for a plain file.
            argv += ["-r", remote_spec, local_path]

        if progress_cb is None:
            try:
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=_SCP_TIMEOUT,
                    env=self._askpass_env(),
                )
                self._last_stderr = (result.stderr or "").strip()
                return result.returncode == 0
            except Exception as e:
                logger.debug("SCP failed", exc_info=True)
                self._last_stderr = str(e)
                return False
        return self._scp_with_progress(argv, local_path, remote_path,
                                       direction, progress_cb)

    def _scp_with_progress(self, argv, local_path, remote_path,
                           direction, progress_cb) -> bool:
        """Run scp, reporting progress by watching the destination grow.

        scp only draws a progress bar to a terminal, so rather than allocate a
        pty we poll the size of the file being written. Model files are large
        enough that a silent transfer is not acceptable.
        """
        total = self._transfer_total_bytes(local_path, remote_path, direction)
        try:
            process = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=self._askpass_env(),
            )
        except Exception as e:
            logger.debug("SCP failed to start", exc_info=True)
            self._last_stderr = str(e)
            return False

        try:
            while process.poll() is None:
                time.sleep(_PROGRESS_POLL_SECONDS)
                done = self._destination_size(remote_path, local_path, direction)
                # A raising callback is how the UI cancels.
                progress_cb(done, total)
            _, stderr = process.communicate(timeout=30)
            self._last_stderr = (stderr or "").strip()
            if process.returncode == 0:
                progress_cb(total, total)
                return True
            return False
        except Exception as e:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            self._last_stderr = str(e)
            # A half-copied model is indistinguishable from a real one by name
            # and size alone, so do not leave it lying around.
            self._remove_partial(remote_path, local_path, direction)
            raise

    def _remove_partial(self, remote_path, local_path, direction) -> None:
        """Delete the incomplete destination file after an aborted transfer."""
        try:
            if direction == "download":
                Path(local_path).unlink(missing_ok=True)
            else:
                self._run_remote(f"rm -f {shlex.quote(remote_path)}", timeout=15)
        except Exception:
            logger.debug("Could not remove partial transfer", exc_info=True)

    def _transfer_total_bytes(self, local_path, remote_path, direction) -> int:
        """Size of the source file, for percentage reporting."""
        try:
            if direction == "upload":
                return os.path.getsize(local_path)
            success, output = self._run_remote(
                f"stat -c %s {shlex.quote(remote_path)}", capture=True, timeout=15)
            return int(output.strip()) if success and output.strip() else 0
        except (OSError, ValueError):
            return 0

    def _destination_size(self, remote_path, local_path, direction) -> int:
        """Bytes written so far at the destination."""
        try:
            if direction == "download":
                return os.path.getsize(local_path)
            success, output = self._run_remote(
                f"stat -c %s {shlex.quote(remote_path)} 2>/dev/null || echo 0",
                capture=True, timeout=15)
            return int(output.strip()) if success and output.strip() else 0
        except (OSError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def connection_state(self) -> ConnectionState:
        """Last known state. Deliberately does not probe.

        Reading a server's state used to open an SSH connection, so merely
        listing servers reached out over the network and an unreachable host
        stalled the UI for a full connect timeout. Connecting is an explicit
        action now.
        """
        return self._state

    def connect(self) -> bool:
        """Test SSH connectivity to the remote host."""
        ok, _ = self.test_connection()
        return ok

    def test_connection(self) -> Tuple[bool, str]:
        """Probe the host, returning ``(ok, message)`` with ssh's own diagnosis."""
        if not self._config.hostname.strip():
            return (False, "No hostname configured")
        if not self._config.username.strip():
            return (False, "No username configured")
        success, _ = self._run_remote("exit 0", timeout=20)
        if success:
            self._state = ConnectionState.CONNECTED
            return (True, f"Connected to {self._config.username}@{self._config.hostname}")
        self._state = ConnectionState.ERROR
        return (False, self._error_message())

    def _error_message(self) -> str:
        """Turn ssh's stderr into a single actionable line."""
        noise = ("warning: permanently added", "pseudo-terminal")
        lines = [
            line.strip() for line in self._last_stderr.splitlines()
            if line.strip() and not any(n in line.lower() for n in noise)
        ]
        return lines[-1] if lines else "Connection failed (no details from ssh)"

    def disconnect(self) -> None:
        """Mark disconnected and drop the shared SSH master connection.

        Closing the master matters: leaving it open would let later commands
        keep working against a server the user believes is disconnected.
        """
        self._state = ConnectionState.DISCONNECTED
        self.shutdown()

    def shutdown(self) -> None:
        """Tear down the shared master connection (called on app close)."""
        try:
            subprocess.run(
                [*self._ssh_base_cmd[:-1], "-O", "exit", self._ssh_base_cmd[-1]],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            logger.debug("Could not close SSH master connection", exc_info=True)

    def execute(self, command: str) -> bool:
        success, _ = self._run_remote(command)
        return success

    def execute_and_capture(self, command: str) -> str:
        success, output = self._run_remote(command, capture=True)
        if not success:
            raise RuntimeError(f"Command failed: {command}")
        return output

    def upload_file(self, local_path: str, remote_path: str, progress_cb=None) -> bool:
        parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
        self._run_remote(f"mkdir -p {shlex.quote(parent)}")
        return self._scp_remote(local_path, remote_path, direction="upload",
                                progress_cb=progress_cb)

    def download_file(self, remote_path: str, local_path: str, progress_cb=None) -> bool:
        os.makedirs(str(Path(local_path).parent), exist_ok=True)
        return self._scp_remote(local_path, remote_path, direction="download",
                                progress_cb=progress_cb)

    def list_directory(self, remote_path: str) -> List[FileEntry]:
        """List a remote directory.

        Uses ``find -printf`` rather than parsing ``ls -l``: the fields are
        unambiguous, ``%T@`` is a real epoch timestamp, ``%Y`` follows symlinks so
        a link to a directory is reported as one, and there is no ``total`` header
        line to mistake for an entry. Falls back to a name-only ``ls`` for hosts
        without GNU findutils.
        """
        quoted = shlex.quote(remote_path)
        success, output = self._run_remote(
            f"find {quoted} -mindepth 1 -maxdepth 1 -printf '%Y\\t%s\\t%T@\\t%f\\n'",
            capture=True,
        )
        if not success:
            return self._list_directory_fallback(remote_path)

        entries = []
        for line in output.splitlines():
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            kind, size_str, mtime_str, name = parts
            if not name:
                continue
            try:
                size_bytes = int(size_str)
            except ValueError:
                size_bytes = 0
            try:
                modified_time = float(mtime_str)
            except ValueError:
                modified_time = 0.0
            entries.append(FileEntry(
                name=name,
                path=self._join(remote_path, name),
                is_directory=(kind == "d"),
                size_bytes=size_bytes,
                modified_time=modified_time,
            ))
        return entries

    def _list_directory_fallback(self, remote_path: str) -> List[FileEntry]:
        """Name-only listing for remotes without ``find -printf`` (BSD, macOS)."""
        success, output = self._run_remote(
            f"ls -A -p {shlex.quote(remote_path)}", capture=True,
        )
        if not success:
            return []
        entries = []
        for name in output.splitlines():
            name = name.strip()
            if not name:
                continue
            is_dir = name.endswith("/")
            name = name.rstrip("/")
            entries.append(FileEntry(
                name=name,
                path=self._join(remote_path, name),
                is_directory=is_dir,
            ))
        return entries

    @staticmethod
    def _join(directory: str, name: str) -> str:
        """Join remote POSIX path components (never the launcher's local rules)."""
        return f"{directory.rstrip('/')}/{name}"

    def home_directory(self) -> str:
        if self._remote_home is None:
            success, output = self._run_remote(
                'printf %s "$HOME"', capture=True, timeout=15)
            home = output.strip() if success else ""
            if not home:
                # Do not cache a guess; a later call may reach the host.
                return f"/home/{self._config.username}"
            self._remote_home = home
        return self._remote_home

    # ------------------------------------------------------------------
    # Remote paths
    # ------------------------------------------------------------------

    def _remote_working_dir(self) -> str:
        """Working directory as it exists *on the remote host*.

        ``Path.expanduser()`` would resolve ``~`` against the launcher's own home,
        which is the wrong user (and possibly the wrong layout) for the remote
        machine, so a leading ``~`` is resolved from the remote ``$HOME``.
        """
        configured = (self._config.working_directory or "~").strip()
        if configured == "~":
            return self.home_directory()
        if configured.startswith("~/"):
            return f"{self.home_directory().rstrip('/')}/{configured[2:]}"
        return configured

    def _remote_log_path(self) -> str:
        return f"{self._remote_working_dir().rstrip('/')}/{self._config.log_file}"

    def _remote_pid_file(self) -> str:
        return (f"{self._remote_working_dir().rstrip('/')}/"
                f"{pid_file_name(self._config.friendly_name)}")

    def log_follow_argv(self, lines: int = 0) -> List[str]:
        return [*self._ssh_base_cmd,
                f"tail -n {int(lines)} -F {shlex.quote(self._remote_log_path())}"]

    def path_kind(self, path: str) -> str:
        quoted = shlex.quote(path)
        success, output = self._run_remote(
            f'if [ -d {quoted} ]; then echo dir; elif [ -e {quoted} ]; then echo file; '
            f'else echo missing; fi',
            capture=True, timeout=15,
        )
        kind = output.strip() if success else ""
        return kind if kind in ("dir", "file", "missing") else "missing"

    def create_directory(self, remote_path: str) -> bool:
        success, _ = self._run_remote(f"mkdir -p {shlex.quote(remote_path)}")
        return success

    def delete_file(self, remote_path: str) -> bool:
        success, _ = self._run_remote(f"rm -f {shlex.quote(remote_path)}")
        return success

    def rename_file(self, old_path: str, new_path: str) -> bool:
        success, _ = self._run_remote(
            f"mv {shlex.quote(old_path)} {shlex.quote(new_path)}"
        )
        return success

    def get_server_pid(self) -> Optional[int]:
        """PID of this server's llama-server process on the remote host.

        Same resolution order as the local transport: tracked PID -> PID file ->
        the process listening on this server's configured port. A PID found via
        the port is reported but never stored as *tracked*, so stop_server can
        only ever signal a process the launcher itself started.

        No process-name search: besides matching any llama-server on the host,
        the old `pgrep -f llama-server` also matched the very shell running it
        ("sh -c pgrep -f llama-server"), reporting a server that was not there.
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

        return self._pid_on_configured_port()

    def resolve_executable(self, path_or_name: str) -> str:
        name = (path_or_name or "").strip()
        if not name:
            return ""
        quoted = shlex.quote(name)
        if "/" in name:
            # Expand a leading ~ on the remote side, then test executability.
            probe = f'p=$(eval echo {quoted}); [ -x "$p" ] && printf %s "$p"'
        else:
            probe = f"command -v {quoted} 2>/dev/null"
        _, output = self._run_remote(probe, capture=True, timeout=20)
        return (output or "").strip()

    def find_server_binary(self) -> str:
        """Locate llama-server on the remote host in one round trip."""
        checks = " ".join(
            f'p=$(eval echo {shlex.quote(c)}); [ -x "$p" ] && {{ printf %s "$p"; exit 0; }};'
            for c in LLAMA_SERVER_CANDIDATES
        )
        _, output = self._run_remote(
            f"{checks} command -v llama-server 2>/dev/null", capture=True, timeout=30)
        return (output or "").strip()

    def detect_gpu(self) -> GPUInfo:
        """Describe the *remote* GPU(s) with the same probe used locally.

        The probe's exit code is ignored in favour of what it printed: a
        driver/library mismatch makes nvidia-smi exit non-zero, and that text
        is the diagnosis the user needs to see.
        """
        _, output = self._run_remote(build_full_probe(), capture=True, timeout=40)
        parsed = parse_gpu_probe(output, self._last_stderr)
        if parsed is not None:
            return parsed
        return GPUInfo(backend="cpu", vram_mb=0, available_vram_mb=0,
                       device_name="No NVIDIA GPU or AMD GPU found on this host", supported=True)

    def total_ram_mb(self) -> float:
        _, output = self._run_remote("cat /proc/meminfo", capture=True, timeout=20)
        _, total_mb = parse_meminfo(output)
        return total_mb

    def sample_usage(self) -> ResourceSample:
        # The probe's exit status is whatever its last command returned, so a
        # host without a working nvidia-smi would otherwise throw away its
        # perfectly good CPU and RAM figures.
        own = self._own_pids()
        _, output = self._run_remote(build_usage_probe(own), capture=True, timeout=30)
        return parse_usage_probe(output, own)

    def _own_pids(self) -> List[int]:
        pid = self._stored_pid
        return [pid] if pid else []

    def port_listener_pid(self, port: int) -> Optional[int]:
        success, output = self._run_remote(
            f"ss -tlnpH 'sport = :{int(port)}'", capture=True)
        return parse_listener_pid(output) if success else None

    def _pid_on_configured_port(self) -> Optional[int]:
        """PID listening on this server's own port -- the identity that
        separates several llama-servers sharing one host."""
        return self.port_listener_pid(self._config.server_port or 8080)

    def _pid_alive(self, pid: int) -> bool:
        try:
            success, _ = self._run_remote(f"kill -0 {int(pid)}", capture=True, timeout=10)
            return success
        except Exception:
            return False

    def _read_pid_file(self) -> Optional[int]:
        working = self._remote_working_dir().rstrip("/")
        current = shlex.quote(self._remote_pid_file())
        legacy = shlex.quote(f"{working}/{LEGACY_PID_FILE}")
        success, output = self._run_remote(
            f"cat {current} 2>/dev/null || cat {legacy} 2>/dev/null", capture=True)
        if success and output.strip():
            try:
                return int(output.strip().splitlines()[0])
            except (ValueError, IndexError):
                pass
        return None

    def get_status(self) -> ServerStatus:
        pid = self.get_server_pid()
        if not pid:
            return ServerStatus(state=ServerProcessState.STOPPED)

        status = ServerStatus(state=ServerProcessState.RUNNING, pid=pid)
        port = self._config.server_port or 8080

        # One SSH round trip for every field; this runs on a 15s timer and used
        # to cost three separate connections.
        success, output = self._run_remote(
            build_status_probe(pid, port), capture=True)
        if not success:
            return status

        ps_out, ss_out, gpu_out = split_status_probe(output)
        status.cpu_percent, status.ram_mb, status.uptime_seconds = parse_ps_stats(ps_out)
        listener_pid = parse_listener_pid(ss_out)
        if ss_out and (listener_pid is None or listener_pid == pid):
            status.listening_port = port
        status.gpu_utilization, status.gpu_memory_mb = parse_gpu_stats(gpu_out)
        return status

    def start_server(self, command: str) -> bool:
        working_dir = shlex.quote(self._remote_working_dir())
        log_path = shlex.quote(self._remote_log_path())
        pid_file = shlex.quote(self._remote_pid_file())
        # Record the PID remotely too, so the launcher can still find and stop
        # this instance after a restart -- and so co-hosted servers each keep
        # their own PID file rather than sharing one.
        start_cmd = (
            f"(cd {working_dir} && "
            f"exec nohup {environment_prefix(self._config)}{command} > {log_path} 2>&1 < /dev/null) & "
            f"pid=$!; echo $pid > {pid_file}; echo $pid"
        )
        success, output = self._run_remote(start_cmd, capture=True)
        if success and output.strip():
            try:
                pid = int(output.strip().splitlines()[-1])
            except (ValueError, IndexError):
                return False
            if self._pid_alive(pid):
                self._stored_pid = pid
                return True
            logger.warning("start_server: remote PID %s exited immediately", pid)
        return False

    def _tracked_pid(self) -> Optional[int]:
        """PID of a process the launcher started (stored or PID file)."""
        if self._stored_pid and self._pid_alive(self._stored_pid):
            return self._stored_pid
        pid = self._read_pid_file()
        if pid and self._pid_alive(pid):
            return pid
        return None

    def tracked_pid_alive(self) -> bool:
        return self._tracked_pid() is not None

    def stop_server(self) -> bool:
        # Only ever signal a process the launcher started, never one merely
        # discovered by port or pgrep. One round trip: TERM, wait for the
        # process to exit (it has to flush and free the GPU), then KILL -- a
        # Restart that followed an unfinished TERM used to die on the port.
        pid = self._tracked_pid()
        if not pid:
            logger.info("stop_server: no tracked llama-server process to stop")
            return True
        try:
            script = (
                f"kill {int(pid)} 2>/dev/null; "
                f"for i in $(seq 1 {_STOP_GRACE_SECONDS * 2}); do "
                f"kill -0 {int(pid)} 2>/dev/null || break; sleep 0.5; done; "
                f"kill -0 {int(pid)} 2>/dev/null && kill -9 {int(pid)}; "
                f"rm -f {shlex.quote(self._remote_pid_file())}; true"
            )
            self._run_remote(script, timeout=_STOP_GRACE_SECONDS + 15)
            self._stored_pid = None
            return True
        except Exception:
            logger.exception("stop_server failed for pid %s", pid)
            return False

    def restart_server(self, command: str) -> bool:
        self.stop_server()
        return self.start_server(command)

    def tail_log(self, lines: int = 100) -> str:
        log_path = shlex.quote(self._remote_log_path())
        success, output = self._run_remote(f"tail -n {int(lines)} {log_path}", capture=True)
        return output if success else ""

    def scan_model_directory(self, directory: str) -> List[ModelInfo]:
        """Find model files on the remote host.

        Classification is shared with the local scanner so both transports label
        family and quantization identically.
        """
        # Quoted so the remote shell cannot expand the globs against its cwd.
        ext_clauses = " -o ".join(f"-name {shlex.quote(ext)}" for ext in MODEL_EXTENSIONS)
        find_cmd = (f"find {shlex.quote(directory)} -type f \\( {ext_clauses} \\) "
                    f'-printf "%s\\t%p\\n"')
        success, output = self._run_remote(find_cmd, capture=True, timeout=60)
        if not success or not output.strip():
            return []

        models = []
        for line in output.splitlines():
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            try:
                size_bytes = int(parts[0])
            except ValueError:
                continue
            models.append(build_model_info(parts[1], size_bytes))
        return models
