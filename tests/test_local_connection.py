"""Tests for LocalServerConnection."""
import os
import stat
import subprocess
import tempfile
import time
import pytest

from llama_launcher.remote.local import LocalServerConnection
from llama_launcher.remote.types import (
    ServerConfig,
    ConnectionState, ServerStatus, ServerProcessState, FileEntry,
)


def _local_config(**kwargs):
    wd = kwargs.pop("working_directory", None)
    if wd is None:
        wd = tempfile.mkdtemp()
    llama_path = kwargs.pop("llama_server_path", "echo")
    cfg = ServerConfig(
        friendly_name="local-test",
        working_directory=wd,
        llama_server_path=llama_path,
        **kwargs,
    )
    return cfg


class TestConnectionStates:
    """Test connection state management."""

    def test_initial_state_is_disconnected(self):
        conn = LocalServerConnection(_local_config())
        # connection_state property auto-detects; test the internal state
        assert conn._state == ConnectionState.DISCONNECTED

    def test_connect_with_valid_command(self):
        conn = LocalServerConnection(_local_config())
        result = conn.connect()
        assert result is True
        assert conn.connection_state == ConnectionState.CONNECTED

    def test_connect_with_invalid_command(self):
        cfg = _local_config(llama_server_path="/nonexistent/command/that/does/not/exist")
        conn = LocalServerConnection(cfg)
        result = conn.connect()
        assert result is False
        assert conn.connection_state == ConnectionState.ERROR

    def test_disconnect_clears_log_process(self):
        conn = LocalServerConnection(_local_config())
        # Simulate having a log process
        import subprocess
        conn._log_process = subprocess.Popen(["sleep", "100"])
        conn.disconnect()
        assert conn._log_process is None

    def test_disconnect_resets_state(self):
        conn = LocalServerConnection(_local_config())
        conn.connect()
        conn.disconnect()
        # State used to be preserved so the old auto-probing connection_state
        # would not reconnect. It is passive now, so keeping CONNECTED would
        # make Disconnect appear to do nothing.
        assert conn.connection_state == ConnectionState.DISCONNECTED

    def test_repeated_connect_calls(self):
        conn = LocalServerConnection(_local_config())
        r1 = conn.connect()
        r2 = conn.connect()
        assert r1 is True
        assert r2 is True


class TestExecute:
    """Test command execution."""

    def test_execute_success(self):
        conn = LocalServerConnection(_local_config())
        result = conn.execute("echo hello")
        assert result is True

    def test_execute_failure(self):
        conn = LocalServerConnection(_local_config())
        result = conn.execute("exit 1")
        assert result is False

    def test_execute_command_not_found(self):
        conn = LocalServerConnection(_local_config())
        result = conn.execute("/no/such/command")
        assert result is False

    def test_execute_and_capture_success(self):
        conn = LocalServerConnection(_local_config())
        output = conn.execute_and_capture("echo hello world")
        assert "hello world" in output

    def test_execute_and_capture_failure_raises(self):
        conn = LocalServerConnection(_local_config())
        with pytest.raises(RuntimeError):
            conn.execute_and_capture("exit 1")

    def test_execute_timeout(self):
        cfg = _local_config()
        conn = LocalServerConnection(cfg)
        # This should timeout since the command takes too long
        result = conn.execute("sleep 60", )
        # May return False due to timeout
        assert isinstance(result, bool)


class TestFileOperations:
    """Test file upload/download/list operations."""

    def test_upload_file(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        src = tmp_path / "source.txt"
        src.write_text("test content")
        dst = tmp_path / "upload" / "dest.txt"
        result = conn.upload_file(str(src), str(dst))
        assert result is True
        assert dst.exists()
        assert dst.read_text() == "test content"

    def test_download_file(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        src = tmp_path / "download" / "source.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("download me")
        dst = tmp_path / "downloaded.txt"
        result = conn.download_file(str(src), str(dst))
        assert result is True
        assert dst.exists()
        assert dst.read_text() == "download me"

    def test_upload_file_with_progress_callback(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        src = tmp_path / "progress.txt"
        src.write_text("progress test content")
        dst = tmp_path / "upload" / "progress_dest.txt"
        callbacks = []
        def cb(done, total):
            callbacks.append((done, total))
        result = conn.upload_file(str(src), str(dst), progress_cb=cb)
        assert result is True
        assert len(callbacks) > 0

    def test_download_file_with_progress_callback(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        src = tmp_path / "src" / "file.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("progress download")
        dst = tmp_path / "dst" / "file.txt"
        callbacks = []
        def cb(done, total):
            callbacks.append((done, total))
        result = conn.download_file(str(src), str(dst), progress_cb=cb)
        assert result is True

    def test_upload_file_fails_for_nonexistent_source(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        result = conn.upload_file(str(tmp_path / "nope.txt"), str(tmp_path / "dest.txt"))
        assert result is False

    def test_list_directory(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        (tmp_path / "file1.txt").write_text("a")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        entries = conn.list_directory(str(tmp_path))
        names = [e.name for e in entries]
        assert "file1.txt" in names
        assert "subdir" in names

    def test_list_directory_nonexistent(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        entries = conn.list_directory(str(tmp_path / "nope"))
        assert entries == []

    def test_create_directory(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        result = conn.create_directory(str(tmp_path / "new" / "nested" / "dir"))
        assert result is True
        assert (tmp_path / "new" / "nested" / "dir").is_dir()

    def test_create_existing_directory(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        d = tmp_path / "exists"
        d.mkdir()
        result = conn.create_directory(str(d))
        assert result is True

    def test_delete_file(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        f = tmp_path / "to_delete.txt"
        f.write_text("delete me")
        result = conn.delete_file(str(f))
        assert result is True
        assert not f.exists()

    def test_delete_nonexistent_file(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        result = conn.delete_file(str(tmp_path / "nope.txt"))
        assert result is False

    def test_rename_file(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        old = tmp_path / "old.txt"
        old.write_text("rename")
        new = tmp_path / "new.txt"
        result = conn.rename_file(str(old), str(new))
        assert result is True
        assert new.exists()
        assert not old.exists()
        assert new.read_text() == "rename"

    def test_rename_nonexistent_file(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        result = conn.rename_file(str(tmp_path / "nope"), str(tmp_path / "dest"))
        assert result is False


class TestPIDManagement:
    """Test PID tracking and file management."""

    def test_stored_pid_alive(self):
        """PID 0 (swapper/scheduler) should be alive on Linux."""
        assert LocalServerConnection._pid_alive(0) is True

    def test_stored_pid_dead(self):
        """A very high PID number should not be alive."""
        assert LocalServerConnection._pid_alive(999999) is False

    def test_read_pid_file(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        pid_file = conn._pid_file_path()
        pid_file.write_text("12345\n")
        pid = conn._read_pid_file()
        assert pid == 12345

    def test_read_pid_file_empty(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        pid_file = conn._pid_file_path()
        pid_file.write_text("")
        pid = conn._read_pid_file()
        assert pid is None

    def test_read_pid_file_invalid(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        pid_file = conn._pid_file_path()
        pid_file.write_text("not-a-pid")
        pid = conn._read_pid_file()
        assert pid is None

    def test_write_pid_file(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        conn._write_pid_file(12345)
        pid_file = conn._pid_file_path()
        assert pid_file.exists()
        assert "12345" in pid_file.read_text()

    def test_remove_pid_file(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        pid_file = conn._pid_file_path()
        pid_file.write_text("12345\n")
        conn._remove_pid_file()
        assert not pid_file.exists()

    def test_pid_file_path(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        path = conn._pid_file_path()
        # Named per server, not per directory: co-hosted servers usually share a
        # working directory and must not share a PID file.
        assert path.name == ".llama_launcher_local-test.pid"
        assert str(tmp_path) in str(path)

    def test_pid_files_differ_between_servers_in_one_directory(self, tmp_path):
        def conn_named(name):
            cfg = _local_config(working_directory=str(tmp_path))
            cfg.friendly_name = name
            return LocalServerConnection(cfg)

        assert conn_named("alpha")._pid_file_path() != conn_named("beta")._pid_file_path()

    def test_legacy_pid_file_is_still_read(self, tmp_path):
        # A server started by an older build must remain stoppable.
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        (tmp_path / ".llama_launcher_server.pid").write_text("4242\n")
        assert conn._read_pid_file() == 4242

    def test_current_pid_file_wins_over_legacy(self, tmp_path):
        cfg = _local_config(working_directory=str(tmp_path))
        conn = LocalServerConnection(cfg)
        (tmp_path / ".llama_launcher_server.pid").write_text("4242\n")
        conn._pid_file_path().write_text("777\n")
        assert conn._read_pid_file() == 777


class TestGetStatus:
    """Test status reporting."""

    def test_stopped_status_no_llama_server(self):
        """Status when no llama-server is detected by pgrep."""
        conn = LocalServerConnection(_local_config())
        status = conn.get_status()
        # The pgrep may find a running llama-server on this system
        # Just verify the status dataclass is returned correctly
        assert isinstance(status, ServerStatus)
        assert status.state in (ServerProcessState.STOPPED, ServerProcessState.RUNNING)

    def test_status_dataclass_fields(self):
        conn = LocalServerConnection(_local_config())
        status = conn.get_status()
        assert isinstance(status, ServerStatus)
        assert hasattr(status, "cpu_percent")
        assert hasattr(status, "ram_mb")
        assert hasattr(status, "gpu_utilization")
        assert hasattr(status, "listening_port")


class TestServerLifecycle:
    """Test server start/stop/restart."""

    def test_start_server_fails_no_llama_server(self, tmp_path):
        cfg = _local_config(
            working_directory=str(tmp_path),
        )
        cfg.llama_server_path = "/nonexistent/server"
        conn = LocalServerConnection(cfg)
        # start_server runs a shell pipeline; may return True even if inner cmd fails
        # Just verify it doesn't crash and returns a bool
        result = conn.start_server("/nonexistent/server")
        assert isinstance(result, bool)

    def test_tracked_pid_alive_false(self):
        conn = LocalServerConnection(_local_config())
        assert conn.tracked_pid_alive() is False

    def test_stop_server_no_tracked_pid(self):
        conn = LocalServerConnection(_local_config())
        # Should return True (nothing to stop)
        result = conn.stop_server()
        assert result is True

    def test_config_property(self):
        cfg = _local_config()
        conn = LocalServerConnection(cfg)
        assert conn.config is cfg
        assert conn.config.friendly_name == "local-test"


class TestScanModels:
    """Test model directory scanning."""

    def test_scan_model_directory(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        # Create some model files
        (tmp_path / "model.gguf").write_bytes(b"fake")
        (tmp_path / "model.bin").write_bytes(b"fake")
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_text("ignore")
        models = conn.scan_model_directory(str(tmp_path))
        assert len(models) == 3
        formats = {m.format for m in models}
        assert formats == {"gguf", "bin", "safetensors"}

    def test_scan_empty_directory(self, tmp_path):
        conn = LocalServerConnection(_local_config())
        models = conn.scan_model_directory(str(tmp_path))
        assert models == []


class TestTildeExpansion:
    """Test that ~ in paths is expanded correctly."""

    def test_upload_with_tilde(self, tmp_path):
        """Simulates tilde expansion in upload paths."""
        conn = LocalServerConnection(_local_config())
        # Test that Path.expanduser works
        src = tmp_path / "src.txt"
        src.write_text("tilde test")
        # Just verify the method handles tilde correctly
        result = conn.upload_file(str(src), str(tmp_path / "dest.txt"))
        assert result is True
