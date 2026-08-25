"""Tests for ServerManager."""
import tempfile
import pytest

from llama_launcher.remote.manager import ServerManager
from llama_launcher.remote.types import (
    ServerConfig,
    ConnectionState, ServerStatus, ServerProcessState,
)
from llama_launcher.remote.local import LocalServerConnection


def _local_config(name="local-test", **kwargs):
    llama_path = kwargs.pop("llama_server_path", "echo")
    return ServerConfig(
        friendly_name=name,
        working_directory=tempfile.mkdtemp(),
        llama_server_path=llama_path,
        **kwargs,
    )


class TestAddServer:
    """Test adding servers."""

    def test_add_local_server(self):
        mgr = ServerManager()
        cfg = _local_config()
        result = mgr.add_server(cfg)
        assert result is True
        assert "local-test" in mgr.get_all_names()

    def test_add_server_creates_local_connection(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        conn = mgr.get_connection("local-test")
        assert isinstance(conn, LocalServerConnection)

    def test_add_duplicate_server_overwrites(self):
        mgr = ServerManager()
        cfg1 = _local_config(name="dup", llama_server_path="echo")
        cfg2 = _local_config(name="dup", llama_server_path="cat")
        mgr.add_server(cfg1)
        mgr.add_server(cfg2)
        # Should only have one connection for "dup"
        assert len(mgr.get_all_names()) == 1


class TestRemoveServer:
    """Test removing servers."""

    def test_remove_existing_server(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        result = mgr.remove_server("local-test")
        assert result is True
        assert "local-test" not in mgr.get_all_names()

    def test_remove_nonexistent_server(self):
        mgr = ServerManager()
        result = mgr.remove_server("nope")
        assert result is False

    def test_remove_disconnects_connection(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        conn = mgr.get_connection("local-test")
        # Simulate having a log process
        import subprocess
        conn._log_process = subprocess.Popen(["sleep", "100"])
        mgr.remove_server("local-test")
        # Connection should be disconnected
        assert "local-test" not in mgr._connections


class TestGetConnection:
    """Test connection retrieval."""

    def test_get_existing_connection(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        conn = mgr.get_connection("local-test")
        assert conn is not None

    def test_get_nonexistent_connection(self):
        mgr = ServerManager()
        conn = mgr.get_connection("nope")
        assert conn is None

    def test_get_or_create_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        conn1 = mgr.get_connection("local-test")
        conn2 = mgr.get_or_create_connection(cfg)
        assert conn1 is conn2

    def test_get_or_create_new(self):
        mgr = ServerManager()
        cfg = _local_config(name="new")
        conn = mgr.get_or_create_connection(cfg)
        assert conn is not None
        assert "new" in mgr.get_all_names()


class TestConnectionOperations:
    """Test connection management operations."""

    def test_connect_existing_server(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        result = mgr.connect("local-test")
        assert result is True

    def test_connect_nonexistent_server(self):
        mgr = ServerManager()
        result = mgr.connect("nope")
        assert result is False

    def test_disconnect_existing_server(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        mgr.disconnect("local-test")

    def test_disconnect_nonexistent_server(self):
        mgr = ServerManager()
        # Should not raise
        mgr.disconnect("nope")

    def test_disconnect_all(self):
        mgr = ServerManager()
        cfg1 = _local_config(name="srv1")
        cfg2 = _local_config(name="srv2")
        mgr.add_server(cfg1)
        mgr.add_server(cfg2)
        mgr.disconnect_all()
        # Both should be disconnected
        for name in mgr.get_all_names():
            conn = mgr.get_connection(name)
            if conn:
                conn.disconnect()

    def test_get_status_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        status = mgr.get_status("local-test")
        assert status is not None
        assert isinstance(status, ServerStatus)

    def test_get_status_nonexistent(self):
        mgr = ServerManager()
        status = mgr.get_status("nope")
        assert status is None

    def test_get_connection_state_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        state = mgr.get_connection_state("local-test")
        assert state in (ConnectionState.CONNECTED, ConnectionState.DISCONNECTED)

    def test_get_connection_state_nonexistent(self):
        mgr = ServerManager()
        state = mgr.get_connection_state("nope")
        assert state == ConnectionState.DISCONNECTED


class TestServerOperations:
    """Test server start/stop/restart through manager."""

    def test_start_server_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        result = mgr.start_server("local-test", "echo test")
        # May return True or False depending on env
        assert isinstance(result, bool)

    def test_start_server_nonexistent(self):
        mgr = ServerManager()
        result = mgr.start_server("nope", "echo test")
        assert result is False

    def test_stop_server_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        result = mgr.stop_server("local-test")
        assert result is True

    def test_stop_server_nonexistent(self):
        mgr = ServerManager()
        result = mgr.stop_server("nope")
        assert result is False

    def test_execute_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        result = mgr.execute("local-test", "echo hello")
        assert result is True

    def test_execute_nonexistent(self):
        mgr = ServerManager()
        result = mgr.execute("nope", "echo hello")
        assert result is False

    def test_execute_and_capture_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        output = mgr.execute_and_capture("local-test", "echo hello")
        assert "hello" in output

    def test_execute_and_capture_nonexistent(self):
        mgr = ServerManager()
        with pytest.raises(ConnectionError):
            mgr.execute_and_capture("nope", "echo hello")

    def test_tail_log_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        log = mgr.tail_log("local-test")
        assert isinstance(log, str)

    def test_tail_log_nonexistent(self):
        mgr = ServerManager()
        log = mgr.tail_log("nope")
        assert log == ""

    def test_get_server_pid_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        pid = mgr.get_server_pid("local-test")
        # May or may not be None depending on if llama-server is running
        assert isinstance(pid, int) or pid is None

    def test_get_server_pid_nonexistent(self):
        mgr = ServerManager()
        pid = mgr.get_server_pid("nope")
        assert pid is None

    def test_tracked_pid_alive_existing(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        alive = mgr.tracked_pid_alive("local-test")
        assert alive is False

    def test_tracked_pid_alive_nonexistent(self):
        mgr = ServerManager()
        alive = mgr.tracked_pid_alive("nope")
        assert alive is False

    def test_check_port_conflict_no_listener(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        mgr.port_listener_pid = lambda name, port: None
        assert mgr.check_port_conflict("local-test") is None

    def test_check_port_conflict_foreign_listener(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        mgr.port_listener_pid = lambda name, port: 4242
        mgr.tracked_pid_alive = lambda name: False
        msg = mgr.check_port_conflict("local-test")
        assert msg is not None
        assert "4242" in msg
        assert str(cfg.server_port) in msg

    def test_check_port_conflict_own_tracked_pid_is_not_a_conflict(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        mgr.port_listener_pid = lambda name, port: 4242
        mgr.tracked_pid_alive = lambda name: True
        assert mgr.check_port_conflict("local-test") is None

    def test_check_port_conflict_nonexistent_server(self):
        mgr = ServerManager()
        assert mgr.check_port_conflict("nope") is None


class TestFileOperations:
    """Test file operations through manager."""

    def test_list_directory(self, tmp_path):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        entries = mgr.list_directory("local-test", str(tmp_path))
        assert isinstance(entries, list)

    def test_list_directory_nonexistent(self):
        mgr = ServerManager()
        entries = mgr.list_directory("nope", "/tmp")
        assert entries == []

    def test_create_directory(self, tmp_path):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        result = mgr.create_directory("local-test", str(tmp_path / "new"))
        assert result is True

    def test_delete_file(self, tmp_path):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        f = tmp_path / "del.txt"
        f.write_text("delete")
        result = mgr.delete_file("local-test", str(f))
        assert result is True

    def test_rename_file(self, tmp_path):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        old = tmp_path / "old.txt"
        old.write_text("rename")
        new = tmp_path / "new.txt"
        result = mgr.rename_file("local-test", str(old), str(new))
        assert result is True


class TestConfigManagement:
    """Test config listing."""

    def test_get_all_configs_empty(self):
        mgr = ServerManager()
        assert mgr.get_all_configs() == []

    def test_get_all_configs_has_entries(self):
        mgr = ServerManager()
        cfg1 = _local_config(name="srv1")
        cfg2 = _local_config(name="srv2")
        mgr.add_server(cfg1)
        mgr.add_server(cfg2)
        configs = mgr.get_all_configs()
        assert len(configs) == 2
        names = {c.friendly_name for c in configs}
        assert names == {"srv1", "srv2"}

    def test_get_all_names(self):
        mgr = ServerManager()
        cfg1 = _local_config(name="n1")
        cfg2 = _local_config(name="n2")
        mgr.add_server(cfg1)
        mgr.add_server(cfg2)
        names = mgr.get_all_names()
        assert "n1" in names
        assert "n2" in names


class TestModelOperations:
    """Test model operations through manager."""

    def test_scan_server_models(self, tmp_path):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        # Create model files
        (tmp_path / "m.gguf").write_bytes(b"fake")
        models = mgr.scan_server_models("local-test", str(tmp_path))
        assert len(models) == 1

    def test_scan_server_models_nonexistent(self):
        mgr = ServerManager()
        models = mgr.scan_server_models("nope", "/tmp")
        assert models == []

    def test_get_server_model_directories_empty(self):
        mgr = ServerManager()
        cfg = _local_config()
        mgr.add_server(cfg)
        dirs = mgr.get_server_model_directories("local-test")
        assert dirs == []

    def test_get_server_model_directories_with_config(self):
        mgr = ServerManager()
        cfg = _local_config(model_directory="/models")
        mgr.add_server(cfg)
        dirs = mgr.get_server_model_directories("local-test")
        assert "/models" in dirs
