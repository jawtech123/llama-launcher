"""Tests for the remote filesystem browsing support.

Covers the SSH command construction and listing parser, the connection
primitives the path picker relies on, the permanent local-server guard, and the
picker's own rendering helpers.
"""
import os
import time
import subprocess
from unittest.mock import patch

import pytest

from llama_launcher.remote.local import LocalServerConnection
from llama_launcher.remote.manager import (
    ServerManager, create_connection, is_remote_config,
)
from llama_launcher.remote.ssh import SSHServerConnection
from llama_launcher.remote.types import LOCAL_SERVER_NAME, FileEntry, ServerConfig


def _remote_config(**kwargs):
    defaults = dict(
        friendly_name="remote-test",
        hostname="10.0.0.5",
        username="bob",
        ssh_port=2200,
        working_directory="/srv",
    )
    defaults.update(kwargs)
    return ServerConfig(**defaults)


@pytest.fixture
def app():
    """Provide a QApplication instance for Qt widget tests."""
    from PySide6.QtWidgets import QApplication
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


class TestSSHCommandConstruction:
    """The ssh argv must be built so ssh actually honours our options."""

    def test_key_path_precedes_destination(self, tmp_path):
        # ssh stops parsing options at user@host, so -i must come before it.
        key = tmp_path / "id_test"
        key.write_text("x")
        conn = SSHServerConnection(_remote_config(key_path=str(key)))
        argv = conn._ssh_base_cmd
        assert "-i" in argv
        assert argv.index("-i") < argv.index("bob@10.0.0.5")
        assert argv[argv.index("-i") + 1] == str(key)

    def test_key_path_is_not_shell_quoted(self, tmp_path):
        # No shell is involved any more; quoting would corrupt the path.
        key = tmp_path / "my key"
        key.write_text("x")
        conn = SSHServerConnection(_remote_config(key_path=str(key)))
        assert str(key) in conn._ssh_base_cmd

    def test_nonexistent_key_is_skipped(self):
        conn = SSHServerConnection(_remote_config(key_path="/no/such/key"))
        assert "-i" not in conn._ssh_base_cmd

    def test_key_auth_uses_batch_mode(self):
        conn = SSHServerConnection(_remote_config())
        assert "BatchMode=yes" in conn._ssh_base_cmd

    def test_password_auth_disables_batch_mode(self):
        # BatchMode=yes makes ssh fail instead of accepting a password.
        conn = SSHServerConnection(
            _remote_config(auth_method="password", password="secret"))
        argv = conn._ssh_base_cmd
        assert "BatchMode=no" in argv
        assert "BatchMode=yes" not in argv
        assert "NumberOfPasswordPrompts=1" in argv

    def test_password_is_passed_via_environment_not_argv(self):
        conn = SSHServerConnection(
            _remote_config(auth_method="password", password="secret"))
        assert "secret" not in " ".join(conn._ssh_base_cmd)
        env = conn._askpass_env()
        assert env["LL_SSH_PASSWORD"] == "secret"
        assert env["SSH_ASKPASS_REQUIRE"] == "force"
        assert os.access(env["SSH_ASKPASS"], os.X_OK)

    def test_empty_password_field_keeps_key_auth(self):
        conn = SSHServerConnection(_remote_config(auth_method="password", password=""))
        assert "BatchMode=yes" in conn._ssh_base_cmd
        assert conn._askpass_env() is None

    def test_connection_reuse_is_configured(self):
        argv = SSHServerConnection(_remote_config())._ssh_base_cmd
        joined = " ".join(argv)
        assert "ControlMaster=auto" in joined
        assert "ControlPath=" in joined
        assert "ControlPersist=" in joined

    def test_port_and_destination(self):
        argv = SSHServerConnection(_remote_config())._ssh_base_cmd
        assert argv[argv.index("-p") + 1] == "2200"
        assert argv[-1] == "bob@10.0.0.5"


class TestControlPathLength:
    """A ControlPath socket name has a hard ~104 byte limit."""

    def test_default_control_path_leaves_headroom(self):
        from llama_launcher.remote.ssh import _CONTROL_PATH_TOKEN_LEN, _control_path
        template = _control_path()
        assert template is not None
        expanded = len(template) - len("%C") + _CONTROL_PATH_TOKEN_LEN
        assert expanded < 104

    def test_control_dir_is_not_under_the_config_dir(self, monkeypatch, tmp_path):
        # Config dirs follow $HOME and can be arbitrarily long; sockets must not.
        from llama_launcher.utils.paths import ssh_control_dir
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ("deep" * 25)))
        assert str(tmp_path / ("deep" * 25)) not in str(ssh_control_dir())

    def test_reuse_is_dropped_rather_than_breaking_ssh(self, monkeypatch, tmp_path):
        from llama_launcher.remote import ssh as ssh_module
        long_dir = tmp_path / ("x" * 90)
        long_dir.mkdir()
        monkeypatch.setattr(ssh_module, "ssh_control_dir", lambda: long_dir)
        assert ssh_module._control_path() is None

        conn = SSHServerConnection(_remote_config())
        argv = conn._build_ssh_base()
        # Losing reuse costs speed; an over-long path would fail every command.
        assert not any("ControlPath" in a for a in argv)
        assert not any("ControlMaster" in a for a in argv)
        assert argv[-1] == "bob@10.0.0.5"


class TestSSHListDirectory:
    """find -printf output must parse into exact FileEntry values."""

    FIND_OUTPUT = (
        "d\t4096\t1785271122.5372292010\tmodels\n"
        "f\t5368709120\t1782213307.0708110370\tqwen3-30b-Q6_K.gguf\n"
        "d\t4096\t1785186743.3663494610\t.hidden-dir\n"
        "f\t0\t1700000000.0000000000\tempty.bin\n"
    )

    def _conn_with_output(self, output, success=True):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (success, output)
        return conn

    def test_parses_all_fields(self):
        entries = self._conn_with_output(self.FIND_OUTPUT).list_directory("/data")
        by_name = {e.name: e for e in entries}
        assert set(by_name) == {
            "models", "qwen3-30b-Q6_K.gguf", ".hidden-dir", "empty.bin"}

        model_dir = by_name["models"]
        assert model_dir.is_directory is True
        assert model_dir.path == "/data/models"

        gguf = by_name["qwen3-30b-Q6_K.gguf"]
        assert gguf.is_directory is False
        assert gguf.size_bytes == 5368709120
        # The old ls parser produced 0.0 here, so every date showed as 1970.
        assert gguf.modified_time == pytest.approx(1782213307.07, abs=0.1)

    def test_dotfiles_are_included(self):
        entries = self._conn_with_output(self.FIND_OUTPUT).list_directory("/data")
        assert ".hidden-dir" in {e.name for e in entries}

    def test_no_phantom_entry_from_ls_total_header(self):
        # `ls -l` emits "total 48"; find emits nothing of the kind.
        entries = self._conn_with_output(self.FIND_OUTPUT).list_directory("/data")
        assert not any(e.name.isdigit() for e in entries)
        assert len(entries) == 4

    def test_trailing_and_blank_lines_ignored(self):
        entries = self._conn_with_output(
            "d\t4096\t1.0\tone\n\n").list_directory("/data")
        assert [e.name for e in entries] == ["one"]

    def test_names_containing_spaces_survive(self):
        entries = self._conn_with_output(
            "f\t10\t1.0\tmy model file.gguf\n").list_directory("/data")
        assert entries[0].name == "my model file.gguf"
        assert entries[0].path == "/data/my model file.gguf"

    def test_unparseable_size_and_mtime_default_to_zero(self):
        entries = self._conn_with_output(
            "f\tNaN\tnope\tweird.bin\n").list_directory("/data")
        assert entries[0].size_bytes == 0
        assert entries[0].modified_time == 0.0

    def test_root_directory_paths_are_not_doubled(self):
        entries = self._conn_with_output("d\t4096\t1.0\tetc\n").list_directory("/")
        assert entries[0].path == "/etc"

    def test_falls_back_to_ls_when_find_unavailable(self):
        conn = SSHServerConnection(_remote_config())
        calls = []

        def fake_run(cmd, capture=False, timeout=30):
            calls.append(cmd)
            if cmd.startswith("find "):
                return (False, "")
            return (True, "models/\nfile.gguf\n")

        conn._run_remote = fake_run
        entries = conn.list_directory("/data")
        assert any(c.startswith("find ") for c in calls)
        assert any(c.startswith("ls ") for c in calls)
        by_name = {e.name: e.is_directory for e in entries}
        assert by_name == {"models": True, "file.gguf": False}

    def test_returns_empty_when_both_listings_fail(self):
        conn = self._conn_with_output("", success=False)
        assert conn.list_directory("/nope") == []


class TestSSHPathPrimitives:
    def test_path_kind_maps_shell_output(self):
        for output, expected in (("dir\n", "dir"), ("file\n", "file"),
                                 ("missing\n", "missing")):
            conn = SSHServerConnection(_remote_config())
            conn._run_remote = lambda c, capture=False, timeout=30, o=output: (True, o)
            assert conn.path_kind("/x") == expected

    def test_path_kind_missing_on_failure(self):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda c, capture=False, timeout=30: (False, "")
        assert conn.path_kind("/x") == "missing"

    def test_home_directory_uses_remote_home(self):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda c, capture=False, timeout=30: (True, "/home/bob")
        assert conn.home_directory() == "/home/bob"

    def test_home_directory_falls_back_to_username(self):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda c, capture=False, timeout=30: (False, "")
        assert conn.home_directory() == "/home/bob"

    def test_test_connection_requires_host_and_user(self):
        ok, msg = SSHServerConnection(_remote_config(hostname="")).test_connection()
        assert ok is False and "hostname" in msg.lower()
        ok, msg = SSHServerConnection(_remote_config(username="")).test_connection()
        assert ok is False and "username" in msg.lower()

    def test_test_connection_reports_ssh_stderr(self):
        conn = SSHServerConnection(_remote_config())

        def fail(cmd, capture=False, timeout=30):
            conn._last_stderr = (
                "Warning: Permanently added '10.0.0.5' to known hosts.\n"
                "bob@10.0.0.5: Permission denied (publickey,password)."
            )
            return (False, "")

        conn._run_remote = fail
        ok, msg = conn.test_connection()
        assert ok is False
        # The host-key warning is noise; the real reason must surface.
        assert msg == "bob@10.0.0.5: Permission denied (publickey,password)."


class TestSCPDirection:
    """upload_file must copy to the remote host, download_file from it."""

    def _capture_argv(self, fn):
        captured = {}

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return Result()

        with patch.object(subprocess, "run", fake_run):
            fn()
        return captured["argv"]

    def test_upload_puts_local_first_and_remote_second(self, tmp_path):
        src = tmp_path / "model.gguf"
        src.write_text("x")
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "")
        argv = self._capture_argv(
            lambda: conn.upload_file(str(src), "/srv/model.gguf"))
        assert argv[-2] == str(src)
        assert argv[-1] == "bob@10.0.0.5:/srv/model.gguf"

    def test_download_puts_remote_first_and_local_second(self, tmp_path):
        dst = tmp_path / "out.gguf"
        conn = SSHServerConnection(_remote_config())
        argv = self._capture_argv(
            lambda: conn.download_file("/srv/model.gguf", str(dst)))
        assert argv[-2] == "bob@10.0.0.5:/srv/model.gguf"
        assert argv[-1] == str(dst)

    def test_upload_and_download_argv_differ(self, tmp_path):
        # The pre-fix implementation built the identical command for both.
        src = tmp_path / "f.txt"
        src.write_text("x")
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "")
        up = self._capture_argv(lambda: conn.upload_file(str(src), "/srv/f.txt"))
        down = self._capture_argv(lambda: conn.download_file("/srv/f.txt", str(src)))
        assert up != down

    def test_scp_carries_port_and_control_path(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("x")
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "")
        argv = self._capture_argv(lambda: conn.upload_file(str(src), "/srv/f.txt"))
        assert argv[argv.index("-P") + 1] == "2200"
        assert any("ControlPath=" in a for a in argv)


class TestTransferProgress:
    """Model files are large; a silent transfer bar is not acceptable."""

    def _conn(self):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "0")
        return conn

    def test_progress_callback_receives_total_and_completion(self, tmp_path):
        src = tmp_path / "model.gguf"
        src.write_bytes(b"x" * 4096)
        conn = self._conn()
        seen = []

        class Finished:
            returncode = 0

            def poll(self): return 0
            def communicate(self, timeout=None): return ("", "")

        with patch.object(subprocess, "Popen", lambda *a, **k: Finished()):
            ok = conn.upload_file(str(src), "/srv/model.gguf",
                                  progress_cb=lambda d, t: seen.append((d, t)))
        assert ok is True
        # Final callback reports completion against the true source size.
        assert seen[-1] == (4096, 4096)

    def test_cancelling_callback_aborts_and_cleans_up(self, tmp_path):
        src = tmp_path / "model.gguf"
        src.write_bytes(b"x" * 4096)
        conn = self._conn()
        removed = []
        conn._remove_partial = lambda r, l, d: removed.append((r, d))

        class Running:
            returncode = None
            terminated = False

            def poll(self): return None
            def terminate(self): Running.terminated = True
            def wait(self, timeout=None): return 0
            def kill(self): pass

        def cancel(done, total):
            raise RuntimeError("Transfer cancelled")

        with patch.object(subprocess, "Popen", lambda *a, **k: Running()):
            with pytest.raises(RuntimeError, match="cancelled"):
                conn.upload_file(str(src), "/srv/model.gguf", progress_cb=cancel)
        assert Running.terminated, "scp must be killed when the user cancels"
        assert removed == [("/srv/model.gguf", "upload")]

    def test_no_progress_callback_uses_the_simple_path(self, tmp_path):
        src = tmp_path / "model.gguf"
        src.write_bytes(b"x")
        conn = self._conn()
        called = {}

        class Result:
            returncode = 0
            stdout = stderr = ""

        def fake_run(argv, **kwargs):
            called["timeout"] = kwargs.get("timeout")
            return Result()

        with patch.object(subprocess, "run", fake_run):
            assert conn.upload_file(str(src), "/srv/model.gguf") is True
        # A multi-GB model must not hit the ordinary command timeout.
        assert called["timeout"] >= 3600

    def test_download_total_comes_from_the_remote_size(self):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "123456\n")
        assert conn._transfer_total_bytes("/tmp/x", "/srv/m.gguf", "download") == 123456

    def test_upload_total_comes_from_the_local_size(self, tmp_path):
        src = tmp_path / "m.gguf"
        src.write_bytes(b"x" * 777)
        conn = self._conn()
        assert conn._transfer_total_bytes(str(src), "/srv/m.gguf", "upload") == 777


class TestLocalConnectionPrimitives:
    def test_list_directory_survives_broken_symlink(self, tmp_path):
        (tmp_path / "real.gguf").write_text("x")
        (tmp_path / "broken").symlink_to(tmp_path / "does-not-exist")
        conn = LocalServerConnection(ServerConfig(friendly_name="l"))
        names = {e.name for e in conn.list_directory(str(tmp_path))}
        # One unreadable entry must not abort the whole listing.
        assert names == {"real.gguf", "broken"}

    def test_path_kind(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        conn = LocalServerConnection(ServerConfig(friendly_name="l"))
        assert conn.path_kind(str(tmp_path)) == "dir"
        assert conn.path_kind(str(tmp_path / "f.txt")) == "file"
        assert conn.path_kind(str(tmp_path / "nope")) == "missing"

    def test_home_directory(self):
        conn = LocalServerConnection(ServerConfig(friendly_name="l"))
        assert conn.home_directory() == os.path.expanduser("~")

    def test_test_connection_reports_missing_binary(self):
        conn = LocalServerConnection(
            ServerConfig(friendly_name="l", llama_server_path="definitely-not-a-binary"))
        ok, msg = conn.test_connection()
        assert ok is False
        assert "definitely-not-a-binary" in msg

    def test_test_connection_succeeds_for_resolvable_binary(self):
        conn = LocalServerConnection(
            ServerConfig(friendly_name="l", llama_server_path="echo"))
        ok, msg = conn.test_connection()
        assert ok is True
        assert "echo" in msg


class TestCreateConnection:
    def test_dispatches_on_hostname(self):
        assert isinstance(create_connection(_remote_config()), SSHServerConnection)
        assert isinstance(
            create_connection(ServerConfig(friendly_name="l")), LocalServerConnection)

    def test_whitespace_hostname_is_local(self):
        assert is_remote_config(ServerConfig(friendly_name="l", hostname="   ")) is False

    def test_does_not_persist(self):
        from llama_launcher.remote.storage import load_servers
        before = [s.friendly_name for s in load_servers()]
        create_connection(_remote_config(friendly_name="ephemeral"))
        assert [s.friendly_name for s in load_servers()] == before


class TestPermanentLocalServer:
    def test_storage_refuses_to_remove_local_entry(self):
        from llama_launcher.remote.storage import (
            load_servers, remove_server, save_server,
        )
        save_server(ServerConfig(friendly_name=LOCAL_SERVER_NAME,
                                 llama_server_path="echo"))
        assert remove_server(LOCAL_SERVER_NAME) is False
        assert LOCAL_SERVER_NAME in [s.friendly_name for s in load_servers()]

    def test_other_servers_are_still_removable(self):
        from llama_launcher.remote.storage import (
            load_servers, remove_server, save_server,
        )
        save_server(ServerConfig(friendly_name="disposable"))
        assert remove_server("disposable") is True
        assert "disposable" not in [s.friendly_name for s in load_servers()]

    def test_manager_remove_leaves_local_entry_in_storage(self):
        from llama_launcher.remote.storage import load_servers
        mgr = ServerManager()
        mgr.add_server(ServerConfig(friendly_name=LOCAL_SERVER_NAME,
                                    llama_server_path="echo"))
        mgr.remove_server(LOCAL_SERVER_NAME)
        assert LOCAL_SERVER_NAME in [s.friendly_name for s in load_servers()]


class TestManagerPassThroughs:
    def test_unknown_server_returns_safe_defaults(self):
        mgr = ServerManager()
        assert mgr.list_directory("nope", "/") == []
        assert mgr.home_directory("nope") == ""
        assert mgr.path_kind("nope", "/") == "missing"
        ok, msg = mgr.test_connection("nope")
        assert ok is False and "nope" in msg

    def test_primitives_reach_the_connection(self, tmp_path):
        mgr = ServerManager()
        mgr.add_server(ServerConfig(friendly_name="local-primitives",
                                    llama_server_path="echo"), persist=False)
        assert mgr.path_kind("local-primitives", str(tmp_path)) == "dir"
        assert mgr.home_directory("local-primitives") == os.path.expanduser("~")
        ok, _ = mgr.test_connection("local-primitives")
        assert ok is True


class TestPathHelpers:
    def test_posix_parent(self):
        from llama_launcher.gui.dialogs.path_picker import posix_parent
        assert posix_parent("/a/b/c") == "/a/b"
        assert posix_parent("/a/b/c/") == "/a/b"
        assert posix_parent("/a") == "/"
        assert posix_parent("/") == "/"

    def test_posix_join(self):
        from llama_launcher.gui.dialogs.path_picker import posix_join
        assert posix_join("/a/b", "c") == "/a/b/c"
        assert posix_join("/a/b/", "c") == "/a/b/c"
        assert posix_join("/", "c") == "/c"


class TestPopulateFileTree:
    def _entries(self):
        return [
            FileEntry(name="zeta", path="/d/zeta", is_directory=True,
                      modified_time=1700000000.0),
            FileEntry(name="alpha", path="/d/alpha", is_directory=True,
                      modified_time=1700000000.0),
            FileEntry(name="model.gguf", path="/d/model.gguf",
                      size_bytes=2048, modified_time=1700000000.0),
            FileEntry(name="notes.txt", path="/d/notes.txt",
                      size_bytes=10, modified_time=1700000000.0),
        ]

    def _tree(self, app):
        from PySide6.QtWidgets import QTreeWidget
        tree = QTreeWidget()
        tree.setHeaderLabels(["Name", "Size", "Modified"])
        return tree

    def test_directories_first_then_alphabetical(self, app):
        from llama_launcher.gui.dialogs.path_picker import populate_file_tree
        tree = self._tree(app)
        populate_file_tree(tree, self._entries())
        labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert labels == ["[DIR] alpha", "[DIR] zeta", "model.gguf", "notes.txt"]

    def test_dir_mode_hides_files(self, app):
        from llama_launcher.gui.dialogs.path_picker import populate_file_tree
        tree = self._tree(app)
        populate_file_tree(tree, self._entries(), show_files=False)
        labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert labels == ["[DIR] alpha", "[DIR] zeta"]

    def test_name_filter_keeps_directories_and_matching_files(self, app):
        from llama_launcher.gui.dialogs.path_picker import (
            MODEL_PATTERNS, populate_file_tree,
        )
        tree = self._tree(app)
        populate_file_tree(tree, self._entries(), name_filter=MODEL_PATTERNS)
        labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        # Directories stay so the user can navigate; notes.txt is filtered out.
        assert labels == ["[DIR] alpha", "[DIR] zeta", "model.gguf"]

    def test_metadata_is_stashed_on_items(self, app):
        from llama_launcher.gui.dialogs.path_picker import (
            IS_DIR_ROLE, PATH_ROLE, SIZE_ROLE, populate_file_tree,
        )
        tree = self._tree(app)
        populate_file_tree(tree, self._entries())
        item = tree.topLevelItem(2)  # model.gguf
        assert item.data(0, PATH_ROLE) == "/d/model.gguf"
        assert item.data(0, IS_DIR_ROLE) is False
        assert item.data(0, SIZE_ROLE) == 2048

    def test_missing_mtime_renders_placeholder(self, app):
        from llama_launcher.gui.dialogs.path_picker import populate_file_tree
        tree = self._tree(app)
        populate_file_tree(
            tree, [FileEntry(name="f", path="/d/f", modified_time=0.0)])
        assert tree.topLevelItem(0).text(2) == "-"

    def test_directories_stay_on_top_in_both_sort_orders(self, app):
        # setSortingEnabled(True) defaults to DescendingOrder, which silently
        # reversed the dirs-first grouping.
        from PySide6.QtCore import Qt
        from llama_launcher.gui.dialogs.path_picker import (
            enable_name_sorting, populate_file_tree,
        )
        for order in (Qt.AscendingOrder, Qt.DescendingOrder):
            tree = self._tree(app)
            enable_name_sorting(tree)
            populate_file_tree(tree, self._entries())
            tree.sortItems(0, order)
            labels = [tree.topLevelItem(i).text(0)
                      for i in range(tree.topLevelItemCount())]
            assert labels[0].startswith("[DIR] "), (order, labels)
            assert labels[1].startswith("[DIR] "), (order, labels)
            assert not labels[2].startswith("[DIR] "), (order, labels)

    def test_enable_name_sorting_defaults_to_ascending(self, app):
        from PySide6.QtCore import Qt
        from llama_launcher.gui.dialogs.path_picker import (
            enable_name_sorting, populate_file_tree,
        )
        tree = self._tree(app)
        enable_name_sorting(tree)
        assert tree.header().sortIndicatorOrder() == Qt.AscendingOrder
        populate_file_tree(tree, self._entries())
        labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert labels == ["[DIR] alpha", "[DIR] zeta", "model.gguf", "notes.txt"]

    def test_size_column_sorts_numerically(self, app):
        from PySide6.QtCore import Qt
        from llama_launcher.gui.dialogs.path_picker import populate_file_tree
        tree = self._tree(app)
        tree.setSortingEnabled(True)
        populate_file_tree(tree, [
            FileEntry(name="big", path="/d/big", size_bytes=900000),
            FileEntry(name="small", path="/d/small", size_bytes=5),
        ])
        tree.sortItems(1, Qt.AscendingOrder)
        # Text sorting would put "900000 B"-style values before "5 B".
        assert tree.topLevelItem(0).text(0) == "small"


class TestProcInfoParsers:
    def test_parse_listener_pid(self):
        from llama_launcher.remote.procinfo import parse_listener_pid
        line = ('LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:* '
                'users:(("llama-server",pid=4242,fd=3))')
        assert parse_listener_pid(line) == 4242
        assert parse_listener_pid("") is None
        # A listener owned by another user shows no pid= field.
        assert parse_listener_pid("LISTEN 0 4096 0.0.0.0:8080 0.0.0.0:*") is None

    def test_parse_ps_stats(self):
        from llama_launcher.remote.procinfo import parse_ps_stats
        cpu, ram_mb, uptime = parse_ps_stats(" 12.5 2097152 3600 ")
        assert cpu == 12.5
        assert ram_mb == pytest.approx(2048.0)   # RSS is KiB
        assert uptime == 3600.0
        assert parse_ps_stats("") == (0.0, 0.0, 0.0)
        assert parse_ps_stats("garbage output") == (0.0, 0.0, 0.0)

    def test_parse_gpu_stats_multi_gpu(self):
        from llama_launcher.remote.procinfo import parse_gpu_stats
        # Max utilization, summed memory: a split model's true footprint.
        assert parse_gpu_stats("30, 1024\n70, 2048\n") == (70.0, 3072.0)
        assert parse_gpu_stats("") == (0.0, 0.0)

    def test_status_probe_round_trips(self):
        from llama_launcher.remote.procinfo import (
            build_status_probe, split_status_probe,
        )
        probe = build_status_probe(999, 8080)
        assert "ps -p 999" in probe and "sport = :8080" in probe
        ps, ss, gpu = split_status_probe(
            "@@LL@@\n 1.0 100 5\n@@LL@@\nLISTEN pid=7\n@@LL@@\n10, 20\n")
        assert ps == "1.0 100 5"
        assert "pid=7" in ss
        assert gpu == "10, 20"

    def test_status_probe_without_pid_is_still_valid(self):
        from llama_launcher.remote.procinfo import build_status_probe
        assert "ps -p" not in build_status_probe(None, 8080)

    def test_split_handles_truncated_output(self):
        from llama_launcher.remote.procinfo import split_status_probe
        assert split_status_probe("") == ("", "", "")
        assert split_status_probe("@@LL@@\nonly ps\n") == ("only ps", "", "")


class TestMultipleInstancesPerHost:
    """Several llama-servers on one host must stay individually addressable."""

    def test_pid_file_name_is_per_server(self):
        from llama_launcher.remote.types import pid_file_name
        assert pid_file_name("inst-a") != pid_file_name("inst-b")
        assert pid_file_name("inst-a") == ".llama_launcher_inst-a.pid"

    def test_pid_file_name_sanitizes_awkward_names(self):
        from llama_launcher.remote.types import pid_file_name
        name = pid_file_name("My Server / 2 (GPU)")
        assert "/" not in name and " " not in name
        assert name.startswith(".llama_launcher_") and name.endswith(".pid")
        assert pid_file_name("") == ".llama_launcher_server.pid"

    def test_get_server_pid_does_not_fall_back_to_process_name(self, tmp_path):
        # A different llama-server on the box must never be reported as ours:
        # it would show a stopped server as running with someone else's stats.
        conn = LocalServerConnection(ServerConfig(
            friendly_name="mine", working_directory=str(tmp_path), server_port=18999))
        with patch.object(conn, "port_listener_pid", return_value=None):
            assert conn.get_server_pid() is None

    def test_get_server_pid_uses_the_port_listener(self, tmp_path):
        conn = LocalServerConnection(ServerConfig(
            friendly_name="mine", working_directory=str(tmp_path), server_port=18999))
        with patch.object(conn, "port_listener_pid", return_value=os.getpid()):
            assert conn.get_server_pid() == os.getpid()

    def test_port_discovered_pid_is_not_treated_as_tracked(self, tmp_path):
        # Only processes we started may be signalled by stop_server.
        conn = LocalServerConnection(ServerConfig(
            friendly_name="mine", working_directory=str(tmp_path), server_port=18999))
        with patch.object(conn, "port_listener_pid", return_value=os.getpid()):
            assert conn.get_server_pid() == os.getpid()
            assert conn.tracked_pid_alive() is False

    def test_ssh_pid_lookup_never_uses_pgrep(self):
        # `pgrep -f llama-server` also matched the shell running it.
        conn = SSHServerConnection(_remote_config())
        issued = []

        def fake_run(cmd, capture=False, timeout=30):
            issued.append(cmd)
            return (False, "")

        conn._run_remote = fake_run
        conn._remote_home = "/home/bob"
        conn.get_server_pid()
        assert not any("pgrep" in c for c in issued), issued

    def test_find_port_conflicts_same_host(self):
        mgr = ServerManager()
        mgr.add_server(ServerConfig(friendly_name="a", server_port=8080), persist=False)
        mgr.add_server(ServerConfig(friendly_name="b", server_port=8081), persist=False)
        new = ServerConfig(friendly_name="c", server_port=8080)
        assert mgr.find_port_conflicts(new) == ["a"]
        assert mgr.find_port_conflicts(
            ServerConfig(friendly_name="c", server_port=9999)) == []

    def test_find_port_conflicts_ignores_other_hosts(self):
        mgr = ServerManager()
        mgr.add_server(_remote_config(friendly_name="remote-a", server_port=8080),
                       persist=False)
        # Same port, different machine: not a conflict.
        local = ServerConfig(friendly_name="local-a", server_port=8080)
        assert mgr.find_port_conflicts(local) == []

    def test_find_port_conflicts_excludes_self(self):
        mgr = ServerManager()
        cfg = ServerConfig(friendly_name="a", server_port=8080)
        mgr.add_server(cfg, persist=False)
        assert mgr.find_port_conflicts(cfg) == []


class TestRemotePathResolution:
    """`~` must resolve against the remote user's home, not the launcher's."""

    def _conn(self, working_directory):
        conn = SSHServerConnection(_remote_config(
            working_directory=working_directory, friendly_name="rem"))
        conn._remote_home = "/home/bob"       # pre-cached; no ssh call
        return conn

    def test_bare_tilde(self):
        assert self._conn("~")._remote_working_dir() == "/home/bob"

    def test_tilde_prefix(self):
        assert self._conn("~/llama")._remote_working_dir() == "/home/bob/llama"

    def test_absolute_path_untouched(self):
        assert self._conn("/srv/llama")._remote_working_dir() == "/srv/llama"

    def test_log_and_pid_paths_are_remote(self):
        conn = self._conn("~")
        assert conn._remote_log_path() == "/home/bob/llama-server.log"
        assert conn._remote_pid_file() == "/home/bob/.llama_launcher_rem.pid"

    def test_home_is_not_the_launchers_home(self):
        conn = self._conn("~")
        assert conn._remote_working_dir() != os.path.expanduser("~")


class TestLogFollowArgv:
    def test_local_follows_the_configured_log(self, tmp_path):
        conn = LocalServerConnection(ServerConfig(
            friendly_name="l", working_directory=str(tmp_path),
            log_file="server.log"))
        argv = conn.log_follow_argv(25)
        assert argv[0] == "tail"
        assert "-F" in argv                      # -F, not -f: survives restarts
        assert argv[-1] == str(tmp_path / "server.log")
        assert "25" in argv

    def test_remote_follows_through_ssh(self):
        conn = SSHServerConnection(_remote_config(log_file="server.log"))
        conn._remote_home = "/home/bob"
        argv = conn.log_follow_argv(25)
        assert argv[0] == "ssh"
        assert argv[-1].startswith("tail -n 25 -F ")
        assert "/srv/server.log" in argv[-1]

    def test_remote_log_path_uses_remote_home(self):
        conn = SSHServerConnection(
            _remote_config(working_directory="~", log_file="server.log"))
        conn._remote_home = "/home/bob"
        assert "/home/bob/server.log" in conn.log_follow_argv(0)[-1]


class TestBasicTabPickerDispatch:
    """The model Browse buttons must target the selected server's filesystem."""

    def _tab(self, app):
        from llama_launcher.core.config import LlamaConfig
        from llama_launcher.gui.config_controller import ConfigController
        from llama_launcher.gui.tabs.basic_tab import BasicTab
        config = LlamaConfig()
        return BasicTab(ConfigController(config, None))

    def test_remote_server_uses_the_remote_browser(self, app, monkeypatch):
        import llama_launcher.gui.tabs.basic_tab as basic_tab

        mgr = ServerManager()
        mgr.add_server(_remote_config(friendly_name="rem",
                                      model_directory="/srv/models"), persist=False)
        tab = self._tab(app)
        tab.set_browse_context(mgr, lambda: "rem")

        captured = {}

        def fake_get_path(parent, connection, mode="dir", start_path="",
                          title="", name_filter=None):
            captured.update(mode=mode, start_path=start_path,
                            name_filter=name_filter, conn=connection)
            return "/srv/models/remote.gguf"

        monkeypatch.setattr(basic_tab.RemotePathDialog, "get_path", fake_get_path)
        monkeypatch.setattr(
            basic_tab.QFileDialog, "getOpenFileName",
            lambda *a, **k: pytest.fail("must not open a local dialog for a remote server"))

        tab._browse_model()
        assert tab.model_combo.currentText() == "/srv/models/remote.gguf"
        assert captured["mode"] == "file"
        assert captured["start_path"] == "/srv/models"
        assert "*.gguf" in captured["name_filter"]
        assert isinstance(captured["conn"], SSHServerConnection)

    def test_local_server_uses_the_native_dialog(self, app, monkeypatch):
        import llama_launcher.gui.tabs.basic_tab as basic_tab

        mgr = ServerManager()
        mgr.add_server(ServerConfig(friendly_name="loc", llama_server_path="echo"),
                       persist=False)
        tab = self._tab(app)
        tab.set_browse_context(mgr, lambda: "loc")

        monkeypatch.setattr(
            basic_tab.RemotePathDialog, "get_path",
            lambda *a, **k: pytest.fail("must not open the remote browser locally"))
        monkeypatch.setattr(basic_tab.QFileDialog, "getOpenFileName",
                            lambda *a, **k: ("/home/me/local.gguf", ""))

        tab._browse_model()
        assert tab.model_combo.currentText() == "/home/me/local.gguf"

    def test_without_browse_context_falls_back_to_local(self, app, monkeypatch):
        import llama_launcher.gui.tabs.basic_tab as basic_tab

        tab = self._tab(app)  # never given a manager, as in standalone use
        monkeypatch.setattr(basic_tab.QFileDialog, "getOpenFileName",
                            lambda *a, **k: ("/home/me/x.gguf", ""))
        tab._browse_draft()
        assert tab.draft_combo.currentText() == "/home/me/x.gguf"

    def test_cancelled_dialog_leaves_field_untouched(self, app, monkeypatch):
        import llama_launcher.gui.tabs.basic_tab as basic_tab

        tab = self._tab(app)
        tab.model_combo.setEditText("/keep/me.gguf")
        monkeypatch.setattr(basic_tab.QFileDialog, "getOpenFileName",
                            lambda *a, **k: ("", ""))
        tab._browse_model()
        assert tab.model_combo.currentText() == "/keep/me.gguf"


class TestModelCacheMigration:
    def test_remove_models_drops_only_that_server(self):
        from llama_launcher.models.storage import (
            load_models, remove_models, save_models,
        )
        from llama_launcher.models.types import ModelInfo
        model = ModelInfo(file_path="/m/a.gguf", file_name="a.gguf",
                          file_size_mb=1.0, format="gguf")
        save_models("srv-a", [model])
        save_models("srv-b", [model])
        assert remove_models("srv-a") is True
        assert load_models("srv-a") == []
        assert len(load_models("srv-b")) == 1

    def test_remove_models_missing_server_is_false(self):
        from llama_launcher.models.storage import remove_models
        assert remove_models("never-existed") is False


class TestPickerWorkerLifetime:
    """A QThread destroyed while running aborts the process.

    The picker's listing threads must therefore outlive the dialog: a user can
    dismiss it mid-listing, and accept()/reject() never raise closeEvent, so
    there is no reliable moment to wait for them.
    """

    def _dialog(self, app, conn):
        from llama_launcher.gui.dialogs.path_picker import RemotePathDialog
        return RemotePathDialog(None, conn, mode="dir", start_path="/tmp")

    def _slow_conn(self, tmp_path):
        import time as _time
        conn = LocalServerConnection(ServerConfig(friendly_name="slow"))
        real_list = conn.list_directory

        def slow_list(path):
            _time.sleep(0.4)
            return real_list(str(tmp_path))

        conn.list_directory = slow_list
        return conn

    def test_listing_thread_is_not_parented_to_the_dialog(self, app, tmp_path):
        from llama_launcher.gui.dialogs.path_picker import _ACTIVE_WORKERS
        dialog = self._dialog(app, self._slow_conn(tmp_path))
        try:
            assert dialog._worker is not None
            # Parented to the dialog, it would be destroyed along with it.
            assert dialog._worker.parent() is None
            assert dialog._worker in _ACTIVE_WORKERS
        finally:
            dialog._worker.wait(5000)
            dialog.close()

    def test_accept_and_reject_mark_the_dialog_closing(self, app, tmp_path):
        # closeEvent does not fire for either, so done() carries the flag.
        for action in ("accept", "reject"):
            dialog = self._dialog(app, self._slow_conn(tmp_path))
            getattr(dialog, action)()
            assert dialog._closing is True, action
            dialog._worker.wait(5000)

    def test_worker_is_dropped_from_the_registry_when_finished(self, app, tmp_path):
        from llama_launcher.gui.dialogs.path_picker import _ACTIVE_WORKERS
        dialog = self._dialog(app, self._slow_conn(tmp_path))
        worker = dialog._worker
        worker.wait(5000)
        for _ in range(100):
            app.processEvents()
            if worker not in _ACTIVE_WORKERS:
                break
            time.sleep(0.01)
        assert worker not in _ACTIVE_WORKERS
        dialog.close()

    def test_stale_listing_does_not_replace_a_newer_one(self, app, tmp_path):
        # Listings return out of order; an older one must not yank the view back.
        dialog = self._dialog(app, self._slow_conn(tmp_path))
        dialog._worker.wait(5000)
        dialog._current_dir = "/current"
        stale_id = dialog._request_id
        dialog._request_id += 1               # a newer navigation started
        dialog._on_listed((stale_id, "/stale", []))
        assert dialog._current_dir == "/current"
        dialog.close()

    def test_current_listing_is_applied(self, app, tmp_path):
        dialog = self._dialog(app, self._slow_conn(tmp_path))
        dialog._worker.wait(5000)
        dialog._on_listed((dialog._request_id, "/applied", []))
        assert dialog._current_dir == "/applied"
        dialog.close()

    def test_wait_for_active_workers_drains_the_registry(self, app, tmp_path):
        from llama_launcher.gui.dialogs.path_picker import (
            _ACTIVE_WORKERS, wait_for_active_workers,
        )
        dialog = self._dialog(app, self._slow_conn(tmp_path))
        dialog.close()
        wait_for_active_workers(3000)
        assert len(_ACTIVE_WORKERS) == 0
