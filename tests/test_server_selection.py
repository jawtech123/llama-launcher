"""Tests for what changes (and what must not) when a server is selected.

Selecting where a server *runs* must not rewrite what the launch command
*binds to*, and the model browser must never keep showing another machine's
models.
"""
import pytest

from llama_launcher.core.builder import build_command
from llama_launcher.remote.manager import ServerManager
from llama_launcher.remote.types import ServerConfig


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    """A main window with networking and background probes stubbed out."""
    from llama_launcher.gui import main_window as mw

    # No connecting, no polling: these tests are about selection alone.
    monkeypatch.setattr(mw.LlamaLauncherGUI, "_deferred_init", lambda self: None)
    monkeypatch.setattr(mw.LlamaLauncherGUI, "_on_connect_requested",
                        lambda self, name: None)
    win = mw.LlamaLauncherGUI()

    manager = win.server_manager
    manager.add_server(ServerConfig(friendly_name="local-box", llama_server_path="echo"),
                       persist=False)
    manager.add_server(ServerConfig(friendly_name="remote-box", hostname="10.9.9.9",
                                    username="bob", server_port=9099), persist=False)
    win._refresh_server_combo()
    win._refresh_browser_server_combo()
    yield win
    win._closing = True
    win.deleteLater()


class TestHostFieldOnServerSelection:
    def test_ssh_hostname_never_becomes_the_bind_address(self, window):
        window.controller.set("model_path", "/m/x.gguf")
        window._on_server_selected("remote-box")
        assert window.config.host != "10.9.9.9"
        assert "10.9.9.9" not in build_command(window.config)

    def test_loopback_is_promoted_to_all_interfaces_for_a_remote_server(self, window):
        # Binding a remote server to loopback makes it unreachable from here.
        window.controller.set("host", "127.0.0.1")
        window._on_server_selected("remote-box")
        assert window.config.host == "0.0.0.0"

    def test_deliberate_bind_address_is_preserved(self, window):
        window.controller.set("host", "10.1.2.3")
        window._on_server_selected("remote-box")
        assert window.config.host == "10.1.2.3"

    def test_local_server_leaves_loopback_alone(self, window):
        window.controller.set("host", "127.0.0.1")
        window._on_server_selected("local-box")
        assert window.config.host == "127.0.0.1"

    def test_port_still_follows_the_server(self, window):
        window._on_server_selected("remote-box")
        assert window.config.port == 9099


class TestModelBrowserOnServerSelection:
    def _rows(self, window):
        widget = window.models_tab.model_list
        return [widget.item(i).text() for i in range(widget.count())]

    def test_remote_server_without_model_dirs_clears_the_list(self, window, tmp_path):
        # Populate the browser as if a local scan had run.
        from llama_launcher.models.types import ModelInfo
        window.models = [ModelInfo(file_path="/local/a.gguf", file_name="a.gguf",
                                   file_size_mb=1.0, format="gguf")]
        window._populate_model_lists()
        assert any("a.gguf" in r for r in self._rows(window))

        window.basic_tab.server_combo.setCurrentText("remote-box")
        window.models_tab.browser_server_combo.setCurrentText("remote-box")
        window.scan_models()

        rows = self._rows(window)
        assert not any("a.gguf" in r for r in rows), "local models survived the switch"
        assert any("No model directories configured" in r for r in rows), rows

    def test_placeholder_names_the_server_and_where_to_fix_it(self, window):
        window.models_tab.browser_server_combo.setCurrentText("remote-box")
        window.scan_models()
        text = " ".join(self._rows(window))
        assert "remote-box" in text
        assert "Model Dirs" in text

    def test_local_folders_are_not_scanned_for_a_remote_server(self, window, tmp_path):
        (tmp_path / "local-only.gguf").write_bytes(b"x")
        window.models_tab.folders_list.addItem(str(tmp_path))
        conn = window.server_manager.get_connection("remote-box")
        conn.config.model_directories = ["/srv/models"]

        captured = {}

        def fake_worker(manager, server_name, dirs, is_local_only):
            captured["dirs"] = dirs
            raise RuntimeError("stop before starting a thread")

        from llama_launcher.gui import main_window as mw
        original = mw.ModelScanWorker
        mw.ModelScanWorker = fake_worker
        try:
            window.models_tab.browser_server_combo.setCurrentText("remote-box")
            with pytest.raises(RuntimeError):
                window.scan_models()
        finally:
            mw.ModelScanWorker = original
        # This machine's folder list says nothing about the remote host.
        assert captured["dirs"] == ["/srv/models"]
        assert str(tmp_path) not in captured["dirs"]

    def test_placeholder_row_is_not_selectable(self, window):
        from PySide6.QtCore import Qt
        window.models_tab.show_placeholder("nothing here")
        item = window.models_tab.model_list.item(0)
        assert item.flags() == Qt.NoItemFlags
