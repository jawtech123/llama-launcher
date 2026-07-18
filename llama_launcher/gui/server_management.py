"""Server Management GUI Components."""
import logging
import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QDialog, QFileDialog,
    QMessageBox, QInputDialog, QSplitter, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QHeaderView, QAbstractItemView,
    QTreeWidget, QTreeWidgetItem, QProgressBar, QProgressDialog, QFrame,
    QScrollArea, QTableWidget, QTableWidgetItem, QColumnView,
    QListView,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QTextCursor

from llama_launcher.remote.types import (
    ServerConfig, ServerStatus, FileEntry,
    ConnectionType, AuthMethod, ConnectionState,
    ServerProcessState,
)
from llama_launcher.remote.manager import ServerManager
from llama_launcher.remote.ssh_key import (
    generate_key_pair, install_key_to_server,
    default_key_exists, DEFAULT_KEY_DIR, DEFAULT_KEY_NAME,
)

logger = logging.getLogger(__name__)


class _ServerStateWorker(QThread):
    """Check connection states for all servers in background."""
    done = Signal(list)

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def run(self):
        results = []
        for name in self.manager.get_all_names():
            try:
                state = self.manager.get_connection_state(name)
            except Exception:
                state = ConnectionState.DISCONNECTED
            results.append((name, state))
        self.done.emit(results)


class _ServerStatusWorker(QThread):
    """Fetch server status in background."""
    done = Signal(object, str)

    def __init__(self, manager, server_name):
        super().__init__()
        self.manager = manager
        self.server_name = server_name

    def run(self):
        try:
            status = self.manager.get_status(self.server_name)
            self.done.emit(status, "")
        except Exception as e:
            self.done.emit(None, str(e))


class _TaskWorker(QThread):
    """Run an arbitrary callable in a background thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:
            logger.debug("Background task failed: %s", e, exc_info=True)
            self.failed.emit(str(e))


class _TransferWorker(_TaskWorker):
    """A task worker with a progress signal for file transfers."""

    progress = Signal(int)


class _FileTreeItem(QTreeWidgetItem):
    """Tree item that sorts the Size column numerically (bytes), not as text."""

    def __lt__(self, other):
        tree = self.treeWidget()
        if tree is not None and tree.sortColumn() == 1:
            a = self.data(0, Qt.UserRole + 2) or 0
            b = other.data(0, Qt.UserRole + 2) or 0
            return a < b
        return super().__lt__(other)


class ServerEditDialog(QDialog):
    """Dialog for adding or editing a server configuration."""

    def __init__(self, parent=None, config: Optional[ServerConfig] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Server" if not config else "Edit Server")
        self.setMinimumWidth(500)
        self._config = config or ServerConfig()
        self._build_ui()
        self._populate_fields()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        general = QGroupBox("Connection")
        general_layout = QFormLayout(general)
        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Local", "SSH"])
        self.host_edit = QLineEdit("localhost")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.user_edit = QLineEdit()
        self.auth_combo = QComboBox()
        self.auth_combo.addItems(["SSH Key", "Password"])
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.key_path_edit = QLineEdit()
        self.key_browse_btn = QPushButton("Browse...")
        self.key_browse_btn.clicked.connect(self._browse_key)
        key_layout = QHBoxLayout()
        key_layout.addWidget(self.key_path_edit)
        key_layout.addWidget(self.key_browse_btn)
        general_layout.addRow("Name:", self.name_edit)
        general_layout.addRow("Type:", self.type_combo)
        general_layout.addRow("Host:", self.host_edit)
        general_layout.addRow("Port:", self.port_spin)
        general_layout.addRow("Username:", self.user_edit)
        general_layout.addRow("Auth:", self.auth_combo)
        general_layout.addRow("Password:", self.password_edit)
        general_layout.addRow("Key Path:", key_layout)
        layout.addWidget(general)

        paths = QGroupBox("Paths")
        paths_layout = QFormLayout(paths)
        self.working_dir_edit = QLineEdit("~")
        self.server_path_edit = QLineEdit("llama-server")
        self.log_file_edit = QLineEdit("llama-server.log")
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1024, 65535)
        self.server_port_spin.setValue(8080)
        paths_layout.addRow("Working Dir:", self.working_dir_edit)
        paths_layout.addRow("Server Exec:", self.server_path_edit)
        paths_layout.addRow("Log File:", self.log_file_edit)
        paths_layout.addRow("Server Port:", self.server_port_spin)

        model_dirs_widget = QWidget()
        model_dirs_h = QHBoxLayout(model_dirs_widget)
        model_dirs_h.setContentsMargins(0, 0, 0, 0)
        self.model_dirs_list = QListWidget()
        self.model_dirs_list.setMaximumHeight(100)
        model_dirs_btns = QVBoxLayout()
        model_dirs_btns.setSpacing(2)
        self.add_model_dir_btn = QPushButton("Add Dir...")
        self.add_model_dir_btn.clicked.connect(self._add_model_dir)
        self.remove_model_dir_btn = QPushButton("Remove")
        self.remove_model_dir_btn.clicked.connect(self._remove_model_dir)
        model_dirs_btns.addWidget(self.add_model_dir_btn)
        model_dirs_btns.addWidget(self.remove_model_dir_btn)
        model_dirs_btns.addStretch()
        model_dirs_h.addWidget(self.model_dirs_list, 1)
        model_dirs_h.addLayout(model_dirs_btns)
        paths_layout.addRow("Model Dirs:", model_dirs_widget)
        layout.addWidget(paths)

        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        layout.addWidget(self.enabled_check)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.auth_combo.currentIndexChanged.connect(self._on_auth_changed)

    def _on_type_changed(self):
        is_local = self.type_combo.currentIndex() == 0
        self.host_edit.setEnabled(not is_local)
        self.port_spin.setEnabled(not is_local)
        self.user_edit.setEnabled(not is_local)
        self.auth_combo.setEnabled(not is_local)
        self.password_edit.setEnabled(not is_local)
        self.key_path_edit.setEnabled(not is_local)
        self.key_browse_btn.setEnabled(not is_local)

    def _on_auth_changed(self):
        is_key = self.auth_combo.currentIndex() == 0
        self.password_edit.setEnabled(not is_key)
        self.key_path_edit.setEnabled(is_key)
        self.key_browse_btn.setEnabled(is_key)

    def _browse_key(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Private Key", str(Path.home() / ".ssh")
        )
        if path:
            self.key_path_edit.setText(path)

    def _browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path:
            self.model_dirs_list.addItem(path)

    def _add_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Add Model Directory")
        if path:
            existing = {
                self.model_dirs_list.item(i).text()
                for i in range(self.model_dirs_list.count())
            }
            if path not in existing:
                self.model_dirs_list.addItem(path)

    def _remove_model_dir(self):
        current = self.model_dirs_list.currentRow()
        if current >= 0:
            self.model_dirs_list.takeItem(current)

    def _populate_fields(self):
        self.name_edit.setText(self._config.friendly_name)
        self.type_combo.setCurrentIndex(
            0 if self._config.connection_type == ConnectionType.LOCAL else 1
        )
        self.host_edit.setText(self._config.hostname)
        self.port_spin.setValue(self._config.port)
        self.user_edit.setText(self._config.username)
        self.auth_combo.setCurrentIndex(
            0 if self._config.auth_method == AuthMethod.SSH_KEY else 1
        )
        self.password_edit.setText(self._config.password)
        self.key_path_edit.setText(self._config.private_key_path)
        self.working_dir_edit.setText(self._config.working_directory)
        self.server_path_edit.setText(self._config.llama_server_path)
        self.log_file_edit.setText(self._config.log_file)
        self.server_port_spin.setValue(self._config.server_port)
        self.enabled_check.setChecked(self._config.enabled)

        self.model_dirs_list.clear()
        dirs = self._config.model_directories or []
        if self._config.model_directory and self._config.model_directory not in dirs:
            dirs = [self._config.model_directory] + dirs
        for d in dirs:
            self.model_dirs_list.addItem(d)

        self._on_type_changed()

    def get_config(self) -> ServerConfig:
        model_dirs = [
            self.model_dirs_list.item(i).text()
            for i in range(self.model_dirs_list.count())
        ]
        return ServerConfig(
            friendly_name=self.name_edit.text().strip(),
            connection_type=ConnectionType.LOCAL
                if self.type_combo.currentIndex() == 0
                else ConnectionType.SSH,
            hostname=self.host_edit.text().strip(),
            port=self.port_spin.value(),
            username=self.user_edit.text().strip(),
            auth_method=AuthMethod.SSH_KEY
                if self.auth_combo.currentIndex() == 0
                else AuthMethod.PASSWORD,
            password=self.password_edit.text(),
            private_key_path=self.key_path_edit.text().strip(),
            working_directory=self.working_dir_edit.text().strip(),
            llama_server_path=self.server_path_edit.text().strip(),
            model_directory=model_dirs[0] if model_dirs else "",
            model_directories=model_dirs if model_dirs else None,
            enabled=self.enabled_check.isChecked(),
            log_file=self.log_file_edit.text().strip(),
            server_port=self.server_port_spin.value(),
        )


class SSHKeyInstallDialog(QDialog):
    """Dialog to install an SSH key onto a remote server."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install SSH Key")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Enter the server details and password to install your SSH key.\n"
            "After installation, you can switch to key-based authentication."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.host_edit = QLineEdit()
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.user_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.key_path_edit = QLineEdit(str(DEFAULT_KEY_DIR / DEFAULT_KEY_NAME))
        self.key_browse_btn = QPushButton("Browse...")
        self.key_browse_btn.clicked.connect(self._browse_key)
        key_layout = QHBoxLayout()
        key_layout.addWidget(self.key_path_edit)
        key_layout.addWidget(self.key_browse_btn)
        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("Username:", self.user_edit)
        form.addRow("Password:", self.password_edit)
        form.addRow("Key Path:", key_layout)
        layout.addLayout(form)

        self.gen_key_check = QCheckBox("Generate new key if none exists")
        self.gen_key_check.setChecked(True)
        layout.addWidget(self.gen_key_check)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.install_btn = QPushButton("Install Key")
        self.install_btn.clicked.connect(self._install)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.install_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def _browse_key(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Private Key", str(Path.home() / ".ssh")
        )
        if path:
            self.key_path_edit.setText(path)

    def _install(self):
        hostname = self.host_edit.text().strip()
        port = self.port_spin.value()
        username = self.user_edit.text().strip()
        password = self.password_edit.text()

        if not all([hostname, username, password]):
            QMessageBox.warning(self, "Missing Fields",
                                "Host, username, and password are required.")
            return

        self.install_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("Connecting...")

        QTimer.singleShot(50, self._do_install)

    def _do_install(self):
        hostname = self.host_edit.text().strip()
        port = self.port_spin.value()
        username = self.user_edit.text().strip()
        password = self.password_edit.text()
        key_path = self.key_path_edit.text().strip()

        generate_new = self.gen_key_check.isChecked()
        if generate_new and not Path(key_path).expanduser().exists():
            try:
                generate_key_pair(key_path)
            except Exception as e:
                self.status_label.setText(f"Key generation failed: {e}")
                self._done(False)
                return

        self.status_label.setText("Installing key...")
        self._install_worker = _TaskWorker(
            lambda: self._install_task(hostname, port, username, password, key_path),
            self,
        )
        self._install_worker.done.connect(self._on_install_done)
        self._install_worker.failed.connect(self._on_install_failed)
        self._install_worker.finished.connect(self._install_worker.deleteLater)
        self._install_worker.start()

    @staticmethod
    def _install_task(hostname, port, username, password, key_path) -> bool:
        from llama_launcher.remote.ssh_key import read_public_key
        pub_key = read_public_key(key_path)
        return install_key_to_server(hostname, port, username, password, pub_key, key_path)

    def _on_install_done(self, success):
        if success:
            self.status_label.setText("Key installed and verified successfully!")
        else:
            self.status_label.setText("Installation failed or verification failed.")
        self._done(bool(success))

    def _on_install_failed(self, message):
        self.status_label.setText(f"Error: {message}")
        self._done(False)

    def _done(self, success: bool):
        self.progress.setVisible(False)
        self.install_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success",
                                    "SSH key installed successfully.")


class ServerListWidget(QWidget):
    """Left panel: list of configured servers."""

    server_selected = Signal(str)
    servers_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manager: Optional[ServerManager] = None
        self._workers: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.addWidget(QLabel("Servers"))
        header.addStretch()
        self.add_btn = QPushButton("+")
        self.add_btn.setToolTip("Add Server")
        self.add_btn.setFixedWidth(32)
        self.add_btn.clicked.connect(self._on_add)
        header.addWidget(self.add_btn)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(60)
        self.refresh_btn.clicked.connect(self._refresh_list)
        header.addWidget(self.refresh_btn)
        layout.addLayout(header)

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selected)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self._on_edit)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.remove_btn)
        layout.addLayout(btn_layout)

    def set_manager(self, manager: ServerManager):
        self._manager = manager
        # Defer refresh to avoid blocking the main thread with subprocess calls
        QTimer.singleShot(10, self._refresh_list)

    def _refresh_list(self):
        if not self._manager or getattr(self, '_closing', False):
            return
        self._manager.load_saved_servers()
        worker = _ServerStateWorker(self._manager)
        worker.setParent(self)
        worker.done.connect(self._on_states_done)
        worker.finished.connect(worker.deleteLater)
        self._track(worker)
        worker.start()

    def _track(self, worker) -> None:
        self._workers.append(worker)
        worker.finished.connect(lambda: self._untrack(worker))

    def _untrack(self, worker) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    def shutdown(self) -> None:
        """Wait for any running background workers (called on app close)."""
        self._closing = True
        for w in list(self._workers):
            if w.isRunning():
                w.wait(1000)

    def _on_states_done(self, results):
        self.list_widget.clear()
        for name, state in results:
            conn = self._manager.get_connection(name)
            config = conn.config if conn else None
            icon = "○ "
            if state == ConnectionState.CONNECTED:
                icon = "● "
            elif state == ConnectionState.ERROR:
                icon = "✗ "
            type_label = "[SSH]" if config and config.connection_type == ConnectionType.SSH else "[Local]"
            disabled = config is not None and not config.enabled
            text = f"{icon}{type_label} {name}"
            if disabled:
                text += " (disabled)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)  # robust identity; never parse display text
            if disabled:
                item.setForeground(Qt.gray)
            self.list_widget.addItem(item)

    def _on_selected(self, current, previous):
        if current:
            name = current.data(Qt.UserRole)
            if name:
                self.server_selected.emit(name)

    def _on_add(self):
        if not self._manager:
            return
        dialog = ServerEditDialog(self)
        if dialog.exec() == QDialog.Accepted:
            config = dialog.get_config()
            if not config.friendly_name:
                QMessageBox.warning(self, "Validation", "Server name is required.")
                return
            self._manager.add_server(config)
            self._refresh_list()
            self.servers_changed.emit()

    def _on_edit(self):
        current = self.list_widget.currentItem()
        if not current or not self._manager:
            return
        name = current.data(Qt.UserRole)
        conn = self._manager.get_connection(name) if name else None
        if not conn:
            return
        dialog = ServerEditDialog(self, conn.config)
        if dialog.exec() == QDialog.Accepted:
            config = dialog.get_config()
            self._manager.remove_server(name)
            self._manager.add_server(config)
            self._refresh_list()
            self.servers_changed.emit()

    def _on_remove(self):
        current = self.list_widget.currentItem()
        if not current or not self._manager:
            return
        name = current.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Remove Server",
            f"Remove server '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._manager.remove_server(name)
            self._refresh_list()
            self.servers_changed.emit()

    @staticmethod
    def _extract_name(text: str) -> str:
        """Legacy display-text parsing; kept for backwards compatibility."""
        for prefix in ("● ", "✗ ", "○ "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        if text.startswith("[SSH] "):
            text = text[6:]
        if text.startswith("[Local] "):
            text = text[8:]
        if text.endswith(" (disabled)"):
            text = text[:-11]
        return text


class ServerStatusWidget(QWidget):
    """Right panel: status and controls for the selected server."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manager: Optional[ServerManager] = None
        self._selected_server: Optional[str] = None
        self._status_timer: Optional[QTimer] = None
        self._live_console = None
        self._command_provider = None
        self._workers: list = []
        self._build_ui()

    def set_command_provider(self, provider) -> None:
        """Set a callable returning the current llama-server command string."""
        self._command_provider = provider

    def _launch_command(self) -> str:
        """Command used for Start/Restart: the launcher's current config.

        Falls back to a bare server binary if no provider is wired. The
        configured server binary replaces argv[0] when it differs from the
        default "llama-server".
        """
        conn = self._manager.get_connection(self._selected_server) if self._manager else None
        binary = conn.config.llama_server_path if conn else "llama-server"

        if self._command_provider is not None:
            try:
                cmd = self._command_provider()
            except Exception:
                logger.exception("Command provider failed")
                return binary
            if binary and binary != "llama-server":
                parts = cmd.split(" ", 1)
                return binary + (" " + parts[1] if len(parts) > 1 else "")
            return cmd
        return binary

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.server_name_label = QLabel("No server selected")
        self.server_name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.server_name_label)

        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QFormLayout(status_frame)
        self.state_label = QLabel("-")
        self.pid_label = QLabel("-")
        self.cpu_label = QLabel("-")
        self.ram_label = QLabel("-")
        self.gpu_util_label = QLabel("-")
        self.gpu_mem_label = QLabel("-")
        self.port_label = QLabel("-")
        self.uptime_label = QLabel("-")
        self.model_label = QLabel("-")
        status_layout.addRow("State:", self.state_label)
        status_layout.addRow("PID:", self.pid_label)
        status_layout.addRow("CPU:", self.cpu_label)
        status_layout.addRow("RAM:", self.ram_label)
        status_layout.addRow("GPU Util:", self.gpu_util_label)
        status_layout.addRow("GPU Mem:", self.gpu_mem_label)
        status_layout.addRow("Port:", self.port_label)
        status_layout.addRow("Uptime:", self.uptime_label)
        status_layout.addRow("Model:", self.model_label)
        layout.addWidget(status_frame)

        control_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        self.start_btn = QPushButton("Start Server")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("Stop Server")
        self.stop_btn.clicked.connect(self._on_stop)
        self.restart_btn = QPushButton("Restart Server")
        self.restart_btn.clicked.connect(self._on_restart)
        self.refresh_status_btn = QPushButton("Refresh Status")
        self.refresh_status_btn.clicked.connect(self._refresh_status)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.disconnect_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.restart_btn)
        control_layout.addWidget(self.refresh_status_btn)
        layout.addLayout(control_layout)

        ssh_layout = QHBoxLayout()
        self.install_key_btn = QPushButton("Install SSH Key")
        self.install_key_btn.clicked.connect(self._on_install_key)
        self.gen_key_btn = QPushButton("Generate SSH Key")
        self.gen_key_btn.clicked.connect(self._on_generate_key)
        ssh_layout.addWidget(self.install_key_btn)
        ssh_layout.addWidget(self.gen_key_btn)
        layout.addLayout(ssh_layout)

        self.tabs = QTabWidget()

        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setFontFamily("monospace")
        console_layout.addWidget(self.console_text)
        console_btn_layout = QHBoxLayout()
        self.console_start_btn = QPushButton("Start Live Log")
        self.console_start_btn.clicked.connect(self._on_start_console)
        self.console_stop_btn = QPushButton("Stop Live Log")
        self.console_stop_btn.clicked.connect(self._on_stop_console)
        self.console_tail_btn = QPushButton("Tail 100 Lines")
        self.console_tail_btn.clicked.connect(self._on_tail)
        self.console_follow_btn = QPushButton("Follow")
        self.console_follow_btn.setCheckable(True)
        self.console_follow_btn.setChecked(True)
        self.console_follow_btn.setToolTip(
            "Auto-scroll to the newest log lines.\n"
            "Scrolling up pauses following; scroll back down or re-enable to resume.")
        self.console_follow_btn.toggled.connect(self._on_follow_toggled)
        self.console_clear_btn = QPushButton("Clear")
        self.console_clear_btn.clicked.connect(self.console_text.clear)
        console_btn_layout.addWidget(self.console_start_btn)
        console_btn_layout.addWidget(self.console_stop_btn)
        console_btn_layout.addWidget(self.console_tail_btn)
        console_btn_layout.addStretch()
        console_btn_layout.addWidget(self.console_follow_btn)
        console_btn_layout.addWidget(self.console_clear_btn)
        console_layout.addLayout(console_btn_layout)
        self.tabs.addTab(console_tab, "Live Console")

        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        file_toolbar = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Remote directory path")
        self.file_refresh_btn = QPushButton("Refresh")
        self.file_refresh_btn.clicked.connect(self._refresh_file_list)
        self.file_up_btn = QPushButton("Up")
        self.file_up_btn.clicked.connect(self._file_go_up)
        self.file_new_dir_btn = QPushButton("New Dir")
        self.file_new_dir_btn.clicked.connect(self._file_new_dir)
        self.file_delete_btn = QPushButton("Delete")
        self.file_delete_btn.clicked.connect(self._file_delete)
        self.file_upload_btn = QPushButton("Upload")
        self.file_upload_btn.clicked.connect(self._file_upload)
        self.file_download_btn = QPushButton("Download")
        self.file_download_btn.clicked.connect(self._file_download)
        file_toolbar.addWidget(self.file_path_edit)
        file_toolbar.addWidget(self.file_refresh_btn)
        file_toolbar.addWidget(self.file_up_btn)
        file_toolbar.addWidget(self.file_new_dir_btn)
        file_toolbar.addWidget(self.file_delete_btn)
        file_toolbar.addWidget(self.file_upload_btn)
        file_toolbar.addWidget(self.file_download_btn)
        file_layout.addLayout(file_toolbar)
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Size", "Modified"])
        self.file_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_tree.setSortingEnabled(True)
        self.file_tree.itemDoubleClicked.connect(self._on_file_double_clicked)
        file_layout.addWidget(self.file_tree)
        self.tabs.addTab(file_tab, "File Browser")

        layout.addWidget(self.tabs)

    def set_manager(self, manager: ServerManager):
        self._manager = manager
        if not self._status_timer:
            self._status_timer = QTimer(self)
            self._status_timer.timeout.connect(self._auto_refresh)
            self._status_timer.start(15000)

    def select_server(self, name: str):
        self._selected_server = name
        self.server_name_label.setText(name)
        self._refresh_status()

    def _refresh_status(self):
        if not self._manager or not self._selected_server:
            return
        if getattr(self, '_closing', False):
            return
        worker = _ServerStatusWorker(self._manager, self._selected_server)
        worker.setParent(self)
        worker.done.connect(self._on_status_done)
        worker.finished.connect(worker.deleteLater)
        self._workers.append(worker)
        worker.finished.connect(lambda: self._untrack(worker))
        worker.start()

    def _untrack(self, worker) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    def _run_task(self, fn, on_done=None, on_failed=None, worker_class=_TaskWorker):
        """Run ``fn`` in a background thread; route result/failure to callbacks."""
        worker = worker_class(fn, self)
        if on_done is not None:
            worker.done.connect(on_done)
        if on_failed is not None:
            worker.failed.connect(on_failed)
        else:
            worker.failed.connect(
                lambda msg: logger.warning("Background task failed: %s", msg))
        worker.finished.connect(worker.deleteLater)
        self._workers.append(worker)
        worker.finished.connect(lambda: self._untrack(worker))
        worker.start()
        return worker

    def shutdown(self) -> None:
        """Stop timers/console and wait for workers (called on app close)."""
        self._closing = True
        if self._status_timer:
            self._status_timer.stop()
        self._on_stop_console()
        for w in list(self._workers):
            if w.isRunning():
                w.wait(1000)

    def _on_status_done(self, status, error_msg):
        if error_msg:
            self.state_label.setText(f"Error: {error_msg}")
            return
        if status:
            self.state_label.setText(status.state.value)
            self.pid_label.setText(str(status.pid or "-"))
            self.cpu_label.setText(f"{status.cpu_percent:.1f}%")
            self.ram_label.setText(f"{status.ram_mb:.1f} MB")
            self.gpu_util_label.setText(f"{status.gpu_utilization:.1f}%")
            self.gpu_mem_label.setText(f"{status.gpu_memory_mb:.1f} MB")
            self.port_label.setText(str(status.listening_port or "-"))
            self.uptime_label.setText(self._format_uptime(status.uptime_seconds))
            self.model_label.setText(status.loaded_model or "-")
            if status.error_message:
                self.model_label.setText(f"Error: {status.error_message}")

    def _auto_refresh(self):
        if self._selected_server:
            self._refresh_status()

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        if seconds <= 0:
            return "-"
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{mins}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _on_connect(self):
        if not self._selected_server:
            return
        self.connect_btn.setEnabled(False)
        self.state_label.setText("Connecting...")
        self._run_task(
            lambda: self._manager.connect(self._selected_server),
            on_done=self._on_connect_done,
        )

    def _on_connect_done(self, result):
        self.connect_btn.setEnabled(True)
        if result:
            QMessageBox.information(self, "Connected",
                                    f"Connected to {self._selected_server}.")
            self._refresh_status()
        else:
            QMessageBox.warning(self, "Connection Failed",
                                f"Could not connect to {self._selected_server}.")
            self.state_label.setText("Connection failed")

    def _on_disconnect(self):
        if not self._selected_server:
            return
        self._run_task(
            lambda: self._manager.disconnect(self._selected_server),
            on_done=lambda _r: self._refresh_status(),
        )

    def _on_start(self):
        if not self._selected_server:
            return
        cmd = self._launch_command()
        if QMessageBox.question(
            self, "Start Server",
            f"Start llama-server on {self._selected_server}?\n\n{cmd}",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            self.start_btn.setEnabled(False)
            self._run_task(
                lambda: self._manager.start_server(self._selected_server, cmd),
                on_done=self._on_start_done,
            )

    def _on_start_done(self, result):
        self.start_btn.setEnabled(True)
        if not result:
            QMessageBox.warning(self, "Failed", "Could not start server.")
            self._refresh_status()
            return
        # Don't declare victory yet: verify the process survives startup
        # (bad model path/args kill llama-server within the first second).
        self.state_label.setText("Starting...")
        QTimer.singleShot(2500, self._verify_started)

    def _verify_started(self):
        """Confirm the launched process is alive; show the log if it died."""
        if not self._selected_server:
            return
        if self._manager.tracked_pid_alive(self._selected_server):
            self._refresh_status()
            return
        log = (self._manager.tail_log(self._selected_server, 40) or "").strip()
        detail = log[-1500:] if log else "(no log output)"
        QMessageBox.warning(
            self, "Server Failed to Start",
            f"llama-server exited right after launch on '{self._selected_server}'.\n\n"
            f"Log output:\n{detail}",
        )
        self._refresh_status()

    def _on_stop(self):
        if not self._selected_server:
            return
        self.stop_btn.setEnabled(False)
        self._run_task(
            lambda: self._manager.stop_server(self._selected_server),
            on_done=self._on_stop_done,
        )

    def _on_stop_done(self, _result):
        self.stop_btn.setEnabled(True)
        self._refresh_status()

    def _on_restart(self):
        if not self._selected_server:
            return
        cmd = self._launch_command()
        self.restart_btn.setEnabled(False)
        self._run_task(
            lambda: self._manager.restart_server(self._selected_server, cmd),
            on_done=self._on_restart_done,
        )

    def _on_restart_done(self, result):
        self.restart_btn.setEnabled(True)
        if result:
            QMessageBox.information(self, "Restarted", "Server restarted.")
        self._refresh_status()

    def _on_install_key(self):
        dialog = SSHKeyInstallDialog(self)
        dialog.exec()

    def _on_generate_key(self):
        if default_key_exists():
            QMessageBox.information(self, "Key Exists",
                                    "SSH key already exists at ~/.ssh/llama_launcher")
            return
        try:
            priv, pub = generate_key_pair()
            QMessageBox.information(self, "Key Generated",
                                    f"SSH key generated at {priv}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Key generation failed: {e}")

    def _on_start_console(self):
        if not self._selected_server:
            return
        from llama_launcher.remote.console import LiveConsole
        if self._live_console:
            self._live_console.stop()
        self._live_console = LiveConsole(self._manager, self._selected_server)
        self._live_console.line_received.connect(self._append_console_line)
        self._live_console.error_occurred.connect(self._console_error)
        self._live_console.disconnected.connect(self._console_disconnected)
        self._live_console.connected.connect(self._console_connected)
        self._live_console.start()

    def _on_stop_console(self):
        if self._live_console:
            self._live_console.stop()
            self._live_console = None

    def _on_tail(self):
        if not self._selected_server:
            return
        self._run_task(
            lambda: self._manager.tail_log(self._selected_server, 100),
            on_done=lambda log: self.console_text.setPlainText(log or ""),
        )

    def _append_console_line(self, line: str):
        sb = self.console_text.verticalScrollBar()
        at_bottom = sb is None or sb.value() >= sb.maximum() - 4

        # Always insert at the document end, not at the user-visible cursor.
        cursor = self.console_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_text.setTextCursor(cursor)
        self.console_text.insertPlainText(line + "\n")

        # Follow only when enabled and the user hasn't scrolled up.
        if self.console_follow_btn.isChecked() and at_bottom and sb is not None:
            sb.setValue(sb.maximum())

    def _on_follow_toggled(self, checked: bool):
        if checked:
            sb = self.console_text.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

    def _console_error(self, error: str):
        self._append_console_line(f"[ERROR] {error}")

    def _console_disconnected(self):
        self._append_console_line("[CONNECTION LOST]")

    def _console_connected(self):
        self._append_console_line("[CONNECTED - streaming logs]")

    def _refresh_file_list(self):
        if not self._selected_server:
            return
        remote_path = self.file_path_edit.text().strip()
        if not remote_path:
            conn = self._manager.get_connection(self._selected_server)
            if conn:
                remote_path = conn.config.model_directory or conn.config.working_directory
        if not remote_path:
            return
        self.file_path_edit.setText(remote_path)
        self.file_tree.setEnabled(False)
        self._run_task(
            lambda: self._manager.list_directory(self._selected_server, remote_path),
            on_done=self._populate_file_tree,
        )

    def _populate_file_tree(self, entries):
        self.file_tree.setEnabled(True)
        self.file_tree.clear()
        for entry in sorted(entries, key=lambda e: (not e.is_directory, e.name.lower())):
            item = _FileTreeItem(self.file_tree)
            item.setText(0, ("[DIR] " if entry.is_directory else "") + entry.name)
            if entry.is_directory:
                item.setText(1, "-")
            else:
                from llama_launcher.utils.helpers import format_bytes
                item.setText(1, format_bytes(entry.size_bytes))
            import datetime
            item.setText(2, datetime.datetime.fromtimestamp(
                entry.modified_time).strftime("%Y-%m-%d %H:%M"))
            item.setData(0, Qt.UserRole, entry.path)
            item.setData(0, Qt.UserRole + 1, entry.is_directory)
            item.setData(0, Qt.UserRole + 2, entry.size_bytes)

    def _on_file_double_clicked(self, item, _column=0):
        """Navigate into a directory on double-click."""
        is_dir = item.data(0, Qt.UserRole + 1)
        path = item.data(0, Qt.UserRole)
        if is_dir and path:
            self.file_path_edit.setText(path)
            self._refresh_file_list()

    def _file_go_up(self):
        path = self.file_path_edit.text().strip()
        if path:
            parent = str(Path(path).parent)
            self.file_path_edit.setText(parent)
            self._refresh_file_list()

    def _file_new_dir(self):
        path = self.file_path_edit.text().strip()
        if not path:
            return
        name, ok = QInputDialog.getItem(self, "New Directory", "Directory name:", [], 0, False)
        if ok and name:
            self._run_task(
                lambda: self._manager.create_directory(
                    self._selected_server, f"{path}/{name}"),
                on_done=lambda result: self._refresh_file_list() if result else None,
            )

    def _file_delete(self):
        item = self.file_tree.currentItem()
        if not item:
            return
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        reply = QMessageBox.question(
            self, "Delete", f"Delete '{item.text(0)}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._run_task(
                lambda: self._manager.delete_file(self._selected_server, path),
                on_done=lambda _r: self._refresh_file_list(),
            )

    def _file_upload(self):
        local, _ = QFileDialog.getOpenFileName(self, "Upload File")
        if not local:
            return
        remote = self.file_path_edit.text().strip()
        if not remote:
            conn = self._manager.get_connection(self._selected_server)
            remote = conn.config.model_directory if conn else "~"
        remote = f"{remote}/{Path(local).name}"
        try:
            total = os.path.getsize(local)
        except OSError:
            total = 0
        self._start_transfer(
            f"Uploading {Path(local).name}", total,
            lambda cb: self._manager.upload_file(
                self._selected_server, local, remote, progress_cb=cb),
        )

    def _file_download(self):
        item = self.file_tree.currentItem()
        if not item:
            return
        is_dir = item.data(0, Qt.UserRole + 1)
        if is_dir:
            QMessageBox.warning(self, "Error", "Cannot download directories.")
            return
        remote = item.data(0, Qt.UserRole)
        if not remote:
            return
        size_bytes = item.data(0, Qt.UserRole + 2) or 0
        local_dir = QFileDialog.getExistingDirectory(self, "Download to")
        if not local_dir:
            return
        self._start_transfer(
            f"Downloading {item.text(0)}", size_bytes,
            lambda cb: self._manager.download_file(
                self._selected_server, remote, f"{local_dir}/{item.text(0)}",
                progress_cb=cb),
        )

    def _start_transfer(self, title: str, total_bytes: int, fn):
        """Run a transfer in a worker with a cancellable progress dialog."""
        progress = QProgressDialog(title, "Cancel", 0, 100, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)
        progress.setValue(0)

        holder = {}

        def cb(done, total):
            if progress.wasCanceled():
                raise RuntimeError("Transfer cancelled")
            worker = holder.get("worker")
            if worker is not None:
                denom = total or total_bytes or 1
                worker.progress.emit(int(done * 100 / max(denom, 1)))

        worker = _TransferWorker(lambda: fn(cb), self)
        holder["worker"] = worker
        worker.progress.connect(progress.setValue)
        worker.done.connect(lambda result: self._on_transfer_done(progress, result))
        worker.failed.connect(lambda msg: self._on_transfer_failed(progress, msg))
        worker.finished.connect(worker.deleteLater)
        self._workers.append(worker)
        worker.finished.connect(lambda: self._untrack(worker))
        worker.start()

    def _on_transfer_done(self, progress, result):
        progress.close()
        if result:
            self._refresh_file_list()
        else:
            QMessageBox.warning(self, "Transfer Failed",
                                "The file transfer did not complete.")

    def _on_transfer_failed(self, progress, message):
        progress.close()
        if "cancelled" not in message.lower():
            QMessageBox.warning(self, "Transfer Error", message)


class ServerManagementWidget(QWidget):
    """Main server management panel: list + details side by side."""

    def __init__(self, manager: ServerManager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._build_ui()

    def set_command_provider(self, provider) -> None:
        """Wire a callable returning the current llama-server command string."""
        self.status_widget.set_command_provider(provider)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Horizontal)

        self.list_widget = ServerListWidget(self)
        self.list_widget.set_manager(self._manager)
        self.list_widget.server_selected.connect(self._on_server_selected)
        splitter.addWidget(self.list_widget)

        self.status_widget = ServerStatusWidget(self)
        self.status_widget.set_manager(self._manager)
        splitter.addWidget(self.status_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 750])

        layout.addWidget(splitter)

    def _on_server_selected(self, name: str):
        self.status_widget.select_server(name)

    def shutdown(self) -> None:
        """Stop child widgets' timers and background workers."""
        self.list_widget.shutdown()
        self.status_widget.shutdown()

    @Slot()
    def refresh(self):
        self.list_widget._refresh_list()
        self.status_widget._refresh_status()
