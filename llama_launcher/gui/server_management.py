"""Server Management GUI Components."""
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox, QPlainTextEdit,
    QGroupBox, QFormLayout, QDialog, QFileDialog,
    QMessageBox, QInputDialog, QSplitter, QListWidget, QListWidgetItem,
    QTextEdit, QTabWidget, QHeaderView,
    QTreeWidget, QProgressDialog, QFrame,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QColor, QTextCursor

from llama_launcher.gui import theme

from llama_launcher.remote.types import (
    LOCAL_SERVER_NAME,
    ServerConfig,
    ConnectionState,
    environment_prefix,
    parse_environment,
)
from llama_launcher.remote.manager import ServerManager, create_connection
from llama_launcher.gui.dialogs.path_picker import (
    _ACTIVE_WORKERS, IS_DIR_ROLE, PATH_ROLE, SIZE_ROLE, RemotePathDialog,
    enable_name_sorting, populate_file_tree, posix_join, posix_parent,
)
from llama_launcher.gui.workers import TaskWorker, TransferWorker

logger = logging.getLogger(__name__)

#: Scrollback cap for the live console, in lines.
MAX_LOG_LINES = 5000


class _ServerStateWorker(QThread):
    """Check connection states for all servers in background.

    Probes run concurrently: an unreachable host costs a full connect timeout,
    so probing serially made the list take (hosts x timeout) seconds to appear.
    """
    done = Signal(list)

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def run(self):
        names = self.manager.get_all_names()
        if not names:
            self.done.emit([])
            return

        def probe(name):
            try:
                return (name, self.manager.get_connection_state(name))
            except Exception:
                logger.debug("State probe failed for %s", name, exc_info=True)
                return (name, ConnectionState.DISCONNECTED)

        with ThreadPoolExecutor(max_workers=min(8, len(names))) as pool:
            results = list(pool.map(probe, names))
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


class ServerEditDialog(QDialog):
    """Dialog for adding or editing a server configuration."""

    def __init__(self, parent=None, config: Optional[ServerConfig] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Server" if not config else "Edit Server")
        self.setMinimumWidth(480)
        self._config = config or ServerConfig()
        self._is_local_entry = self._config.friendly_name == LOCAL_SERVER_NAME
        # Connection used for Test Connection and browsing, rebuilt whenever the
        # connection fields change. Never registered with the manager, so nothing
        # is persisted until the user saves.
        self._probe_conn = None
        self._probe_signature = None
        self._probe_worker = None
        self._build_ui()
        self._populate_fields()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        paths = QGroupBox("Paths")
        paths_layout = QFormLayout(paths)
        self.name_edit = QLineEdit()
        self.working_dir_edit = QLineEdit("~")
        self.server_path_edit = QLineEdit("llama-server")
        self.log_file_edit = QLineEdit("llama-server.log")
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1024, 65535)
        self.server_port_spin.setValue(8080)
        paths_layout.addRow("Name:", self.name_edit)
        paths_layout.addRow("Working Dir:", self._with_browse(
            self.working_dir_edit, self._browse_working_dir,
            "Browse for the directory llama-server runs in"))
        paths_layout.addRow("Server Exec:", self._with_browse(
            self.server_path_edit, self._browse_server_exec,
            "Browse for the llama-server binary (needed when it is not on PATH)"))
        paths_layout.addRow("Log File:", self.log_file_edit)
        paths_layout.addRow("Server Port:", self.server_port_spin)
        self.env_edit = QPlainTextEdit()
        self.env_edit.setPlaceholderText(
            "One KEY=VALUE per line, e.g.\nHSA_OVERRIDE_GFX_VERSION=11.0.0\nCUDA_VISIBLE_DEVICES=0")
        self.env_edit.setMaximumHeight(64)
        self.env_edit.setToolTip(
            "Environment variables set for llama-server when this launcher starts it.\n"
            "AMD cards that ROCm does not officially support usually need "
            "HSA_OVERRIDE_GFX_VERSION; CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES pick a GPU.")
        paths_layout.addRow("Environment:", self.env_edit)
        layout.addWidget(paths)

        connection = QGroupBox("Connection")
        conn_layout = QFormLayout(connection)
        self.conn_type_combo = QComboBox()
        self.conn_type_combo.addItems(["Local", "Remote (SSH)"])
        self.hostname_edit = QLineEdit()
        self.hostname_edit.setPlaceholderText("e.g. 192.168.1.100 or my-server.com")
        self.ssh_port_spin = QSpinBox()
        self.ssh_port_spin.setRange(1, 65535)
        self.ssh_port_spin.setValue(22)
        self.username_edit = QLineEdit()
        self.auth_method_combo = QComboBox()
        self.auth_method_combo.addItems(["SSH Key", "Password"])
        self.key_path_edit = QLineEdit()
        self.key_path_edit.setPlaceholderText("~/.ssh/id_rsa")
        self.key_path_browse_btn = QPushButton("...")
        self.key_path_browse_btn.setFixedWidth(32)
        self.key_path_browse_btn.clicked.connect(self._browse_key_path)
        self.key_path_h = QHBoxLayout()
        self.key_path_h.addWidget(self.key_path_edit)
        self.key_path_h.addWidget(self.key_path_browse_btn)
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("Server password")
        self.remember_password_check = QCheckBox("Save password")
        self.remember_password_check.setToolTip(
            "Passwords are saved in plain text in servers.json (readable only by\n"
            "you). Prefer 'Set Up Key Auth', which removes the need to store one.")
        self.password_h = QHBoxLayout()
        self.password_h.addWidget(self.password_edit, 1)
        self.password_h.addWidget(self.remember_password_check)

        conn_layout.addRow("Type:", self.conn_type_combo)
        conn_layout.addRow("Host:", self.hostname_edit)
        conn_layout.addRow("SSH Port:", self.ssh_port_spin)
        conn_layout.addRow("Username:", self.username_edit)
        conn_layout.addRow("Auth:", self.auth_method_combo)
        conn_layout.addRow("Key Path:", self.key_path_h)
        conn_layout.addRow("Password:", self.password_h)

        test_row = QHBoxLayout()
        self.test_conn_btn = QPushButton("Test Connection")
        self.test_conn_btn.clicked.connect(self._on_test_connection)
        self.setup_key_btn = QPushButton("Set Up Key Auth")
        self.setup_key_btn.setToolTip(
            "Install your public key on this host using the password once, then\n"
            "switch to key authentication so no password has to be stored.")
        self.setup_key_btn.clicked.connect(self._on_setup_key_auth)
        test_row.addWidget(self.test_conn_btn)
        test_row.addWidget(self.setup_key_btn)
        test_row.addStretch()
        conn_layout.addRow("", test_row)
        self.test_result_label = QLabel("")
        self.test_result_label.setWordWrap(True)
        conn_layout.addRow("", self.test_result_label)
        layout.addWidget(connection)

        self.conn_type_combo.currentTextChanged.connect(self._on_conn_type_changed)
        self.auth_method_combo.currentTextChanged.connect(self._on_auth_changed)

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

    def _with_browse(self, line_edit: QLineEdit, handler, tooltip: str) -> QWidget:
        """Wrap a line edit in a row with a browse button."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line_edit, 1)
        btn = QPushButton("...")
        btn.setFixedWidth(32)
        btn.setToolTip(tooltip)
        btn.clicked.connect(handler)
        h.addWidget(btn)
        return row

    def _browse_key_path(self):
        # The key lives on this machine even when the server is remote.
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Key File", str(Path.home() / ".ssh"),
            "SSH Keys (*.pem *.key id_*);;All Files (*)")
        if path:
            self.key_path_edit.setText(path)

    # ------------------------------------------------------------------
    # Browsing the target machine
    # ------------------------------------------------------------------

    def _is_remote_selected(self) -> bool:
        return "Remote" in self.conn_type_combo.currentText()

    def _probe_connection(self):
        """Connection for the config as currently edited, or None if incomplete.

        Rebuilt only when a connection-relevant field changes so the SSH master
        socket is reused across browse operations.
        """
        config = self.get_config()
        signature = (
            config.hostname, config.ssh_port, config.username,
            config.auth_method, config.key_path, config.password,
        )
        if self._is_remote_selected():
            if not config.hostname or not config.username:
                QMessageBox.information(
                    self, "Connection Details Needed",
                    "Enter the host and username first, then browse the remote "
                    "filesystem.")
                return None
        if self._probe_conn is None or signature != self._probe_signature:
            self._probe_conn = create_connection(config)
            self._probe_signature = signature
        return self._probe_conn

    def _pick_path(self, mode: str, title: str, current: str,
                   name_filter=None) -> Optional[str]:
        """Pick a path on the target machine: native dialog local, browser remote."""
        if not self._is_remote_selected():
            if mode == "dir":
                return QFileDialog.getExistingDirectory(self, title, current) or None
            filters = "All Files (*)"
            if name_filter:
                globs = " ".join(name_filter)
                filters = f"Model Files ({globs});;All Files (*)"
            path, _ = QFileDialog.getOpenFileName(self, title, current, filters)
            return path or None

        conn = self._probe_connection()
        if conn is None:
            return None
        return RemotePathDialog.get_path(
            self, conn, mode=mode, start_path=current, title=title,
            name_filter=name_filter,
        )

    def _browse_working_dir(self):
        chosen = self._pick_path(
            "dir", "Select Working Directory", self.working_dir_edit.text().strip())
        if chosen:
            self.working_dir_edit.setText(chosen)

    def _browse_server_exec(self):
        current = self.server_path_edit.text().strip()
        # A bare command name is not a browsable location.
        start = current if "/" in current else self.working_dir_edit.text().strip()
        chosen = self._pick_path("file", "Select llama-server Binary", start)
        if chosen:
            self.server_path_edit.setText(chosen)

    def _on_test_connection(self):
        conn = self._probe_connection()
        if conn is None:
            return
        self.test_conn_btn.setEnabled(False)
        self.test_result_label.setStyleSheet("")
        self.test_result_label.setText("Testing…")

        self._start_probe(conn.test_connection, self._on_test_done,
                          lambda msg: self._on_test_done((False, msg)))

    def _on_test_done(self, result):
        ok, message = result
        self.test_conn_btn.setEnabled(True)
        self._show_result(ok, message)

    def _show_result(self, ok: bool, message: str):
        self.test_result_label.setStyleSheet(
            "color: #2e7d32;" if ok else "color: #c62828;")
        self.test_result_label.setText(message)

    def _on_setup_key_auth(self):
        """Install a key on the host so no password needs to be stored."""
        config = self.get_config()
        if not self._is_remote_selected():
            QMessageBox.information(self, "Key Authentication",
                                    "Key authentication applies to remote servers.")
            return
        if not config.hostname or not config.username:
            QMessageBox.information(self, "Key Authentication",
                                    "Enter the host and username first.")
            return

        password = self.password_edit.text()
        if not password:
            password, ok = QInputDialog.getText(
                self, "Account Password",
                f"Password for {config.username}@{config.hostname}.\n"
                "Used once to install the key, then discarded:",
                QLineEdit.EchoMode.Password)
            if not ok or not password:
                return

        self.setup_key_btn.setEnabled(False)
        self.test_conn_btn.setEnabled(False)
        self._show_result(True, "Installing key…")
        key_path = self.key_path_edit.text().strip()

        def run():
            from llama_launcher.remote.ssh_key import setup_key_auth
            return setup_key_auth(config, password, key_path or None)

        self._start_probe(run, self._on_setup_key_done,
                          lambda msg: self._on_setup_key_done((False, msg, "")))

    def _start_probe(self, fn, on_done, on_failed):
        """Run a connection probe without tying its thread to this dialog.

        A QThread parented to the dialog is destroyed with it, which segfaults
        when the user closes the dialog before the probe returns -- and
        accept()/reject() never raise closeEvent, so there is nowhere to wait.
        """
        worker = TaskWorker(fn)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        _ACTIVE_WORKERS.add(worker)
        worker.finished.connect(lambda: _ACTIVE_WORKERS.discard(worker))
        self._probe_worker = worker
        worker.start()

    def _on_setup_key_done(self, result):
        ok, message, key_path = result
        self.setup_key_btn.setEnabled(True)
        self.test_conn_btn.setEnabled(True)
        if key_path:
            self.key_path_edit.setText(key_path)
        if ok:
            # Key auth is proven to work, so the password is no longer needed.
            self.auth_method_combo.setCurrentText("SSH Key")
            self.password_edit.clear()
            self.remember_password_check.setChecked(False)
            self._probe_conn = None      # rebuild with the new credentials
        self._show_result(ok, message)

    def _on_conn_type_changed(self, text: str):
        is_remote = "Remote" in text
        self.hostname_edit.setEnabled(is_remote)
        self.ssh_port_spin.setEnabled(is_remote)
        self.username_edit.setEnabled(is_remote)
        self.auth_method_combo.setEnabled(is_remote)
        self.key_path_edit.setEnabled(is_remote)
        self.key_path_browse_btn.setEnabled(is_remote)
        self.password_edit.setEnabled(is_remote)
        self.remember_password_check.setEnabled(is_remote)
        self.setup_key_btn.setEnabled(is_remote)

    def _on_auth_changed(self, text: str):
        is_key = "Key" in text
        self.key_path_edit.setVisible(is_key)
        self.key_path_browse_btn.setVisible(is_key)
        self.password_edit.setVisible(not is_key)
        self.remember_password_check.setVisible(not is_key)

    def _add_model_dir(self):
        current = ""
        if self.model_dirs_list.count():
            current = self.model_dirs_list.item(self.model_dirs_list.count() - 1).text()
        path = self._pick_path("dir", "Add Model Directory", current)
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
        self.working_dir_edit.setText(self._config.working_directory)
        self.server_path_edit.setText(self._config.llama_server_path)
        self.log_file_edit.setText(self._config.log_file)
        self.server_port_spin.setValue(self._config.server_port)
        self.env_edit.setPlainText(self._config.environment or "")
        self.enabled_check.setChecked(self._config.enabled)

        self.model_dirs_list.clear()
        dirs = self._config.model_directories or []
        if self._config.model_directory and self._config.model_directory not in dirs:
            dirs = [self._config.model_directory] + dirs
        for d in dirs:
            self.model_dirs_list.addItem(d)

        # Connection fields
        if self._config.hostname:
            self.conn_type_combo.setCurrentText("Remote (SSH)")
        else:
            self.conn_type_combo.setCurrentText("Local")
        self.hostname_edit.setText(self._config.hostname)
        self.ssh_port_spin.setValue(self._config.ssh_port)
        self.username_edit.setText(self._config.username)
        if self._config.auth_method == "password":
            self.auth_method_combo.setCurrentText("Password")
        else:
            self.auth_method_combo.setCurrentText("SSH Key")
        self.key_path_edit.setText(self._config.key_path)
        self.password_edit.setText(self._config.password)
        self.remember_password_check.setChecked(self._config.remember_password)
        self._on_conn_type_changed(self.conn_type_combo.currentText())
        self._on_auth_changed(self.auth_method_combo.currentText())

        if self._is_local_entry:
            # The local machine entry is permanent: its name is the key for its
            # own stored config and model cache, and it cannot become remote.
            self.name_edit.setReadOnly(True)
            self.name_edit.setToolTip(
                "The local machine entry is permanent and cannot be renamed.")
            self.conn_type_combo.setEnabled(False)
            self.conn_type_combo.setToolTip(
                "The local machine entry always uses a local connection.")

    def get_config(self) -> ServerConfig:
        model_dirs = [
            self.model_dirs_list.item(i).text()
            for i in range(self.model_dirs_list.count())
        ]
        is_remote = "Remote" in self.conn_type_combo.currentText()
        auth_key = "Key" in self.auth_method_combo.currentText()
        return ServerConfig(
            friendly_name=self.name_edit.text().strip(),
            working_directory=self.working_dir_edit.text().strip(),
            llama_server_path=self.server_path_edit.text().strip(),
            model_directory=model_dirs[0] if model_dirs else "",
            model_directories=model_dirs if model_dirs else None,
            enabled=self.enabled_check.isChecked(),
            log_file=self.log_file_edit.text().strip(),
            server_port=self.server_port_spin.value(),
            hostname=self.hostname_edit.text().strip() if is_remote else "",
            ssh_port=self.ssh_port_spin.value() if is_remote else 22,
            username=self.username_edit.text().strip() if is_remote else "",
            auth_method="key" if auth_key else "password" if is_remote else "key",
            key_path=self.key_path_edit.text().strip() if is_remote else "",
            password=self.password_edit.text() if is_remote else "",
            remember_password=(
                self.remember_password_check.isChecked() and is_remote and not auth_key),
            environment=self.env_edit.toPlainText().strip(),
        )

    def accept(self):
        try:
            parse_environment(self.env_edit.toPlainText())
        except ValueError as e:
            QMessageBox.warning(self, "Environment", f"{e}\n\nUse one KEY=VALUE per line.")
            return
        super().accept()


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
        QTimer.singleShot(10, self._refresh_list)

    def refresh_states(self):
        """Re-read connection states, e.g. after connecting from another tab."""
        self._refresh_list()

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
            if conn is None:
                continue
            config = conn.config
            icon = "○ "
            # Connected is the only state that earns a colour; an idle server
            # keeps the default so it does not read as a problem.
            colour = None
            if state == ConnectionState.CONNECTED:
                icon, colour = "● ", QColor(theme.SUCCESS)
            elif state == ConnectionState.ERROR:
                icon, colour = "✗ ", QColor(theme.ERROR)
            disabled = not config.enabled
            text = f"{icon}{name}"
            if disabled:
                text += " (disabled)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            if disabled:
                item.setForeground(Qt.gray)
            elif colour is not None:
                item.setForeground(colour)
            self.list_widget.addItem(item)

    def _on_selected(self, current, previous):
        name = current.data(Qt.UserRole) if current else None
        # The local machine entry is permanent; make that visible rather than
        # letting the user click Remove and get refused.
        self.remove_btn.setEnabled(bool(name) and name != LOCAL_SERVER_NAME)
        if name:
            self.server_selected.emit(name)

    def _confirm_port_conflicts(self, config: ServerConfig) -> bool:
        """Warn when another server on the same host already claims this port."""
        conflicts = self._manager.find_port_conflicts(config)
        if not conflicts:
            return True
        where = "this machine" if not config.hostname else config.hostname
        return QMessageBox.warning(
            self, "Port Already Used",
            f"Port {config.server_port} on {where} is already configured for "
            f"{', '.join(repr(c) for c in conflicts)}.\n\n"
            "Several llama-servers can share a host, but each needs its own "
            "port -- whichever starts second will fail to bind.\n\nSave anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def _on_add(self):
        if not self._manager:
            return
        dialog = ServerEditDialog(self)
        if dialog.exec() == QDialog.Accepted:
            config = dialog.get_config()
            if not config.friendly_name:
                QMessageBox.warning(self, "Validation", "Server name is required.")
                return
            if config.friendly_name in self._manager.get_all_names():
                QMessageBox.warning(
                    self, "Validation",
                    f"A server named '{config.friendly_name}' already exists.")
                return
            if not self._confirm_port_conflicts(config):
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
            if not config.friendly_name:
                QMessageBox.warning(self, "Validation", "Server name is required.")
                return
            if not self._confirm_port_conflicts(config):
                return
            if config.friendly_name != name:
                # Entries are keyed by name, so a rename is a remove plus an add.
                # Carry the cached model list over instead of orphaning it.
                self._migrate_model_cache(name, config.friendly_name)
                self._manager.remove_server(name)
            self._manager.add_server(config)
            self._refresh_list()
            self.servers_changed.emit()

    @staticmethod
    def _migrate_model_cache(old_name: str, new_name: str) -> None:
        """Move a renamed server's cached model list to its new key."""
        from llama_launcher.models.storage import (
            load_models, remove_models, save_models,
        )
        try:
            cached = load_models(old_name)
            if cached:
                save_models(new_name, cached)
            remove_models(old_name)
        except Exception:
            logger.debug("Could not migrate model cache %s -> %s",
                         old_name, new_name, exc_info=True)

    def _on_remove(self):
        current = self.list_widget.currentItem()
        if not current or not self._manager:
            return
        name = current.data(Qt.UserRole)
        if name == LOCAL_SERVER_NAME:
            QMessageBox.information(
                self, "Permanent Entry",
                f"'{LOCAL_SERVER_NAME}' represents the machine the launcher is "
                "running on and cannot be removed.")
            return
        reply = QMessageBox.question(
            self, "Remove Server",
            f"Remove server '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._manager.remove_server(name)
            self._refresh_list()
            parent = self.parent()
            if parent is not None:
                status_w = getattr(parent, "status_widget", None)
                if status_w is not None and status_w._selected_server == name:
                    status_w._selected_server = None
                    status_w.server_name_label.setText("No server selected")
            self.servers_changed.emit()


class ServerStatusWidget(QWidget):
    """Right panel: status and controls for the selected server."""

    connect_requested = Signal(str)
    disconnect_requested = Signal(str)

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
        """Command used for Start/Restart."""
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

        command_group = QGroupBox("Command")
        command_layout = QVBoxLayout(command_group)
        self.command_preview_text = QTextEdit()
        self.command_preview_text.setReadOnly(True)
        self.command_preview_text.setMaximumHeight(80)
        self.command_preview_text.setFontFamily("monospace")
        command_layout.addWidget(self.command_preview_text)
        self.env_label = QLabel("")
        self.env_label.setStyleSheet(f"color: {theme.FG_DIM}; font-size: 11px;")
        self.env_label.setWordWrap(True)
        self.env_label.setVisible(False)
        command_layout.addWidget(self.env_label)
        layout.addWidget(command_group)

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

        self.tabs = QTabWidget()

        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        # A server can log indefinitely; keep a bounded scrollback so memory and
        # redraw cost stay flat over a long session.
        self.console_text.document().setMaximumBlockCount(MAX_LOG_LINES)
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
        self.file_path_edit.setPlaceholderText("Directory path on the selected server")
        self.file_refresh_btn = QPushButton("Refresh")
        self.file_refresh_btn.clicked.connect(self._refresh_file_list)
        self.file_up_btn = QPushButton("Up")
        self.file_up_btn.clicked.connect(self._file_go_up)
        self.file_new_dir_btn = QPushButton("New Dir")
        self.file_new_dir_btn.clicked.connect(self._file_new_dir)
        self.file_upload_btn = QPushButton("Upload...")
        self.file_upload_btn.setToolTip(
            "Copy a file from this machine into the directory shown above")
        self.file_upload_btn.clicked.connect(self._file_upload)
        self.file_download_btn = QPushButton("Download...")
        self.file_download_btn.setToolTip("Copy the selected file to this machine")
        self.file_download_btn.clicked.connect(self._file_download)
        self.file_delete_btn = QPushButton("Delete")
        self.file_delete_btn.clicked.connect(self._file_delete)
        file_toolbar.addWidget(self.file_path_edit)
        file_toolbar.addWidget(self.file_refresh_btn)
        file_toolbar.addWidget(self.file_up_btn)
        file_toolbar.addWidget(self.file_new_dir_btn)
        file_toolbar.addWidget(self.file_upload_btn)
        file_toolbar.addWidget(self.file_download_btn)
        file_toolbar.addWidget(self.file_delete_btn)
        file_layout.addLayout(file_toolbar)
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Size", "Modified"])
        self.file_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        enable_name_sorting(self.file_tree)
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

    def _reset_to_config(self, config: ServerConfig):
        """Reset status labels to show config-based info for a selected server."""
        self.state_label.setText("not connected")
        self.pid_label.setText("-")
        self.cpu_label.setText("-")
        self.ram_label.setText("-")
        self.gpu_util_label.setText("-")
        self.gpu_mem_label.setText("-")
        self.port_label.setText(str(config.server_port))
        self.uptime_label.setText("-")
        self.model_label.setText("-")

    def _reset_to_placeholder(self):
        """Reset all status labels to placeholder values."""
        self.state_label.setText("-")
        self.pid_label.setText("-")
        self.cpu_label.setText("-")
        self.ram_label.setText("-")
        self.gpu_util_label.setText("-")
        self.gpu_mem_label.setText("-")
        self.port_label.setText("-")
        self.uptime_label.setText("-")
        self.model_label.setText("-")

    def select_server(self, name: str):
        self._selected_server = name
        self.server_name_label.setText(name)
        conn = self._manager.get_connection(name) if self._manager else None
        if conn:
            # Reset labels to show server info immediately
            self._reset_to_config(conn.config)
        else:
            self._reset_to_placeholder()
        self._refresh_status()

    def _refresh_status(self):
        if not self._manager or not self._selected_server:
            return
        if getattr(self, '_closing', False):
            return
        self.command_preview_text.setPlainText(self._launch_command())
        conn = self._manager.get_connection(self._selected_server)
        prefix = environment_prefix(conn.config) if conn else ""
        self.env_label.setText(f"Runs with: {prefix.strip()}" if prefix else "")
        self.env_label.setVisible(bool(prefix))
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

    def _run_task(self, fn, on_done=None, on_failed=None, worker_class=TaskWorker):
        """Run ``fn`` in a background thread; route result/failure to callbacks."""
        if getattr(self, '_closing', False):
            return None
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
        if not self._selected_server:
            return
        conn = self._manager.get_connection(self._selected_server) if self._manager else None
        is_connected = False
        if conn:
            try:
                is_connected = conn.connection_state == ConnectionState.CONNECTED
            except Exception:
                is_connected = False

        if error_msg:
            self.state_label.setText(f"Error: {error_msg}")
            return

        if status:
            if not is_connected:
                self.state_label.setText("not connected")
                self.pid_label.setText("-")
                self.cpu_label.setText("-")
                self.ram_label.setText("-")
                self.gpu_util_label.setText("-")
                self.gpu_mem_label.setText("-")
                self.port_label.setText("-")
                self.uptime_label.setText("-")
                self.model_label.setText("-")
                return

            self.state_label.setText(status.state.value)
            self.pid_label.setText(str(status.pid or "-"))
            self.cpu_label.setText(f"{status.cpu_percent:.1f}%")
            self.ram_label.setText(f"{status.ram_mb:.1f} MB")
            self.gpu_util_label.setText(f"{status.gpu_utilization:.1f}%")
            self.gpu_mem_label.setText(f"{status.gpu_memory_mb:.1f} MB")

            if status.listening_port:
                self.port_label.setText(str(status.listening_port))
            else:
                self.port_label.setText(str(conn.config.server_port))

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
        """Hand off to the application's single connect flow.

        Connecting must read the host's GPU, RAM and models and update every
        server selector, so this cannot call manager.connect() itself -- doing
        so left this tab and the others disagreeing about what was connected.
        """
        if not self._selected_server or not self._ensure_password():
            return
        self.connect_btn.setEnabled(False)
        self.state_label.setText("Connecting...")
        self.connect_requested.emit(self._selected_server)

    def on_connection_changed(self, server_name: str):
        """Called when the app finishes connecting or disconnecting a server."""
        self.connect_btn.setEnabled(True)
        if server_name == self._selected_server:
            self._refresh_status()

    def _on_disconnect(self):
        if not self._selected_server:
            return
        self.disconnect_requested.emit(self._selected_server)

    def _ensure_password(self) -> bool:
        """Ask for a password-auth server's password if none is held.

        Passwords are no longer written to disk unless the user opts in, so a
        session may start without one. Kept in memory for this run only.
        """
        conn = self._manager.get_connection(self._selected_server) if self._manager else None
        if conn is None:
            return False
        config = conn.config
        if config.auth_method != "password" or config.password:
            return True
        password, ok = QInputDialog.getText(
            self, "Password Required",
            f"Password for {config.username}@{config.hostname}.\n"
            "Used for this session only:",
            QLineEdit.EchoMode.Password)
        if not ok or not password:
            return False
        config.password = password
        # The connection caches its ssh options from the config it was built
        # with, so rebuild it around the now-complete credentials.
        self._manager.add_server(config, persist=False)
        return True

    def _on_start(self):
        if not self._selected_server or not self._ensure_password():
            return
        cmd = self._launch_command()
        if QMessageBox.question(
            self, "Start Server",
            f"Start llama-server on {self._selected_server}?\n\n{cmd}",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.start_btn.setEnabled(False)
        # Probe the port off the UI thread before starting: llama-server would
        # otherwise just die at bind time with the reason buried in the log.
        # cmd is resolved here because it reads GUI widgets.
        self._run_task(lambda: self._check_port_then_start(cmd),
                       on_done=self._on_start_done,
                       on_failed=self._on_start_failed)

    def _check_port_then_start(self, cmd: str):
        """Refuse to start onto an occupied port; otherwise start the server."""
        name = self._selected_server
        conflict = self._manager.check_port_conflict(name)
        if conflict:
            raise RuntimeError(conflict)
        return self._manager.start_server(name, cmd)

    def _on_start_failed(self, message: str):
        self.start_btn.setEnabled(True)
        QMessageBox.warning(self, "Cannot Start Server", message)
        self._refresh_status()

    def _on_start_done(self, result):
        self.start_btn.setEnabled(True)
        if not result:
            QMessageBox.warning(self, "Failed", "Could not start server.")
            self._refresh_status()
            return
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

    def _on_start_console(self):
        if not self._selected_server:
            return
        from llama_launcher.remote.console import LogFollowWorker
        if self._live_console:
            self._live_console.stop()
        self._live_console = LogFollowWorker(self._manager, self._selected_server)
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

        cursor = self.console_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_text.setTextCursor(cursor)
        self.console_text.insertPlainText(line + "\n")

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
        target_path = self.file_path_edit.text().strip()
        if not target_path:
            conn = self._manager.get_connection(self._selected_server)
            if conn:
                target_path = conn.config.model_directory or conn.config.working_directory
        if not target_path:
            return
        self.file_path_edit.setText(target_path)
        self.file_tree.setEnabled(False)
        self._run_task(
            lambda: self._manager.list_directory(self._selected_server, target_path),
            on_done=self._populate_file_tree,
        )

    def _populate_file_tree(self, entries):
        self.file_tree.setEnabled(True)
        populate_file_tree(self.file_tree, entries)

    def _on_file_double_clicked(self, item, _column=0):
        """Navigate into a directory on double-click."""
        is_dir = item.data(0, IS_DIR_ROLE)
        path = item.data(0, PATH_ROLE)
        if is_dir and path:
            self.file_path_edit.setText(path)
            self._refresh_file_list()

    def _file_go_up(self):
        path = self.file_path_edit.text().strip()
        if path:
            self.file_path_edit.setText(posix_parent(path))
            self._refresh_file_list()

    def _file_new_dir(self):
        path = self.file_path_edit.text().strip()
        if not path:
            return
        # getItem with an empty, non-editable list offered nothing to type into.
        name, ok = QInputDialog.getText(self, "New Directory", "Directory name:")
        name = (name or "").strip()
        if ok and name:
            self._run_task(
                lambda: self._manager.create_directory(
                    self._selected_server, posix_join(path, name)),
                on_done=lambda result: self._refresh_file_list() if result else None,
            )

    def _file_upload(self):
        """Copy a file from this machine into the directory being browsed."""
        if not self._selected_server:
            return
        remote_dir = self.file_path_edit.text().strip()
        if not remote_dir:
            QMessageBox.information(self, "Upload",
                                    "Open the destination directory first.")
            return
        local_path, _ = QFileDialog.getOpenFileName(
            self, "Select File to Upload", str(Path.home()))
        if not local_path:
            return

        remote_path = posix_join(remote_dir, Path(local_path).name)
        if self._manager.path_kind(self._selected_server, remote_path) == "file":
            if QMessageBox.question(
                self, "Overwrite?",
                f"'{Path(local_path).name}' already exists in {remote_dir}.\n\nReplace it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) != QMessageBox.Yes:
                return

        try:
            total_bytes = Path(local_path).stat().st_size
        except OSError:
            total_bytes = 0
        self._start_transfer(
            f"Uploading {Path(local_path).name}…", total_bytes,
            lambda cb: self._manager.upload_file(
                self._selected_server, local_path, remote_path, progress_cb=cb),
        )

    def _file_download(self):
        """Copy the selected file from the server to this machine."""
        item = self.file_tree.currentItem()
        if not item or not self._selected_server:
            return
        if item.data(0, IS_DIR_ROLE):
            QMessageBox.information(self, "Download",
                                    "Select a file to download, not a directory.")
            return
        remote_path = item.data(0, PATH_ROLE)
        if not remote_path:
            return

        name = Path(remote_path).name
        local_path, _ = QFileDialog.getSaveFileName(
            self, "Save Downloaded File As", str(Path.home() / name))
        if not local_path:
            return
        total_bytes = item.data(0, SIZE_ROLE) or 0
        self._start_transfer(
            f"Downloading {name}…", total_bytes,
            lambda cb: self._manager.download_file(
                self._selected_server, remote_path, local_path, progress_cb=cb),
        )

    def _file_delete(self):
        item = self.file_tree.currentItem()
        if not item:
            return
        path = item.data(0, PATH_ROLE)
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

        worker = TransferWorker(lambda: fn(cb), self)
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

    #: Raised so the application can run its one connect flow; this tab does
    #: not connect on its own.
    connect_requested = Signal(str)
    disconnect_requested = Signal(str)
    #: Raised on row selection so this tab's choice of server participates in
    #: the same shared "current server" the other tabs' ServerBars agree on --
    #: otherwise a server picked (and scanned) here stays invisible elsewhere
    #: until the user re-selects it in one of the synced bars.
    server_selected = Signal(str)

    def __init__(self, manager: ServerManager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._build_ui()

    def on_connection_changed(self, server_name: str):
        """Refresh the list and details after any server connects/disconnects."""
        self.list_widget.refresh_states()
        self.status_widget.on_connection_changed(server_name)

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
        self.status_widget.connect_requested.connect(self.connect_requested.emit)
        self.status_widget.disconnect_requested.connect(self.disconnect_requested.emit)
        splitter.addWidget(self.status_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 750])

        layout.addWidget(splitter)

    def _on_server_selected(self, name: str):
        self.status_widget.select_server(name)
        self.server_selected.emit(name)

    def shutdown(self) -> None:
        """Stop child widgets' timers and background workers."""
        self.list_widget.shutdown()
        self.status_widget.shutdown()

    @Slot()
    def refresh(self):
        self.list_widget._refresh_list()
        self.status_widget._refresh_status()
