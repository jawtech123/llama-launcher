"""Transport-agnostic filesystem browser for picking remote or local paths.

Everything here talks to an :class:`IServerConnection`, so the same dialog browses
the machine the launcher runs on and a host reached over SSH. It takes the
connection object rather than a ``(manager, name)`` pair because the Add Server
dialog has to browse a server the user has not saved yet.

Listings always run on a worker thread -- an SSH round trip on the GUI thread
would freeze the window.
"""
import fnmatch
import logging
from datetime import datetime
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from llama_launcher.gui.workers import TaskWorker
from llama_launcher.remote.connection import IServerConnection
from llama_launcher.remote.types import FileEntry
from llama_launcher.utils.helpers import format_bytes

logger = logging.getLogger(__name__)

#: Listing workers in flight, kept alive here rather than parented to the dialog.
#: A QThread destroyed while running segfaults, and a dialog can be dismissed
#: mid-listing -- accept()/reject() do not raise closeEvent, so there is no
#: reliable moment to wait for it. Qt drops the signal connections when the
#: dialog dies, so a late result is simply ignored.
_ACTIVE_WORKERS = set()


def wait_for_active_workers(timeout_ms: int = 1500) -> None:
    """Let in-flight listing/probe threads finish before the app tears down.

    Qt aborts if a running QThread is destroyed, which is what happens at exit
    if the user quits while a probe is still blocked on its connect timeout.
    Listings normally finish in milliseconds; the budget stays short so quitting
    never feels like a hang, and a straggler is force-stopped rather than left to
    abort the process on the way out.
    """
    for worker in list(_ACTIVE_WORKERS):
        try:
            if worker.isRunning() and not worker.wait(max(0, timeout_ms)):
                logger.debug("Force-stopping a background probe at shutdown")
                worker.setTerminationEnabled(True)
                worker.terminate()
                worker.wait(500)
        except RuntimeError:
            continue               # already reaped
    _ACTIVE_WORKERS.clear()

#: Roles used to stash entry metadata on a tree item.
PATH_ROLE = Qt.UserRole
IS_DIR_ROLE = Qt.UserRole + 1
SIZE_ROLE = Qt.UserRole + 2

#: Glob patterns matching the model files llama-server can load.
MODEL_PATTERNS = ("*.gguf", "*.bin", "*.safetensors")


class FileTreeItem(QTreeWidgetItem):
    """Tree item for a filesystem listing.

    Keeps directories grouped above files whichever column and direction the user
    sorts by, and sorts the Size column by actual byte count rather than by the
    formatted text (where "9.00 KB" would precede "10.00 B").
    """

    def __lt__(self, other):
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        self_is_dir = bool(self.data(0, IS_DIR_ROLE))
        other_is_dir = bool(other.data(0, IS_DIR_ROLE))
        if self_is_dir != other_is_dir:
            # Qt reverses the comparison for a descending sort, so invert to keep
            # directories on top either way.
            descending = (
                tree is not None
                and tree.header().sortIndicatorOrder() == Qt.DescendingOrder
            )
            return other_is_dir if descending else self_is_dir
        if column == 1:
            a = self.data(0, SIZE_ROLE) or 0
            b = other.data(0, SIZE_ROLE) or 0
            try:
                return a < b
            except TypeError:
                return False
        if column < 0:
            column = 0
        return self.text(column).lower() < other.text(column).lower()


def enable_name_sorting(tree: QTreeWidget) -> None:
    """Turn on sorting with an explicit name-ascending default.

    ``setSortingEnabled(True)`` alone leaves the indicator on *descending*, which
    reverses the listing the moment items are inserted.
    """
    tree.setSortingEnabled(True)
    tree.sortByColumn(0, Qt.AscendingOrder)


def populate_file_tree(
    tree: QTreeWidget,
    entries: Sequence[FileEntry],
    show_files: bool = True,
    name_filter: Optional[Sequence[str]] = None,
) -> None:
    """Fill ``tree`` with ``entries``, directories first.

    Args:
        show_files: When False only directories are listed (directory picking).
        name_filter: Glob patterns; non-matching *files* are omitted. Directories
            are always kept so the user can navigate through them.
    """
    tree.clear()
    for entry in sorted(entries, key=lambda e: (not e.is_directory, e.name.lower())):
        if not entry.is_directory:
            if not show_files:
                continue
            if name_filter and not any(
                fnmatch.fnmatch(entry.name.lower(), pattern.lower())
                for pattern in name_filter
            ):
                continue
        if entry.modified_time:
            modified = datetime.fromtimestamp(entry.modified_time).strftime(
                "%Y-%m-%d %H:%M")
        else:
            modified = "-"
        item = FileTreeItem([
            f"[DIR] {entry.name}" if entry.is_directory else entry.name,
            "" if entry.is_directory else format_bytes(entry.size_bytes),
            modified,
        ])
        item.setData(0, PATH_ROLE, entry.path)
        item.setData(0, IS_DIR_ROLE, entry.is_directory)
        item.setData(0, SIZE_ROLE, entry.size_bytes)
        tree.addTopLevelItem(item)


def posix_parent(path: str) -> str:
    """Parent of a POSIX path, computed without the launcher's local path rules.

    ``pathlib`` would apply Windows semantics when the launcher runs on Windows
    and the host being browsed is Linux.
    """
    trimmed = path.rstrip("/")
    if "/" not in trimmed:
        return "/"
    parent = trimmed.rsplit("/", 1)[0]
    return parent or "/"


def posix_join(directory: str, name: str) -> str:
    """Join POSIX path components for the host being browsed."""
    return f"{directory.rstrip('/')}/{name}"


class RemotePathDialog(QDialog):
    """Browse a connection's filesystem and return a chosen path.

    Args:
        connection: The server to browse (local or SSH).
        mode: ``"dir"`` to choose a directory, ``"file"`` to choose a file.
        start_path: Directory to open at; falls back to the server's configured
            model/working directory, then its home directory.
        name_filter: Glob patterns limiting which files are shown in file mode.
    """

    def __init__(self, parent=None, connection: IServerConnection = None,
                 mode: str = "dir", start_path: str = "", title: str = "",
                 name_filter: Optional[Sequence[str]] = None):
        super().__init__(parent)
        self._conn = connection
        self._mode = mode if mode in ("dir", "file") else "dir"
        self._name_filter = list(name_filter) if name_filter else None
        self._selected: Optional[str] = None
        self._current_dir = ""
        self._worker: Optional[TaskWorker] = None
        self._request_id = 0
        self._closing = False

        self.setWindowTitle(title or (
            "Select Directory" if self._mode == "dir" else "Select File"))
        self.setMinimumSize(680, 460)
        self._build_ui()
        self._navigate(start_path or self._default_start_path())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        host = QLabel(self._host_label())
        host.setStyleSheet("font-weight: bold;")
        layout.addWidget(host)

        toolbar = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path — type one and press Enter")
        self.path_edit.returnPressed.connect(self._on_path_entered)
        self.up_btn = QPushButton("Up")
        self.up_btn.clicked.connect(self._go_up)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        toolbar.addWidget(self.path_edit, 1)
        toolbar.addWidget(self.up_btn)
        toolbar.addWidget(self.refresh_btn)
        if self._mode == "dir":
            self.new_dir_btn = QPushButton("New Folder")
            self.new_dir_btn.clicked.connect(self._new_folder)
            toolbar.addWidget(self.new_dir_btn)
        layout.addLayout(toolbar)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Size", "Modified"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        enable_name_sorting(self.tree)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemSelectionChanged.connect(self._update_choose_state)
        layout.addWidget(self.tree)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.choose_btn = QPushButton("Choose")
        self.choose_btn.setDefault(True)
        self.choose_btn.clicked.connect(self._on_choose)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.choose_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def _host_label(self) -> str:
        if self._conn is None:
            return "No connection"
        cfg = self._conn.config
        if cfg.hostname:
            return f"Browsing {cfg.username}@{cfg.hostname}"
        return "Browsing this machine"

    def _default_start_path(self) -> str:
        cfg = self._conn.config if self._conn else None
        if cfg is not None:
            dirs = cfg.model_directories or []
            for candidate in ([cfg.model_directory] + list(dirs)):
                if candidate:
                    return candidate
            if cfg.working_directory and cfg.working_directory != "~":
                return cfg.working_directory
        return ""

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, path: str):
        """Resolve ``path`` to a directory and list it, off the UI thread."""
        self._set_busy(True, "Opening…")
        # Listings return out of order when the user keeps navigating, and a
        # slow earlier one would otherwise replace the view they are now on.
        self._request_id += 1
        request_id = self._request_id

        def resolve_and_list():
            target = path
            if not target or target == "~":
                target = self._conn.home_directory()
            kind = self._conn.path_kind(target)
            if kind == "file":
                target = posix_parent(target)
            elif kind == "missing":
                fallback = self._conn.home_directory()
                if self._conn.path_kind(fallback) == "dir":
                    target = fallback
                else:
                    target = "/"
            return request_id, target, self._conn.list_directory(target)

        self._run(resolve_and_list, self._on_listed)

    def _on_listed(self, result):
        request_id, target, entries = result
        if request_id != self._request_id:
            return                      # superseded by a later navigation
        self._current_dir = target
        self.path_edit.setText(target)
        populate_file_tree(
            self.tree, entries,
            show_files=(self._mode == "file"),
            name_filter=self._name_filter if self._mode == "file" else None,
        )
        shown = self.tree.topLevelItemCount()
        self._set_busy(False, f"{shown} item{'' if shown == 1 else 's'}")
        self._update_choose_state()

    def _on_failed(self, message: str):
        self._set_busy(False, "")
        QMessageBox.warning(self, "Browse Failed", message)

    def _refresh(self):
        self._navigate(self._current_dir)

    def _go_up(self):
        if self._current_dir and self._current_dir != "/":
            self._navigate(posix_parent(self._current_dir))

    def _on_path_entered(self):
        typed = self.path_edit.text().strip()
        if typed:
            self._navigate(typed)

    def _on_double_clicked(self, item, _column):
        if item.data(0, IS_DIR_ROLE):
            self._navigate(item.data(0, PATH_ROLE))
        elif self._mode == "file":
            self._selected = item.data(0, PATH_ROLE)
            self.accept()

    def _new_folder(self):
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        new_path = posix_join(self._current_dir, name)
        self._set_busy(True, "Creating…")
        self._run(
            lambda: self._conn.create_directory(new_path),
            lambda created: self._refresh() if created else self._on_failed(
                f"Could not create {new_path}"),
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _current_selection(self) -> Optional[QTreeWidgetItem]:
        items = self.tree.selectedItems()
        return items[0] if items else None

    def _update_choose_state(self):
        if self._mode == "dir":
            # Choosing applies to the directory currently open.
            self.choose_btn.setEnabled(bool(self._current_dir))
            self.choose_btn.setText("Choose This Folder")
            return
        item = self._current_selection()
        self.choose_btn.setEnabled(bool(item) and not item.data(0, IS_DIR_ROLE))

    def _on_choose(self):
        if self._mode == "dir":
            selected = self._current_selection()
            # A highlighted subdirectory is the more specific intent.
            if selected is not None and selected.data(0, IS_DIR_ROLE):
                self._selected = selected.data(0, PATH_ROLE)
            else:
                self._selected = self._current_dir
            self.accept()
            return
        item = self._current_selection()
        if item is None or item.data(0, IS_DIR_ROLE):
            return
        self._selected = item.data(0, PATH_ROLE)
        self.accept()

    def selected_path(self) -> Optional[str]:
        return self._selected

    # ------------------------------------------------------------------
    # Worker plumbing
    # ------------------------------------------------------------------

    def _run(self, fn, on_done):
        if self._closing:
            return
        # Deliberately unparented: see _ACTIVE_WORKERS.
        worker = TaskWorker(fn)
        worker.done.connect(on_done)
        worker.failed.connect(self._on_failed)
        _ACTIVE_WORKERS.add(worker)
        worker.finished.connect(lambda: _ACTIVE_WORKERS.discard(worker))
        worker.finished.connect(lambda: self._forget_worker(worker))
        self._worker = worker
        worker.start()

    def _forget_worker(self, worker):
        if self._worker is worker:
            self._worker = None

    def _set_busy(self, busy: bool, message: str):
        self.tree.setEnabled(not busy)
        self.up_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
        self.path_edit.setEnabled(not busy)
        if busy:
            self.choose_btn.setEnabled(False)
        self.status_label.setText(message)

    def done(self, result):
        """Both accept() and reject() route through here, unlike closeEvent."""
        self._closing = True
        super().done(result)

    def closeEvent(self, event):
        self._closing = True
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def get_path(parent, connection: IServerConnection, mode: str = "dir",
                 start_path: str = "", title: str = "",
                 name_filter: Optional[Sequence[str]] = None) -> Optional[str]:
        """Show the dialog; return the chosen path or None if cancelled."""
        dialog = RemotePathDialog(
            parent, connection, mode=mode, start_path=start_path,
            title=title, name_filter=name_filter,
        )
        if dialog.exec() == QDialog.Accepted:
            return dialog.selected_path()
        return None
