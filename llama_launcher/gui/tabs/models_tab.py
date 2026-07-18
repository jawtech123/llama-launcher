"""Model Browser tab: source folders, scanning, filtering, and selection."""
import json
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from llama_launcher.models.types import ModelInfo
from llama_launcher.utils.paths import config_file

logger = logging.getLogger(__name__)


class ModelsTab(QWidget):
    """Browse scanned models from multiple source folders on any server."""

    scan_requested = Signal()
    model_selected = Signal(str)          # emits model file path
    browser_server_changed = Signal(str)  # emits server name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.load_folders()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Server selector
        server_row = QHBoxLayout()
        server_row.addWidget(QLabel("Server:"))
        self.browser_server_combo = QComboBox()
        self.browser_server_combo.setMinimumWidth(140)
        self.browser_server_combo.currentTextChanged.connect(
            self.browser_server_changed.emit)
        server_row.addWidget(self.browser_server_combo, 1)
        server_row.addStretch()
        layout.addLayout(server_row)

        # Source folders
        folders_group = QGroupBox("Source Folders")
        folders_layout = QVBoxLayout(folders_group)
        self.folders_list = QListWidget()
        self.folders_list.setMaximumHeight(100)
        folders_layout.addWidget(self.folders_list)

        folders_btns = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self._browse_add_folder)
        self.remove_folder_btn = QPushButton("Remove Selected")
        self.remove_folder_btn.clicked.connect(self._remove_selected_folder)
        folders_btns.addWidget(self.add_folder_btn)
        folders_btns.addWidget(self.remove_folder_btn)
        folders_btns.addStretch()
        folders_layout.addLayout(folders_btns)
        layout.addWidget(folders_group)

        # Scan row with busy indicator
        scan_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan All Folders for Models")
        self.scan_btn.clicked.connect(self.scan_requested.emit)
        scan_row.addWidget(self.scan_btn, 1)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # indeterminate
        self.busy_bar.setFixedWidth(140)
        self.busy_bar.setVisible(False)
        scan_row.addWidget(self.busy_bar)
        layout.addLayout(scan_row)

        # Filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name, format, or quant...")
        self.filter_input.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_input, 1)
        layout.addLayout(filter_row)

        # Model list
        layout.addWidget(QLabel("Detected Models:"))
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self._on_item_clicked)
        self.model_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.model_list)

    # ------------------------------------------------------------------
    # Folders persistence
    # ------------------------------------------------------------------

    def get_folders(self) -> list:
        return [self.folders_list.item(i).text() for i in range(self.folders_list.count())]

    def load_folders(self):
        folders_file = config_file("folders.json")
        if not folders_file.exists():
            return
        try:
            with open(folders_file, "r") as f:
                folders = json.load(f)
            for folder in folders:
                if not self.folders_list.findItems(folder, Qt.MatchExactly):
                    self.folders_list.addItem(folder)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load folders file %s: %s", folders_file, e)

    def save_folders(self):
        try:
            with open(config_file("folders.json"), "w") as f:
                json.dump(self.get_folders(), f, indent=2)
        except OSError:
            logger.exception("Could not save folders file")

    def _browse_add_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if dir_path and not self.folders_list.findItems(dir_path, Qt.MatchExactly):
            self.folders_list.addItem(dir_path)
            self.save_folders()
            self.scan_requested.emit()  # auto-scan the newly added folder

    def _remove_selected_folder(self):
        for item in self.folders_list.selectedItems():
            self.folders_list.takeItem(self.folders_list.row(item))
        self.save_folders()

    # ------------------------------------------------------------------
    # Scan state + population
    # ------------------------------------------------------------------

    def set_scanning(self, scanning: bool):
        """Toggle busy UI while a scan is running."""
        self.scan_btn.setEnabled(not scanning)
        self.busy_bar.setVisible(scanning)

    def populate(self, models: list):
        """Fill the model list from scan results (ModelInfo objects)."""
        self.model_list.clear()
        for model in models:
            item_text = f"{model.file_name} ({model.file_size_mb:.2f}MB) [{model.format}]"
            if model.quant_type:
                item_text += f" - {model.quant_type}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model.file_path)
            item.setToolTip(model.file_path)
            self.model_list.addItem(item)
        self._apply_filter()

    def _apply_filter(self):
        needle = self.filter_input.text().strip().lower()
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.model_selected.emit(path)

    def _on_item_double_clicked(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.model_selected.emit(path)
