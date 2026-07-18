"""Preset manager dialog (CRUD operations)."""
import logging

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget, QMessageBox,
    QPushButton, QVBoxLayout,
)

from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.presets import PresetManager

logger = logging.getLogger(__name__)


class PresetManagerDialog(QDialog):
    """Dialog for managing presets (add, delete, save current config)."""

    def __init__(self, get_current_config, parent=None):
        """Args:
            get_current_config: zero-arg callable returning the live LlamaConfig.
        """
        super().__init__(parent)
        self._get_current_config = get_current_config
        self.setWindowTitle("Manage Presets")
        self.setGeometry(300, 300, 500, 400)

        layout = QVBoxLayout(self)

        self.preset_list = QListWidget()
        self.refresh_preset_list()

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add New")
        self.delete_btn = QPushButton("Delete Selected")
        self.save_btn = QPushButton("Save Current Config as Preset")

        self.add_btn.clicked.connect(self.add_preset)
        self.delete_btn.clicked.connect(self.delete_preset)
        self.save_btn.clicked.connect(self.save_current_as_preset)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()

        layout.addWidget(QLabel("Existing Presets:"))
        layout.addWidget(self.preset_list)
        layout.addLayout(btn_layout)

    def refresh_preset_list(self):
        self.preset_list.clear()
        for preset_name in PresetManager.get_all_presets():
            self.preset_list.addItem(preset_name)

    def add_preset(self):
        name, ok = QInputDialog.getText(self, "New Preset", "Preset Name:")
        if ok and name:
            if name in PresetManager.get_all_presets():
                QMessageBox.warning(self, "Error", f"Preset '{name}' already exists.")
                return
            new_config = LlamaConfig(name=name, description=f"Custom preset: {name}")
            PresetManager.save_custom_preset(name, new_config)
            self.refresh_preset_list()

    def delete_preset(self):
        current_items = self.preset_list.selectedItems()
        if not current_items:
            QMessageBox.warning(self, "Error", "Please select a preset to delete.")
            return

        preset_name = current_items[0].text()

        custom_presets = PresetManager._load_custom_presets()
        is_builtin_override = (
            preset_name in custom_presets and preset_name in PresetManager.BUILTIN_PRESETS
        )

        if not is_builtin_override and preset_name in PresetManager.BUILTIN_PRESETS:
            QMessageBox.warning(self, "Error", "Built-in presets cannot be deleted.")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete preset '{preset_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                PresetManager.delete_custom_preset(preset_name)
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))
            self.refresh_preset_list()

    def save_current_as_preset(self):
        name, ok = QInputDialog.getText(self, "Save as Preset", "Preset Name:")
        if not ok or not name:
            return

        if name in PresetManager.get_all_presets():
            QMessageBox.warning(self, "Error", f"Preset '{name}' already exists.")
            return

        config = self._get_current_config()
        config.name = name
        config.description = f"Custom preset: {name}"
        PresetManager.save_custom_preset(name, config)
        self.refresh_preset_list()
