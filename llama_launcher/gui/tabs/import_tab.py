"""Import Launch Command tab."""
import dataclasses
import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QInputDialog, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from llama_launcher.core.config import LlamaConfig, migrate_config_dict
from llama_launcher.core.parser import parse_launch_command
from llama_launcher.core.presets import PresetManager

logger = logging.getLogger(__name__)

_PLACEHOLDER = (
    "Paste llama-server command here, e.g.:\n"
    "llama-server --model /path/to/model.gguf --ctx-size 4096 "
    "--gpu-layers 33 --threads 8 --port 8080 --host 127.0.0.1"
)


class ImportTab(QWidget):
    """Parse a pasted llama-server command into a saved preset."""

    preset_created = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        command_group = QGroupBox("Launch Command")
        command_layout = QVBoxLayout(command_group)
        self.import_command_text = QTextEdit()
        self.import_command_text.setPlaceholderText(_PLACEHOLDER)
        self.import_command_text.setMinimumHeight(150)
        command_layout.addWidget(self.import_command_text)
        layout.addWidget(command_group)

        self.import_parse_btn = QPushButton("Parse Command and Create Preset")
        self.import_parse_btn.clicked.connect(self._do_import)
        layout.addWidget(self.import_parse_btn)

        result_group = QGroupBox("Parsed Parameters")
        result_layout = QVBoxLayout(result_group)
        self.import_result_text = QTextEdit()
        self.import_result_text.setReadOnly(True)
        self.import_result_text.setMinimumHeight(150)
        result_layout.addWidget(self.import_result_text)
        layout.addWidget(result_group)
        layout.addStretch()

    def _do_import(self):
        command_str = self.import_command_text.toPlainText().strip()
        if not command_str:
            QMessageBox.warning(self, "Error", "Please enter a launch command.")
            return

        params, unrecognized = parse_launch_command(command_str)

        if not params and not unrecognized:
            QMessageBox.warning(self, "Error", "No parameters detected in the command.")
            return

        # Report what was recognized and what was skipped
        result_lines = ["Parsed Parameters:"]
        for key, value in params.items():
            result_lines.append(f"  {key}: {value}")
        if unrecognized:
            result_lines.append("")
            result_lines.append("Skipped (unrecognized) flags:")
            for flag in unrecognized:
                result_lines.append(f"  {flag}")
        self.import_result_text.setPlainText("\n".join(result_lines))

        if not params:
            QMessageBox.warning(
                self, "Import Error",
                "No recognized llama-server parameters found in the command.",
            )
            return

        name, ok = QInputDialog.getText(self, "Create Preset", "Preset Name:")
        if not (ok and name):
            return
        if name in PresetManager.get_all_presets():
            QMessageBox.warning(self, "Error", f"Preset '{name}' already exists.")
            return

        migrated = migrate_config_dict(params)
        valid_fields = {f.name for f in dataclasses.fields(LlamaConfig)}
        filtered = {k: v for k, v in migrated.items() if k in valid_fields}
        dropped = sorted(set(migrated) - valid_fields)
        if dropped:
            logger.warning("Import: ignoring unknown parameters: %s", ", ".join(dropped))

        config = LlamaConfig(
            **filtered,
            name=name,
            description=f"Imported from command: {command_str[:50]}...",
        )
        PresetManager.save_custom_preset(name, config)

        if unrecognized:
            QMessageBox.information(
                self, "Preset Created",
                f"Preset '{name}' created.\n\n"
                f"Note: {len(unrecognized)} unrecognized flag(s) were skipped:\n"
                + "\n".join(unrecognized),
            )
        else:
            QMessageBox.information(self, "Success", f"Preset '{name}' created successfully.")
        self.preset_created.emit()
