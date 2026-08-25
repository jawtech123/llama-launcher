"""Command Preview & Launch tab."""
import logging

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

from llama_launcher.gui.server_bar import ServerBar

logger = logging.getLogger(__name__)

#: Scrollback cap for the launch log, in lines.
MAX_LOG_LINES = 5000


class PreviewTab(QWidget):
    """Shows the generated command, launch controls, and server log output."""

    launch_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edited = False
        self._command_builder = None
        self.config = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Target server -- lets Launch target any configured server, not just local.
        server_row = QHBoxLayout()
        self.server_bar = ServerBar()
        server_row.addWidget(self.server_bar)
        layout.addLayout(server_row)

        # Generated command
        cmd_group = QGroupBox("Generated Command")
        cmd_layout = QVBoxLayout(cmd_group)
        self.command_text = QTextEdit()
        self.command_text.setReadOnly(False)
        self.command_text.setMaximumHeight(110)
        self.command_text.textChanged.connect(self._on_command_changed)
        cmd_layout.addWidget(self.command_text)
        self._regenerate_hint_label = QLabel()
        self._regenerate_hint_label.setStyleSheet("color: #FF9800; font-style: italic;")
        cmd_layout.addWidget(self._regenerate_hint_label)
        layout.addWidget(cmd_group)

        # Controls
        controls = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Command")
        self.copy_btn.clicked.connect(self._copy_command)
        controls.addWidget(self.copy_btn)

        self.regenerate_btn = QPushButton("Regenerate")
        self.regenerate_btn.clicked.connect(self._regenerate_command)
        controls.addWidget(self.regenerate_btn)

        controls.addStretch()

        self.status_label = QLabel("● Stopped")
        controls.addWidget(self.status_label)

        self.launch_btn = QPushButton("Launch llama-server")
        self.launch_btn.setProperty("accent", True)
        self.launch_btn.clicked.connect(self.launch_requested.emit)
        controls.addWidget(self.launch_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        controls.addWidget(self.stop_btn)
        layout.addLayout(controls)

        # Log output
        log_group = QGroupBox("Server Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Bounded scrollback: a launched server can log for hours.
        self.log_text.document().setMaximumBlockCount(MAX_LOG_LINES)
        self.log_text.setFontFamily("monospace")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

    # ------------------------------------------------------------------

    def set_command(self, cmd: str):
        self.command_text.setPlainText(cmd)

    def _copy_command(self):
        QApplication.clipboard().setText(self.command_text.toPlainText())

    def _regenerate_command(self):
        if self._command_builder:
            cmd = self._command_builder(self.config)
            self.command_text.setPlainText(cmd)
            self._regenerate_hint_label.setText("Command regenerated.")
            self._regenerate_hint_label.setStyleSheet("color: #4CAF50;")
            self._edited = False
            self._update_hint()

    def _update_hint(self):
        if self._edited:
            self._regenerate_hint_label.setText("✎ Edited command — click Regenerate to restore")
            self._regenerate_hint_label.setStyleSheet("color: #FF9800;")
        else:
            self._regenerate_hint_label.setText("")

    def set_command(self, cmd: str, config=None, command_builder=None):
        self.config = config if config else self.config
        if command_builder:
            self._command_builder = command_builder
        self.command_text.setPlainText(cmd)
        self._edited = False
        self._update_hint()

    def _on_command_changed(self):
        self._edited = True
        self._update_hint()

    def set_running(self, running: bool):
        """Reflect the launch process state in the controls."""
        self.launch_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.server_bar.setEnabled(not running)
        if running:
            self.status_label.setText("● Running")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.status_label.setText("● Stopped")
            self.status_label.setStyleSheet("color: #9d9d9d;")

    def clear_log(self):
        self.log_text.clear()

    def append_log(self, text: str):
        sb = self.log_text.verticalScrollBar()
        at_bottom = sb is None or sb.value() >= sb.maximum() - 4

        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
        self.log_text.insertPlainText(text)

        # Auto-scroll only if the user is already at the bottom.
        if at_bottom and sb is not None:
            sb.setValue(sb.maximum())
