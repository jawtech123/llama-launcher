"""Command Preview & Launch tab."""
import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)


class PreviewTab(QWidget):
    """Shows the generated command, launch controls, and server log output."""

    launch_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Generated command
        cmd_group = QGroupBox("Generated Command")
        cmd_layout = QVBoxLayout(cmd_group)
        self.command_text = QTextEdit()
        self.command_text.setReadOnly(True)
        self.command_text.setMaximumHeight(110)
        cmd_layout.addWidget(self.command_text)
        layout.addWidget(cmd_group)

        # Controls
        controls = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Command")
        self.copy_btn.clicked.connect(self._copy_command)
        controls.addWidget(self.copy_btn)

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
        self.log_text.setFontFamily("monospace")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

    # ------------------------------------------------------------------

    def set_command(self, cmd: str):
        self.command_text.setPlainText(cmd)

    def _copy_command(self):
        QApplication.clipboard().setText(self.command_text.toPlainText())

    def set_running(self, running: bool):
        """Reflect the launch process state in the controls."""
        self.launch_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.status_label.setText("● Running")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.status_label.setText("● Stopped")
            self.status_label.setStyleSheet("color: #9d9d9d;")

    def clear_log(self):
        self.log_text.clear()

    def append_log(self, text: str):
        self.log_text.moveCursor(self.log_text.textCursor().End)
        self.log_text.insertPlainText(text)
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
