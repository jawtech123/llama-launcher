"""Command Preview & Launch tab.

The command box is editable: a hand-tweaked command is what gets launched,
and it is left alone until Regenerate is pressed (or a preset is loaded), so
a config change elsewhere never silently discards an edit.
"""
import logging

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

from llama_launcher.gui import theme
from llama_launcher.gui.server_bar import ServerBar

logger = logging.getLogger(__name__)

#: Scrollback cap for the launch log, in lines.
MAX_LOG_LINES = 5000


class PreviewTab(QWidget):
    """Shows the generated command, launch controls, and server log output."""

    launch_requested = Signal()
    stop_requested = Signal()
    #: The user pressed Regenerate: the owner should push a fresh command.
    regenerate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edited = False
        self._generated = ""
        self._stale = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Target server -- lets Launch target any configured server, not just local.
        server_row = QHBoxLayout()
        self.server_bar = ServerBar()
        server_row.addWidget(self.server_bar)
        layout.addLayout(server_row)

        cmd_group = QGroupBox("Launch Command (editable)")
        cmd_layout = QVBoxLayout(cmd_group)
        self.command_text = QTextEdit()
        self.command_text.setReadOnly(False)
        self.command_text.setAcceptRichText(False)
        self.command_text.setMaximumHeight(110)
        self.command_text.setFontFamily("monospace")
        self.command_text.setToolTip(
            "Edit freely; the edited command is what Launch runs.\n"
            "Regenerate rebuilds it from the current configuration.")
        self.command_text.textChanged.connect(self._on_command_changed)
        cmd_layout.addWidget(self.command_text)
        self._hint_label = QLabel()
        self._hint_label.setStyleSheet(f"color: {theme.WARNING}; font-style: italic;")
        self._hint_label.setVisible(False)
        cmd_layout.addWidget(self._hint_label)
        cmd_group.setSizePolicy(cmd_group.sizePolicy().horizontalPolicy(),
                                cmd_group.sizePolicy().verticalPolicy().Maximum)
        layout.addWidget(cmd_group)

        controls = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Command")
        self.copy_btn.clicked.connect(self._copy_command)
        controls.addWidget(self.copy_btn)

        self.regenerate_btn = QPushButton("Regenerate")
        self.regenerate_btn.setToolTip("Rebuild the command from the current configuration")
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

        log_group = QGroupBox("Server Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Bounded scrollback: a launched server can log for hours.
        self.log_text.document().setMaximumBlockCount(MAX_LOG_LINES)
        self.log_text.setFontFamily("monospace")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group, 1)

    # ------------------------------------------------------------------
    # Command text
    # ------------------------------------------------------------------

    def set_command(self, cmd: str, force: bool = False):
        """Show a freshly generated command.

        While the user has edits in the box the new command is held back
        (and a hint shown) unless ``force`` is set, e.g. after Regenerate or
        loading a preset.
        """
        self._generated = cmd
        if self._edited and not force:
            self._stale = True
            self._update_hint()
            return
        self.command_text.blockSignals(True)
        self.command_text.setPlainText(cmd)
        self.command_text.blockSignals(False)
        self._edited = False
        self._stale = False
        self._update_hint()

    def command(self) -> str:
        """The command Launch will run: the box's content, edited or not."""
        return self.command_text.toPlainText().strip()

    def is_edited(self) -> bool:
        return self._edited

    def _copy_command(self):
        QApplication.clipboard().setText(self.command())

    def _regenerate_command(self):
        self._edited = False
        self._stale = False
        self.regenerate_requested.emit()
        # If no owner is wired (tests), fall back to the last generated text.
        if self.command_text.toPlainText() != self._generated and self._generated:
            self.set_command(self._generated, force=True)
        self._update_hint()

    def _on_command_changed(self):
        self._edited = self.command_text.toPlainText() != self._generated
        if not self._edited:
            self._stale = False
        self._update_hint()

    def _update_hint(self):
        if self._edited and self._stale:
            text = ("✎ Edited command; the configuration has changed since. "
                    "Regenerate to rebuild from the configuration.")
        elif self._edited:
            text = "✎ Edited command: Launch runs this text as-is. Regenerate to restore."
        else:
            text = ""
        self._hint_label.setText(text)
        self._hint_label.setVisible(bool(text))

    # ------------------------------------------------------------------
    # Run state + log
    # ------------------------------------------------------------------

    def set_running(self, running: bool):
        """Reflect the launch process state in the controls."""
        self.launch_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.server_bar.setEnabled(not running)
        if running:
            self.status_label.setText("● Running")
            self.status_label.setStyleSheet(f"color: {theme.SUCCESS}; font-weight: bold;")
        else:
            self.status_label.setText("● Stopped")
            self.status_label.setStyleSheet(f"color: {theme.FG_DIM};")

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
