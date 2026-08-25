"""Shared server selector with connection state and a Connect button.

The same control appears on every tab that acts on a server, so wherever you
are you can see whether that machine is actually reachable and connect it
without going to Server Management.

Connecting is manual: nothing here touches the network until the button is
pressed.
"""
import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget,
)

from llama_launcher.gui import theme
from llama_launcher.remote.types import ConnectionState

logger = logging.getLogger(__name__)

#: Connected is the only state that gets a colour -- an idle server should not
#: look like a warning.
_STATE_STYLE = {
    ConnectionState.CONNECTED: (f"color: {theme.SUCCESS};", "● connected"),
    ConnectionState.ERROR: (f"color: {theme.ERROR};", "✗ unreachable"),
    ConnectionState.CONNECTING: ("", "… connecting"),
    ConnectionState.DISCONNECTED: ("", "○ not connected"),
}


class ServerBar(QWidget):
    """"Server: [combo] ● connected [Disconnect]" row."""

    server_changed = Signal(str)
    connect_requested = Signal(str)
    disconnect_requested = Signal(str)

    def __init__(self, parent=None, show_label: bool = True):
        super().__init__(parent)
        self._state = ConnectionState.DISCONNECTED
        self._busy = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if show_label:
            layout.addWidget(QLabel("Server:"))

        self.combo = QComboBox()
        self.combo.setMinimumWidth(140)
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addWidget(self.combo)

        self.status_label = QLabel("")
        self.status_label.setMinimumWidth(110)
        layout.addWidget(self.status_label)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_button_clicked)
        layout.addWidget(self.connect_btn)
        layout.addStretch()

        self.set_state(ConnectionState.DISCONNECTED)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def current_server(self) -> str:
        return self.combo.currentText()

    def set_servers(self, names, current: str = ""):
        """Repopulate the list, keeping the current selection where possible."""
        previous = current or self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in names:
            self.combo.addItem(name)
        if previous and previous in names:
            self.combo.setCurrentText(previous)
        elif names:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    def set_current_server(self, name: str):
        """Select a server without re-emitting server_changed."""
        if name and name != self.combo.currentText():
            self.combo.blockSignals(True)
            self.combo.setCurrentText(name)
            self.combo.blockSignals(False)

    def _on_combo_changed(self, name: str):
        if name:
            self.server_changed.emit(name)

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    def set_state(self, state: ConnectionState, detail: str = ""):
        """Show a connection state, and swap Connect/Disconnect to match."""
        self._state = state
        style, text = _STATE_STYLE.get(
            state, _STATE_STYLE[ConnectionState.DISCONNECTED])
        self.status_label.setStyleSheet(style)
        self.status_label.setText(text)
        self.status_label.setToolTip(detail)

        connected = state == ConnectionState.CONNECTED
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setEnabled(not self._busy)
        self.combo.setEnabled(not self._busy)

    def state(self) -> ConnectionState:
        return self._state

    def set_busy(self, busy: bool, message: str = "Connecting…"):
        """Disable interaction while a connection attempt is in flight."""
        self._busy = busy
        if busy:
            self.status_label.setStyleSheet("")
            self.status_label.setText(message)
        self.connect_btn.setEnabled(not busy)
        self.combo.setEnabled(not busy)

    def _on_button_clicked(self):
        name = self.current_server()
        if not name:
            return
        if self._state == ConnectionState.CONNECTED:
            self.disconnect_requested.emit(name)
        else:
            self.connect_requested.emit(name)
