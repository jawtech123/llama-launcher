"""GPU Info tab: hardware of whichever server is selected."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from llama_launcher.gui import theme
from llama_launcher.gui.server_bar import ServerBar


class GpuTab(QWidget):
    """Shows the selected server's GPU information and live stats text."""

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # The GPU that matters is the one on the machine the model will run on.
        self.server_bar = ServerBar()
        layout.addWidget(self.server_bar)

        self.staleness_label = QLabel("")
        self.staleness_label.setStyleSheet(f"color: {theme.FG_MUTED};")
        self.staleness_label.setVisible(False)
        layout.addWidget(self.staleness_label)

        self.gpu_info_label = QLabel("Detecting GPU...")
        self.gpu_info_label.setWordWrap(True)
        self.gpu_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.gpu_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.gpu_info_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.refresh_gpu_btn = QPushButton("Refresh GPU Info")
        self.refresh_gpu_btn.clicked.connect(self.refresh_requested.emit)
        btn_row.addWidget(self.refresh_gpu_btn)
        layout.addLayout(btn_row)

    def set_info_text(self, text: str):
        self.gpu_info_label.setText(text)

    def info_text(self) -> str:
        return self.gpu_info_label.text()

    def set_staleness(self, message: str):
        """Say when the figures came from a host we are no longer connected to."""
        self.staleness_label.setText(message)
        self.staleness_label.setVisible(bool(message))
