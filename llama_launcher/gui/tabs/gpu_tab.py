"""GPU Info tab."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class GpuTab(QWidget):
    """Shows detected GPU information and live stats text."""

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

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
