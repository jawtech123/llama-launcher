"""Shared custom widgets for the llama-launcher GUI."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QStyle, QToolButton, QWidget


def info_label(text: str, tooltip: str) -> QWidget:
    """Create a label with an information icon button showing a tooltip."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    label = QLabel(text)
    label.setToolTip(tooltip)
    layout.addWidget(label)

    info_btn = QToolButton()
    info_btn.setIcon(widget.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
    info_btn.setToolTip(tooltip)
    info_btn.setFixedSize(16, 16)
    info_btn.setCursor(Qt.CursorShape.WhatsThisCursor)
    info_btn.setAutoRaise(True)
    layout.addWidget(info_btn)

    return widget


class BinaryContextSpinBox(QSpinBox):
    """QSpinBox that steps between powers-of-two context sizes."""

    BINARY_CONTEXTS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                       65536, 131072, 262144, 524288, 1048576]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(256, 1048576)
        self.setSingleStep(1)

    def stepBy(self, steps: int):
        current = self.value()
        if current not in self.BINARY_CONTEXTS:
            current = self.round_to_binary_context(current)

        idx = self.BINARY_CONTEXTS.index(current)

        if steps > 0:
            new_val = self.BINARY_CONTEXTS[min(idx + 1, len(self.BINARY_CONTEXTS) - 1)]
        elif steps < 0:
            new_val = self.BINARY_CONTEXTS[max(idx - 1, 0)]
        else:
            return super().stepBy(steps)

        self.setValue(new_val)  # emits valueChanged, propagating via the controller

    def round_to_binary_context(self, size: int) -> int:
        """Round an arbitrary size to the nearest power-of-two context size."""
        import math

        if size <= 256:
            return 256
        if size >= 1048576:
            return 1048576

        p = int(math.log2(size))
        lower = 1 << p
        upper = lower << 1

        if upper > 1048576:
            return 524288

        return lower if (size - lower) < (upper - size) else upper
