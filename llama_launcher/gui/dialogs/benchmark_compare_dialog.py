"""Side-by-side comparison of two or more benchmark sessions."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHeaderView, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from llama_launcher.gui import theme


def _gen_tps(s: dict) -> float:
    gen_ms = float(s.get("gen_ms", 0))
    return s.get("gen_tokens", 0) / (gen_ms / 1000.0) if gen_ms > 0 else 0.0


def _prompt_tps(s: dict) -> float:
    prompt_ms = float(s.get("prompt_ms", 0))
    return s.get("prompt_tokens", 0) / (prompt_ms / 1000.0) if prompt_ms > 0 else 0.0


def _context_size(s: dict) -> float:
    return float((s.get("config_snapshot") or {}).get("context_size") or 0)


#: Metric name -> extractor function, in display order.
_METRICS = {
    "Gen Speed (t/s)": _gen_tps,
    "Prompt Speed (t/s)": _prompt_tps,
    "TTFT p50 (ms)": lambda s: s.get("ttft_ms_p50", 0.0),
    "Context Size": _context_size,
    "Requests": lambda s: float(s.get("requests", 0)),
}


class _BarChartWidget(QWidget):
    """Simple hand-painted horizontal bar chart, one bar per session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []  # list[(label, value)]
        self.setMinimumHeight(32)

    def set_data(self, items) -> None:
        self._items = items
        self.setMinimumHeight(max(32, 28 * len(items)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._items:
            painter.end()
            return

        label_col_w = 150
        row_h = max(24, self.height() // len(self._items))
        max_val = max((v for _, v in self._items), default=0.0) or 1.0

        for i, (label, value) in enumerate(self._items):
            y = i * row_h
            bar_w = int((value / max_val) * max(1, self.width() - label_col_w - 60))
            painter.setPen(QColor(theme.FG_TEXT))
            painter.drawText(4, y + row_h - 8, label_col_w - 8, row_h,
                             Qt.AlignmentFlag.AlignVCenter, label[:22])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawRoundedRect(label_col_w, y + 4, max(2, bar_w), row_h - 10, 4, 4)
            painter.setPen(QColor(theme.FG_TEXT))
            painter.drawText(label_col_w + bar_w + 6, y + row_h - 8, f"{value:.1f}")
        painter.end()


class SessionCompareDialog(QDialog):
    """Compare selected benchmark sessions on a chosen metric."""

    def __init__(self, sessions: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Sessions")
        self.resize(640, 400)
        self._sessions = sessions

        layout = QVBoxLayout(self)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(_METRICS))
        self.metric_combo.currentTextChanged.connect(self._refresh_chart)
        layout.addWidget(self.metric_combo)

        self.chart = _BarChartWidget()
        layout.addWidget(self.chart)

        self.table = QTableWidget(len(sessions), 6)
        self.table.setHorizontalHeaderLabels(
            ["Label", "Context", "Speed In (t/s)", "Speed Out (t/s)",
             "TTFT p50 (ms)", "Requests"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self._populate_table()
        self._refresh_chart()

    def _populate_table(self):
        for row, s in enumerate(self._sessions):
            context = (s.get("config_snapshot") or {}).get("context_size")
            values = [
                str(s.get("label") or s.get("source") or "--"),
                str(context) if context else "--",
                f"{_prompt_tps(s):.1f}",
                f"{_gen_tps(s):.1f}",
                f"{s.get('ttft_ms_p50', 0):.0f}",
                str(int(s.get("requests", 0))),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)

    def _refresh_chart(self):
        fn = _METRICS[self.metric_combo.currentText()]
        data = [(str(s.get("label") or s.get("source") or "--"), fn(s))
                for s in self._sessions]
        self.chart.set_data(data)
