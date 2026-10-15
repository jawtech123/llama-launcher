"""GPU Info tab: hardware of whichever server is selected, live.

Static facts (name, driver, compute target) come from the host profile cached
at connect time; utilization, memory, temperature and power are refreshed
from the same usage samples that drive the live bars at the bottom of the
window, so the tab costs no extra round trips.
"""
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QProgressBar, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from llama_launcher.gpu.advice import suggest_backend
from llama_launcher.gpu.probe import merge_live
from llama_launcher.gpu.types import GPUInfo
from llama_launcher.gui import theme
from llama_launcher.gui.server_bar import ServerBar


def _fmt(value: float, unit: str, digits: int = 0) -> str:
    if value is None or value < 0:
        return "--"
    return f"{value:.{digits}f}{unit}"


def _bar(text_visible: bool = True) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setTextVisible(text_visible)
    bar.setFixedHeight(18)
    return bar


class _GpuCard(QGroupBox):
    """One physical GPU."""

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)

        self.vram_bar = _bar()
        self.util_bar = _bar()
        grid.addWidget(QLabel("VRAM"), 0, 0)
        grid.addWidget(self.vram_bar, 0, 1, 1, 3)
        grid.addWidget(QLabel("Utilization"), 1, 0)
        grid.addWidget(self.util_bar, 1, 1, 1, 3)

        self._values = {}
        cells = [("temp", "Temperature"), ("power", "Power"), ("clock", "Clocks"),
                 ("cc", "Compute target"), ("pci", "PCI"), ("driver", "Driver")]
        for i, (key, label) in enumerate(cells):
            row, col = 2 + i // 2, (i % 2) * 2
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color: {theme.FG_DIM};")
            val = QLabel("--")
            val.setStyleSheet(theme.monospace_style(theme.FG_TEXT, 11))
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(lbl, row, col)
            grid.addWidget(val, row, col + 1)
            self._values[key] = val
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

    def update(self, gpu: GPUInfo):
        title = f"GPU {gpu.index}: {gpu.device_name or 'Unknown'}"
        if gpu.integrated:
            title += "  (integrated)"
        self.setTitle(title)

        if gpu.vram_mb > 0 and gpu.used_vram_mb >= 0:
            pct = int(round(gpu.used_vram_mb / gpu.vram_mb * 100))
            self.vram_bar.setValue(min(max(pct, 0), 100))
            self.vram_bar.setFormat(
                f"{gpu.used_vram_mb / 1024:.1f} / {gpu.vram_mb / 1024:.1f} GB  ({pct}%)")
        elif gpu.vram_mb > 0:
            free = gpu.available_vram_mb
            self.vram_bar.setValue(int(round((gpu.vram_mb - free) / gpu.vram_mb * 100)))
            self.vram_bar.setFormat(f"{free / 1024:.1f} GB free of {gpu.vram_mb / 1024:.1f} GB")
        else:
            self.vram_bar.setValue(0)
            self.vram_bar.setFormat("VRAM size unknown")

        if gpu.utilization_pct >= 0:
            self.util_bar.setValue(int(round(gpu.utilization_pct)))
            self.util_bar.setFormat(f"{gpu.utilization_pct:.0f}%")
        else:
            self.util_bar.setValue(0)
            self.util_bar.setFormat("--")

        self._values["temp"].setText(_fmt(gpu.temperature_c, " °C"))
        power = _fmt(gpu.power_w, " W", 1 if 0 <= gpu.power_w < 10 else 0)
        if gpu.power_limit_w > 0:
            power += f" / {gpu.power_limit_w:.0f} W"
        self._values["power"].setText(power)
        clocks = _fmt(gpu.clock_mhz, " MHz")
        if gpu.mem_clock_mhz > 0:
            clocks += f" core, {gpu.mem_clock_mhz:.0f} MHz mem"
        self._values["clock"].setText(clocks)
        self._values["cc"].setText(gpu.compute_capability or "--")
        pci = gpu.pci_bus_id or "--"
        if gpu.pcie_link:
            pci += f"  ({gpu.pcie_link})"
        self._values["pci"].setText(pci)
        self._values["driver"].setText(gpu.driver_version or "--")


class GpuTab(QWidget):
    """Shows the selected server's GPUs, with live readings while connected."""

    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile: Optional[GPUInfo] = None
        self._cards = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # The GPU that matters is the one on the machine the model will run on.
        self.server_bar = ServerBar()
        layout.addWidget(self.server_bar)

        self.staleness_label = QLabel("")
        self.staleness_label.setStyleSheet(f"color: {theme.FG_MUTED};")
        self.staleness_label.setVisible(False)
        self.staleness_label.setWordWrap(True)
        layout.addWidget(self.staleness_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        container = QWidget()
        scroll.setWidget(container)
        body = QVBoxLayout(container)

        summary = QGroupBox("Summary")
        form = QFormLayout(summary)
        self._summary = {}
        for key, label in [("backend", "Backend"), ("gpus", "GPUs"), ("vram", "VRAM"),
                           ("ram", "System RAM"), ("rocm", "ROCm"), ("build", "llama.cpp build")]:
            val = QLabel("--")
            val.setWordWrap(True)
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(f"{label}:", val)
            self._summary[key] = val
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {theme.FG_DIM};")
        form.addRow("", self.message_label)
        body.addWidget(summary)

        self._cards_box = QVBoxLayout()
        body.addLayout(self._cards_box)

        procs = QGroupBox("Processes using the GPU")
        procs_layout = QVBoxLayout(procs)
        self.process_table = QTableWidget(0, 4)
        self.process_table.setHorizontalHeaderLabels(["PID", "Process", "VRAM", "GPU"])
        self.process_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.process_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.process_table.verticalHeader().setVisible(False)
        self.process_table.setMaximumHeight(180)
        procs_layout.addWidget(self.process_table)
        self.process_note = QLabel("")
        self.process_note.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 11px;")
        procs_layout.addWidget(self.process_note)
        body.addWidget(procs)
        body.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.refresh_gpu_btn = QPushButton("Re-detect GPU")
        self.refresh_gpu_btn.setToolTip("Re-read the selected server's hardware (requires a connection)")
        self.refresh_gpu_btn.clicked.connect(self.refresh_requested.emit)
        btn_row.addWidget(self.refresh_gpu_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def set_info_text(self, text: str):
        """Show a message instead of hardware facts (no profile yet)."""
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def info_text(self) -> str:
        return self.message_label.text()

    def set_staleness(self, message: str):
        """Say when the figures came from a host we are no longer connected to."""
        self.staleness_label.setText(message)
        self.staleness_label.setVisible(bool(message))

    def set_profile(self, gpu: Optional[GPUInfo], total_ram_mb: float = -1.0):
        """Show the cached hardware description of the selected server."""
        self._profile = gpu
        self.set_info_text("")
        if gpu is None:
            for v in self._summary.values():
                v.setText("--")
            self._render_cards([])
            self._render_processes([])
            return
        self._render(gpu, total_ram_mb)

    def update_live(self, live: Optional[GPUInfo]):
        """Overlay a live reading (utilization, memory, temperature, power)."""
        if live is None or self._profile is None:
            return
        merged = merge_live(self._profile, live)
        self._render_cards(merged.gpu_infos or [merged])
        self._render_processes(merged.processes)
        self._summary["vram"].setText(self._vram_text(merged))

    def clear_live(self):
        """Back to the cached profile (server disconnected)."""
        if self._profile is not None:
            self._render_cards(self._profile.gpu_infos or [self._profile])
            self._render_processes(self._profile.processes)

    # ------------------------------------------------------------------

    @staticmethod
    def _vram_text(gpu: GPUInfo) -> str:
        if gpu.vram_mb <= 0:
            return "unknown"
        text = f"{gpu.vram_mb / 1024:.1f} GB total"
        if gpu.used_vram_mb >= 0:
            text += f", {(gpu.vram_mb - gpu.used_vram_mb) / 1024:.1f} GB free"
        elif gpu.available_vram_mb and gpu.available_vram_mb != gpu.vram_mb:
            text += f", {gpu.available_vram_mb / 1024:.1f} GB free at last check"
        if gpu.gpu_count > 1:
            text += f" across {gpu.gpu_count} GPUs"
        return text

    def _render(self, gpu: GPUInfo, total_ram_mb: float):
        title, lines = suggest_backend(gpu)
        self._summary["backend"].setText(
            f"{title}" + ("" if gpu.supported else "  (driver problem, see below)"))
        cards = gpu.gpu_infos or [gpu]
        self._summary["gpus"].setText(
            "; ".join(f"{c.device_name or 'Unknown'} ({c.vram_mb / 1024:.1f} GB)" for c in cards)
            if gpu.vram_mb > 0 else (gpu.device_name or "none"))
        self._summary["vram"].setText(self._vram_text(gpu))
        self._summary["ram"].setText(f"{total_ram_mb / 1024:.1f} GB" if total_ram_mb and total_ram_mb > 0 else "--")
        self._summary["rocm"].setText(
            "available" if gpu.rocm_available else ("not installed" if gpu.vendor == "amd" else "n/a"))
        self._summary["build"].setText("\n".join(lines))
        self._render_cards(cards if gpu.vram_mb > 0 or gpu.backend != "cpu" else [])
        self._render_processes(gpu.processes)

    def _render_cards(self, cards):
        while len(self._cards) < len(cards):
            card = _GpuCard()
            self._cards_box.addWidget(card)
            self._cards.append(card)
        for widget, gpu in zip(self._cards, cards):
            widget.update(gpu)
            widget.setVisible(True)
        for widget in self._cards[len(cards):]:
            widget.setVisible(False)

    def _render_processes(self, processes):
        processes = sorted(processes or [], key=lambda p: -p.vram_mb)
        self.process_table.setRowCount(len(processes))
        for row, proc in enumerate(processes):
            for col, text in enumerate((str(proc.pid), proc.name or "(unknown)",
                                        f"{proc.vram_mb:.0f} MB", str(proc.gpu_index))):
                item = QTableWidgetItem(text)
                if col in (0, 2, 3):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.process_table.setItem(row, col, item)
        if processes:
            total = sum(p.vram_mb for p in processes)
            self.process_note.setText(
                f"{total / 1024:.1f} GB held by {len(processes)} process(es). "
                "VRAM held by other programs is not available to llama-server.")
        else:
            self.process_note.setText("No per-process VRAM information (none running, or the "
                                      "driver does not report it).")
