"""Resource Usage Panel - displays estimated VRAM and RAM usage with animated bars."""
from typing import Optional

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QFrame)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QObject
from PySide6.QtGui import QColor, QPalette, QFont

from llama_launcher.estimator.types import ResourceEstimate
from llama_launcher.gui import theme


class _AnimatedValue(QObject):
    """Animated float value for smooth bar transitions."""

    def __init__(self, initial=0.0, parent=None):
        super().__init__(parent)
        self._current = initial
        self._target = initial

    @property
    def current(self) -> float:
        return self._current

    @property
    def target(self) -> float:
        return self._target

    @target.setter
    def target(self, value: float):
        self._target = value

    def step(self):
        diff = self._target - self._current
        if abs(diff) < 0.1:
            self._current = self._target
        else:
            self._current += diff * 0.15


class _ProgressBar(QFrame):
    """Custom progress bar with color thresholds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self._percentage = 0.0
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)

    def set_percentage(self, percentage: float):
        self._percentage = percentage
        self.update()

    def _get_color(self, pct: float) -> QColor:
        if pct <= 70:
            return QColor(76, 175, 80)
        elif pct <= 85:
            r = int(76 + (255 - 76) * (pct - 70) / 15)
            g = int(175 + (207 - 175) * (pct - 70) / 15)
            b = int(80 + (0 - 80) * (pct - 70) / 15)
            return QColor(r, g, b)
        elif pct <= 95:
            return QColor(255, int(207 - 57 * (pct - 85) / 10), 0)
        else:
            return QColor(244, 67, 54)

    def paintEvent(self, event):
        super().paintEvent(event)
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bar_color = self._get_color(self._percentage)
        fill_width = self.width() * min(self._percentage, 100) / 100

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bar_color)

        margin = 1
        rect_height = self.height() - margin * 2
        rect_width = max(2, int(fill_width) - margin)
        if rect_width > 0 and rect_height > 0:
            painter.drawRoundedRect(
                margin, margin,
                rect_width, rect_height,
                6, 6
            )
        painter.end()


class _ResourceRow(QWidget):
    """Single row showing label, progress bar, and value text."""

    def __init__(self, label: str, tooltip: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._tooltip = tooltip
        self._animated = _AnimatedValue(parent=self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(6)

        self._label_widget = QLabel(label)
        self._label_widget.setFixedWidth(85)
        self._label_widget.setToolTip(tooltip)
        self._label_widget.setStyleSheet(f"font-weight: bold; color: {theme.FG_TEXT}; font-size: 11px;")
        layout.addWidget(self._label_widget)

        self._bar = _ProgressBar()
        self._bar.setToolTip(tooltip)
        self._bar.setMinimumWidth(120)
        layout.addWidget(self._bar, 1)

        self._value_label = QLabel("0 / 0 GB")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value_label.setMinimumWidth(120)
        self._value_label.setToolTip(tooltip)
        self._value_label.setStyleSheet(theme.monospace_style(theme.FG_TEXT, 11))
        layout.addWidget(self._value_label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)

    def update_value(self, used_mb: float, total_mb: float):
        self._bar.setVisible(True)
        self._timer.stop()
        pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
        # Set bar immediately for responsiveness, then animate
        self._animated.target = pct
        self._bar.set_percentage(self._animated.current)
        self._bar.repaint()  # Force immediate repaint
        self._timer.start(16)
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024

        if total_mb <= 0:
            self._value_label.setText(f"{used_gb:.1f} GB")
        elif pct > 100:
            self._value_label.setText(f"{used_gb:.1f} / {total_gb:.1f} GB ({pct:.0f}%)")
        else:
            self._value_label.setText(f"{used_gb:.1f} / {total_gb:.1f} GB ({pct:.0f}%)")

    def _animate(self):
        self._animated.step()
        self._bar.set_percentage(self._animated.current)
        self._bar.repaint()
        if abs(self._animated.current - self._animated.target) < 0.1:
            self._bar.set_percentage(self._animated.target)
            self._bar.repaint()
            self._timer.stop()

    def set_static_value(self, text: str, tooltip: str = ""):
        self._timer.stop()
        self._bar.setVisible(False)
        self._value_label.setText(text)
        self._value_label.setToolTip(tooltip)


class _UsageRow(QWidget):
    """Compact row showing a realtime utilization percentage bar."""

    def __init__(self, label: str, tooltip: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(6)

        self._label_widget = QLabel(label)
        self._label_widget.setFixedWidth(85)
        self._label_widget.setToolTip(tooltip)
        self._label_widget.setStyleSheet(
            f"font-weight: bold; color: {theme.FG_TEXT}; font-size: 11px;")
        layout.addWidget(self._label_widget)

        self._bar = _ProgressBar()
        self._bar.setToolTip(tooltip)
        self._bar.setMinimumWidth(120)
        layout.addWidget(self._bar, 1)

        self._value_label = QLabel("--")
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value_label.setMinimumWidth(120)
        self._value_label.setToolTip(tooltip)
        self._value_label.setStyleSheet(theme.monospace_style(theme.FG_TEXT, 11))
        layout.addWidget(self._value_label)

    def set_usage(self, pct: float, text: str = ""):
        if pct < 0:
            self._bar.set_percentage(0)
            self._bar.repaint()
            self._value_label.setText("N/A")
            return
        self._bar.set_percentage(pct)
        self._bar.repaint()
        self._value_label.setText(text or f"{pct:.0f}%")


class _ModelInfoPanel(QWidget):
    """Compact single-line panel showing model metadata from GGUF."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 4, 0)
        layout.setSpacing(4)

        self._labels = {}
        for key, label in [("arch", "Arch"), ("params", "Params"), ("quant", "Quant"),
                           ("layers", "Layers"), ("native_ctx", "Native Ctx")]:
            container = QWidget()
            cl = QHBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)

            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color: {theme.FG_MUTED}; font-size: 9px;")
            cl.addWidget(lbl)

            val = QLabel("--")
            val.setStyleSheet(theme.monospace_style(theme.FG_DIM, 9))
            cl.addWidget(val)
            self._labels[key] = val

            layout.addWidget(container)

        layout.addStretch()

    def update(self, estimate: ResourceEstimate, selected_ctx: int = 0):
        meta = estimate.model_metadata

        arch_str = estimate.arch_name
        if not isinstance(arch_str, str):
            arch_str = str(arch_str)
        self._labels["arch"].setText(arch_str)

        params_str = estimate.total_params_display
        if not isinstance(params_str, str):
            params_str = str(params_str)
        self._labels["params"].setText(params_str)

        # Determine dominant quant type
        if meta.quant_types:
            dominant = max(meta.quant_types, key=meta.quant_types.get)
            self._labels["quant"].setText(str(dominant))
        elif estimate.quant_type and estimate.quant_type != "unknown":
            self._labels["quant"].setText(str(estimate.quant_type))
        else:
            self._labels["quant"].setText("--")

        if isinstance(meta.layers, int) and meta.layers > 0:
            self._labels["layers"].setText(str(meta.layers))
        else:
            self._labels["layers"].setText("--")

        if isinstance(meta.context_length, int) and meta.context_length > 0:
            self._labels["native_ctx"].setText(f"{meta.context_length:,}")
        else:
            self._labels["native_ctx"].setText("--")


class ResourceUsagePanel(QWidget):
    """Panel displaying estimated resource usage for the current configuration.

    Updates in real time as settings change. Informational only - never prevents launch.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._estimate: Optional[ResourceEstimate] = None
        self._build_ui()
        self.setStyleSheet(
            f"ResourceUsagePanel {{ background-color: {theme.BG_BASE}; border-top: 1px solid {theme.BORDER}; }}"
        )

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(1)

        # Header row with model info
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)

        header = QLabel("Resource Usage")
        header.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {theme.FG_DIM};")
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Model info panel (compact, right side)
        self._model_info = _ModelInfoPanel()
        self._model_info.setVisible(False)
        header_layout.addWidget(self._model_info)

        layout.addLayout(header_layout)

        # Primary usage bars
        self._vram_row = _ResourceRow(
            "VRAM",
            "VRAM consists of:\n"
            "- Model Weights (GPU offloaded layers)\n"
            "- KV Cache (context * batch * layers)\n"
            "- Inference Buffers (layer activations)\n"
            "- Backend Overhead (CUDA/Metal/Vulkan runtime)\n"
            "- Safety Margin"
        )
        layout.addWidget(self._vram_row)

        self._ram_row = _ResourceRow(
            "System RAM",
            "System RAM consists of:\n"
            "- Model Weights (CPU layers)\n"
            "- KV Cache (CPU side)\n"
            "- Inference Buffers (CPU side)\n"
            "- mmap File Overhead\n"
            "- Runtime Overhead"
        )
        layout.addWidget(self._ram_row)

        # Detail values row (compact)
        detail_layout = QHBoxLayout()
        detail_layout.setSpacing(10)
        detail_layout.setContentsMargins(8, 0, 4, 0)

        self._kv_label = QLabel("KV Cache: --")
        self._kv_label.setStyleSheet(theme.monospace_style(theme.FG_DIM, 10))
        self._kv_label.setToolTip(
            "KV Cache size = layers * n_kv_heads * head_size * ctx_size * parallel * (type_k + type_v)\n"
            "Depends on context size, parallel streams, cache type, and model architecture."
        )
        detail_layout.addWidget(self._kv_label)

        self._model_label = QLabel("Model: --")
        self._model_label.setStyleSheet(theme.monospace_style(theme.FG_DIM, 10))
        self._model_label.setToolTip("Total model weights size (from GGUF tensor info or file size).")
        detail_layout.addWidget(self._model_label)

        self._cpu_label = QLabel("CPU: --")
        self._cpu_label.setStyleSheet(theme.monospace_style(theme.FG_DIM, 10))
        self._cpu_label.setToolTip("Estimated CPU memory for offloaded layers and CPU-side buffers.")
        detail_layout.addWidget(self._cpu_label)

        self._gpu_label = QLabel("GPU: --")
        self._gpu_label.setStyleSheet(theme.monospace_style(theme.FG_DIM, 10))
        self._gpu_label.setToolTip("Estimated GPU memory for offloaded layers and GPU-side buffers.")
        detail_layout.addWidget(self._gpu_label)

        self._remaining_label = QLabel("Remaining: --")
        self._remaining_label.setStyleSheet(theme.monospace_style(theme.FG_MUTED, 10))
        self._remaining_label.setToolTip("Estimated VRAM remaining after loading this configuration.")
        detail_layout.addWidget(self._remaining_label)

        self._compute_label = QLabel("Buffers: --")
        self._compute_label.setStyleSheet(theme.monospace_style(theme.FG_MUTED, 10))
        self._compute_label.setToolTip("Compute buffers for inference (activations, attention workspace).")
        detail_layout.addWidget(self._compute_label)

        self._native_ctx_label = QLabel("Native Ctx: --")
        self._native_ctx_label.setStyleSheet(theme.monospace_style(theme.FG_MUTED, 10))
        self._native_ctx_label.setToolTip("Model's native training context length from GGUF metadata.")
        detail_layout.addWidget(self._native_ctx_label)

        detail_layout.addStretch()
        layout.addLayout(detail_layout)

        # Warning label
        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet(
            f"color: {theme.WARNING}; font-size: 10px; padding: 3px; "
            f"background-color: {theme.WARNING_BG}; border-radius: 4px;"
        )
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        # Live (realtime) usage section
        live_sep = QFrame()
        live_sep.setFrameShape(QFrame.Shape.HLine)
        live_sep.setStyleSheet(f"color: {theme.BORDER};")
        layout.addWidget(live_sep)

        live_header = QLabel("Live Usage")
        live_header.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        live_header.setStyleSheet(f"color: {theme.FG_DIM};")
        layout.addWidget(live_header)

        self._cpu_row = _UsageRow(
            "CPU",
            "Realtime CPU utilization of this machine (all cores, sampled over 0.25s)."
        )
        layout.addWidget(self._cpu_row)

        self._gpu_live_row = _UsageRow(
            "GPU",
            "Realtime GPU utilization and VRAM usage of this machine (nvidia-smi).\n"
            "Multi-GPU: shows the highest utilization and summed memory."
        )
        layout.addWidget(self._gpu_live_row)

    def update_live_usage(self, cpu_pct: float, gpu_pct: float,
                          gpu_mem_used_mb: float = -1.0,
                          gpu_mem_total_mb: float = -1.0):
        """Update the realtime CPU/GPU utilization bars (local machine)."""
        self._cpu_row.set_usage(cpu_pct)
        if gpu_pct < 0:
            self._gpu_live_row.set_usage(-1)
            return
        if gpu_mem_total_mb > 0:
            text = (f"{gpu_pct:.0f}%  |  "
                    f"{gpu_mem_used_mb / 1024:.1f}/{gpu_mem_total_mb / 1024:.1f} GB")
        else:
            text = f"{gpu_pct:.0f}%"
        self._gpu_live_row.set_usage(gpu_pct, text)

    def update_estimate(self, estimate: Optional[ResourceEstimate], selected_ctx: int = 0):
        """Update the panel with a new resource estimate."""
        if estimate is None:
            self._estimate = None
            self._model_info.setVisible(False)
            self._vram_row.set_static_value("VRAM: Select a model to estimate usage", "No model selected.")
            self._ram_row.set_static_value("RAM: --", "No model selected.")
            self._kv_label.setText("KV Cache: --")
            self._model_label.setText("Model: --")
            self._cpu_label.setText("CPU: --")
            self._gpu_label.setText("GPU: --")
            self._remaining_label.setText("Remaining: --")
            self._compute_label.setText("Buffers: --")
            self._native_ctx_label.setText("Native Ctx: --")
            self._remaining_label.setStyleSheet(
                theme.monospace_style(theme.FG_MUTED, 10)
            )
            self._warning_label.setVisible(False)
            return

        self._estimate = estimate

        # Model info panel
        if estimate.model_metadata.arch != "unknown" or estimate.layer_count > 0:
            try:
                self._model_info.setVisible(True)
                self._model_info.update(estimate, selected_ctx)
            except Exception:
                self._model_info.setVisible(False)

        # VRAM bar
        est_vram = float(estimate.estimated_vram_mb or estimate.total_vram_mb)
        gpu_total = float(estimate.gpu_total_vram_mb)
        if gpu_total > 0:
            self._vram_row.update_value(float(estimate.total_vram_mb), gpu_total)
        else:
            if est_vram > 0:
                self._vram_row.update_value(est_vram, est_vram)
            else:
                self._vram_row.update_value(0, 1)

        # RAM bar
        sys_ram = float(estimate.system_total_ram_mb)
        if sys_ram > 0:
            self._ram_row.update_value(float(estimate.total_ram_mb), sys_ram)
        else:
            self._ram_row.set_static_value(
                f"RAM: {float(estimate.total_ram_mb) / 1024:.1f} GB",
                "System RAM information not available."
            )

        # Multi-GPU info in VRAM tooltip
        if estimate.gpu_breakdowns and len(estimate.gpu_breakdowns) > 1:
            gpu_info_text = "\n".join(
                f"GPU {gb.gpu_index} ({str(gb.gpu_name)}): "
                f"{gb.total_used_mb / 1024:.1f}/{gb.total_vram_mb / 1024:.1f} GB"
                for gb in estimate.gpu_breakdowns
            )
            split_mode = str(estimate.split_mode) if estimate.split_mode else "layer"
            self._vram_row._label_widget.setToolTip(
                f"Multi-GPU ({len(estimate.gpu_breakdowns)} GPUs, {split_mode} split):\n{gpu_info_text}"
            )

        # Detail labels
        self._kv_label.setText(f"KV Cache: {self._fmt(float(estimate.kv_cache_total_mb))}")
        self._model_label.setText(f"Model: {self._fmt(float(estimate.model_weights_mb))}")
        self._cpu_label.setText(f"CPU: {self._fmt(float(estimate.total_ram_mb))}")
        self._gpu_label.setText(f"GPU: {self._fmt(float(estimate.total_vram_mb))}")
        self._compute_label.setText(f"Buffers: {self._fmt(float(estimate.compute_buffers_mb))}")
        self._set_native_context(int(estimate.training_context_length) if estimate.training_context_length else 0)

        # Remaining VRAM
        if float(estimate.gpu_total_vram_mb) > 0:
            remaining = float(estimate.gpu_remaining_vram_mb)
            remaining_gb = remaining / 1024
            if remaining < 0:
                self._remaining_label.setText(f"Remaining: {remaining_gb:.1f} GB (OVER)")
                self._remaining_label.setStyleSheet(
                    theme.monospace_style(theme.ERROR, 10) + " font-weight: bold;"
                )
            else:
                self._remaining_label.setText(f"Remaining: {remaining_gb:.1f} GB")
                self._remaining_label.setStyleSheet(
                    theme.monospace_style(theme.SUCCESS, 10)
                )
        else:
            self._remaining_label.setText("Remaining: N/A")
            self._remaining_label.setStyleSheet(
                theme.monospace_style(theme.FG_MUTED, 10)
            )

        # Warnings
        if estimate.warnings:
            warning_text = " | ".join(str(w) for w in estimate.warnings)
            self._warning_label.setText(warning_text)
            self._warning_label.setVisible(True)
            if estimate.vram_exceeded:
                self._warning_label.setStyleSheet(
                    f"color: {theme.ERROR}; font-size: 10px; padding: 3px; "
                    f"background-color: {theme.ERROR_BG}; border-radius: 4px;"
                )
            else:
                self._warning_label.setStyleSheet(
                    f"color: {theme.WARNING}; font-size: 10px; padding: 3px; "
                    f"background-color: {theme.WARNING_BG}; border-radius: 4px;"
                )
        else:
            self._warning_label.setVisible(False)

    @staticmethod
    def _fmt(mb: float) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    @staticmethod
    def _fmt_ctx(length: int) -> str:
        if length >= 1024:
            return f"{length / 1024:.0f}K"
        return str(length)

    def _set_native_context(self, ctx_length: int):
        if ctx_length > 0:
            self._native_ctx_label.setText(f"Native Ctx: {self._fmt_ctx(ctx_length)}")
            self._native_ctx_label.setStyleSheet(
                theme.monospace_style(theme.SUCCESS, 10)
            )
        else:
            self._native_ctx_label.setText("Native Ctx: unknown")
            self._native_ctx_label.setStyleSheet(
                theme.monospace_style(theme.FG_MUTED, 10)
            )
