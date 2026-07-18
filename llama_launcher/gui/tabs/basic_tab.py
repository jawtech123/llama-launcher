"""Basic Config tab: preset bar, server selector, and the most-used options."""
import logging

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from llama_launcher.core.fields import CACHE_TYPES
from llama_launcher.gui.config_controller import ConfigController
from llama_launcher.gui.widgets import BinaryContextSpinBox, info_label

logger = logging.getLogger(__name__)

_MODEL_FILTER = "Model Files (*.gguf *.bin *.safetensors);;All Files (*)"


class BasicTab(QWidget):
    """Preset selector, server selector, and core launch options.

    All editor widgets are bound to the shared :class:`ConfigController`;
    cross-widget sync (including with the Advanced tab) is automatic.
    """

    def __init__(self, controller: ConfigController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()
        self._bind_fields()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Preset / server bar
        bar = QHBoxLayout()
        bar.setSpacing(6)

        bar.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        bar.addWidget(self.preset_combo)

        self.save_preset_btn = QPushButton("Save Preset")
        bar.addWidget(self.save_preset_btn)

        bar.addWidget(QLabel("Server:"))
        self.server_combo = QComboBox()
        self.server_combo.setMinimumWidth(140)
        bar.addWidget(self.server_combo)
        bar.addStretch()

        self.manage_presets_btn = QPushButton("Manage Presets...")
        bar.addWidget(self.manage_presets_btn)
        layout.addLayout(bar)

        # Basic configuration group
        config_group = QGroupBox("Basic Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(4)

        form = QFormLayout()
        form.setSpacing(6)

        self.model_combo = self._path_combo("Select or enter main model path (.gguf)")
        form.addRow(
            info_label("Main Model:", "Path to the main model file (.gguf, .bin, .safetensors)."),
            self._with_browse(self.model_combo, self._browse_model),
        )

        self.mmproj_combo = self._path_combo("mmproj model path (.gguf) - optional for vision models")
        form.addRow(
            info_label("mmproj Model:", "Path to mmproj model for vision-language models."),
            self._with_browse(self.mmproj_combo, self._browse_mmproj),
        )

        self.draft_combo = self._path_combo("Draft model path (.gguf) - optional for speculative decoding")
        form.addRow(
            info_label("Draft Model:", "Path to draft model for speculative decoding."),
            self._with_browse(self.draft_combo, self._browse_draft),
        )

        self.ctx_spin = BinaryContextSpinBox()
        self.ctx_spin.setToolTip("Size of the prompt context in tokens. Default: 4096.")
        self.ctx_spin.editingFinished.connect(self._on_ctx_editing_finished)
        self.ctx_spin.setMinimumWidth(120)
        form.addRow(info_label("Context Size:", "Size of the prompt context in tokens."),
                    self.ctx_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 256)
        self.threads_spin.setSpecialValueText("auto")
        self.threads_spin.setToolTip("CPU threads for generation. 0 = auto-detect.")
        self.threads_spin.setMinimumWidth(120)
        form.addRow(info_label("Threads:", "Number of CPU threads (0 = auto-detect)."),
                    self.threads_spin)

        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(-1, 999)
        self.gpu_spin.setSpecialValueText("all")
        self.gpu_spin.setToolTip("Max number of layers to store in VRAM. -1 = all.")
        self.gpu_spin.setMinimumWidth(120)
        form.addRow(info_label("GPU Layers:", "Max layers to VRAM (-1 = all)."),
                    self.gpu_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 65536)
        self.batch_spin.setToolTip("Logical maximum batch size (-b). Default: 2048.")
        self.batch_spin.setMinimumWidth(120)
        form.addRow(info_label("Batch Size:", "Logical maximum batch size. Default: 2048."),
                    self.batch_spin)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setToolTip("Port for llama-server to listen on. Default: 8080.")
        self.port_spin.setMinimumWidth(120)
        form.addRow(info_label("Port:", "Server listen port."), self.port_spin)

        self.host_input = QLineEdit()
        self.host_input.setToolTip("Host address for llama-server to bind to.")
        self.host_input.setMinimumWidth(150)
        form.addRow(info_label("Host:", "Server bind address."), self.host_input)

        self.cache_combo = QComboBox()
        self.cache_combo.addItems(list(CACHE_TYPES))
        self.cache_combo.setCurrentText("f16")
        self.cache_combo.setToolTip("KV cache data type for K and V. Default: f16.")
        self.cache_combo.setMinimumWidth(120)
        form.addRow(info_label("KV Cache Type:", "KV cache data type (K and V)."),
                    self.cache_combo)

        self.mmap_check = QCheckBox("Memory-map model (--mmap)")
        self.mmap_check.setToolTip(
            "Memory-map the model file. Disable for slower load but fewer pageouts."
        )
        form.addRow(info_label("mmap:", "Memory-map the model file. Default: enabled."),
                    self.mmap_check)

        config_layout.addLayout(form)
        layout.addWidget(config_group)
        layout.addStretch()

    def _path_combo(self, placeholder: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setPlaceholderText(placeholder)
        combo.setMinimumWidth(300)
        return combo

    def _with_browse(self, combo: QComboBox, handler) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(combo, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(handler)
        h.addWidget(btn)
        return row

    # ------------------------------------------------------------------
    # Controller binding
    # ------------------------------------------------------------------

    def _bind_fields(self):
        c = self.controller
        c.bind("model_path", self.model_combo.currentText,
               self.model_combo.setCurrentText, self.model_combo.currentTextChanged)
        c.bind("mmproj_model_path", self.mmproj_combo.currentText,
               self.mmproj_combo.setCurrentText, self.mmproj_combo.currentTextChanged)
        c.bind("draft_model_path", self.draft_combo.currentText,
               self.draft_combo.setCurrentText, self.draft_combo.currentTextChanged)
        c.bind("context_size", self.ctx_spin.value,
               self.ctx_spin.setValue, self.ctx_spin.valueChanged)
        c.bind("threads", self.threads_spin.value,
               self.threads_spin.setValue, self.threads_spin.valueChanged)
        c.bind("gpu_layers", self.gpu_spin.value,
               self.gpu_spin.setValue, self.gpu_spin.valueChanged)
        c.bind("batch_size", self.batch_spin.value,
               self.batch_spin.setValue, self.batch_spin.valueChanged)
        c.bind("port", self.port_spin.value,
               self.port_spin.setValue, self.port_spin.valueChanged)
        c.bind("host", self.host_input.text,
               self.host_input.setText, self.host_input.textChanged)
        c.bind("use_mmap", self.mmap_check.isChecked,
               self.mmap_check.setChecked, self.mmap_check.stateChanged)
        # The basic KV cache combo drives both K and V cache types.
        c.bind("cache_type_k", self.cache_combo.currentText,
               self.cache_combo.setCurrentText, self.cache_combo.currentTextChanged)
        self.cache_combo.currentTextChanged.connect(
            lambda v: c.set("cache_type_v", v))

    # ------------------------------------------------------------------
    # Browse handlers + context rounding
    # ------------------------------------------------------------------

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Main Model File", "", _MODEL_FILTER)
        if path:
            self.model_combo.setEditText(path)

    def _browse_mmproj(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select mmproj Model File", "", _MODEL_FILTER)
        if path:
            self.mmproj_combo.setEditText(path)

    def _browse_draft(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Draft Model File", "", _MODEL_FILTER)
        if path:
            self.draft_combo.setEditText(path)

    def _on_ctx_editing_finished(self):
        """Snap a hand-typed context size to the nearest binary size."""
        value = self.ctx_spin.value()
        rounded = self.ctx_spin.round_to_binary_context(value)
        if rounded != value:
            self.ctx_spin.setValue(rounded)
