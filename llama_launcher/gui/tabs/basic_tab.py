"""Basic Config tab: preset bar, server selector, and the most-used options."""
import logging

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from llama_launcher.core.fields import CACHE_TYPES
from llama_launcher.gui.config_controller import ConfigController
from llama_launcher.gui.dialogs.path_picker import MODEL_PATTERNS, RemotePathDialog
from llama_launcher.gui.server_bar import ServerBar
from llama_launcher.gui.widgets import info_label

logger = logging.getLogger(__name__)

_MODEL_FILTER = "Model Files (*.gguf *.bin *.safetensors);;All Files (*)"
_TEMPLATE_FILTER = "Jinja templates (*.jinja *.j2 *.txt);;All Files (*)"

#: Built-in template names llama-server accepts in --chat-template, for the hint.
_BUILTIN_TEMPLATES = "chatml, llama3, llama4, gemma, deepseek3, mistral-v7, phi4, gpt-oss, kimi-k2 ..."


class BasicTab(QWidget):
    """Preset selector, server selector, and core launch options.

    All editor widgets are bound to the shared :class:`ConfigController`;
    cross-widget sync (including with the Advanced tab) is automatic.
    """

    def __init__(self, controller: ConfigController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._manager = None
        self._server_name_provider = None
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

        self.benchmark_info_btn = QPushButton("ℹ")
        self.benchmark_info_btn.setFixedWidth(32)
        self.benchmark_info_btn.setToolTip(
            "Show benchmark info for this preset (tokens/sec from your last run).\n"
            "The info resets when any parameter is changed.")
        bar.addWidget(self.benchmark_info_btn)

        self.save_preset_btn = QPushButton("Save Preset")
        bar.addWidget(self.save_preset_btn)

        self.server_bar = ServerBar()
        # Kept as an alias: plenty of call sites drive the combo directly.
        self.server_combo = self.server_bar.combo
        bar.addWidget(self.server_bar)
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

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(256, 1048576)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setSpecialValueText("")
        self.ctx_spin.setToolTip("Size of the prompt context in tokens. Default: 4096.")
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
        self.gpu_spin.setSpecialValueText("auto")
        self.gpu_spin.setToolTip(
            "Layers to keep in VRAM. auto = llama-server offloads everything that fits "
            "(--fit); the estimate below assumes all layers.")
        self.gpu_spin.setMinimumWidth(120)
        form.addRow(info_label("GPU Layers:", "Layers kept in VRAM (auto = all that fit)."),
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

        # Chat template: some models (and some clients, e.g. coding agents that
        # rely on tool calls) need a template other than the one baked into the
        # GGUF. Jinja templates are multi-line, hence a real editor.
        template_group = QGroupBox("Chat Template (optional)")
        template_layout = QVBoxLayout(template_group)
        template_layout.setSpacing(4)

        template_hint = QLabel(
            "Leave empty to use the template embedded in the model. Enter a built-in "
            f"name ({_BUILTIN_TEMPLATES}) or paste a full Jinja template; or point at a "
            "template file on the server.")
        template_hint.setWordWrap(True)
        template_hint.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        template_layout.addWidget(template_hint)

        self.chat_template_edit = QPlainTextEdit()
        self.chat_template_edit.setPlaceholderText(
            "chatml   -or-   {%- for message in messages %} ... {%- endfor %}")
        self.chat_template_edit.setMaximumHeight(140)
        self.chat_template_edit.setTabChangesFocus(False)
        self.chat_template_edit.setToolTip(
            "Passed as --chat-template. A built-in name or a complete Jinja template.")
        template_layout.addWidget(self.chat_template_edit)

        template_buttons = QHBoxLayout()
        self.load_template_btn = QPushButton("Load Model's Template")
        self.load_template_btn.setToolTip(
            "Copy the chat template embedded in the selected (local) model into the "
            "editor so it can be adjusted.")
        template_buttons.addWidget(self.load_template_btn)
        self.import_template_btn = QPushButton("Import From File...")
        self.import_template_btn.setToolTip("Paste a template file from this machine into the editor.")
        self.import_template_btn.clicked.connect(self._import_template_file)
        template_buttons.addWidget(self.import_template_btn)
        self.clear_template_btn = QPushButton("Clear")
        self.clear_template_btn.clicked.connect(self.chat_template_edit.clear)
        template_buttons.addWidget(self.clear_template_btn)
        template_buttons.addStretch()
        template_layout.addLayout(template_buttons)

        file_row = QHBoxLayout()
        file_row.addWidget(info_label(
            "Template file:",
            "Passed as --chat-template-file: a Jinja file that exists on the machine "
            "running llama-server."))
        self.chat_template_file_edit = QLineEdit()
        self.chat_template_file_edit.setPlaceholderText("/path/on/server/template.jinja")
        file_row.addWidget(self.chat_template_file_edit, 1)
        browse_template_btn = QPushButton("Browse...")
        browse_template_btn.clicked.connect(self._browse_template_file)
        file_row.addWidget(browse_template_btn)
        template_layout.addLayout(file_row)

        layout.addWidget(template_group)
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
        c.bind("chat_template", self.chat_template_edit.toPlainText,
               self._set_template_text, self.chat_template_edit.textChanged)
        c.bind("chat_template_file", self.chat_template_file_edit.text,
               self.chat_template_file_edit.setText, self.chat_template_file_edit.textChanged)

    def _set_template_text(self, text: str):
        # setPlainText resets the cursor; skip it when nothing changed so typing
        # (which echoes back through the controller) is not disturbed.
        if self.chat_template_edit.toPlainText() != (text or ""):
            self.chat_template_edit.setPlainText(text or "")

    # ------------------------------------------------------------------
    # Browse handlers + context rounding
    # ------------------------------------------------------------------

    def set_browse_context(self, manager, server_name_provider) -> None:
        """Supply the server context used to decide where Browse looks.

        Without this the pickers stay local-only, which is what standalone
        (test) construction of the tab wants.
        """
        self._manager = manager
        self._server_name_provider = server_name_provider

    def _pick_model_file(self, title: str) -> str:
        """Pick a model file on whichever machine the selected server runs on.

        Browsing the launcher's own filesystem for a remote server would hand
        llama-server a path that does not exist on that host.
        """
        manager = getattr(self, "_manager", None)
        provider = getattr(self, "_server_name_provider", None)
        server_name = ""
        if manager is not None and provider is not None:
            try:
                server_name = provider() or ""
            except Exception:
                logger.debug("Could not determine the selected server", exc_info=True)

        if server_name and manager is not None and manager.is_remote(server_name):
            conn = manager.get_connection(server_name)
            if conn is not None:
                model_dirs = manager.get_server_model_directories(server_name)
                return RemotePathDialog.get_path(
                    self, conn, mode="file",
                    start_path=model_dirs[0] if model_dirs else "",
                    title=f"{title} on {server_name}",
                    name_filter=MODEL_PATTERNS,
                ) or ""

        path, _ = QFileDialog.getOpenFileName(self, title, "", _MODEL_FILTER)
        return path or ""

    def _browse_model(self):
        path = self._pick_model_file("Select Main Model File")
        if path:
            self.model_combo.setEditText(path)

    def _browse_mmproj(self):
        path = self._pick_model_file("Select mmproj Model File")
        if path:
            self.mmproj_combo.setEditText(path)

    def _browse_draft(self):
        path = self._pick_model_file("Select Draft Model File")
        if path:
            self.draft_combo.setEditText(path)

    def _browse_template_file(self):
        """Pick a template file on the machine that will run llama-server."""
        manager = getattr(self, "_manager", None)
        provider = getattr(self, "_server_name_provider", None)
        server_name = ""
        if manager is not None and provider is not None:
            try:
                server_name = provider() or ""
            except Exception:
                logger.debug("Could not determine the selected server", exc_info=True)
        if server_name and manager is not None and manager.is_remote(server_name):
            conn = manager.get_connection(server_name)
            if conn is not None:
                path = RemotePathDialog.get_path(
                    self, conn, mode="file", title=f"Select chat template on {server_name}")
                if path:
                    self.chat_template_file_edit.setText(path)
                return
        path, _ = QFileDialog.getOpenFileName(self, "Select Chat Template File", "", _TEMPLATE_FILTER)
        if path:
            self.chat_template_file_edit.setText(path)

    def _import_template_file(self):
        """Load a local template file's text into the inline editor."""
        path, _ = QFileDialog.getOpenFileName(self, "Import Chat Template", "", _TEMPLATE_FILTER)
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.chat_template_edit.setPlainText(f.read())
        except OSError as e:
            logger.warning("Could not read template %s: %s", path, e)
