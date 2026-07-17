"""PySide6 GUI Main Window."""
import sys
import subprocess
import json
import shlex
from pathlib import Path

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                   QHBoxLayout, QTabWidget, QPushButton, QTextEdit,
                                   QLineEdit, QLabel, QComboBox, QFileDialog, QGroupBox,
                                   QListWidget, QListWidgetItem, QSpinBox, QCheckBox,
                                   QDialog, QFormLayout, QInputDialog, QMessageBox,
                                   QScrollArea, QDoubleSpinBox, QToolButton, QStyle, QGridLayout)
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt, QProcess, QTimer, Slot, QThread, Signal
except ImportError:
    print("PySide6 not installed. Install with: pip install PySide6")
    sys.exit(1)

from core.config import LlamaConfig
from core.validator import validate_config
from core.builder import build_command
from core.presets import PresetManager
from gpu.detector import detect_gpu
from gpu.types import GPUInfo
from models.scanner import scan_models
from models.storage import load_models as load_cached_models, save_models as save_cached_models
from remote.manager import ServerManager
from remote.types import ServerConfig, ConnectionType, ConnectionState
from gui.server_management import ServerManagementWidget
from gui.resource_usage_panel import ResourceUsagePanel
from estimator import ResourceEstimator
from utils.helpers import get_file_size_mb


class _ModelScanWorker(QThread):
    """Runs model scan in a background thread to avoid blocking the UI."""
    scan_done = Signal(str, list)

    def __init__(self, manager, server_name, dirs_to_scan, is_local_only):
        super().__init__()
        self.manager = manager
        self.server_name = server_name
        self.dirs = dirs_to_scan
        self.is_local_only = is_local_only

    def run(self):
        results = []
        for dir_path in self.dirs:
            if self.is_local_only:
                if Path(dir_path).exists():
                    results.extend(scan_models(dir_path))
            else:
                try:
                    results.extend(self.manager.scan_server_models(self.server_name, dir_path))
                except Exception:
                    pass
        self.scan_done.emit(self.server_name, results)


class _PresetStyleWorker(QThread):
    """Refreshes preset combo styles in background to avoid blocking the UI."""
    style_done = Signal(list)

    def __init__(self, preset_names, server_names, manager):
        super().__init__()
        self.preset_names = preset_names
        self.server_names = server_names
        self.manager = manager

    def run(self):
        results = []
        for name in self.preset_names:
            try:
                preset_config = PresetManager.get_preset(name)
            except Exception:
                results.append((name, False))
                continue
            target = getattr(preset_config, 'target_server', None)
            offline = False
            if target:
                state = self.manager.get_connection_state(target)
                offline = state in (ConnectionState.DISCONNECTED, ConnectionState.ERROR)
            results.append((name, offline))
        self.style_done.emit(results)


class _GpuDetectWorker(QThread):
    """Detect GPU info in a background thread."""
    detect_done = Signal(str, object)

    def __init__(self):
        super().__init__()

    def run(self):
        import traceback
        gpu_info_dict = {}
        try:
            gpu_info = detect_gpu()
            gpu_info_dict = {
                'backend': gpu_info.backend,
                'device_name': gpu_info.device_name,
                'vram_mb': gpu_info.vram_mb,
                'available_vram_mb': gpu_info.available_vram_mb,
                'driver_version': gpu_info.driver_version,
                'supported': gpu_info.supported,
            }
            gi = gpu_info
            info_text = (
                f"Backend: {gi.backend}\n"
                f"Device: {gi.device_name or 'Unknown'}\n"
                f"VRAM: {gi.vram_mb} MB\n"
            )
            if gi.available_vram_mb > 0 and gi.available_vram_mb != gi.vram_mb:
                info_text += f"Available VRAM: {gi.available_vram_mb} MB\n"
            if gi.driver_version:
                info_text += f"Driver: {gi.driver_version}\n"
            info_text += f"Supported: {gi.supported}"

            if gi.backend == "cuda":
                try:
                    nvidia_sm = subprocess.run(
                        ['nvidia-smi', '--query-gpu=driver_version,name,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu,utilization.memory,power.draw,power.limit',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=2
                    )
                    if nvidia_sm.returncode == 0 and nvidia_sm.stdout.strip():
                        parts = [p.strip() for p in nvidia_sm.stdout.strip().split(',')[0].replace('\n', ',').split(',')]
                        if len(parts) >= 9:
                            info_text += (
                                f"\n--- NVIDIA-SMI Info ---\n"
                                f"Driver Version: {parts[0]}\n"
                                f"Device: {parts[1]}\n"
                                f"Temperature: {parts[2]}°C\n"
                                f"GPU Clock: {parts[3]} MHz | Memory Clock: {parts[4]} MHz\n"
                                f"GPU Util: {parts[5]}% | Mem Util: {parts[6]}%\n"
                                f"Power: {parts[7]}W / {parts[8]}W\n"
                            )
                except Exception:
                    pass
        except Exception as e:
            info_text = f"GPU detection failed: {e}\n{traceback.format_exc()}"
        self.detect_done.emit(info_text, gpu_info_dict)


class _GpuStatsWorker(QThread):
    """Lightweight worker to fetch live GPU stats (temp, util, power) only."""
    stats_done = Signal(str)

    def run(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(',')[0].replace('\n', ',').split(',')]
                if len(parts) >= 5:
                    line = (
                        f"\n--- Live Stats ---\n"
                        f"Temperature: {parts[0]}°C  |  GPU Util: {parts[1]}%  |  Mem Util: {parts[2]}%\n"
                        f"Power: {parts[3]}W / {parts[4]}W"
                    )
                    self.stats_done.emit(line)
                    return
        except Exception:
            pass
        self.stats_done.emit("")


class BinaryContextSpinBox(QSpinBox):
    """QSpinBox that jumps between valid binary context sizes."""
    BINARY_CONTEXTS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(256, 1048576)
        self.setSingleStep(1)
        
    def stepBy(self, steps: int):
        current = self.value()
        if current not in self.BINARY_CONTEXTS:
            current = self.round_to_binary_context(current)
        
        try:
            idx = self.BINARY_CONTEXTS.index(current)
        except ValueError:
            current = self.round_to_binary_context(current)
            idx = self.BINARY_CONTEXTS.index(current)
        
        if steps > 0:
            if idx < len(self.BINARY_CONTEXTS) - 1:
                new_val = self.BINARY_CONTEXTS[idx + 1]
            else:
                new_val = self.BINARY_CONTEXTS[-1]
        elif steps < 0:
            if idx > 0:
                new_val = self.BINARY_CONTEXTS[idx - 1]
            else:
                new_val = self.BINARY_CONTEXTS[0]
        else:
            return super().stepBy(steps)
            
        self.blockSignals(True)
        super().setValue(new_val)
        self.blockSignals(False)
        self.valueChanged.emit(new_val)
        
    def round_to_binary_context(self, size):
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
        
        if size - lower < upper - size:
            return lower
        else:
            return upper


class LlamaLauncherGUI(QMainWindow):
    """Main GUI Window for llama-launcher."""
    
    FOLDERS_FILE = Path.home() / ".llama_launcher_folders.json"
    
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig(model_path="", context_size=4096, gpu_layers=-1,
                                  threads=0, port=8080, host="127.0.0.1", backend="auto",
                                  use_mmap=True, kv_cache_type="f16", batch_size=512)
        self.models = []
        self.server_manager = ServerManager()
        self.gpu_info: GPUInfo = GPUInfo(backend="cpu", device_name="Detecting...")
        self.resource_estimator = ResourceEstimator()
        self._ctx_auto_set = True  # Allow auto-set on first model load
        self._active_workers = []
        self._ensure_local_server()
        self.init_ui()

    def _ensure_local_server(self):
        """Ensure a default local server exists."""
        configs = self.server_manager.get_all_configs()
        has_local = any(
            c.friendly_name == "Local Machine" and c.connection_type == ConnectionType.LOCAL
            for c in configs
        )
        if not has_local:
            local_config = ServerConfig(
                friendly_name="Local Machine",
                connection_type=ConnectionType.LOCAL,
                hostname="localhost",
                llama_server_path="llama-server",
                working_directory="~",
                enabled=True,
            )
            self.server_manager.add_server(local_config)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("llama-launcher")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget and tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Resource Usage Panel at the bottom
        self.resource_panel = ResourceUsagePanel()
        self.resource_panel.setMaximumHeight(160)
        self.resource_panel.setMinimumHeight(120)
        main_layout.addWidget(self.resource_panel)

        # Create tabs
        self.create_basic_config_tab()
        self.create_advanced_config_tab()
        self.create_gpu_info_tab()
        self.create_model_browser_tab()
        self.create_command_preview_tab()
        self.create_server_management_tab()
        self.create_import_launch_command_tab()
        
        # Load saved folders and rescan
        self.load_folders()

        # Defer GPU detection and model scan to after UI is shown
        QTimer.singleShot(200, self._deferred_init)
        
        # Load first preset at launch time
        presets = PresetManager.get_all_presets()
        if presets:
            QTimer.singleShot(100, lambda: self.load_preset(presets[0]))

    def closeEvent(self, event):
        """Clean up server connections on exit."""
        self.server_manager.disconnect_all()
        if hasattr(self, '_preset_style_timer'):
            self._preset_style_timer.stop()
        if hasattr(self, '_gpu_stats_timer'):
            self._gpu_stats_timer.stop()
        for w in getattr(self, '_active_workers', []):
            if w.isRunning():
                w.wait(1000)
        if hasattr(self, '_scan_worker') and self._scan_worker.isRunning():
            self._scan_worker.wait(1000)
        event.accept()

    def _create_info_label(self, text, tooltip):
        """Create a label with an information icon."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        label = QLabel(text)
        layout.addWidget(label)
        
        info_btn = QToolButton()
        info_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        info_btn.setToolTip(tooltip)
        info_btn.setFixedSize(16, 16)
        info_btn.setCursor(Qt.CursorShape.WhatsThisCursor)
        info_btn.setStyleSheet(
            "QToolButton { border: none; background: transparent; color: #2196F3; }"
            "QToolButton:hover { background-color: #ffffe0; color: #000000; border-radius: 4px; }"
        )
        layout.addWidget(info_btn)
        
        return widget
        
    def create_model_browser_tab(self):
        """Create model browser tab with multiple source folders."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Server selector for model browser
        browser_server_layout = QHBoxLayout()
        browser_server_layout.addWidget(QLabel("Server:"))
        self.browser_server_combo = QComboBox()
        self.browser_server_combo.setMinimumWidth(140)
        self._refresh_browser_server_combo()
        self.browser_server_combo.currentTextChanged.connect(self.on_browser_server_changed)
        browser_server_layout.addWidget(self.browser_server_combo, 1)
        browser_server_layout.addStretch()
        layout.addLayout(browser_server_layout)

        # Source folders section
        folders_group = QGroupBox("Source Folders")
        folders_layout = QVBoxLayout()

        self.folders_list = QListWidget()
        self.folders_list.setMaximumHeight(100)

        folders_btn_layout = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self.browse_add_folder)
        self.remove_folder_btn = QPushButton("Remove Selected")
        self.remove_folder_btn.clicked.connect(self.remove_selected_folder)

        folders_btn_layout.addWidget(self.add_folder_btn)
        folders_btn_layout.addWidget(self.remove_folder_btn)

        folders_layout.addWidget(self.folders_list)
        folders_layout.addLayout(folders_btn_layout)
        folders_group.setLayout(folders_layout)

        # Scan button
        self.scan_btn = QPushButton("Scan All Folders for Models")
        self.scan_btn.clicked.connect(self.scan_models)

        # Model list
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_selected)

        layout.addWidget(folders_group)
        layout.addWidget(self.scan_btn)
        layout.addWidget(QLabel("Detected Models:"))
        layout.addWidget(self.model_list)

        self.tab_widget.addTab(tab, "Model Browser")
        
    def create_gpu_info_tab(self):
        """Create GPU info panel tab."""
        tab = QWidget()
        tab.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.gpu_info_label = QLabel("Detecting GPU...")
        self.gpu_info_label.setWordWrap(True)
        self.gpu_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.gpu_info_label.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.gpu_info_label)
        
        # Refresh button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.refresh_gpu_btn = QPushButton("Refresh GPU Info")
        self.refresh_gpu_btn.clicked.connect(self._detect_gpu_async)
        btn_layout.addWidget(self.refresh_gpu_btn)
        
        layout.addLayout(btn_layout)
        
        self.tab_widget.addTab(tab, "GPU Info")
        
    def create_basic_config_tab(self):
        """Create preset selector and basic config editor tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Preset selector - compact horizontal layout at the top
        preset_layout = QHBoxLayout()
        preset_layout.setContentsMargins(0, 0, 0, 4)
        preset_layout.setSpacing(6)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PresetManager.get_all_presets())
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        QTimer.singleShot(500, self._refresh_preset_combo_styles)
        
        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.clicked.connect(self.save_current_preset)

        self.server_combo = QComboBox()
        self.server_combo.setMinimumWidth(140)
        self._refresh_server_combo()
        self.server_combo.currentTextChanged.connect(self.on_server_changed)

        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.save_preset_btn)
        preset_layout.addWidget(QLabel("Server:"))
        preset_layout.addWidget(self.server_combo)
        preset_layout.addStretch()
        
        self.manage_presets_btn = QPushButton("Manage Presets...")
        self.manage_presets_btn.clicked.connect(self.open_preset_manager)
        preset_layout.addWidget(self.manage_presets_btn)
        
        layout.addLayout(preset_layout)
        
        # Basic config editor
        config_group = QGroupBox("Basic Configuration")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(4)
        config_layout.setContentsMargins(10, 10, 10, 10)
        
        # Model paths and parameters - using single form layout
        config_form = QFormLayout()
        config_form.setSpacing(6)
        config_form.setContentsMargins(0, 0, 0, 0)
        
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText("Select or enter main model path (.gguf)")
        self.model_combo.setMinimumWidth(300)
        self.model_browse_btn = QPushButton("Browse...")
        self.model_browse_btn.clicked.connect(self.browse_model)
        config_form.addRow(self._create_info_label("Main Model:", "Path to the main model file (.gguf, .bin, .safetensors)."), self.model_combo)
        
        self.mmproj_combo = QComboBox()
        self.mmproj_combo.setEditable(True)
        self.mmproj_combo.setPlaceholderText("Select or enter mmproj model path (.gguf) - optional for vision models")
        self.mmproj_combo.setMinimumWidth(300)
        self.mmproj_browse_btn = QPushButton("Browse...")
        self.mmproj_browse_btn.clicked.connect(self.browse_mmproj_model)
        config_form.addRow(self._create_info_label("mmproj Model:", "Path to mmproj model for vision-language models."), self.mmproj_combo)
        
        self.draft_combo = QComboBox()
        self.draft_combo.setEditable(True)
        self.draft_combo.setPlaceholderText("Select or enter draft model path (.gguf) - optional for speculative decoding")
        self.draft_combo.setMinimumWidth(300)
        self.draft_browse_btn = QPushButton("Browse...")
        self.draft_browse_btn.clicked.connect(self.browse_draft_model)
        config_form.addRow(self._create_info_label("Draft Model:", "Path to draft model for speculative decoding."), self.draft_combo)
        
        # Context Size
        self.ctx_spin = BinaryContextSpinBox()
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setToolTip("Size of the prompt context in tokens. Default: 4096.")
        self.ctx_spin.editingFinished.connect(self.on_context_editing_finished)
        self.ctx_spin.valueChanged.connect(self._sync_ctx_to_adv)
        self.ctx_spin.setMinimumWidth(120)
        config_form.addRow(self._create_info_label("Context Size:", "Size of the prompt context in tokens."), self.ctx_spin)
        
        # Threads
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 256)
        self.threads_spin.setValue(0)
        self.threads_spin.setToolTip("Number of CPU threads to use during generation. Default: 0 (auto-detect).")
        self.threads_spin.setMinimumWidth(120)
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(self.threads_spin)
        threads_layout.addWidget(QLabel("(0=auto)"))
        threads_layout.addStretch()
        config_form.addRow(self._create_info_label("Threads:", "Number of CPU threads (0=auto)."), threads_layout)
        
        # GPU Layers
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(-1, 999)
        self.gpu_spin.setValue(-1)
        self.gpu_spin.setToolTip("Max number of layers to store in VRAM. Default: -1 (all layers to GPU).")
        self.gpu_spin.setMinimumWidth(120)
        gpu_layers_layout = QHBoxLayout()
        gpu_layers_layout.addWidget(self.gpu_spin)
        gpu_layers_layout.addWidget(QLabel("(-1=all)"))
        gpu_layers_layout.addStretch()
        config_form.addRow(self._create_info_label("GPU Layers:", "Max layers to VRAM (-1=all)."), gpu_layers_layout)
        
        # Batch Size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 4096)
        self.batch_spin.setValue(512)
        self.batch_spin.setToolTip("Physical maximum batch size for internal batching operations. Default: 512.")
        self.batch_spin.setMinimumWidth(120)
        config_form.addRow(self._create_info_label("Batch Size:", "Physical max batch size."), self.batch_spin)
        
        # Port
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8080)
        self.port_spin.setToolTip("Port for the llama-server to listen on. Default: 8080.")
        self.port_spin.setMinimumWidth(120)
        config_form.addRow(self._create_info_label("Port:", "Server listen port."), self.port_spin)
        
        # Host
        self.host_input = QLineEdit()
        self.host_input.setText("127.0.0.1")
        self.host_input.setToolTip("Host address for the llama-server to bind to. Default: 127.0.0.1.")
        self.host_input.setMinimumWidth(150)
        config_form.addRow(self._create_info_label("Host:", "Server bind address."), self.host_input)
        
        # KV Cache Type
        self.cache_combo = QComboBox()
        self.cache_combo.addItems(["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"])
        self.cache_combo.setToolTip("KV cache data type. Default: f16.")
        self.cache_combo.setMinimumWidth(120)
        config_form.addRow(self._create_info_label("KV Cache Type:", "KV cache data type."), self.cache_combo)
        
        # mmap checkbox
        self.mmap_check = QCheckBox("--mmap / --no-mmap")
        self.mmap_check.setChecked(True)
        self.mmap_check.setToolTip("Whether to memory-map model. Default: True (enabled). If disabled, slower load but may reduce pageouts if not using mlock.")
        config_form.addRow(self._create_info_label("--mmap / --no-mmap", "Whether to memory-map model. Default: True (enabled). If disabled, slower load but may reduce pageouts if not using mlock."), self.mmap_check)
        
        config_layout.addLayout(config_form)
        config_group.setLayout(config_layout)
        
        layout.addWidget(config_group)
        
        self.tab_widget.addTab(tab, "Basic Config")
        
    def create_advanced_config_tab(self):
        """Create advanced configuration tab with all llama-server options."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Preset selector - compact horizontal layout at the top
        preset_layout = QHBoxLayout()
        
        self.adv_preset_combo = QComboBox()
        self.adv_preset_combo.addItems(PresetManager.get_all_presets())
        self.adv_preset_combo.currentTextChanged.connect(self.load_preset)
        
        self.adv_save_preset_btn = QPushButton("Save Preset")
        self.adv_save_preset_btn.clicked.connect(self.save_current_preset)
        
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.adv_preset_combo)
        preset_layout.addWidget(self.adv_save_preset_btn)
        preset_layout.addStretch()
        
        self.manage_presets_btn_adv = QPushButton("Manage Presets...")
        self.manage_presets_btn_adv.clicked.connect(self.open_preset_manager)
        preset_layout.addWidget(self.manage_presets_btn_adv)
        
        layout.addLayout(preset_layout)
        
        # Use a scroll area for the advanced config
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Common params group
        common_group = QGroupBox("Common Parameters")
        common_layout = QFormLayout()
        
        self.adv_threads_spin = QSpinBox()
        self.adv_threads_spin.setRange(-1, 256)
        self.adv_threads_spin.setValue(0)
        self.adv_threads_spin.setToolTip("Number of CPU threads to use during generation. Default: 0 (auto-detect based on physical cores). Set to -1 for auto-detect based on physical cores. Example: 8 for 8 threads.")
        common_layout.addRow(self._create_info_label("Threads (-t, --threads):", "Number of CPU threads to use during generation. Default: 0 (auto-detect based on physical cores). Set to -1 for auto-detect based on physical cores. Example: 8 for 8 threads."), self.adv_threads_spin)
        
        self.adv_threads_batch_spin = QSpinBox()
        self.adv_threads_batch_spin.setRange(-1, 256)
        self.adv_threads_batch_spin.setValue(-1)
        self.adv_threads_batch_spin.setToolTip("Number of threads to use during batch and prompt processing. Default: -1 (same as --threads). Example: 4 for batch processing.")
        common_layout.addRow(self._create_info_label("Threads Batch (-tb, --threads-batch):", "Number of threads to use during batch and prompt processing. Default: -1 (same as --threads). Example: 4 for batch processing."), self.adv_threads_batch_spin)
        
        self.adv_ctx_spin = QSpinBox()
        self.adv_ctx_spin.setRange(0, 524288)
        self.adv_ctx_spin.setValue(0)
        self.adv_ctx_spin.setToolTip("Size of the prompt context in tokens. Default: 0 (load from model). Example: 4096 for 4K context, 131072 for 128K context.")
        common_layout.addRow(self._create_info_label("Context Size (-c, --ctx-size):", "Size of the prompt context in tokens. Default: 0 (load from model). Example: 4096 for 4K context, 131072 for 128K context."), self.adv_ctx_spin)
        
        self.adv_predict_spin = QSpinBox()
        self.adv_predict_spin.setRange(-1, 65536)
        self.adv_predict_spin.setValue(-1)
        self.adv_predict_spin.setToolTip("Number of tokens to predict. Default: -1 (infinity, generate until stop token or context limit). Example: 256 for 256 tokens.")
        common_layout.addRow(self._create_info_label("Predict (-n, --predict/--n-predict):", "Number of tokens to predict. Default: -1 (infinity, generate until stop token or context limit). Example: 256 for 256 tokens."), self.adv_predict_spin)
        
        self.adv_batch_spin = QSpinBox()
        self.adv_batch_spin.setRange(1, 65536)
        self.adv_batch_spin.setValue(2048)
        self.adv_batch_spin.setToolTip("Logical maximum batch size for processing. Default: 2048. Example: 2048 for standard batching.")
        common_layout.addRow(self._create_info_label("Batch Size (-b, --batch-size):", "Logical maximum batch size for processing. Default: 2048. Example: 2048 for standard batching."), self.adv_batch_spin)
        
        self.adv_ubatch_spin = QSpinBox()
        self.adv_ubatch_spin.setRange(1, 4096)
        self.adv_ubatch_spin.setValue(512)
        self.adv_ubatch_spin.setToolTip("Physical maximum batch size. Used for internal batching operations. Default: 512. Example: 512 for physical batch size.")
        common_layout.addRow(self._create_info_label("UBatch Size (-ub, --ubatch-size):", "Physical maximum batch size. Used for internal batching operations. Default: 512. Example: 512 for physical batch size."), self.adv_ubatch_spin)
        
        self.adv_keep_spin = QSpinBox()
        self.adv_keep_spin.setRange(-1, 131072)
        self.adv_keep_spin.setValue(0)
        self.adv_keep_spin.setToolTip("Number of tokens to keep from the initial prompt. Default: 0 (keep none). Set to -1 to keep all. Example: 100 to keep first 100 tokens.")
        common_layout.addRow(self._create_info_label("Keep (--keep):", "Number of tokens to keep from the initial prompt. Default: 0 (keep none). Set to -1 to keep all. Example: 100 to keep first 100 tokens."), self.adv_keep_spin)
        
        self.adv_rope_scaling_combo = QComboBox()
        self.adv_rope_scaling_combo.addItems(["none", "linear", "yarn"])
        self.adv_rope_scaling_combo.setToolTip("RoPE frequency scaling method for extending context: 'none' (default), 'linear', or 'yarn' (YaRN). Example: yarn for models trained on 4K but needing 32K context.")
        common_layout.addRow(self._create_info_label("RoPE Scaling (--rope-scaling):", "RoPE frequency scaling method for extending context: 'none' (default), 'linear', or 'yarn' (YaRN). Example: yarn for models trained on 4K but needing 32K context."), self.adv_rope_scaling_combo)
        
        self.adv_rope_scale_spin = QSpinBox()
        self.adv_rope_scale_spin.setRange(1, 100)
        self.adv_rope_scale_spin.setValue(1)
        self.adv_rope_scale_spin.setToolTip("RoPE context scaling factor, expands context by a factor of N. Default: 1. Example: 8 to expand 4K context to 32K.")
        common_layout.addRow(self._create_info_label("RoPE Scale (--rope-scale):", "RoPE context scaling factor, expands context by a factor of N. Default: 1. Example: 8 to expand 4K context to 32K."), self.adv_rope_scale_spin)
        
        self.adv_rope_freq_base_spin = QSpinBox()
        self.adv_rope_freq_base_spin.setRange(0, 10000000)
        self.adv_rope_freq_base_spin.setValue(0)
        self.adv_rope_freq_base_spin.setToolTip("RoPE base frequency, used by NTK-aware scaling. Default: 0 (load from model). Example: 1000000 for 1M context models.")
        common_layout.addRow(self._create_info_label("RoPE Freq Base (--rope-freq-base):", "RoPE base frequency, used by NTK-aware scaling. Default: 0 (load from model). Example: 1000000 for 1M context models."), self.adv_rope_freq_base_spin)
        
        self.adv_rope_freq_scale_spin = QSpinBox()
        self.adv_rope_freq_scale_spin.setRange(0, 100)
        self.adv_rope_freq_scale_spin.setValue(0)
        self.adv_rope_freq_scale_spin.setToolTip("RoPE frequency scaling factor, expands context by a factor of 1/N. Default: 0 (load from model). Example: 2 to halve the frequency.")
        common_layout.addRow(self._create_info_label("RoPE Freq Scale (--rope-freq-scale):", "RoPE frequency scaling factor, expands context by a factor of 1/N. Default: 0 (load from model). Example: 2 to halve the frequency."), self.adv_rope_freq_scale_spin)
        
        self.adv_cache_type_k_combo = QComboBox()
        self.adv_cache_type_k_combo.addItems(["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"])
        self.adv_cache_type_k_combo.setCurrentIndex(1)
        self.adv_cache_type_k_combo.setToolTip("KV cache data type for K (Key). Default: f16. Options: f32 (highest precision, most VRAM), f16/bf16 (balanced), q8_0/q5_1/q5_0/q4_1/q4_0 (quantized, less VRAM). Example: q8_0 for good balance.")
        common_layout.addRow(self._create_info_label("Cache Type K (-ctk, --cache-type-k):", "KV cache data type for K (Key). Default: f16. Options: f32 (highest precision, most VRAM), f16/bf16 (balanced), q8_0/q5_1/q5_0/q4_1/q4_0 (quantized, less VRAM). Example: q8_0 for good balance."), self.adv_cache_type_k_combo)

        self.adv_cache_type_v_combo = QComboBox()
        self.adv_cache_type_v_combo.addItems(["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"])
        self.adv_cache_type_v_combo.setCurrentIndex(1)
        self.adv_cache_type_v_combo.setToolTip("KV cache data type for V (Value). Default: f16. Same options as Cache Type K. Example: f16 for standard, q4_0 to save VRAM.")
        common_layout.addRow(self._create_info_label("Cache Type V (-ctv, --cache-type-v):", "KV cache data type for V (Value). Default: f16. Same options as Cache Type K. Example: f16 for standard, q4_0 to save VRAM."), self.adv_cache_type_v_combo)
        
        self.adv_mmap_check = QCheckBox("--mmap / --no-mmap")
        self.adv_mmap_check.setChecked(True)
        self.adv_mmap_check.setToolTip("Whether to memory-map model. Default: True (enabled). If disabled, slower load but may reduce pageouts if not using mlock.")
        common_layout.addRow(self._create_info_label("--mmap / --no-mmap", "Whether to memory-map model. Default: True (enabled). If disabled, slower load but may reduce pageouts if not using mlock."), self.adv_mmap_check)
        
        self.adv_mlock_check = QCheckBox("--mlock")
        self.adv_mlock_check.setToolTip("Force system to keep model in RAM rather than swapping or compressing. Prevents paging to disk. Default: disabled.")
        common_layout.addRow(self._create_info_label("--mlock", "Force system to keep model in RAM rather than swapping or compressing. Prevents paging to disk. Default: disabled."), self.adv_mlock_check)
        
        common_group.setLayout(common_layout)
        scroll_layout.addWidget(common_group)
        
        # GPU params group
        gpu_group = QGroupBox("GPU Parameters")
        gpu_layout = QFormLayout()
        
        self.adv_gpu_layers_spin = QSpinBox()
        self.adv_gpu_layers_spin.setRange(-1, 999)
        self.adv_gpu_layers_spin.setValue(-1)
        self.adv_gpu_layers_spin.setToolTip("Max number of layers to store in VRAM. Default: -1 (all layers to GPU). Set to -1 or 'all' for all layers to GPU. Example: 33 for 33 layers, -1 for all.")
        gpu_layout.addRow(self._create_info_label("GPU Layers (-ngl, --gpu-layers/--n-gpu-layers):", "Max number of layers to store in VRAM. Default: -1 (all layers to GPU). Set to -1 or 'all' for all layers to GPU. Example: 33 for 33 layers, -1 for all."), self.adv_gpu_layers_spin)
        
        self.adv_split_mode_combo = QComboBox()
        self.adv_split_mode_combo.addItems(["none", "layer", "row", "tensor"])
        self.adv_split_mode_combo.setCurrentIndex(1)
        self.adv_split_mode_combo.setToolTip("How to split model across multiple GPUs: 'none' (one GPU), 'layer' (default, split layers and KV), 'row' (split weights by rows), 'tensor' (experimental). Default: layer.")
        gpu_layout.addRow(self._create_info_label("Split Mode (-sm, --split-mode):", "How to split model across multiple GPUs: 'none' (one GPU), 'layer' (default, split layers and KV), 'row' (split weights by rows), 'tensor' (experimental). Default: layer."), self.adv_split_mode_combo)
        
        self.adv_tensor_split_input = QLineEdit()
        self.adv_tensor_split_input.setPlaceholderText("e.g. 3,1")
        self.adv_tensor_split_input.setToolTip("Fraction of model to offload to each GPU, comma-separated list of proportions. Default: not set (use split-mode=none or layer). Example: '3,1' for 75% on GPU 0, 25% on GPU 1.")
        gpu_layout.addRow(self._create_info_label("Tensor Split (-ts, --tensor-split):", "Fraction of model to offload to each GPU, comma-separated list of proportions. Default: not set (use split-mode=none or layer). Example: '3,1' for 75% on GPU 0, 25% on GPU 1."), self.adv_tensor_split_input)
        
        self.adv_main_gpu_spin = QSpinBox()
        self.adv_main_gpu_spin.setRange(0, 256)
        self.adv_main_gpu_spin.setValue(0)
        self.adv_main_gpu_spin.setToolTip("The GPU to use for the model (with split-mode=none), or for intermediate results and KV (with split-mode=row). Default: 0.")
        gpu_layout.addRow(self._create_info_label("Main GPU (-mg, --main-gpu):", "The GPU to use for the model (with split-mode=none), or for intermediate results and KV (with split-mode=row). Default: 0."), self.adv_main_gpu_spin)
        
        gpu_group.setLayout(gpu_layout)
        scroll_layout.addWidget(gpu_group)
        
        # Sampling params group
        sampling_group = QGroupBox("Sampling Parameters")
        sampling_layout = QFormLayout()
        
        self.adv_temp_spin = QDoubleSpinBox()
        self.adv_temp_spin.setRange(0.0, 2.0)
        self.adv_temp_spin.setSingleStep(0.05)
        self.adv_temp_spin.setValue(0.8)
        self.adv_temp_spin.setToolTip("Controls randomness: lower (0.1) = more deterministic/focused, higher (1.5) = more creative/random. Default: 0.8. Example: 0.7 for balanced.")
        sampling_layout.addRow(self._create_info_label("Temperature (-temp, --temperature):", "Controls randomness: lower (0.1) = more deterministic/focused, higher (1.5) = more creative/random. Default: 0.8. Example: 0.7 for balanced."), self.adv_temp_spin)
        
        self.adv_top_k_spin = QSpinBox()
        self.adv_top_k_spin.setRange(0, 128)
        self.adv_top_k_spin.setValue(40)
        self.adv_top_k_spin.setToolTip("Top-K sampling: restricts to top K most probable tokens. Set to 0 to disable. Default: 40. Example: 20 for more focused output.")
        sampling_layout.addRow(self._create_info_label("Top K (-top-k):", "Top-K sampling: restricts to top K most probable tokens. Set to 0 to disable. Default: 40. Example: 20 for more focused output."), self.adv_top_k_spin)
        
        self.adv_top_p_spin = QDoubleSpinBox()
        self.adv_top_p_spin.setRange(0.0, 1.0)
        self.adv_top_p_spin.setSingleStep(0.01)
        self.adv_top_p_spin.setValue(0.95)
        self.adv_top_p_spin.setToolTip("Top-P (nucleus) sampling: restricts to tokens with cumulative probability P. Set to 1.0 to disable. Default: 0.95. Example: 0.9 for balanced diversity.")
        sampling_layout.addRow(self._create_info_label("Top P (--top-p):", "Top-P (nucleus) sampling: restricts to tokens with cumulative probability P. Set to 1.0 to disable. Default: 0.95. Example: 0.9 for balanced diversity."), self.adv_top_p_spin)
        
        self.adv_min_p_spin = QDoubleSpinBox()
        self.adv_min_p_spin.setRange(0.0, 1.0)
        self.adv_min_p_spin.setSingleStep(0.01)
        self.adv_min_p_spin.setValue(0.05)
        self.adv_min_p_spin.setToolTip("Min-P sampling: filters tokens with probability below min-p * highest token probability. Set to 0.0 to disable. Default: 0.05.")
        sampling_layout.addRow(self._create_info_label("Min P (--min-p):", "Min-P sampling: filters tokens with probability below min-p * highest token probability. Set to 0.0 to disable. Default: 0.05."), self.adv_min_p_spin)
        
        self.adv_repeat_last_n_spin = QSpinBox()
        self.adv_repeat_last_n_spin.setRange(-1, 131072)
        self.adv_repeat_last_n_spin.setValue(64)
        self.adv_repeat_last_n_spin.setToolTip("Last N tokens to consider for penalty. Set to 0 to disable, -1 for context size. Default: 64. Example: 128 for longer context penalty.")
        sampling_layout.addRow(self._create_info_label("Repeat Last N (--repeat-last-n):", "Last N tokens to consider for penalty. Set to 0 to disable, -1 for context size. Default: 64. Example: 128 for longer context penalty."), self.adv_repeat_last_n_spin)
        
        self.adv_repeat_penalty_spin = QDoubleSpinBox()
        self.adv_repeat_penalty_spin.setRange(0.0, 2.0)
        self.adv_repeat_penalty_spin.setSingleStep(0.05)
        self.adv_repeat_penalty_spin.setValue(1.0)
        self.adv_repeat_penalty_spin.setToolTip("Penalize repeat sequence of tokens. Set to 1.0 to disable. >1.0 penalizes repeats. Example: 1.2 to reduce repetition.")
        sampling_layout.addRow(self._create_info_label("Repeat Penalty (--repeat-penalty):", "Penalize repeat sequence of tokens. Set to 1.0 to disable. >1.0 penalizes repeats. Example: 1.2 to reduce repetition."), self.adv_repeat_penalty_spin)
        
        self.adv_presence_penalty_spin = QDoubleSpinBox()
        self.adv_presence_penalty_spin.setRange(-2.0, 2.0)
        self.adv_presence_penalty_spin.setSingleStep(0.05)
        self.adv_presence_penalty_spin.setValue(0.0)
        self.adv_presence_penalty_spin.setToolTip("Repeat alpha presence penalty: >0 penalizes new tokens based on whether they appear in text. Default: 0.0. Example: 0.5 for more topics.")
        sampling_layout.addRow(self._create_info_label("Presence Penalty (--presence-penalty):", "Repeat alpha presence penalty: >0 penalizes new tokens based on whether they appear in text. Default: 0.0. Example: 0.5 for more topics."), self.adv_presence_penalty_spin)
        
        self.adv_frequency_penalty_spin = QDoubleSpinBox()
        self.adv_frequency_penalty_spin.setRange(-2.0, 2.0)
        self.adv_frequency_penalty_spin.setSingleStep(0.05)
        self.adv_frequency_penalty_spin.setValue(0.0)
        self.adv_frequency_penalty_spin.setToolTip("Repeat alpha frequency penalty: >0 penalizes new tokens based on their frequency in text. Default: 0.0. Example: 0.5 for less repetition.")
        sampling_layout.addRow(self._create_info_label("Frequency Penalty (--frequency-penalty):", "Repeat alpha frequency penalty: >0 penalizes new tokens based on their frequency in text. Default: 0.0. Example: 0.5 for less repetition."), self.adv_frequency_penalty_spin)

        self.adv_ignore_eos_check = QCheckBox("--ignore-eos/--no-ignore-eos")
        self.adv_ignore_eos_check.setToolTip("Ignore end-of-sequence token. Allows generation beyond EOS. Default: disabled.")
        sampling_layout.addRow(self._create_info_label("--ignore-eos/--no-ignore-eos", "Ignore end-of-sequence token. Default: disabled."), self.adv_ignore_eos_check)

        self.adv_backend_sampling_check = QCheckBox("--backend-sampling/--no-backend-sampling")
        self.adv_backend_sampling_check.setToolTip("Use backend (GPU) sampling for faster generation. Default: disabled.")
        sampling_layout.addRow(self._create_info_label("--backend-sampling/--no-backend-sampling", "Use backend (GPU) sampling. Default: disabled."), self.adv_backend_sampling_check)

        sampling_group.setLayout(sampling_layout)
        scroll_layout.addWidget(sampling_group)

        # Extra Sampling group
        extra_sampling_group = QGroupBox("Extra Sampling Parameters")
        extra_sampling_layout = QFormLayout()

        self.adv_samplers_input = QLineEdit()
        self.adv_samplers_input.setPlaceholderText("e.g. penalties;dry;top_k;top_p;temperature")
        self.adv_samplers_input.setToolTip("Semi-colon separated sampler chain. Default: penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature.")
        extra_sampling_layout.addRow(self._create_info_label("Samplers (--samplers):", "Semi-colon separated sampler chain."), self.adv_samplers_input)

        self.adv_sampler_seq_input = QLineEdit()
        self.adv_sampler_seq_input.setPlaceholderText("e.g. edskypmxt")
        self.adv_sampler_seq_input.setToolTip("Sampler sequence shorthand. Default: edskypmxt.")
        extra_sampling_layout.addRow(self._create_info_label("Sampler Seq (--sampler-seq):", "Sampler sequence shorthand."), self.adv_sampler_seq_input)

        self.adv_seed_spin = QSpinBox()
        self.adv_seed_spin.setRange(-1, 2147483647)
        self.adv_seed_spin.setValue(-1)
        self.adv_seed_spin.setToolTip("Random seed for reproducibility. -1=random. Default: -1.")
        extra_sampling_layout.addRow(self._create_info_label("Seed (--seed):", "Random seed. -1=random. Default: -1."), self.adv_seed_spin)

        self.adv_top_n_sigma_spin = QDoubleSpinBox()
        self.adv_top_n_sigma_spin.setRange(-1.0, 10.0)
        self.adv_top_n_sigma_spin.setSingleStep(0.1)
        self.adv_top_n_sigma_spin.setValue(-1.0)
        self.adv_top_n_sigma_spin.setToolTip("Top-N-Sigma sampling. -1=disabled. Default: -1.")
        extra_sampling_layout.addRow(self._create_info_label("Top N Sigma (--top-n-sigma):", "Top-N-Sigma sampling. -1=disabled. Default: -1."), self.adv_top_n_sigma_spin)

        self.adv_typical_p_spin = QDoubleSpinBox()
        self.adv_typical_p_spin.setRange(0.0, 1.0)
        self.adv_typical_p_spin.setSingleStep(0.01)
        self.adv_typical_p_spin.setValue(1.0)
        self.adv_typical_p_spin.setToolTip("Locally typical sampling. 1.0=disabled. Default: 1.0.")
        extra_sampling_layout.addRow(self._create_info_label("Typical P (--typical-p):", "Locally typical sampling. 1.0=disabled. Default: 1.0."), self.adv_typical_p_spin)

        self.adv_xtc_probability_spin = QDoubleSpinBox()
        self.adv_xtc_probability_spin.setRange(0.0, 1.0)
        self.adv_xtc_probability_spin.setSingleStep(0.01)
        self.adv_xtc_probability_spin.setValue(0.0)
        self.adv_xtc_probability_spin.setToolTip("XTC turn probability. 0=disabled. Default: 0.")
        extra_sampling_layout.addRow(self._create_info_label("XTC Probability (--xtc-probability):", "XTC turn probability. 0=disabled. Default: 0."), self.adv_xtc_probability_spin)

        self.adv_xtc_threshold_spin = QDoubleSpinBox()
        self.adv_xtc_threshold_spin.setRange(0.0, 1.0)
        self.adv_xtc_threshold_spin.setSingleStep(0.01)
        self.adv_xtc_threshold_spin.setValue(0.1)
        self.adv_xtc_threshold_spin.setToolTip("XTC token probability threshold. Default: 0.1.")
        extra_sampling_layout.addRow(self._create_info_label("XTC Threshold (--xtc-threshold):", "XTC token probability threshold. Default: 0.1."), self.adv_xtc_threshold_spin)

        self.adv_logit_bias_input = QLineEdit()
        self.adv_logit_bias_input.setPlaceholderText("e.g. 15032:1.0,15033:-1.0")
        self.adv_logit_bias_input.setToolTip("Logit bias for token control. Default: not set.")
        extra_sampling_layout.addRow(self._create_info_label("Logit Bias (--logit-bias):", "Logit bias for token control. Default: not set."), self.adv_logit_bias_input)

        extra_sampling_group.setLayout(extra_sampling_layout)
        scroll_layout.addWidget(extra_sampling_group)

        # Adaptive & DynaTemp group
        adaptive_group = QGroupBox("Adaptive & DynaTemp")
        adaptive_layout = QFormLayout()

        self.adv_adaptive_target_spin = QDoubleSpinBox()
        self.adv_adaptive_target_spin.setRange(-1.0, 10.0)
        self.adv_adaptive_target_spin.setSingleStep(0.1)
        self.adv_adaptive_target_spin.setValue(-1.0)
        self.adv_adaptive_target_spin.setToolTip("Adaptive temperature target. -1=disabled. Default: -1.")
        adaptive_layout.addRow(self._create_info_label("Adaptive Target (--adaptive-target):", "Adaptive temperature target. -1=disabled. Default: -1."), self.adv_adaptive_target_spin)

        self.adv_adaptive_decay_spin = QDoubleSpinBox()
        self.adv_adaptive_decay_spin.setRange(0.0, 1.0)
        self.adv_adaptive_decay_spin.setSingleStep(0.05)
        self.adv_adaptive_decay_spin.setValue(0.9)
        self.adv_adaptive_decay_spin.setToolTip("Adaptive temperature decay rate. Default: 0.9.")
        adaptive_layout.addRow(self._create_info_label("Adaptive Decay (--adaptive-decay):", "Adaptive temperature decay rate. Default: 0.9."), self.adv_adaptive_decay_spin)

        self.adv_dynatemp_range_spin = QDoubleSpinBox()
        self.adv_dynatemp_range_spin.setRange(0.0, 5.0)
        self.adv_dynatemp_range_spin.setSingleStep(0.1)
        self.adv_dynatemp_range_spin.setValue(0.0)
        self.adv_dynatemp_range_spin.setToolTip("DynaTemp range. 0=disabled. Default: 0.")
        adaptive_layout.addRow(self._create_info_label("DynaTemp Range (--dynatemp-range):", "DynaTemp range. 0=disabled. Default: 0."), self.adv_dynatemp_range_spin)

        self.adv_dynatemp_exp_spin = QDoubleSpinBox()
        self.adv_dynatemp_exp_spin.setRange(0.0, 10.0)
        self.adv_dynatemp_exp_spin.setSingleStep(0.1)
        self.adv_dynatemp_exp_spin.setValue(1.0)
        self.adv_dynatemp_exp_spin.setToolTip("DynaTemp exponent. Default: 1.0.")
        adaptive_layout.addRow(self._create_info_label("DynaTemp Exp (--dynatemp-exp):", "DynaTemp exponent. Default: 1.0."), self.adv_dynatemp_exp_spin)

        adaptive_group.setLayout(adaptive_layout)
        scroll_layout.addWidget(adaptive_group)

        # Mirostat group
        mirostat_group = QGroupBox("Mirostat Sampling")
        mirostat_layout = QFormLayout()

        self.adv_mirostat_spin = QSpinBox()
        self.adv_mirostat_spin.setRange(0, 2)
        self.adv_mirostat_spin.setValue(0)
        self.adv_mirostat_spin.setToolTip("Mirostat mode: 0=disabled, 1=Mirostat, 2=Mirostat 2.0. Default: 0.")
        mirostat_layout.addRow(self._create_info_label("Mirostat (--mirostat):", "Mirostat mode: 0=disabled, 1=Mirostat, 2=Mirostat 2.0. Default: 0."), self.adv_mirostat_spin)

        self.adv_mirostat_lr_spin = QDoubleSpinBox()
        self.adv_mirostat_lr_spin.setRange(0.0, 1.0)
        self.adv_mirostat_lr_spin.setSingleStep(0.01)
        self.adv_mirostat_lr_spin.setValue(0.1)
        self.adv_mirostat_lr_spin.setToolTip("Mirostat learning rate. Default: 0.1.")
        mirostat_layout.addRow(self._create_info_label("Mirostat LR (--mirostat-lr):", "Mirostat learning rate. Default: 0.1."), self.adv_mirostat_lr_spin)

        self.adv_mirostat_ent_spin = QDoubleSpinBox()
        self.adv_mirostat_ent_spin.setRange(0.0, 20.0)
        self.adv_mirostat_ent_spin.setSingleStep(0.1)
        self.adv_mirostat_ent_spin.setValue(5.0)
        self.adv_mirostat_ent_spin.setToolTip("Target Mirostat entropy. Default: 5.0.")
        mirostat_layout.addRow(self._create_info_label("Mirostat Entropy (--mirostat-ent):", "Target Mirostat entropy. Default: 5.0."), self.adv_mirostat_ent_spin)

        mirostat_group.setLayout(mirostat_layout)
        scroll_layout.addWidget(mirostat_group)

        # DRY (Don't Repeat Yourself) controls group
        dry_group = QGroupBox("DRY (Anti-Repetition) Controls")
        dry_layout = QFormLayout()

        self.adv_dry_multiplier_spin = QDoubleSpinBox()
        self.adv_dry_multiplier_spin.setRange(0.0, 10.0)
        self.adv_dry_multiplier_spin.setSingleStep(0.05)
        self.adv_dry_multiplier_spin.setValue(0.0)
        self.adv_dry_multiplier_spin.setToolTip("DRY penalty multiplier. 0=disabled. Default: 0.0.")
        dry_layout.addRow(self._create_info_label("DRY Multiplier (--dry-mul):", "DRY penalty multiplier. 0=disabled. Default: 0.0."), self.adv_dry_multiplier_spin)

        self.adv_dry_base_spin = QDoubleSpinBox()
        self.adv_dry_base_spin.setRange(0.0, 10.0)
        self.adv_dry_base_spin.setSingleStep(0.05)
        self.adv_dry_base_spin.setValue(1.75)
        self.adv_dry_base_spin.setToolTip("DRY penalty base value. Default: 1.75.")
        dry_layout.addRow(self._create_info_label("DRY Base (--dry-base):", "DRY penalty base value. Default: 1.75."), self.adv_dry_base_spin)

        self.adv_dry_allowed_length_spin = QSpinBox()
        self.adv_dry_allowed_length_spin.setRange(0, 32)
        self.adv_dry_allowed_length_spin.setValue(2)
        self.adv_dry_allowed_length_spin.setToolTip("DRY allowed repetition length. Default: 2.")
        dry_layout.addRow(self._create_info_label("DRY Allowed Length (--dry-allowed-length):", "DRY allowed repetition length. Default: 2."), self.adv_dry_allowed_length_spin)

        self.adv_dry_penalty_last_n_spin = QSpinBox()
        self.adv_dry_penalty_last_n_spin.setRange(-1, 131072)
        self.adv_dry_penalty_last_n_spin.setValue(-1)
        self.adv_dry_penalty_last_n_spin.setToolTip("Tokens to look back for DRY penalty. -1=context size. Default: -1.")
        dry_layout.addRow(self._create_info_label("DRY Penalty Last N (--dry-penalty-last-n):", "Tokens to look back for DRY penalty. -1=context size. Default: -1."), self.adv_dry_penalty_last_n_spin)

        self.adv_dry_sequence_breaker_input = QLineEdit()
        self.adv_dry_sequence_breaker_input.setPlaceholderText("e.g. \\n")
        self.adv_dry_sequence_breaker_input.setToolTip("Sequence breaker token for DRY penalty. Default: not set.")
        dry_layout.addRow(self._create_info_label("DRY Sequence Breaker (--dry-sequence-breaker):", "Sequence breaker token for DRY penalty. Default: not set."), self.adv_dry_sequence_breaker_input)

        dry_group.setLayout(dry_layout)
        scroll_layout.addWidget(dry_group)

        # Speculative params group
        spec_group = QGroupBox("Speculative Decoding Parameters")
        spec_layout = QFormLayout()
        
        self.adv_draft_n_max_spin = QSpinBox()
        self.adv_draft_n_max_spin.setRange(1, 32)
        self.adv_draft_n_max_spin.setValue(3)
        self.adv_draft_n_max_spin.setToolTip("Number of tokens to draft for speculative decoding. Higher = more aggressive speculation but more validation cost. Default: 3. Example: 5 for faster generation.")
        spec_layout.addRow(self._create_info_label("Draft N Max (--spec-draft-n-max):", "Number of tokens to draft for speculative decoding. Higher = more aggressive speculation but more validation cost. Default: 3. Example: 5 for faster generation."), self.adv_draft_n_max_spin)
        
        self.adv_draft_n_min_spin = QSpinBox()
        self.adv_draft_n_min_spin.setRange(0, 32)
        self.adv_draft_n_min_spin.setValue(0)
        self.adv_draft_n_min_spin.setToolTip("Minimum number of draft tokens to use for speculative decoding. Set to 0 to disable minimum. Default: 0.")
        spec_layout.addRow(self._create_info_label("Draft N Min (--spec-draft-n-min):", "Minimum number of draft tokens to use for speculative decoding. Set to 0 to disable minimum. Default: 0."), self.adv_draft_n_min_spin)
        
        self.adv_spec_type_combo = QComboBox()
        self.adv_spec_type_combo.addItems(["none", "ngram", "mod", "simple", "map_k", "map_k4v"])
        self.adv_spec_type_combo.setToolTip("Speculative decoding type: 'none' (disabled), 'ngram', 'mod', 'simple', 'map_k', or 'map_k4v'. Default: none.")
        spec_layout.addRow(self._create_info_label("Spec Type (--spec-type):", "Speculative decoding type: 'none' (disabled), 'ngram', 'mod', 'simple', 'map_k', or 'map_k4v'. Default: none."), self.adv_spec_type_combo)

        self.adv_spec_draft_model_input = QLineEdit()
        self.adv_spec_draft_model_input.setPlaceholderText("e.g. /path/to/draft-model.gguf")
        self.adv_spec_draft_model_input.setToolTip("Path to speculative draft model. Default: not set.")
        spec_layout.addRow(self._create_info_label("Spec Draft Model (--spec-draft-model):", "Path to speculative draft model. Default: not set."), self.adv_spec_draft_model_input)

        self.adv_spec_draft_threads_spin = QSpinBox()
        self.adv_spec_draft_threads_spin.setRange(0, 256)
        self.adv_spec_draft_threads_spin.setValue(0)
        self.adv_spec_draft_threads_spin.setToolTip("Number of threads for speculative draft model. 0=auto. Default: 0.")
        spec_layout.addRow(self._create_info_label("Spec Draft Threads (--spec-draft-threads):", "Threads for speculative draft model. 0=auto. Default: 0."), self.adv_spec_draft_threads_spin)

        self.adv_spec_draft_threads_batch_spin = QSpinBox()
        self.adv_spec_draft_threads_batch_spin.setRange(-1, 256)
        self.adv_spec_draft_threads_batch_spin.setValue(-1)
        self.adv_spec_draft_threads_batch_spin.setToolTip("Threads for speculative draft batch processing. -1=same as draft threads. Default: -1.")
        spec_layout.addRow(self._create_info_label("Spec Draft Threads Batch (--spec-draft-threads-batch):", "Threads for speculative draft batch. -1=same as draft threads. Default: -1."), self.adv_spec_draft_threads_batch_spin)

        self.adv_spec_draft_hf_input = QLineEdit()
        self.adv_spec_draft_hf_input.setPlaceholderText("e.g. org/model-name")
        self.adv_spec_draft_hf_input.setToolTip("Hugging Face repo for speculative draft model. Default: not set.")
        spec_layout.addRow(self._create_info_label("Spec Draft HF Repo (--spec-draft-hf):", "Hugging Face repo for speculative draft model. Default: not set."), self.adv_spec_draft_hf_input)

        self.adv_spec_draft_override_tensor_input = QLineEdit()
        self.adv_spec_draft_override_tensor_input.setPlaceholderText("e.g. KQV=1,O=1")
        self.adv_spec_draft_override_tensor_input.setToolTip("Override tensor split for speculative draft model. Default: not set.")
        spec_layout.addRow(self._create_info_label("Spec Draft Override Tensor (--spec-draft-override-tensor):", "Override tensor split for speculative draft model. Default: not set."), self.adv_spec_draft_override_tensor_input)

        self.adv_spec_draft_device_input = QLineEdit()
        self.adv_spec_draft_device_input.setPlaceholderText("e.g. cuda:0,cpu")
        self.adv_spec_draft_device_input.setToolTip("Device for speculative draft model. Default: not set.")
        spec_layout.addRow(self._create_info_label("Spec Draft Device (--spec-draft-device):", "Device for speculative draft model. Default: not set."), self.adv_spec_draft_device_input)

        self.adv_spec_draft_n_max_spin = QSpinBox()
        self.adv_spec_draft_n_max_spin.setRange(1, 32)
        self.adv_spec_draft_n_max_spin.setValue(3)
        self.adv_spec_draft_n_max_spin.setToolTip("Max draft tokens for spec decoding. Default: 3.")
        spec_layout.addRow(self._create_info_label("Spec Draft N Max (--spec-draft-n-max):", "Max draft tokens for spec decoding. Default: 3."), self.adv_spec_draft_n_max_spin)

        self.adv_spec_draft_n_min_spin = QSpinBox()
        self.adv_spec_draft_n_min_spin.setRange(0, 32)
        self.adv_spec_draft_n_min_spin.setValue(0)
        self.adv_spec_draft_n_min_spin.setToolTip("Min draft tokens for spec decoding. Default: 0.")
        spec_layout.addRow(self._create_info_label("Spec Draft N Min (--spec-draft-n-min):", "Min draft tokens for spec decoding. Default: 0."), self.adv_spec_draft_n_min_spin)

        self.adv_spec_draft_p_split_spin = QDoubleSpinBox()
        self.adv_spec_draft_p_split_spin.setRange(0.0, 1.0)
        self.adv_spec_draft_p_split_spin.setSingleStep(0.05)
        self.adv_spec_draft_p_split_spin.setValue(0.1)
        self.adv_spec_draft_p_split_spin.setToolTip("Spec draft p-split threshold. Default: 0.1.")
        spec_layout.addRow(self._create_info_label("Spec Draft P Split (--spec-draft-p-split):", "Spec draft p-split threshold. Default: 0.1."), self.adv_spec_draft_p_split_spin)

        self.adv_spec_draft_p_min_spin = QDoubleSpinBox()
        self.adv_spec_draft_p_min_spin.setRange(0.0, 1.0)
        self.adv_spec_draft_p_min_spin.setSingleStep(0.05)
        self.adv_spec_draft_p_min_spin.setValue(0.0)
        self.adv_spec_draft_p_min_spin.setToolTip("Spec draft p-min threshold. Default: 0.0.")
        spec_layout.addRow(self._create_info_label("Spec Draft P Min (--spec-draft-p-min):", "Spec draft p-min threshold. Default: 0.0."), self.adv_spec_draft_p_min_spin)

        self.adv_spec_draft_backend_sampling_check = QCheckBox("--spec-draft-backend-sampling/--no-spec-draft-backend-sampling")
        self.adv_spec_draft_backend_sampling_check.setChecked(True)
        self.adv_spec_draft_backend_sampling_check.setToolTip("Use backend sampling for spec draft. Default: enabled.")
        spec_layout.addRow(self._create_info_label("--spec-draft-backend-sampling", "Use backend sampling for spec draft. Default: enabled."), self.adv_spec_draft_backend_sampling_check)

        self.adv_spec_draft_ngl_spin = QSpinBox()
        self.adv_spec_draft_ngl_spin.setRange(-1, 999)
        self.adv_spec_draft_ngl_spin.setValue(-1)
        self.adv_spec_draft_ngl_spin.setToolTip("GPU layers for spec draft model. -1=all. Default: -1.")
        spec_layout.addRow(self._create_info_label("Spec Draft NGL (--spec-draft-ngl):", "GPU layers for spec draft model. Default: -1."), self.adv_spec_draft_ngl_spin)

        spec_group.setLayout(spec_layout)
        scroll_layout.addWidget(spec_group)
        
        # Spec draft cache types group
        spec_draft_cache_group = QGroupBox("Spec Draft Cache Types")
        spec_draft_cache_layout = QFormLayout()
        
        self.adv_spec_draft_type_k_combo = QComboBox()
        self.adv_spec_draft_type_k_combo.addItems(["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"])
        self.adv_spec_draft_type_k_combo.setCurrentIndex(1)
        self.adv_spec_draft_type_k_combo.setToolTip("KV cache data type for K (Key) in spec draft. Default: f16.")
        spec_draft_cache_layout.addRow(self._create_info_label("Spec Draft Type K (--spec-draft-type-k):", "KV cache data type for K (Key) in spec draft. Default: f16."), self.adv_spec_draft_type_k_combo)
        
        self.adv_spec_draft_type_v_combo = QComboBox()
        self.adv_spec_draft_type_v_combo.addItems(["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"])
        self.adv_spec_draft_type_v_combo.setCurrentIndex(1)
        self.adv_spec_draft_type_v_combo.setToolTip("KV cache data type for V (Value) in spec draft. Default: f16.")
        spec_draft_cache_layout.addRow(self._create_info_label("Spec Draft Type V (--spec-draft-type-v):", "KV cache data type for V (Value) in spec draft. Default: f16."), self.adv_spec_draft_type_v_combo)
        
        spec_draft_cache_group.setLayout(spec_draft_cache_layout)
        scroll_layout.addWidget(spec_draft_cache_group)

        # Grammar & Schema group
        grammar_schema_group = QGroupBox("Grammar & Schema Constraints")
        grammar_schema_layout = QFormLayout()

        self.adv_grammar_input = QLineEdit()
        self.adv_grammar_input.setPlaceholderText("e.g. [a-z]+")
        self.adv_grammar_input.setToolTip("GBNF grammar string for constrained generation. Default: not set.")
        grammar_schema_layout.addRow(self._create_info_label("Grammar (--grammar):", "GBNF grammar string for constrained generation. Default: not set."), self.adv_grammar_input)

        self.adv_grammar_file_input = QLineEdit()
        self.adv_grammar_file_input.setPlaceholderText("e.g. /path/to/grammar.gbnf")
        self.adv_grammar_file_input.setToolTip("Path to GBNF grammar file. Default: not set.")
        grammar_schema_layout.addRow(self._create_info_label("Grammar File (--grammar-file):", "Path to GBNF grammar file. Default: not set."), self.adv_grammar_file_input)

        self.adv_json_schema_input = QLineEdit()
        self.adv_json_schema_input.setPlaceholderText("e.g. {...}")
        self.adv_json_schema_input.setToolTip("JSON schema string for constrained generation. Default: not set.")
        grammar_schema_layout.addRow(self._create_info_label("JSON Schema (--json-schema):", "JSON schema string for constrained generation. Default: not set."), self.adv_json_schema_input)

        self.adv_json_schema_file_input = QLineEdit()
        self.adv_json_schema_file_input.setPlaceholderText("e.g. /path/to/schema.json")
        self.adv_json_schema_file_input.setToolTip("Path to JSON schema file. Default: not set.")
        grammar_schema_layout.addRow(self._create_info_label("JSON Schema File (--json-schema-file):", "Path to JSON schema file. Default: not set."), self.adv_json_schema_file_input)

        grammar_schema_group.setLayout(grammar_schema_layout)
        scroll_layout.addWidget(grammar_schema_group)

        # Spec N-gram & Mod Parameters group
        spec_ngram_group = QGroupBox("Speculative N-Gram / Mod Parameters")
        spec_ngram_layout = QFormLayout()

        self.adv_spec_ngram_mod_n_min_spin = QSpinBox()
        self.adv_spec_ngram_mod_n_min_spin.setRange(1, 128)
        self.adv_spec_ngram_mod_n_min_spin.setValue(48)
        self.adv_spec_ngram_mod_n_min_spin.setToolTip("Min n-gram mod sequence length. Default: 48.")
        spec_ngram_layout.addRow(self._create_info_label("N-Gram Mod N Min (--spec-ngram-mod-n-min):", "Min n-gram mod sequence length. Default: 48."), self.adv_spec_ngram_mod_n_min_spin)

        self.adv_spec_ngram_mod_n_max_spin = QSpinBox()
        self.adv_spec_ngram_mod_n_max_spin.setRange(1, 128)
        self.adv_spec_ngram_mod_n_max_spin.setValue(64)
        self.adv_spec_ngram_mod_n_max_spin.setToolTip("Max n-gram mod sequence length. Default: 64.")
        spec_ngram_layout.addRow(self._create_info_label("N-Gram Mod N Max (--spec-ngram-mod-n-max):", "Max n-gram mod sequence length. Default: 64."), self.adv_spec_ngram_mod_n_max_spin)

        self.adv_spec_ngram_mod_n_match_spin = QSpinBox()
        self.adv_spec_ngram_mod_n_match_spin.setRange(1, 128)
        self.adv_spec_ngram_mod_n_match_spin.setValue(24)
        self.adv_spec_ngram_mod_n_match_spin.setToolTip("N-gram mod match length. Default: 24.")
        spec_ngram_layout.addRow(self._create_info_label("N-Gram Mod N Match (--spec-ngram-mod-n-match):", "N-gram mod match length. Default: 24."), self.adv_spec_ngram_mod_n_match_spin)

        self.adv_spec_ngram_simple_size_n_spin = QSpinBox()
        self.adv_spec_ngram_simple_size_n_spin.setRange(1, 128)
        self.adv_spec_ngram_simple_size_n_spin.setValue(12)
        self.adv_spec_ngram_simple_size_n_spin.setToolTip("Simple n-gram size N. Default: 12.")
        spec_ngram_layout.addRow(self._create_info_label("Simple Size N (--spec-ngram-simple-size-n):", "Simple n-gram size N. Default: 12."), self.adv_spec_ngram_simple_size_n_spin)

        self.adv_spec_ngram_simple_size_m_spin = QSpinBox()
        self.adv_spec_ngram_simple_size_m_spin.setRange(1, 128)
        self.adv_spec_ngram_simple_size_m_spin.setValue(48)
        self.adv_spec_ngram_simple_size_m_spin.setToolTip("Simple n-gram size M. Default: 48.")
        spec_ngram_layout.addRow(self._create_info_label("Simple Size M (--spec-ngram-simple-size-m):", "Simple n-gram size M. Default: 48."), self.adv_spec_ngram_simple_size_m_spin)

        self.adv_spec_ngram_simple_min_hits_spin = QSpinBox()
        self.adv_spec_ngram_simple_min_hits_spin.setRange(0, 32)
        self.adv_spec_ngram_simple_min_hits_spin.setValue(1)
        self.adv_spec_ngram_simple_min_hits_spin.setToolTip("Simple n-gram min hits. Default: 1.")
        spec_ngram_layout.addRow(self._create_info_label("Simple Min Hits (--spec-ngram-simple-min-hits):", "Simple n-gram min hits. Default: 1."), self.adv_spec_ngram_simple_min_hits_spin)

        self.adv_spec_ngram_map_k_size_n_spin = QSpinBox()
        self.adv_spec_ngram_map_k_size_n_spin.setRange(1, 128)
        self.adv_spec_ngram_map_k_size_n_spin.setValue(12)
        self.adv_spec_ngram_map_k_size_n_spin.setToolTip("Map K size N. Default: 12.")
        spec_ngram_layout.addRow(self._create_info_label("Map K Size N (--spec-ngram-map-k-size-n):", "Map K size N. Default: 12."), self.adv_spec_ngram_map_k_size_n_spin)

        self.adv_spec_ngram_map_k_size_m_spin = QSpinBox()
        self.adv_spec_ngram_map_k_size_m_spin.setRange(1, 128)
        self.adv_spec_ngram_map_k_size_m_spin.setValue(48)
        self.adv_spec_ngram_map_k_size_m_spin.setToolTip("Map K size M. Default: 48.")
        spec_ngram_layout.addRow(self._create_info_label("Map K Size M (--spec-ngram-map-k-size-m):", "Map K size M. Default: 48."), self.adv_spec_ngram_map_k_size_m_spin)

        self.adv_spec_ngram_map_k_min_hits_spin = QSpinBox()
        self.adv_spec_ngram_map_k_min_hits_spin.setRange(0, 32)
        self.adv_spec_ngram_map_k_min_hits_spin.setValue(1)
        self.adv_spec_ngram_map_k_min_hits_spin.setToolTip("Map K min hits. Default: 1.")
        spec_ngram_layout.addRow(self._create_info_label("Map K Min Hits (--spec-ngram-map-k-min-hits):", "Map K min hits. Default: 1."), self.adv_spec_ngram_map_k_min_hits_spin)

        self.adv_spec_ngram_map_k4v_size_n_spin = QSpinBox()
        self.adv_spec_ngram_map_k4v_size_n_spin.setRange(1, 128)
        self.adv_spec_ngram_map_k4v_size_n_spin.setValue(12)
        self.adv_spec_ngram_map_k4v_size_n_spin.setToolTip("Map K4V size N. Default: 12.")
        spec_ngram_layout.addRow(self._create_info_label("Map K4V Size N (--spec-ngram-map-k4v-size-n):", "Map K4V size N. Default: 12."), self.adv_spec_ngram_map_k4v_size_n_spin)

        self.adv_spec_ngram_map_k4v_size_m_spin = QSpinBox()
        self.adv_spec_ngram_map_k4v_size_m_spin.setRange(1, 128)
        self.adv_spec_ngram_map_k4v_size_m_spin.setValue(48)
        self.adv_spec_ngram_map_k4v_size_m_spin.setToolTip("Map K4V size M. Default: 48.")
        spec_ngram_layout.addRow(self._create_info_label("Map K4V Size M (--spec-ngram-map-k4v-size-m):", "Map K4V size M. Default: 48."), self.adv_spec_ngram_map_k4v_size_m_spin)

        self.adv_spec_ngram_map_k4v_min_hits_spin = QSpinBox()
        self.adv_spec_ngram_map_k4v_min_hits_spin.setRange(0, 32)
        self.adv_spec_ngram_map_k4v_min_hits_spin.setValue(1)
        self.adv_spec_ngram_map_k4v_min_hits_spin.setToolTip("Map K4V min hits. Default: 1.")
        spec_ngram_layout.addRow(self._create_info_label("Map K4V Min Hits (--spec-ngram-map-k4v-min-hits):", "Map K4V min hits. Default: 1."), self.adv_spec_ngram_map_k4v_min_hits_spin)

        spec_ngram_group.setLayout(spec_ngram_layout)
        scroll_layout.addWidget(spec_ngram_group)

        # Threads & Batching params group
        threads_batch_group = QGroupBox("Threads & Batching Parameters")
        threads_batch_layout = QFormLayout()
        
        self.adv_cpu_mask_input = QLineEdit()
        self.adv_cpu_mask_input.setPlaceholderText("e.g. 0x1,0x2")
        self.adv_cpu_mask_input.setToolTip("CPU affinity mask for main thread. Default: not set.")
        threads_batch_layout.addRow(self._create_info_label("CPU Mask (--cpu-mask):", "CPU affinity mask for main thread. Default: not set."), self.adv_cpu_mask_input)
        
        self.adv_cpu_range_input = QLineEdit()
        self.adv_cpu_range_input.setPlaceholderText("e.g. 0-3,4-7")
        self.adv_cpu_range_input.setToolTip("CPU range for main thread. Default: not set.")
        threads_batch_layout.addRow(self._create_info_label("CPU Range (--cpu-range):", "CPU range for main thread. Default: not set."), self.adv_cpu_range_input)
        
        self.adv_cpu_strict_spin = QSpinBox()
        self.adv_cpu_strict_spin.setRange(0, 1)
        self.adv_cpu_strict_spin.setValue(0)
        self.adv_cpu_strict_spin.setToolTip("Strict CPU affinity for main thread. Default: 0 (disabled).")
        threads_batch_layout.addRow(self._create_info_label("CPU Strict (--cpu-strict):", "Strict CPU affinity for main thread. Default: 0 (disabled)."), self.adv_cpu_strict_spin)
        
        self.adv_prio_spin = QSpinBox()
        self.adv_prio_spin.setRange(-2, 2)
        self.adv_prio_spin.setValue(0)
        self.adv_prio_spin.setToolTip("Thread priority for main thread: -2=lowest, -1=low, 0=default, 1=high, 2=highest. Default: 0.")
        threads_batch_layout.addRow(self._create_info_label("Prio (--prio):", "Thread priority for main thread: -2=lowest, -1=low, 0=default, 1=high, 2=highest. Default: 0."), self.adv_prio_spin)
        
        self.adv_poll_spin = QSpinBox()
        self.adv_poll_spin.setRange(0, 1000)
        self.adv_poll_spin.setValue(50)
        self.adv_poll_spin.setToolTip("Polling interval for main thread in ms. Default: 50.")
        threads_batch_layout.addRow(self._create_info_label("Poll (--poll):", "Polling interval for main thread in ms. Default: 50."), self.adv_poll_spin)
        
        self.adv_cpu_mask_batch_input = QLineEdit()
        self.adv_cpu_mask_batch_input.setPlaceholderText("e.g. 0x1,0x2")
        self.adv_cpu_mask_batch_input.setToolTip("CPU affinity mask for batch thread. Default: not set.")
        threads_batch_layout.addRow(self._create_info_label("CPU Mask Batch (--cpu-mask-batch):", "CPU affinity mask for batch thread. Default: not set."), self.adv_cpu_mask_batch_input)
        
        self.adv_cpu_range_batch_input = QLineEdit()
        self.adv_cpu_range_batch_input.setPlaceholderText("e.g. 0-3,4-7")
        self.adv_cpu_range_batch_input.setToolTip("CPU range for batch thread. Default: not set.")
        threads_batch_layout.addRow(self._create_info_label("CPU Range Batch (--cpu-range-batch):", "CPU range for batch thread. Default: not set."), self.adv_cpu_range_batch_input)
        
        self.adv_cpu_strict_batch_spin = QSpinBox()
        self.adv_cpu_strict_batch_spin.setRange(0, 1)
        self.adv_cpu_strict_batch_spin.setValue(0)
        self.adv_cpu_strict_batch_spin.setToolTip("Strict CPU affinity for batch thread. Default: 0 (disabled).")
        threads_batch_layout.addRow(self._create_info_label("CPU Strict Batch (--cpu-strict-batch):", "Strict CPU affinity for batch thread. Default: 0 (disabled)."), self.adv_cpu_strict_batch_spin)
        
        self.adv_prio_batch_spin = QSpinBox()
        self.adv_prio_batch_spin.setRange(-2, 2)
        self.adv_prio_batch_spin.setValue(0)
        self.adv_prio_batch_spin.setToolTip("Thread priority for batch thread: -2=lowest, -1=low, 0=default, 1=high, 2=highest. Default: 0.")
        threads_batch_layout.addRow(self._create_info_label("Prio Batch (--prio-batch):", "Thread priority for batch thread: -2=lowest, -1=low, 0=default, 1=high, 2=highest. Default: 0."), self.adv_prio_batch_spin)
        
        self.adv_poll_batch_spin = QSpinBox()
        self.adv_poll_batch_spin.setRange(0, 1000)
        self.adv_poll_batch_spin.setValue(0)
        self.adv_poll_batch_spin.setToolTip("Polling interval for batch thread in ms. Default: 0 (inherit from main).")
        threads_batch_layout.addRow(self._create_info_label("Poll Batch (--poll-batch):", "Polling interval for batch thread in ms. Default: 0 (inherit from main)."), self.adv_poll_batch_spin)
        
        threads_batch_group.setLayout(threads_batch_layout)
        scroll_layout.addWidget(threads_batch_group)

        # Spec Draft CPU Affinity group
        spec_cpu_group = QGroupBox("Spec Draft CPU Affinity")
        spec_cpu_layout = QFormLayout()

        self.adv_spec_draft_cpu_mask_input = QLineEdit()
        self.adv_spec_draft_cpu_mask_input.setPlaceholderText("e.g. 0x1,0x2")
        self.adv_spec_draft_cpu_mask_input.setToolTip("CPU affinity mask for spec draft. Default: not set.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft CPU Mask (--spec-draft-cpu-mask):", "CPU affinity mask for spec draft. Default: not set."), self.adv_spec_draft_cpu_mask_input)

        self.adv_spec_draft_cpu_range_input = QLineEdit()
        self.adv_spec_draft_cpu_range_input.setPlaceholderText("e.g. 0-3,4-7")
        self.adv_spec_draft_cpu_range_input.setToolTip("CPU range for spec draft. Default: not set.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft CPU Range (--spec-draft-cpu-range):", "CPU range for spec draft. Default: not set."), self.adv_spec_draft_cpu_range_input)

        self.adv_spec_draft_cpu_strict_spin = QSpinBox()
        self.adv_spec_draft_cpu_strict_spin.setRange(0, 1)
        self.adv_spec_draft_cpu_strict_spin.setValue(0)
        self.adv_spec_draft_cpu_strict_spin.setToolTip("Strict CPU affinity for spec draft. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft CPU Strict (--spec-draft-cpu-strict):", "Strict CPU affinity for spec draft. Default: 0."), self.adv_spec_draft_cpu_strict_spin)

        self.adv_spec_draft_prio_spin = QSpinBox()
        self.adv_spec_draft_prio_spin.setRange(-2, 2)
        self.adv_spec_draft_prio_spin.setValue(0)
        self.adv_spec_draft_prio_spin.setToolTip("Thread priority for spec draft. 0=default. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft Prio (--spec-draft-prio):", "Thread priority for spec draft. Default: 0."), self.adv_spec_draft_prio_spin)

        self.adv_spec_draft_poll_spin = QSpinBox()
        self.adv_spec_draft_poll_spin.setRange(0, 1000)
        self.adv_spec_draft_poll_spin.setValue(0)
        self.adv_spec_draft_poll_spin.setToolTip("Polling interval for spec draft in ms. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft Poll (--spec-draft-poll):", "Polling interval for spec draft in ms. Default: 0."), self.adv_spec_draft_poll_spin)

        self.adv_spec_draft_cpu_mask_batch_input = QLineEdit()
        self.adv_spec_draft_cpu_mask_batch_input.setPlaceholderText("e.g. 0x1,0x2")
        self.adv_spec_draft_cpu_mask_batch_input.setToolTip("CPU mask for spec draft batch. Default: not set.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft CPU Mask Batch (--spec-draft-cpu-mask-batch):", "CPU mask for spec draft batch. Default: not set."), self.adv_spec_draft_cpu_mask_batch_input)

        self.adv_spec_draft_cpu_strict_batch_spin = QSpinBox()
        self.adv_spec_draft_cpu_strict_batch_spin.setRange(0, 1)
        self.adv_spec_draft_cpu_strict_batch_spin.setValue(0)
        self.adv_spec_draft_cpu_strict_batch_spin.setToolTip("Strict CPU affinity for spec draft batch. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft CPU Strict Batch (--spec-draft-cpu-strict-batch):", "Strict CPU affinity for spec draft batch. Default: 0."), self.adv_spec_draft_cpu_strict_batch_spin)

        self.adv_spec_draft_prio_batch_spin = QSpinBox()
        self.adv_spec_draft_prio_batch_spin.setRange(-2, 2)
        self.adv_spec_draft_prio_batch_spin.setValue(0)
        self.adv_spec_draft_prio_batch_spin.setToolTip("Priority for spec draft batch. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft Prio Batch (--spec-draft-prio-batch):", "Priority for spec draft batch. Default: 0."), self.adv_spec_draft_prio_batch_spin)

        self.adv_spec_draft_poll_batch_spin = QSpinBox()
        self.adv_spec_draft_poll_batch_spin.setRange(0, 1000)
        self.adv_spec_draft_poll_batch_spin.setValue(0)
        self.adv_spec_draft_poll_batch_spin.setToolTip("Polling interval for spec draft batch. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft Poll Batch (--spec-draft-poll-batch):", "Polling interval for spec draft batch. Default: 0."), self.adv_spec_draft_poll_batch_spin)

        self.adv_spec_draft_cpu_moe_check = QCheckBox("--spec-draft-cpu-moe/--no-spec-draft-cpu-moe")
        self.adv_spec_draft_cpu_moe_check.setToolTip("Offload spec draft MoE layers to CPU. Default: disabled.")
        spec_cpu_layout.addRow(self._create_info_label("--spec-draft-cpu-moe", "Offload spec draft MoE layers to CPU. Default: disabled."), self.adv_spec_draft_cpu_moe_check)

        self.adv_spec_draft_n_cpu_moe_spin = QSpinBox()
        self.adv_spec_draft_n_cpu_moe_spin.setRange(0, 256)
        self.adv_spec_draft_n_cpu_moe_spin.setValue(0)
        self.adv_spec_draft_n_cpu_moe_spin.setToolTip("CPU threads for spec draft MoE. 0=auto. Default: 0.")
        spec_cpu_layout.addRow(self._create_info_label("Spec Draft N CPU MoE (--spec-draft-n-cpu-moe):", "CPU threads for spec draft MoE. Default: 0."), self.adv_spec_draft_n_cpu_moe_spin)

        spec_cpu_group.setLayout(spec_cpu_layout)
        scroll_layout.addWidget(spec_cpu_group)

        # Context & Predict params group
        context_predict_group = QGroupBox("Context & Predict Parameters")
        context_predict_layout = QFormLayout()
        
        self.adv_swa_full_check = QCheckBox("--swa-full/--no-swa-full")
        self.adv_swa_full_check.setToolTip("Enable full SWA (Sliding Window Attention). Default: disabled.")
        context_predict_layout.addRow(self._create_info_label("--swa-full/--no-swa-full", "Enable full SWA (Sliding Window Attention). Default: disabled."), self.adv_swa_full_check)
        
        self.adv_flash_attn_combo = QComboBox()
        self.adv_flash_attn_combo.addItems(["auto", "enable", "disable"])
        self.adv_flash_attn_combo.setToolTip("Flash attention mode: 'auto', 'enable', or 'disable'. Default: auto.")
        context_predict_layout.addRow(self._create_info_label("Flash Attention (--flash-attn):", "Flash attention mode: 'auto', 'enable', or 'disable'. Default: auto."), self.adv_flash_attn_combo)
        
        self.adv_perf_check = QCheckBox("--perf/--no-perf")
        self.adv_perf_check.setToolTip("Print performance information. Default: disabled.")
        context_predict_layout.addRow(self._create_info_label("--perf/--no-perf", "Print performance information. Default: disabled."), self.adv_perf_check)
        
        self.adv_escape_check = QCheckBox("--escape/--no-escape")
        self.adv_escape_check.setChecked(True)
        self.adv_escape_check.setToolTip("Process escape sequences in input. Default: enabled.")
        context_predict_layout.addRow(self._create_info_label("--escape/--no-escape", "Process escape sequences in input. Default: enabled."), self.adv_escape_check)
        
        context_predict_group.setLayout(context_predict_layout)
        scroll_layout.addWidget(context_predict_group)
        
        # YaRN params group
        yarn_group = QGroupBox("YaRN RoPE Extension Parameters")
        yarn_layout = QFormLayout()
        
        self.adv_yarn_orig_ctx_spin = QSpinBox()
        self.adv_yarn_orig_ctx_spin.setRange(0, 131072)
        self.adv_yarn_orig_ctx_spin.setValue(0)
        self.adv_yarn_orig_ctx_spin.setToolTip("Original context size for YaRN scaling. Default: 0 (load from model). Example: 4096 for 4K original context.")
        yarn_layout.addRow(self._create_info_label("YaRN Orig Context (--yarn-orig-ctx):", "Original context size for YaRN scaling. Default: 0 (load from model). Example: 4096 for 4K original context."), self.adv_yarn_orig_ctx_spin)
        
        self.adv_yarn_ext_factor_spin = QDoubleSpinBox()
        self.adv_yarn_ext_factor_spin.setRange(-1.0, 100.0)
        self.adv_yarn_ext_factor_spin.setSingleStep(0.1)
        self.adv_yarn_ext_factor_spin.setValue(-1.0)
        self.adv_yarn_ext_factor_spin.setToolTip("YaRN extrapolation mix factor. Default: -1.0 (auto). Example: 1.0 for full extrapolation.")
        yarn_layout.addRow(self._create_info_label("YaRN Ext Factor (--yarn-ext-factor):", "YaRN extrapolation mix factor. Default: -1.0 (auto). Example: 1.0 for full extrapolation."), self.adv_yarn_ext_factor_spin)
        
        self.adv_yarn_attn_factor_spin = QDoubleSpinBox()
        self.adv_yarn_attn_factor_spin.setRange(-1.0, 100.0)
        self.adv_yarn_attn_factor_spin.setSingleStep(0.1)
        self.adv_yarn_attn_factor_spin.setValue(-1.0)
        self.adv_yarn_attn_factor_spin.setToolTip("YaRN attention scaling factor. Default: -1.0 (auto). Example: 1.0 for full scaling.")
        yarn_layout.addRow(self._create_info_label("YaRN Attn Factor (--yarn-attn-factor):", "YaRN attention scaling factor. Default: -1.0 (auto). Example: 1.0 for full scaling."), self.adv_yarn_attn_factor_spin)
        
        self.adv_yarn_beta_slow_spin = QDoubleSpinBox()
        self.adv_yarn_beta_slow_spin.setRange(-1.0, 100.0)
        self.adv_yarn_beta_slow_spin.setSingleStep(0.1)
        self.adv_yarn_beta_slow_spin.setValue(-1.0)
        self.adv_yarn_beta_slow_spin.setToolTip("YaRN beta slow factor for scaling. Default: -1.0 (auto).")
        yarn_layout.addRow(self._create_info_label("YaRN Beta Slow (--yarn-beta-slow):", "YaRN beta slow factor for scaling. Default: -1.0 (auto)."), self.adv_yarn_beta_slow_spin)
        
        self.adv_yarn_beta_fast_spin = QDoubleSpinBox()
        self.adv_yarn_beta_fast_spin.setRange(-1.0, 100.0)
        self.adv_yarn_beta_fast_spin.setSingleStep(0.1)
        self.adv_yarn_beta_fast_spin.setValue(-1.0)
        self.adv_yarn_beta_fast_spin.setToolTip("YaRN beta fast factor for scaling. Default: -1.0 (auto).")
        yarn_layout.addRow(self._create_info_label("YaRN Beta Fast (--yarn-beta-fast):", "YaRN beta fast factor for scaling. Default: -1.0 (auto)."), self.adv_yarn_beta_fast_spin)
        
        yarn_group.setLayout(yarn_layout)
        scroll_layout.addWidget(yarn_group)
        
        # Cache & Memory params group
        cache_mem_group = QGroupBox("Cache & Memory Parameters")
        cache_mem_layout = QFormLayout()
        
        self.adv_kv_offload_check = QCheckBox("--kv-offload/--no-kv-offload")
        self.adv_kv_offload_check.setChecked(True)
        self.adv_kv_offload_check.setToolTip("Offload KV cache to system RAM if VRAM is full. Default: enabled.")
        cache_mem_layout.addRow(self._create_info_label("--kv-offload/--no-kv-offload", "Offload KV cache to system RAM if VRAM is full. Default: enabled."), self.adv_kv_offload_check)
        
        self.adv_repack_check = QCheckBox("--repack/--no-repack")
        self.adv_repack_check.setChecked(True)
        self.adv_repack_check.setToolTip("Repack KV cache for better memory efficiency. Default: enabled.")
        cache_mem_layout.addRow(self._create_info_label("--repack/--no-repack", "Repack KV cache for better memory efficiency. Default: enabled."), self.adv_repack_check)
        
        self.adv_no_host_check = QCheckBox("--no-host/--host")
        self.adv_no_host_check.setToolTip("Do not keep model in host RAM. Default: disabled (keep in host RAM).")
        cache_mem_layout.addRow(self._create_info_label("--no-host/--host", "Do not keep model in host RAM. Default: disabled (keep in host RAM)."), self.adv_no_host_check)
        
        self.adv_defrag_thold_spin = QSpinBox()
        self.adv_defrag_thold_spin.setRange(0, 100)
        self.adv_defrag_thold_spin.setValue(0)
        self.adv_defrag_thold_spin.setToolTip("KV cache defragmentation threshold (0=disabled). Default: 0.")
        cache_mem_layout.addRow(self._create_info_label("Defrag Thold (--defrag-thold):", "KV cache defragmentation threshold (0=disabled). Default: 0."), self.adv_defrag_thold_spin)
        
        self.adv_direct_io_check = QCheckBox("--direct-io/--no-direct-io")
        self.adv_direct_io_check.setToolTip("Use direct I/O for model loading. Default: disabled.")
        cache_mem_layout.addRow(self._create_info_label("--direct-io/--no-direct-io", "Use direct I/O for model loading. Default: disabled."), self.adv_direct_io_check)
        
        cache_mem_group.setLayout(cache_mem_layout)
        scroll_layout.addWidget(cache_mem_group)

        # Extra Cache & Context group
        extra_cache_group = QGroupBox("Extra Cache & Context")
        extra_cache_layout = QFormLayout()

        self.adv_ctx_checkpoints_spin = QSpinBox()
        self.adv_ctx_checkpoints_spin.setRange(0, 1024)
        self.adv_ctx_checkpoints_spin.setValue(32)
        self.adv_ctx_checkpoints_spin.setToolTip("Number of context checkpoints. Default: 32.")
        extra_cache_layout.addRow(self._create_info_label("Context Checkpoints (--ctx-checkpoints):", "Number of context checkpoints. Default: 32."), self.adv_ctx_checkpoints_spin)

        self.adv_checkpoint_min_step_spin = QSpinBox()
        self.adv_checkpoint_min_step_spin.setRange(0, 4096)
        self.adv_checkpoint_min_step_spin.setValue(256)
        self.adv_checkpoint_min_step_spin.setToolTip("Minimum checkpoint step size. Default: 256.")
        extra_cache_layout.addRow(self._create_info_label("Checkpoint Min Step (--checkpoint-min-step):", "Minimum checkpoint step size. Default: 256."), self.adv_checkpoint_min_step_spin)

        self.adv_cache_ram_spin = QSpinBox()
        self.adv_cache_ram_spin.setRange(0, 65536)
        self.adv_cache_ram_spin.setValue(8192)
        self.adv_cache_ram_spin.setToolTip("Cache RAM size in MB. Default: 8192.")
        extra_cache_layout.addRow(self._create_info_label("Cache RAM (--cache-ram):", "Cache RAM size in MB. Default: 8192."), self.adv_cache_ram_spin)

        self.adv_cache_idle_slots_check = QCheckBox("--cache-idle-slots/--no-cache-idle-slots")
        self.adv_cache_idle_slots_check.setChecked(True)
        self.adv_cache_idle_slots_check.setToolTip("Enable idle slot caching. Default: enabled.")
        extra_cache_layout.addRow(self._create_info_label("--cache-idle-slots/--no-cache-idle-slots", "Enable idle slot caching. Default: enabled."), self.adv_cache_idle_slots_check)

        self.adv_context_shift_check = QCheckBox("--context-shift/--no-context-shift")
        self.adv_context_shift_check.setToolTip("Enable context shifting. Default: disabled.")
        extra_cache_layout.addRow(self._create_info_label("--context-shift/--no-context-shift", "Enable context shifting. Default: disabled."), self.adv_context_shift_check)

        extra_cache_group.setLayout(extra_cache_layout)
        scroll_layout.addWidget(extra_cache_group)

        # GPU & NUMA params group
        gpu_numa_group = QGroupBox("GPU & NUMA Parameters")
        gpu_numa_layout = QFormLayout()
        
        self.adv_numa_input = QLineEdit()
        self.adv_numa_input.setPlaceholderText("e.g. 0,1 or 'prefer-numa-node'")
        self.adv_numa_input.setToolTip("NUMA policy: '0,1' for specific nodes, 'prefer-numa-node' for NUMA-aware allocation. Default: not set.")
        gpu_numa_layout.addRow(self._create_info_label("NUMA (--numa):", "NUMA policy: '0,1' for specific nodes, 'prefer-numa-node' for NUMA-aware allocation. Default: not set."), self.adv_numa_input)
        
        self.adv_device_input = QLineEdit()
        self.adv_device_input.setPlaceholderText("e.g. cuda:0,cpu")
        self.adv_device_input.setToolTip("Device to use: 'cuda:0', 'metal', 'vulkan', 'cpu', etc. Default: not set (auto-detect).")
        gpu_numa_layout.addRow(self._create_info_label("Device (--device):", "Device to use: 'cuda:0', 'metal', 'vulkan', 'cpu', etc. Default: not set (auto-detect)."), self.adv_device_input)
        
        self.adv_override_tensor_input = QLineEdit()
        self.adv_override_tensor_input.setPlaceholderText("e.g. KQV=1,O=1")
        self.adv_override_tensor_input.setToolTip("Override tensor split: 'KQV=1,O=1' to force specific tensors to GPU. Default: not set.")
        gpu_numa_layout.addRow(self._create_info_label("Override Tensor (--override-tensor):", "Override tensor split: 'KQV=1,O=1' to force specific tensors to GPU. Default: not set."), self.adv_override_tensor_input)
        
        self.adv_cpu_moe_check = QCheckBox("--cpu-moe/--no-cpu-moe")
        self.adv_cpu_moe_check.setToolTip("Offload MoE layers to CPU. Default: disabled.")
        gpu_numa_layout.addRow(self._create_info_label("--cpu-moe/--no-cpu-moe", "Offload MoE layers to CPU. Default: disabled."), self.adv_cpu_moe_check)
        
        self.adv_n_cpu_moe_spin = QSpinBox()
        self.adv_n_cpu_moe_spin.setRange(0, 256)
        self.adv_n_cpu_moe_spin.setValue(0)
        self.adv_n_cpu_moe_spin.setToolTip("Number of CPU threads for MoE layers. Default: 0 (auto).")
        gpu_numa_layout.addRow(self._create_info_label("N CPU MoE (--n-cpu-moe):", "Number of CPU threads for MoE layers. Default: 0 (auto)."), self.adv_n_cpu_moe_spin)
        
        self.adv_fit_combo = QComboBox()
        self.adv_fit_combo.addItems(["on", "off"])
        self.adv_fit_combo.setToolTip("Fit model to VRAM: 'on' or 'off'. Default: on.")
        gpu_numa_layout.addRow(self._create_info_label("Fit (--fit):", "Fit model to VRAM: 'on' or 'off'. Default: on."), self.adv_fit_combo)
        
        self.adv_fit_target_input = QLineEdit()
        self.adv_fit_target_input.setPlaceholderText("e.g. 80%")
        self.adv_fit_target_input.setToolTip("Target VRAM usage percentage for fit mode. Default: not set.")
        gpu_numa_layout.addRow(self._create_info_label("Fit Target (--fit-target):", "Target VRAM usage percentage for fit mode. Default: not set."), self.adv_fit_target_input)
        
        self.adv_fit_ctx_spin = QSpinBox()
        self.adv_fit_ctx_spin.setRange(512, 131072)
        self.adv_fit_ctx_spin.setValue(4096)
        self.adv_fit_ctx_spin.setToolTip("Context size for fit mode calculation. Default: 4096.")
        gpu_numa_layout.addRow(self._create_info_label("Fit Context (--fit-ctx):", "Context size for fit mode calculation. Default: 4096."), self.adv_fit_ctx_spin)
        
        self.adv_check_tensors_check = QCheckBox("--check-tensors/--no-check-tensors")
        self.adv_check_tensors_check.setToolTip("Check tensor sizes and types. Default: disabled.")
        gpu_numa_layout.addRow(self._create_info_label("--check-tensors/--no-check-tensors", "Check tensor sizes and types. Default: disabled."), self.adv_check_tensors_check)
        
        self.adv_override_kv_input = QLineEdit()
        self.adv_override_kv_input.setPlaceholderText("e.g. k_scale=0.5,v_scale=0.5")
        self.adv_override_kv_input.setToolTip("Override KV scale factors: 'k_scale=0.5,v_scale=0.5'. Default: not set.")
        gpu_numa_layout.addRow(self._create_info_label("Override KV (--override-kv):", "Override KV scale factors: 'k_scale=0.5,v_scale=0.5'. Default: not set."), self.adv_override_kv_input)
        
        self.adv_op_offload_check = QCheckBox("--op-offload/--no-op-offload")
        self.adv_op_offload_check.setChecked(True)
        self.adv_op_offload_check.setToolTip("Offload operations to GPU. Default: enabled.")
        gpu_numa_layout.addRow(self._create_info_label("--op-offload/--no-op-offload", "Offload operations to GPU. Default: enabled."), self.adv_op_offload_check)
        
        gpu_numa_group.setLayout(gpu_numa_layout)
        scroll_layout.addWidget(gpu_numa_group)
        
        # LoRA & Control Vectors params group
        lora_ctrl_group = QGroupBox("LoRA & Control Vectors Parameters")
        lora_ctrl_layout = QFormLayout()
        
        self.adv_lora_input = QLineEdit()
        self.adv_lora_input.setPlaceholderText("e.g. /path/to/lora.gguf")
        self.adv_lora_input.setToolTip("LoRA adapter path. Default: not set.")
        lora_ctrl_layout.addRow(self._create_info_label("LoRA (--lora):", "LoRA adapter path. Default: not set."), self.adv_lora_input)
        
        self.adv_lora_scaled_input = QLineEdit()
        self.adv_lora_scaled_input.setPlaceholderText("e.g. /path/to/lora.gguf=1.5")
        self.adv_lora_scaled_input.setToolTip("LoRA adapter with scale: '/path/to/lora.gguf=1.5'. Default: not set.")
        lora_ctrl_layout.addRow(self._create_info_label("LoRA Scaled (--lora-scaled):", "LoRA adapter with scale: '/path/to/lora.gguf=1.5'. Default: not set."), self.adv_lora_scaled_input)
        
        self.adv_control_vector_input = QLineEdit()
        self.adv_control_vector_input.setPlaceholderText("e.g. /path/to/cv.gguf")
        self.adv_control_vector_input.setToolTip("Control vector path. Default: not set.")
        lora_ctrl_layout.addRow(self._create_info_label("Control Vector (--control-vector):", "Control vector path. Default: not set."), self.adv_control_vector_input)
        
        self.adv_control_vector_scaled_input = QLineEdit()
        self.adv_control_vector_scaled_input.setPlaceholderText("e.g. /path/to/cv.gguf=1.5")
        self.adv_control_vector_scaled_input.setToolTip("Control vector with scale: '/path/to/cv.gguf=1.5'. Default: not set.")
        lora_ctrl_layout.addRow(self._create_info_label("Control Vector Scaled (--control-vector-scaled):", "Control vector with scale: '/path/to/cv.gguf=1.5'. Default: not set."), self.adv_control_vector_scaled_input)
        
        self.adv_control_vector_layer_range_input = QLineEdit()
        self.adv_control_vector_layer_range_input.setPlaceholderText("e.g. 0-20")
        self.adv_control_vector_layer_range_input.setToolTip("Control vector layer range: '0-20'. Default: not set.")
        lora_ctrl_layout.addRow(self._create_info_label("Control Vector Layer Range (--control-vector-layer-range):", "Control vector layer range: '0-20'. Default: not set."), self.adv_control_vector_layer_range_input)
        
        lora_ctrl_group.setLayout(lora_ctrl_layout)
        scroll_layout.addWidget(lora_ctrl_group)
        
        # Model Sources params group
        model_sources_group = QGroupBox("Model Sources Parameters")
        model_sources_layout = QFormLayout()
        
        self.adv_model_url_input = QLineEdit()
        self.adv_model_url_input.setPlaceholderText("e.g. https://example.com/model.gguf")
        self.adv_model_url_input.setToolTip("Model download URL. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("Model URL (--model-url):", "Model download URL. Default: not set."), self.adv_model_url_input)
        
        self.adv_docker_repo_input = QLineEdit()
        self.adv_docker_repo_input.setPlaceholderText("e.g. ghcr.io/ggml-org/llama.cpp")
        self.adv_docker_repo_input.setToolTip("Docker repository for model. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("Docker Repo (--docker-repo):", "Docker repository for model. Default: not set."), self.adv_docker_repo_input)
        
        self.adv_hf_repo_input = QLineEdit()
        self.adv_hf_repo_input.setPlaceholderText("e.g. org/model-name")
        self.adv_hf_repo_input.setToolTip("Hugging Face repository: 'org/model-name'. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("HF Repo (--hf-repo):", "Hugging Face repository: 'org/model-name'. Default: not set."), self.adv_hf_repo_input)
        
        self.adv_hf_file_input = QLineEdit()
        self.adv_hf_file_input.setPlaceholderText("e.g. model.gguf")
        self.adv_hf_file_input.setToolTip("Hugging Face file name. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("HF File (--hf-file):", "Hugging Face file name. Default: not set."), self.adv_hf_file_input)
        
        self.adv_hf_repo_v_input = QLineEdit()
        self.adv_hf_repo_v_input.setPlaceholderText("e.g. org/validator-model")
        self.adv_hf_repo_v_input.setToolTip("Hugging Face validator repository: 'org/validator-model'. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("HF Repo V (--hf-repo-v):", "Hugging Face validator repository: 'org/validator-model'. Default: not set."), self.adv_hf_repo_v_input)
        
        self.adv_hf_file_v_input = QLineEdit()
        self.adv_hf_file_v_input.setPlaceholderText("e.g. validator.gguf")
        self.adv_hf_file_v_input.setToolTip("Hugging Face validator file name. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("HF File V (--hf-file-v):", "Hugging Face validator file name. Default: not set."), self.adv_hf_file_v_input)
        
        self.adv_hf_token_input = QLineEdit()
        self.adv_hf_token_input.setPlaceholderText("e.g. hf_...")
        self.adv_hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.adv_hf_token_input.setToolTip("Hugging Face access token. Default: not set.")
        model_sources_layout.addRow(self._create_info_label("HF Token (--hf-token):", "Hugging Face access token. Default: not set."), self.adv_hf_token_input)
        
        model_sources_group.setLayout(model_sources_layout)
        scroll_layout.addWidget(model_sources_group)
        
        # Logging params group
        logging_group = QGroupBox("Logging Parameters")
        logging_layout = QFormLayout()
        
        self.adv_log_disable_check = QCheckBox("--log-disable/--no-log-disable")
        self.adv_log_disable_check.setToolTip("Disable logging. Default: disabled (logging enabled).")
        logging_layout.addRow(self._create_info_label("--log-disable/--no-log-disable", "Disable logging. Default: disabled (logging enabled)."), self.adv_log_disable_check)
        
        self.adv_log_file_input = QLineEdit()
        self.adv_log_file_input.setPlaceholderText("e.g. /path/to/log.txt")
        self.adv_log_file_input.setToolTip("Log file path. Default: not set (stdout).")
        logging_layout.addRow(self._create_info_label("Log File (--log-file):", "Log file path. Default: not set (stdout)."), self.adv_log_file_input)
        
        self.adv_log_colors_combo = QComboBox()
        self.adv_log_colors_combo.addItems(["auto", "enable", "disable"])
        self.adv_log_colors_combo.setToolTip("Log colors: 'auto', 'enable', or 'disable'. Default: auto.")
        logging_layout.addRow(self._create_info_label("Log Colors (--log-colors):", "Log colors: 'auto', 'enable', or 'disable'. Default: auto."), self.adv_log_colors_combo)
        
        self.adv_verbose_check = QCheckBox("--verbose/--no-verbose")
        self.adv_verbose_check.setToolTip("Enable verbose logging. Default: disabled.")
        logging_layout.addRow(self._create_info_label("--verbose/--no-verbose", "Enable verbose logging. Default: disabled."), self.adv_verbose_check)
        
        self.adv_offline_check = QCheckBox("--offline/--no-offline")
        self.adv_offline_check.setToolTip("Disable network access (offline mode). Default: disabled (online).")
        logging_layout.addRow(self._create_info_label("--offline/--no-offline", "Disable network access (offline mode). Default: disabled (online)."), self.adv_offline_check)
        
        self.adv_verbosity_spin = QSpinBox()
        self.adv_verbosity_spin.setRange(0, 5)
        self.adv_verbosity_spin.setValue(3)
        self.adv_verbosity_spin.setToolTip("Log verbosity level: 0=off, 1=error, 2=warn, 3=info, 4=debug, 5=trace. Default: 3.")
        logging_layout.addRow(self._create_info_label("Verbosity (--verbosity):", "Log verbosity level: 0=off, 1=error, 2=warn, 3=info, 4=debug, 5=trace. Default: 3."), self.adv_verbosity_spin)
        
        self.adv_log_prefix_check = QCheckBox("--log-prefix/--no-log-prefix")
        self.adv_log_prefix_check.setChecked(True)
        self.adv_log_prefix_check.setToolTip("Include log prefix (timestamp, level). Default: enabled.")
        logging_layout.addRow(self._create_info_label("--log-prefix/--no-log-prefix", "Include log prefix (timestamp, level). Default: enabled."), self.adv_log_prefix_check)
        
        self.adv_log_timestamps_check = QCheckBox("--log-timestamps/--no-log-timestamps")
        self.adv_log_timestamps_check.setChecked(True)
        self.adv_log_timestamps_check.setToolTip("Include timestamps in logs. Default: enabled.")
        logging_layout.addRow(self._create_info_label("--log-timestamps/--no-log-timestamps", "Include timestamps in logs. Default: enabled."), self.adv_log_timestamps_check)
        
        logging_group.setLayout(logging_layout)
        scroll_layout.addWidget(logging_group)

        # Server Options group
        server_group = QGroupBox("Server Options")
        server_layout = QFormLayout()

        self.adv_parallel_spin = QSpinBox()
        self.adv_parallel_spin.setRange(-1, 128)
        self.adv_parallel_spin.setValue(-1)
        self.adv_parallel_spin.setToolTip("Number of parallel request streams. Default: -1 (auto).")
        server_layout.addRow(self._create_info_label("Parallel (--parallel):", "Number of parallel request streams. Default: -1 (auto)."), self.adv_parallel_spin)

        self.adv_backend_combo = QComboBox()
        self.adv_backend_combo.addItems(["auto", "cuda", "metal", "vulkan", "cpu", "sycl"])
        self.adv_backend_combo.setToolTip("Compute backend: 'auto' (default), 'cuda', 'metal', 'vulkan', 'cpu', or 'sycl'.")
        server_layout.addRow(self._create_info_label("Backend (--backend):", "Compute backend: 'auto' (default), 'cuda', 'metal', 'vulkan', 'cpu', or 'sycl'."), self.adv_backend_combo)

        self.adv_jinja_check = QCheckBox("--jinja/--no-jinja")
        self.adv_jinja_check.setToolTip("Use Jinja template for prompt formatting. Default: disabled.")
        server_layout.addRow(self._create_info_label("--jinja/--no-jinja", "Use Jinja template for prompt formatting. Default: disabled."), self.adv_jinja_check)

        self.adv_warmup_check = QCheckBox("--warmup/--no-warmup")
        self.adv_warmup_check.setChecked(True)
        self.adv_warmup_check.setToolTip("Warm up the model on startup. Default: enabled.")
        server_layout.addRow(self._create_info_label("--warmup/--no-warmup", "Warm up the model on startup. Default: enabled."), self.adv_warmup_check)

        self.adv_embeddings_check = QCheckBox("--embeddings/--no-embeddings")
        self.adv_embeddings_check.setToolTip("Return embeddings for prompts. Default: disabled.")
        server_layout.addRow(self._create_info_label("--embeddings/--no-embeddings", "Return embeddings for prompts. Default: disabled."), self.adv_embeddings_check)

        self.adv_reranking_check = QCheckBox("--reranking/--no-reranking")
        self.adv_reranking_check.setToolTip("Enable reranking mode. Default: disabled.")
        server_layout.addRow(self._create_info_label("--reranking/--no-reranking", "Enable reranking mode. Default: disabled."), self.adv_reranking_check)

        self.adv_ui_check = QCheckBox("--ui/--no-ui")
        self.adv_ui_check.setChecked(True)
        self.adv_ui_check.setToolTip("Enable the web UI. Default: enabled.")
        server_layout.addRow(self._create_info_label("--ui/--no-ui", "Enable the web UI. Default: enabled."), self.adv_ui_check)

        self.adv_agent_check = QCheckBox("--agent/--no-agent")
        self.adv_agent_check.setToolTip("Enable agent mode. Default: disabled.")
        server_layout.addRow(self._create_info_label("--agent/--no-agent", "Enable agent mode. Default: disabled."), self.adv_agent_check)

        self.adv_cont_batching_check = QCheckBox("--cont-batching/--no-cont-batching")
        self.adv_cont_batching_check.setChecked(True)
        self.adv_cont_batching_check.setToolTip("Enable continuous batching. Default: enabled.")
        server_layout.addRow(self._create_info_label("--cont-batching/--no-cont-batching", "Enable continuous batching. Default: enabled."), self.adv_cont_batching_check)

        self.adv_api_prefix_input = QLineEdit()
        self.adv_api_prefix_input.setPlaceholderText("e.g. /v1")
        self.adv_api_prefix_input.setToolTip("API endpoint prefix. Default: not set.")
        server_layout.addRow(self._create_info_label("API Prefix (--api-prefix):", "API endpoint prefix. Default: not set."), self.adv_api_prefix_input)

        self.adv_ui_config_input = QLineEdit()
        self.adv_ui_config_input.setPlaceholderText("e.g. {...}")
        self.adv_ui_config_input.setToolTip("UI configuration JSON string. Default: not set.")
        server_layout.addRow(self._create_info_label("UI Config (--ui-config):", "UI configuration JSON string. Default: not set."), self.adv_ui_config_input)

        self.adv_ui_config_file_input = QLineEdit()
        self.adv_ui_config_file_input.setPlaceholderText("e.g. /path/to/ui-config.json")
        self.adv_ui_config_file_input.setToolTip("Path to UI configuration file. Default: not set.")
        server_layout.addRow(self._create_info_label("UI Config File (--ui-config-file):", "Path to UI configuration file. Default: not set."), self.adv_ui_config_file_input)

        self.adv_tools_input = QLineEdit()
        self.adv_tools_input.setPlaceholderText("e.g. tool1,tool2")
        self.adv_tools_input.setToolTip("Comma-separated list of available tools. Default: not set.")
        server_layout.addRow(self._create_info_label("Tools (--tools):", "Comma-separated list of available tools. Default: not set."), self.adv_tools_input)

        server_group.setLayout(server_layout)
        scroll_layout.addWidget(server_group)

        # Generation & Cache group
        gen_cache_group = QGroupBox("Generation & Cache Options")
        gen_cache_layout = QFormLayout()

        self.adv_reverse_prompt_input = QLineEdit()
        self.adv_reverse_prompt_input.setPlaceholderText("e.g. User:")
        self.adv_reverse_prompt_input.setToolTip("Reverse prompt to stop generation. Default: not set.")
        gen_cache_layout.addRow(self._create_info_label("Reverse Prompt (--reverse-prompt):", "Reverse prompt to stop generation. Default: not set."), self.adv_reverse_prompt_input)

        self.adv_pooling_combo = QComboBox()
        self.adv_pooling_combo.addItems(["none", "mean", "cls", "last"])
        self.adv_pooling_combo.setToolTip("Pooling method for embeddings: 'none' (default), 'mean', 'cls', or 'last'.")
        gen_cache_layout.addRow(self._create_info_label("Pooling (--pooling):", "Pooling method for embeddings: 'none' (default), 'mean', 'cls', or 'last'."), self.adv_pooling_combo)

        self.adv_kv_unified_check = QCheckBox("--kv-unified/--no-kv-unified")
        self.adv_kv_unified_check.setChecked(True)
        self.adv_kv_unified_check.setToolTip("Use unified KV cache. Default: enabled.")
        gen_cache_layout.addRow(self._create_info_label("--kv-unified/--no-kv-unified", "Use unified KV cache. Default: enabled."), self.adv_kv_unified_check)

        self.adv_lookup_cache_static_input = QLineEdit()
        self.adv_lookup_cache_static_input.setPlaceholderText("e.g. /path/to/static-cache")
        self.adv_lookup_cache_static_input.setToolTip("Static lookup cache path. Default: not set.")
        gen_cache_layout.addRow(self._create_info_label("Lookup Cache Static (--lookup-cache-static):", "Static lookup cache path. Default: not set."), self.adv_lookup_cache_static_input)

        self.adv_lookup_cache_dynamic_input = QLineEdit()
        self.adv_lookup_cache_dynamic_input.setPlaceholderText("e.g. /path/to/dynamic-cache")
        self.adv_lookup_cache_dynamic_input.setToolTip("Dynamic lookup cache path. Default: not set.")
        gen_cache_layout.addRow(self._create_info_label("Lookup Cache Dynamic (--lookup-cache-dynamic):", "Dynamic lookup cache path. Default: not set."), self.adv_lookup_cache_dynamic_input)

        self.adv_special_check = QCheckBox("--special/--no-special")
        self.adv_special_check.setToolTip("Render special tokens in output. Default: disabled.")
        gen_cache_layout.addRow(self._create_info_label("--special/--no-special", "Render special tokens in output. Default: disabled."), self.adv_special_check)

        self.adv_spm_infill_check = QCheckBox("--spm-infill/--no-spm-infill")
        self.adv_spm_infill_check.setToolTip("Enable SPM infill mode for fill-in-the-middle. Default: disabled.")
        gen_cache_layout.addRow(self._create_info_label("--spm-infill/--no-spm-infill", "Enable SPM infill mode. Default: disabled."), self.adv_spm_infill_check)

        gen_cache_group.setLayout(gen_cache_layout)
        scroll_layout.addWidget(gen_cache_group)

        # Multimodal group
        multimodal_group = QGroupBox("Multimodal Parameters")
        multimodal_layout = QFormLayout()

        self.adv_mmproj_input = QLineEdit()
        self.adv_mmproj_input.setPlaceholderText("e.g. /path/to/mmproj.gguf")
        self.adv_mmproj_input.setToolTip("Path to multimodal projector model. Default: not set.")
        multimodal_layout.addRow(self._create_info_label("MMProj (--mmproj):", "Path to multimodal projector model. Default: not set."), self.adv_mmproj_input)

        self.adv_mmproj_url_input = QLineEdit()
        self.adv_mmproj_url_input.setPlaceholderText("e.g. https://example.com/mmproj.gguf")
        self.adv_mmproj_url_input.setToolTip("URL to download multimodal projector. Default: not set.")
        multimodal_layout.addRow(self._create_info_label("MMProj URL (--mmproj-url):", "URL to download multimodal projector. Default: not set."), self.adv_mmproj_url_input)

        self.adv_mmproj_auto_check = QCheckBox("--mmproj-auto/--no-mmproj-auto")
        self.adv_mmproj_auto_check.setChecked(True)
        self.adv_mmproj_auto_check.setToolTip("Auto-detect multimodal projector. Default: enabled.")
        multimodal_layout.addRow(self._create_info_label("--mmproj-auto/--no-mmproj-auto", "Auto-detect multimodal projector. Default: enabled."), self.adv_mmproj_auto_check)

        self.adv_mmproj_offload_check = QCheckBox("--mmproj-offload/--no-mmproj-offload")
        self.adv_mmproj_offload_check.setChecked(True)
        self.adv_mmproj_offload_check.setToolTip("Offload multimodal projector to GPU. Default: enabled.")
        multimodal_layout.addRow(self._create_info_label("--mmproj-offload/--no-mmproj-offload", "Offload multimodal projector to GPU. Default: enabled."), self.adv_mmproj_offload_check)

        self.adv_image_min_tokens_spin = QSpinBox()
        self.adv_image_min_tokens_spin.setRange(0, 65536)
        self.adv_image_min_tokens_spin.setValue(0)
        self.adv_image_min_tokens_spin.setToolTip("Minimum tokens for image processing. Default: 0.")
        multimodal_layout.addRow(self._create_info_label("Image Min Tokens (--image-min-tokens):", "Minimum tokens for image processing. Default: 0."), self.adv_image_min_tokens_spin)

        self.adv_image_max_tokens_spin = QSpinBox()
        self.adv_image_max_tokens_spin.setRange(0, 65536)
        self.adv_image_max_tokens_spin.setValue(0)
        self.adv_image_max_tokens_spin.setToolTip("Maximum tokens for image processing. Default: 0 (unlimited).")
        multimodal_layout.addRow(self._create_info_label("Image Max Tokens (--image-max-tokens):", "Maximum tokens for image processing. Default: 0."), self.adv_image_max_tokens_spin)

        self.adv_mtmd_batch_max_tokens_spin = QSpinBox()
        self.adv_mtmd_batch_max_tokens_spin.setRange(0, 65536)
        self.adv_mtmd_batch_max_tokens_spin.setValue(1024)
        self.adv_mtmd_batch_max_tokens_spin.setToolTip("Maximum tokens per batch for mtmd. Default: 1024.")
        multimodal_layout.addRow(self._create_info_label("MTMD Batch Max Tokens (--mtmd-batch-max-tokens):", "Maximum tokens per batch for mtmd. Default: 1024."), self.adv_mtmd_batch_max_tokens_spin)

        multimodal_group.setLayout(multimodal_layout)
        scroll_layout.addWidget(multimodal_group)

        # Server & Embeddings Extras
        server_extras_group = QGroupBox("Server & Embeddings Extras")
        server_extras_layout = QFormLayout()

        self.adv_alias_input = QLineEdit()
        self.adv_alias_input.setPlaceholderText("e.g. my-model")
        self.adv_alias_input.setToolTip("Model alias for the API. Default: not set.")
        server_extras_layout.addRow(self._create_info_label("Alias (--alias):", "Model alias for the API. Default: not set."), self.adv_alias_input)

        self.adv_tags_input = QLineEdit()
        self.adv_tags_input.setPlaceholderText("e.g. coding,chat")
        self.adv_tags_input.setToolTip("Comma-separated tags for the model. Default: not set.")
        server_extras_layout.addRow(self._create_info_label("Tags (--tags):", "Comma-separated tags for the model. Default: not set."), self.adv_tags_input)

        self.adv_embd_normalize_spin = QSpinBox()
        self.adv_embd_normalize_spin.setRange(0, 2)
        self.adv_embd_normalize_spin.setValue(2)
        self.adv_embd_normalize_spin.setToolTip("Embedding normalization type: 0=none, 1=max absolute, 2=L2. Default: 2.")
        server_extras_layout.addRow(self._create_info_label("Embedding Normalize (--embd-normalize):", "Embedding normalization: 0=none, 1=max abs, 2=L2. Default: 2."), self.adv_embd_normalize_spin)

        self.adv_path_input = QLineEdit()
        self.adv_path_input.setPlaceholderText("e.g. /api")
        self.adv_path_input.setToolTip("Custom API path. Default: not set.")
        server_extras_layout.addRow(self._create_info_label("Path (--path):", "Custom API path. Default: not set."), self.adv_path_input)

        self.adv_reuse_port_check = QCheckBox("--reuse-port/--no-reuse-port")
        self.adv_reuse_port_check.setToolTip("Allow port reuse. Default: disabled.")
        server_extras_layout.addRow(self._create_info_label("--reuse-port/--no-reuse-port", "Allow port reuse. Default: disabled."), self.adv_reuse_port_check)

        self.adv_ui_mcp_proxy_check = QCheckBox("--ui-mcp-proxy/--no-ui-mcp-proxy")
        self.adv_ui_mcp_proxy_check.setToolTip("Enable UI MCP proxy. Default: disabled.")
        server_extras_layout.addRow(self._create_info_label("--ui-mcp-proxy/--no-ui-mcp-proxy", "Enable UI MCP proxy. Default: disabled."), self.adv_ui_mcp_proxy_check)

        server_extras_group.setLayout(server_extras_layout)
        scroll_layout.addWidget(server_extras_group)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        self.tab_widget.addTab(tab, "Advanced Config")
        
    def create_command_preview_tab(self):
        """Create command preview and launch tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.command_text = QTextEdit()
        self.command_text.setReadOnly(True)
        
        self.launch_btn = QPushButton("Launch llama-server")
        self.launch_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.launch_btn.clicked.connect(self.launch_server)
        
        layout.addWidget(QLabel("Generated Command:"))
        layout.addWidget(self.command_text)
        layout.addWidget(self.launch_btn)
        
        self.tab_widget.addTab(tab, "Command Preview & Launch")
        
        # Update command preview on config changes
        self.model_combo.currentTextChanged.connect(self.update_command_preview)
        self.mmproj_combo.currentTextChanged.connect(self.update_command_preview)
        self.draft_combo.currentTextChanged.connect(self.update_command_preview)
        self.ctx_spin.valueChanged.connect(self.update_command_preview)
        self.gpu_spin.valueChanged.connect(self.update_command_preview)
        self.threads_spin.valueChanged.connect(self.update_command_preview)
        self.port_spin.valueChanged.connect(self.update_command_preview)
        self.host_input.textChanged.connect(self.update_command_preview)
        self.cache_combo.currentTextChanged.connect(self._sync_cache_to_adv)
        self.cache_combo.currentTextChanged.connect(self.update_command_preview)
        self.batch_spin.valueChanged.connect(self.update_command_preview)
        self.mmap_check.stateChanged.connect(self.update_command_preview)

        # Wire advanced tab signals to update command preview
        for attr, signal_name in [
            ('adv_ctx_spin', 'valueChanged'),
            ('adv_threads_spin', 'valueChanged'),
            ('adv_threads_batch_spin', 'valueChanged'),
            ('adv_predict_spin', 'valueChanged'),
            ('adv_batch_spin', 'valueChanged'),
            ('adv_ubatch_spin', 'valueChanged'),
            ('adv_keep_spin', 'valueChanged'),
            ('adv_rope_scaling_combo', 'currentTextChanged'),
            ('adv_rope_scale_spin', 'valueChanged'),
            ('adv_rope_freq_base_spin', 'valueChanged'),
            ('adv_rope_freq_scale_spin', 'valueChanged'),
            ('adv_cache_type_k_combo', 'currentTextChanged'),
            ('adv_cache_type_v_combo', 'currentTextChanged'),
            ('adv_mmap_check', 'stateChanged'),
            ('adv_mlock_check', 'stateChanged'),
            ('adv_gpu_layers_spin', 'valueChanged'),
            ('adv_split_mode_combo', 'currentTextChanged'),
            ('adv_tensor_split_input', 'textChanged'),
            ('adv_main_gpu_spin', 'valueChanged'),
            ('adv_temp_spin', 'valueChanged'),
            ('adv_top_k_spin', 'valueChanged'),
            ('adv_top_p_spin', 'valueChanged'),
            ('adv_min_p_spin', 'valueChanged'),
            ('adv_repeat_last_n_spin', 'valueChanged'),
            ('adv_repeat_penalty_spin', 'valueChanged'),
            ('adv_presence_penalty_spin', 'valueChanged'),
            ('adv_frequency_penalty_spin', 'valueChanged'),
            ('adv_draft_n_max_spin', 'valueChanged'),
            ('adv_draft_n_min_spin', 'valueChanged'),
            ('adv_spec_type_combo', 'currentTextChanged'),
            ('adv_spec_draft_type_k_combo', 'currentTextChanged'),
            ('adv_spec_draft_type_v_combo', 'currentTextChanged'),
            ('adv_cpu_mask_input', 'textChanged'),
            ('adv_cpu_range_input', 'textChanged'),
            ('adv_cpu_strict_spin', 'valueChanged'),
            ('adv_prio_spin', 'valueChanged'),
            ('adv_poll_spin', 'valueChanged'),
            ('adv_cpu_mask_batch_input', 'textChanged'),
            ('adv_cpu_range_batch_input', 'textChanged'),
            ('adv_cpu_strict_batch_spin', 'valueChanged'),
            ('adv_prio_batch_spin', 'valueChanged'),
            ('adv_poll_batch_spin', 'valueChanged'),
            ('adv_swa_full_check', 'stateChanged'),
            ('adv_flash_attn_combo', 'currentTextChanged'),
            ('adv_perf_check', 'stateChanged'),
            ('adv_escape_check', 'stateChanged'),
            ('adv_yarn_orig_ctx_spin', 'valueChanged'),
            ('adv_yarn_ext_factor_spin', 'valueChanged'),
            ('adv_yarn_attn_factor_spin', 'valueChanged'),
            ('adv_yarn_beta_slow_spin', 'valueChanged'),
            ('adv_yarn_beta_fast_spin', 'valueChanged'),
            ('adv_kv_offload_check', 'stateChanged'),
            ('adv_repack_check', 'stateChanged'),
            ('adv_no_host_check', 'stateChanged'),
            ('adv_defrag_thold_spin', 'valueChanged'),
            ('adv_direct_io_check', 'stateChanged'),
            ('adv_numa_input', 'textChanged'),
            ('adv_device_input', 'textChanged'),
            ('adv_override_tensor_input', 'textChanged'),
            ('adv_cpu_moe_check', 'stateChanged'),
            ('adv_n_cpu_moe_spin', 'valueChanged'),
            ('adv_fit_combo', 'currentTextChanged'),
            ('adv_fit_target_input', 'textChanged'),
            ('adv_fit_ctx_spin', 'valueChanged'),
            ('adv_check_tensors_check', 'stateChanged'),
            ('adv_override_kv_input', 'textChanged'),
            ('adv_op_offload_check', 'stateChanged'),
            ('adv_lora_input', 'textChanged'),
            ('adv_lora_scaled_input', 'textChanged'),
            ('adv_control_vector_input', 'textChanged'),
            ('adv_control_vector_scaled_input', 'textChanged'),
            ('adv_control_vector_layer_range_input', 'textChanged'),
            ('adv_model_url_input', 'textChanged'),
            ('adv_docker_repo_input', 'textChanged'),
            ('adv_hf_repo_input', 'textChanged'),
            ('adv_hf_file_input', 'textChanged'),
            ('adv_hf_repo_v_input', 'textChanged'),
            ('adv_hf_file_v_input', 'textChanged'),
            ('adv_hf_token_input', 'textChanged'),
            ('adv_log_disable_check', 'stateChanged'),
            ('adv_log_file_input', 'textChanged'),
            ('adv_log_colors_combo', 'currentTextChanged'),
            ('adv_verbose_check', 'stateChanged'),
            ('adv_offline_check', 'stateChanged'),
            ('adv_verbosity_spin', 'valueChanged'),
            ('adv_log_prefix_check', 'stateChanged'),
            ('adv_log_timestamps_check', 'stateChanged'),
            ('adv_grammar_input', 'textChanged'),
            ('adv_grammar_file_input', 'textChanged'),
            ('adv_json_schema_input', 'textChanged'),
            ('adv_json_schema_file_input', 'textChanged'),
            ('adv_spec_draft_hf_input', 'textChanged'),
            ('adv_spec_draft_override_tensor_input', 'textChanged'),
            ('adv_spec_draft_device_input', 'textChanged'),
            ('adv_reverse_prompt_input', 'textChanged'),
            ('adv_dry_sequence_breaker_input', 'textChanged'),
            ('adv_api_prefix_input', 'textChanged'),
            ('adv_ui_config_input', 'textChanged'),
            ('adv_ui_config_file_input', 'textChanged'),
            ('adv_tools_input', 'textChanged'),
            ('adv_lookup_cache_static_input', 'textChanged'),
            ('adv_lookup_cache_dynamic_input', 'textChanged'),
            ('adv_mirostat_spin', 'valueChanged'),
            ('adv_mirostat_lr_spin', 'valueChanged'),
            ('adv_mirostat_ent_spin', 'valueChanged'),
            ('adv_dry_multiplier_spin', 'valueChanged'),
            ('adv_dry_base_spin', 'valueChanged'),
            ('adv_dry_allowed_length_spin', 'valueChanged'),
            ('adv_dry_penalty_last_n_spin', 'valueChanged'),
            ('adv_ignore_eos_check', 'stateChanged'),
            ('adv_backend_sampling_check', 'stateChanged'),
            ('adv_spec_draft_model_input', 'textChanged'),
            ('adv_spec_draft_threads_spin', 'valueChanged'),
            ('adv_spec_draft_threads_batch_spin', 'valueChanged'),
            ('adv_parallel_spin', 'valueChanged'),
            ('adv_backend_combo', 'currentTextChanged'),
            ('adv_jinja_check', 'stateChanged'),
            ('adv_warmup_check', 'stateChanged'),
            ('adv_embeddings_check', 'stateChanged'),
            ('adv_reranking_check', 'stateChanged'),
            ('adv_ui_check', 'stateChanged'),
            ('adv_agent_check', 'stateChanged'),
            ('adv_cont_batching_check', 'stateChanged'),
            ('adv_pooling_combo', 'currentTextChanged'),
            ('adv_kv_unified_check', 'stateChanged'),
            # Extra sampling
            ('adv_samplers_input', 'textChanged'),
            ('adv_sampler_seq_input', 'textChanged'),
            ('adv_seed_spin', 'valueChanged'),
            ('adv_top_n_sigma_spin', 'valueChanged'),
            ('adv_typical_p_spin', 'valueChanged'),
            ('adv_xtc_probability_spin', 'valueChanged'),
            ('adv_xtc_threshold_spin', 'valueChanged'),
            ('adv_logit_bias_input', 'textChanged'),
            # Adaptive & DynaTemp
            ('adv_adaptive_target_spin', 'valueChanged'),
            ('adv_adaptive_decay_spin', 'valueChanged'),
            ('adv_dynatemp_range_spin', 'valueChanged'),
            ('adv_dynatemp_exp_spin', 'valueChanged'),
            # Spec draft extras
            ('adv_spec_draft_n_max_spin', 'valueChanged'),
            ('adv_spec_draft_n_min_spin', 'valueChanged'),
            ('adv_spec_draft_p_split_spin', 'valueChanged'),
            ('adv_spec_draft_p_min_spin', 'valueChanged'),
            ('adv_spec_draft_backend_sampling_check', 'stateChanged'),
            ('adv_spec_draft_ngl_spin', 'valueChanged'),
            # Spec draft CPU affinity
            ('adv_spec_draft_cpu_mask_input', 'textChanged'),
            ('adv_spec_draft_cpu_range_input', 'textChanged'),
            ('adv_spec_draft_cpu_strict_spin', 'valueChanged'),
            ('adv_spec_draft_prio_spin', 'valueChanged'),
            ('adv_spec_draft_poll_spin', 'valueChanged'),
            ('adv_spec_draft_cpu_mask_batch_input', 'textChanged'),
            ('adv_spec_draft_cpu_strict_batch_spin', 'valueChanged'),
            ('adv_spec_draft_prio_batch_spin', 'valueChanged'),
            ('adv_spec_draft_poll_batch_spin', 'valueChanged'),
            ('adv_spec_draft_cpu_moe_check', 'stateChanged'),
            ('adv_spec_draft_n_cpu_moe_spin', 'valueChanged'),
            # Spec N-gram
            ('adv_spec_ngram_mod_n_min_spin', 'valueChanged'),
            ('adv_spec_ngram_mod_n_max_spin', 'valueChanged'),
            ('adv_spec_ngram_mod_n_match_spin', 'valueChanged'),
            ('adv_spec_ngram_simple_size_n_spin', 'valueChanged'),
            ('adv_spec_ngram_simple_size_m_spin', 'valueChanged'),
            ('adv_spec_ngram_simple_min_hits_spin', 'valueChanged'),
            ('adv_spec_ngram_map_k_size_n_spin', 'valueChanged'),
            ('adv_spec_ngram_map_k_size_m_spin', 'valueChanged'),
            ('adv_spec_ngram_map_k_min_hits_spin', 'valueChanged'),
            ('adv_spec_ngram_map_k4v_size_n_spin', 'valueChanged'),
            ('adv_spec_ngram_map_k4v_size_m_spin', 'valueChanged'),
            ('adv_spec_ngram_map_k4v_min_hits_spin', 'valueChanged'),
            # Extra cache
            ('adv_ctx_checkpoints_spin', 'valueChanged'),
            ('adv_checkpoint_min_step_spin', 'valueChanged'),
            ('adv_cache_ram_spin', 'valueChanged'),
            ('adv_cache_idle_slots_check', 'stateChanged'),
            ('adv_context_shift_check', 'stateChanged'),
            # Generation extras
            ('adv_special_check', 'stateChanged'),
            ('adv_spm_infill_check', 'stateChanged'),
            # Multimodal
            ('adv_mmproj_input', 'textChanged'),
            ('adv_mmproj_url_input', 'textChanged'),
            ('adv_mmproj_auto_check', 'stateChanged'),
            ('adv_mmproj_offload_check', 'stateChanged'),
            ('adv_image_min_tokens_spin', 'valueChanged'),
            ('adv_image_max_tokens_spin', 'valueChanged'),
            ('adv_mtmd_batch_max_tokens_spin', 'valueChanged'),
            # Server extras
            ('adv_alias_input', 'textChanged'),
            ('adv_tags_input', 'textChanged'),
            ('adv_embd_normalize_spin', 'valueChanged'),
            ('adv_path_input', 'textChanged'),
            ('adv_reuse_port_check', 'stateChanged'),
            ('adv_ui_mcp_proxy_check', 'stateChanged'),
        ]:
            widget = getattr(self, attr, None)
            if widget is not None:
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    signal.connect(self.update_command_preview)

        # Sync advanced context spin back to basic
        adv_ctx = getattr(self, 'adv_ctx_spin', None)
        if adv_ctx is not None:
            adv_ctx.valueChanged.connect(self._sync_adv_ctx_to_basic)
        
    def browse_add_folder(self):
        """Browse and add a model directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if dir_path:
            # Check if folder already exists in list
            for i in range(self.folders_list.count()):
                if self.folders_list.item(i).text() == dir_path:
                    return
            self.folders_list.addItem(dir_path)
            self.save_folders()
            # Automatically scan after adding a folder
            self.scan_models()
            
    def remove_selected_folder(self):
        """Remove selected folder from list."""
        current_items = self.folders_list.selectedItems()
        for item in current_items:
            self.folders_list.takeItem(self.folders_list.row(item))
        self.save_folders()
        
    def load_folders(self):
        """Load saved folders from file."""
        if self.FOLDERS_FILE.exists():
            try:
                with open(self.FOLDERS_FILE, 'r') as f:
                    folders = json.load(f)
                for folder in folders:
                    # Check if folder already exists in list
                    exists = False
                    for i in range(self.folders_list.count()):
                        if self.folders_list.item(i).text() == folder:
                            exists = True
                            break
                    if not exists:
                        self.folders_list.addItem(folder)
            except Exception:
                pass

    def _post_init_scan(self):
        """Run model scan after window is shown (non-blocking)."""
        if self.folders_list.count() > 0:
            self.scan_models()
                
    def save_folders(self):
        """Save folders to file."""
        folders = []
        for i in range(self.folders_list.count()):
            folders.append(self.folders_list.item(i).text())
        with open(self.FOLDERS_FILE, 'w') as f:
            json.dump(folders, f, indent=2)
            
    def browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Main Model File", "", "Model Files (*.gguf *.bin *.safetensors);;All Files (*)")
        if file_path:
            self.model_combo.setEditText(file_path)
            self.config.model_path = file_path
            
    def browse_mmproj_model(self):
        """Browse for mmproj model file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select mmproj Model File", "", "Model Files (*.gguf *.bin *.safetensors);;All Files (*)")
        if file_path:
            self.mmproj_combo.setEditText(file_path)
            self.config.mmproj_model_path = file_path
            
    def browse_draft_model(self):
        """Browse for draft model file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Draft Model File", "", "Model Files (*.gguf *.bin *.safetensors);;All Files (*)")
        if file_path:
            self.draft_combo.setEditText(file_path)
            self.config.draft_model_path = file_path
            
    def scan_models(self):
        """Scan models in a background thread for the selected server."""
        server_name = self._get_browser_server()
        dirs_to_scan = []

        for i in range(self.folders_list.count()):
            dirs_to_scan.append(self.folders_list.item(i).text())

        if server_name:
            server_dirs = self.server_manager.get_server_model_directories(server_name)
            for d in server_dirs:
                if d not in dirs_to_scan:
                    dirs_to_scan.append(d)

        if not dirs_to_scan:
            return

        is_local_only = not server_name
        self._scan_worker = _ModelScanWorker(
            self.server_manager, server_name, dirs_to_scan, is_local_only
        )
        self._scan_worker.setParent(self)
        self._scan_worker.scan_done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, scan_server_name, models):
        server_name = self._get_browser_server()
        if scan_server_name != server_name:
            return
        self.models = models
        if server_name:
            save_cached_models(server_name, models)
        self._populate_model_lists()
            
    def on_model_selected(self, item):
        """Handle model selection."""
        file_path = item.data(Qt.UserRole)
        if file_path:
            self.model_combo.setEditText(file_path)
            self.config.model_path = file_path
            self.update_command_preview()

    # --- Server selection helpers ---

    def _refresh_server_combo(self):
        names = self.server_manager.get_all_names()
        current = self.server_combo.currentText()
        self.server_combo.blockSignals(True)
        self.server_combo.clear()
        for name in names:
            self.server_combo.addItem(name)
        if current and current in names:
            self.server_combo.setCurrentText(current)
        elif names:
            self.server_combo.setCurrentIndex(0)
        self.server_combo.blockSignals(False)

    def _refresh_browser_server_combo(self):
        names = self.server_manager.get_all_names()
        current = self.browser_server_combo.currentText()
        self.browser_server_combo.blockSignals(True)
        self.browser_server_combo.clear()
        for name in names:
            self.browser_server_combo.addItem(name)
        if current and current in names:
            self.browser_server_combo.setCurrentText(current)
        elif names:
            self.browser_server_combo.setCurrentIndex(0)
        self.browser_server_combo.blockSignals(False)

    def _sync_server_combos(self, source_text: str):
        if self.server_combo.currentText() != source_text:
            self.server_combo.blockSignals(True)
            self.server_combo.setCurrentText(source_text)
            self.server_combo.blockSignals(False)
        if self.browser_server_combo.currentText() != source_text:
            self.browser_server_combo.blockSignals(True)
            self.browser_server_combo.setCurrentText(source_text)
            self.browser_server_combo.blockSignals(False)

    def _get_current_server(self) -> str:
        return self.server_combo.currentText() or (
            self.browser_server_combo.currentText() or
            self.server_manager.get_all_names()[0] if self.server_manager.get_all_names() else ""
        )

    def _get_browser_server(self) -> str:
        return self.browser_server_combo.currentText() or self._get_current_server()

    def on_server_changed(self, server_name: str):
        self.config.target_server = server_name if server_name else None
        self._sync_server_combos(server_name)
        self._on_server_selected(server_name)

    def on_browser_server_changed(self, server_name: str):
        self.config.target_server = server_name if server_name else None
        self._sync_server_combos(server_name)
        self._on_server_selected(server_name)

    def _on_server_selected(self, server_name: str):
        if not server_name:
            return
        conn = self.server_manager.get_connection(server_name)
        if conn:
            cfg = conn.config
            if cfg.hostname and cfg.hostname != "localhost":
                self.host_input.setText(cfg.hostname)
                self.config.host = cfg.hostname
            if cfg.server_port:
                self.port_spin.setValue(cfg.server_port)
                self.config.port = cfg.server_port
        self._reload_server_models()

    def _on_servers_changed(self):
        self._refresh_server_combo()
        self._refresh_browser_server_combo()
        current = self.config.target_server
        if current and current not in self.server_manager.get_all_names():
            names = self.server_manager.get_all_names()
            current = names[0] if names else None
            self.config.target_server = current
        self._refresh_preset_combo_styles()

    def _is_server_offline(self, server_name):
        if not server_name:
            return False
        state = self.server_manager.get_connection_state(server_name)
        return state in (ConnectionState.DISCONNECTED, ConnectionState.ERROR)

    def _refresh_preset_combo_styles(self):
        preset_names = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
        server_names = self.server_manager.get_all_names()
        worker = _PresetStyleWorker(preset_names, server_names, self.server_manager)
        worker.setParent(self)
        worker.style_done.connect(self._on_preset_style_done)
        self._active_workers.append(worker)
        worker.start()

    def _on_preset_style_done(self, results):
        offline_set = {name for name, offline in results if offline}
        for combo in (self.preset_combo, getattr(self, 'adv_preset_combo', None)):
            if not combo:
                continue
            for i in range(combo.count()):
                name = combo.itemText(i)
                offline = name in offline_set
                font = QFont()
                if offline:
                    font.setItalic(True)
                    combo.setItemData(i, Qt.GlobalColor.gray, Qt.ItemDataRole.ForegroundRole)
                else:
                    combo.setItemData(i, None, Qt.ItemDataRole.ForegroundRole)
                combo.setFont(font)

    def _reload_server_models(self):
        server_name = self._get_current_server()
        if not server_name:
            return
        cached = load_cached_models(server_name)
        if cached:
            self.models = cached
            self._populate_model_lists()
        else:
            self.scan_models()

    def _populate_model_lists(self):
        self.model_list.clear()
        self.model_combo.clear()
        for model in self.models:
            item_text = f"{model.file_name} ({model.file_size_mb:.2f}MB) [{model.format}]"
            if model.quant_type:
                item_text += f" - {model.quant_type}"
            list_item = QListWidgetItem(item_text)
            list_item.setData(Qt.UserRole, model.file_path)
            self.model_list.addItem(list_item)
            self.model_combo.addItem(model.file_path, model.file_path)

    def detect_gpu_info(self):
        """Detect and display GPU information."""
        self._detect_gpu_worker()

    def _deferred_init(self):
        """Run blocking init tasks after UI is shown."""
        self.scan_models()
        self._preset_style_timer = QTimer(self)
        self._preset_style_timer.timeout.connect(self._refresh_preset_combo_styles)
        self._preset_style_timer.start(15000)
        QTimer.singleShot(300, self._detect_gpu_async)

    def auto_detect_gpu(self):
        """Auto-detect GPU info non-blocking."""
        QTimer.singleShot(100, self._detect_gpu_async)

    def _detect_gpu_async(self):
        """Run GPU detection in a background thread (skip if already detected)."""
        if self.gpu_info.backend != "cpu" or self.gpu_info.device_name != "Detecting...":
            return
        worker = _GpuDetectWorker()
        worker.setParent(self)
        worker.detect_done.connect(self._on_gpu_detect_done)
        self._active_workers.append(worker)
        worker.start()

    def _on_gpu_detect_done(self, info_text, gpu_info_dict):
        self.gpu_info_label.setText(info_text)
        if gpu_info_dict:
            self.gpu_info = GPUInfo(**gpu_info_dict)
            if gpu_info_dict.get('backend') == 'cuda':
                self._gpu_stats_timer = QTimer(self)
                self._gpu_stats_timer.timeout.connect(self._gpu_poll_stats)
                self._gpu_stats_timer.start(30000)
        self._update_resource_estimate()

    def _gpu_poll_stats(self):
        worker = _GpuStatsWorker()
        worker.setParent(self)
        worker.stats_done.connect(self._on_gpu_stats_done)
        self._active_workers.append(worker)
        worker.start()

    def _on_gpu_stats_done(self, stats_text):
        if not stats_text:
            return
        existing = self.gpu_info_label.text()
        idx = existing.find("\n--- Live Stats ---")
        if idx >= 0:
            existing = existing[:idx] + stats_text
        else:
            existing = existing + stats_text
        self.gpu_info_label.setText(existing)

    def _detect_gpu_worker(self):
        """Detect GPU and update UI."""
        gpu_info = detect_gpu()
        self.gpu_info = gpu_info
        gi = gpu_info
        info_text = (
            f"Backend: {gi.backend}\n"
            f"Device: {gi.device_name or 'Unknown'}\n"
            f"VRAM: {gi.vram_mb} MB\n"
        )
        if gi.available_vram_mb > 0 and gi.available_vram_mb != gi.vram_mb:
            info_text += f"Available VRAM: {gi.available_vram_mb} MB\n"
        if gi.driver_version:
            info_text += f"Driver: {gi.driver_version}\n"
        info_text += f"Supported: {gi.supported}"

        # Extended nvidia-smi info
        if gi.backend == "cuda":
            try:
                nvidia_sm = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version,name,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu,utilization.memory,power.draw,power.limit',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if nvidia_sm.returncode == 0 and nvidia_sm.stdout.strip():
                    parts = [p.strip() for p in nvidia_sm.stdout.strip().split(',')[0].replace('\n', ',').split(',')]
                    if len(parts) >= 9:
                        info_text += (
                            f"\n--- NVIDIA-SMI Info ---\n"
                            f"Driver Version: {parts[0]}\n"
                            f"Device: {parts[1]}\n"
                            f"Temperature: {parts[2]}°C\n"
                            f"GPU Clock: {parts[3]} MHz | Memory Clock: {parts[4]} MHz\n"
                            f"GPU Util: {parts[5]}% | Mem Util: {parts[6]}%\n"
                            f"Power: {parts[7]}W / {parts[8]}W\n"
                        )
            except Exception:
                pass

        self.gpu_info_label.setText(info_text)
        self._update_resource_estimate()

    def load_preset(self, preset_name):
        """Load a preset configuration."""
        try:
            # Update preset combo selections
            if hasattr(self, 'preset_combo'):
                self.preset_combo.setCurrentText(preset_name)
            if hasattr(self, 'adv_preset_combo'):
                self.adv_preset_combo.setCurrentText(preset_name)
                
            preset_config = PresetManager.get_preset(preset_name)

            # Handle target_server from preset
            target = getattr(preset_config, 'target_server', None)
            if target and target in self.server_manager.get_all_names():
                self.config.target_server = target
                self._sync_server_combos(target)
                self._reload_server_models()
            elif hasattr(self, 'server_combo'):
                target = self.server_combo.currentText() or None
                self.config.target_server = target

            # Load model paths from preset config - clear boxes if no value in preset
            if preset_config.model_path:
                self.config.model_path = preset_config.model_path
                self.model_combo.setCurrentText(self.config.model_path)
            else:
                self.config.model_path = ""
                if hasattr(self, 'model_combo'):
                    self.model_combo.setCurrentText("")
                    
            if preset_config.mmproj_model_path is not None and preset_config.mmproj_model_path != "":
                self.config.mmproj_model_path = preset_config.mmproj_model_path
                self.mmproj_combo.setCurrentText(self.config.mmproj_model_path)
            else:
                self.config.mmproj_model_path = None
                if hasattr(self, 'mmproj_combo'):
                    self.mmproj_combo.setCurrentText("")
                    
            if preset_config.draft_model_path is not None and preset_config.draft_model_path != "":
                self.config.draft_model_path = preset_config.draft_model_path
                self.draft_combo.setCurrentText(self.config.draft_model_path)
            else:
                self.config.draft_model_path = None
                if hasattr(self, 'draft_combo'):
                    self.draft_combo.setCurrentText("")
                
            self.config.context_size = preset_config.context_size
            self.config.gpu_layers = preset_config.gpu_layers
            self.config.threads = preset_config.threads
            self.config.port = preset_config.port
            self.config.host = preset_config.host
            self.config.backend = preset_config.backend
            self.config.use_mmap = preset_config.use_mmap
            self.config.kv_cache_type = preset_config.kv_cache_type
            self.config.batch_size = preset_config.batch_size
            self.config.threads_batch = getattr(preset_config, 'threads_batch', -1)
            self.config.predict = getattr(preset_config, 'predict', -1)
            self.config.logical_batch_size = getattr(preset_config, 'logical_batch_size', 2048)
            self.config.ubatch_size = getattr(preset_config, 'ubatch_size', 512)
            self.config.keep = getattr(preset_config, 'keep', 0)
            self.config.rope_scaling = getattr(preset_config, 'rope_scaling', 'none')
            self.config.rope_scale = getattr(preset_config, 'rope_scale', 1)
            self.config.rope_freq_base = getattr(preset_config, 'rope_freq_base', 0)
            self.config.rope_freq_scale = getattr(preset_config, 'rope_freq_scale', 0)
            self.config.cache_type_k = getattr(preset_config, 'cache_type_k', 'f16')
            self.config.cache_type_v = getattr(preset_config, 'cache_type_v', 'f16')
            self.config.mlock = getattr(preset_config, 'mlock', False)
            self.config.split_mode = getattr(preset_config, 'split_mode', 'layer')
            self.config.tensor_split = getattr(preset_config, 'tensor_split', None)
            self.config.main_gpu = getattr(preset_config, 'main_gpu', 0)
            self.config.temp = getattr(preset_config, 'temp', 0.8)
            self.config.top_k = getattr(preset_config, 'top_k', 40)
            self.config.top_p = getattr(preset_config, 'top_p', 0.95)
            self.config.min_p = getattr(preset_config, 'min_p', 0.05)
            self.config.repeat_last_n = getattr(preset_config, 'repeat_last_n', 64)
            self.config.repeat_penalty = getattr(preset_config, 'repeat_penalty', 1.0)
            self.config.presence_penalty = getattr(preset_config, 'presence_penalty', 0.0)
            self.config.frequency_penalty = getattr(preset_config, 'frequency_penalty', 0.0)
            self.config.draft_n_max = getattr(preset_config, 'draft_n_max', 3)
            self.config.draft_n_min = getattr(preset_config, 'draft_n_min', 0)
            self.config.flash_attn = getattr(preset_config, 'flash_attn', 'auto')
            self.config.parallel = getattr(preset_config, 'parallel', -1)
            self.config.jinja = getattr(preset_config, 'jinja', False)
            # New fields
            self.config.mirostat = getattr(preset_config, 'mirostat', 0)
            self.config.mirostat_lr = getattr(preset_config, 'mirostat_lr', 0.1)
            self.config.mirostat_ent = getattr(preset_config, 'mirostat_ent', 5.0)
            self.config.dry_multiplier = getattr(preset_config, 'dry_multiplier', 0.0)
            self.config.dry_base = getattr(preset_config, 'dry_base', 1.75)
            self.config.dry_allowed_length = getattr(preset_config, 'dry_allowed_length', 2)
            self.config.dry_penalty_last_n = getattr(preset_config, 'dry_penalty_last_n', -1)
            self.config.dry_sequence_breaker = getattr(preset_config, 'dry_sequence_breaker', "")
            self.config.ignore_eos = getattr(preset_config, 'ignore_eos', False)
            self.config.backend_sampling = getattr(preset_config, 'backend_sampling', False)
            self.config.spec_draft_threads = getattr(preset_config, 'spec_draft_threads', 0)
            self.config.spec_draft_threads_batch = getattr(preset_config, 'spec_draft_threads_batch', -1)
            self.config.spec_draft_model = getattr(preset_config, 'spec_draft_model', "")
            self.config.spec_draft_hf = getattr(preset_config, 'spec_draft_hf', "")
            self.config.spec_draft_override_tensor = getattr(preset_config, 'spec_draft_override_tensor', "")
            self.config.spec_draft_device = getattr(preset_config, 'spec_draft_device', "")
            self.config.warmup = getattr(preset_config, 'warmup', True)
            self.config.embeddings = getattr(preset_config, 'embeddings', False)
            self.config.reranking = getattr(preset_config, 'reranking', False)
            self.config.ui = getattr(preset_config, 'ui', True)
            self.config.agent = getattr(preset_config, 'agent', False)
            self.config.cont_batching = getattr(preset_config, 'cont_batching', True)
            self.config.pooling = getattr(preset_config, 'pooling', 'none')
            self.config.kv_unified = getattr(preset_config, 'kv_unified', True)
            self.config.backend = getattr(preset_config, 'backend', 'auto')
            self.config.reverse_prompt = getattr(preset_config, 'reverse_prompt', "")
            self.config.lookup_cache_static = getattr(preset_config, 'lookup_cache_static', "")
            self.config.lookup_cache_dynamic = getattr(preset_config, 'lookup_cache_dynamic', "")
            self.config.api_prefix = getattr(preset_config, 'api_prefix', "")
            self.config.ui_config = getattr(preset_config, 'ui_config', "")
            self.config.ui_config_file = getattr(preset_config, 'ui_config_file', "")
            self.config.tools = getattr(preset_config, 'tools', "")
            self.config.grammar = getattr(preset_config, 'grammar', "")
            self.config.grammar_file = getattr(preset_config, 'grammar_file', "")
            self.config.json_schema = getattr(preset_config, 'json_schema', "")
            self.config.json_schema_file = getattr(preset_config, 'json_schema_file', "")
            # Extra sampling
            self.config.samplers = getattr(preset_config, 'samplers', "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature")
            self.config.sampler_seq = getattr(preset_config, 'sampler_seq', "edskypmxt")
            self.config.seed = getattr(preset_config, 'seed', -1)
            self.config.top_n_sigma = getattr(preset_config, 'top_n_sigma', -1.0)
            self.config.typical_p = getattr(preset_config, 'typical_p', 1.0)
            self.config.xtc_probability = getattr(preset_config, 'xtc_probability', 0.0)
            self.config.xtc_threshold = getattr(preset_config, 'xtc_threshold', 0.1)
            self.config.logit_bias = getattr(preset_config, 'logit_bias', "")
            # Adaptive & DynaTemp
            self.config.adaptive_target = getattr(preset_config, 'adaptive_target', -1.0)
            self.config.adaptive_decay = getattr(preset_config, 'adaptive_decay', 0.9)
            self.config.dynatemp_range = getattr(preset_config, 'dynatemp_range', 0.0)
            self.config.dynatemp_exp = getattr(preset_config, 'dynatemp_exp', 1.0)
            # Spec draft extras
            self.config.spec_draft_n_max = getattr(preset_config, 'spec_draft_n_max', 3)
            self.config.spec_draft_n_min = getattr(preset_config, 'spec_draft_n_min', 0)
            self.config.spec_draft_p_split = getattr(preset_config, 'spec_draft_p_split', 0.1)
            self.config.spec_draft_p_min = getattr(preset_config, 'spec_draft_p_min', 0.0)
            self.config.spec_draft_backend_sampling = getattr(preset_config, 'spec_draft_backend_sampling', True)
            self.config.spec_draft_ngl = getattr(preset_config, 'spec_draft_ngl', -1)
            # Spec draft CPU affinity
            self.config.spec_draft_cpu_mask = getattr(preset_config, 'spec_draft_cpu_mask', "")
            self.config.spec_draft_cpu_range = getattr(preset_config, 'spec_draft_cpu_range', "")
            self.config.spec_draft_cpu_strict = getattr(preset_config, 'spec_draft_cpu_strict', 0)
            self.config.spec_draft_prio = getattr(preset_config, 'spec_draft_prio', 0)
            self.config.spec_draft_poll = getattr(preset_config, 'spec_draft_poll', 0)
            self.config.spec_draft_cpu_mask_batch = getattr(preset_config, 'spec_draft_cpu_mask_batch', "")
            self.config.spec_draft_cpu_strict_batch = getattr(preset_config, 'spec_draft_cpu_strict_batch', 0)
            self.config.spec_draft_prio_batch = getattr(preset_config, 'spec_draft_prio_batch', 0)
            self.config.spec_draft_poll_batch = getattr(preset_config, 'spec_draft_poll_batch', 0)
            self.config.spec_draft_cpu_moe = getattr(preset_config, 'spec_draft_cpu_moe', False)
            self.config.spec_draft_n_cpu_moe = getattr(preset_config, 'spec_draft_n_cpu_moe', 0)
            # Spec N-gram / mod params
            self.config.spec_ngram_mod_n_min = getattr(preset_config, 'spec_ngram_mod_n_min', 48)
            self.config.spec_ngram_mod_n_max = getattr(preset_config, 'spec_ngram_mod_n_max', 64)
            self.config.spec_ngram_mod_n_match = getattr(preset_config, 'spec_ngram_mod_n_match', 24)
            self.config.spec_ngram_simple_size_n = getattr(preset_config, 'spec_ngram_simple_size_n', 12)
            self.config.spec_ngram_simple_size_m = getattr(preset_config, 'spec_ngram_simple_size_m', 48)
            self.config.spec_ngram_simple_min_hits = getattr(preset_config, 'spec_ngram_simple_min_hits', 1)
            self.config.spec_ngram_map_k_size_n = getattr(preset_config, 'spec_ngram_map_k_size_n', 12)
            self.config.spec_ngram_map_k_size_m = getattr(preset_config, 'spec_ngram_map_k_size_m', 48)
            self.config.spec_ngram_map_k_min_hits = getattr(preset_config, 'spec_ngram_map_k_min_hits', 1)
            self.config.spec_ngram_map_k4v_size_n = getattr(preset_config, 'spec_ngram_map_k4v_size_n', 12)
            self.config.spec_ngram_map_k4v_size_m = getattr(preset_config, 'spec_ngram_map_k4v_size_m', 48)
            self.config.spec_ngram_map_k4v_min_hits = getattr(preset_config, 'spec_ngram_map_k4v_min_hits', 1)
            # Extra cache
            self.config.ctx_checkpoints = getattr(preset_config, 'ctx_checkpoints', 32)
            self.config.checkpoint_min_step = getattr(preset_config, 'checkpoint_min_step', 256)
            self.config.cache_ram = getattr(preset_config, 'cache_ram', 8192)
            self.config.cache_idle_slots = getattr(preset_config, 'cache_idle_slots', True)
            self.config.context_shift = getattr(preset_config, 'context_shift', False)
            # Generation extras
            self.config.special = getattr(preset_config, 'special', False)
            self.config.spm_infill = getattr(preset_config, 'spm_infill', False)
            # Multimodal
            self.config.mmproj = getattr(preset_config, 'mmproj', "")
            self.config.mmproj_url = getattr(preset_config, 'mmproj_url', "")
            self.config.mmproj_auto = getattr(preset_config, 'mmproj_auto', True)
            self.config.mmproj_offload = getattr(preset_config, 'mmproj_offload', True)
            self.config.image_min_tokens = getattr(preset_config, 'image_min_tokens', 0)
            self.config.image_max_tokens = getattr(preset_config, 'image_max_tokens', 0)
            self.config.mtmd_batch_max_tokens = getattr(preset_config, 'mtmd_batch_max_tokens', 1024)
            # Server extras
            self.config.alias = getattr(preset_config, 'alias', "")
            self.config.tags = getattr(preset_config, 'tags', "")
            self.config.embd_normalize = getattr(preset_config, 'embd_normalize', 2)
            self.config.path = getattr(preset_config, 'path', "")
            self.config.reuse_port = getattr(preset_config, 'reuse_port', False)
            self.config.ui_mcp_proxy = getattr(preset_config, 'ui_mcp_proxy', False)
                        # Update UI (block signals to prevent sync cascade during bulk update)
            self.ctx_spin.blockSignals(True)
            if hasattr(self, 'adv_ctx_spin'):
                self.adv_ctx_spin.blockSignals(True)
            self.ctx_spin.setValue(self.config.context_size)
            self.gpu_spin.setValue(self.config.gpu_layers)
            self.threads_spin.setValue(self.config.threads)
            self.port_spin.setValue(self.config.port)
            self.host_input.setText(self.config.host)
            self.cache_combo.setCurrentText(self.config.kv_cache_type)
            self.batch_spin.setValue(self.config.batch_size)
            self.mmap_check.setChecked(self.config.use_mmap)
            
            # Update advanced UI controls if they exist
            if hasattr(self, 'adv_threads_spin'):
                self.adv_threads_spin.setValue(self.config.threads if self.config.threads > 0 else -1)
            if hasattr(self, 'adv_threads_batch_spin'):
                self.adv_threads_batch_spin.setValue(self.config.threads_batch if self.config.threads_batch != -1 else -1)
            if hasattr(self, 'adv_ctx_spin'):
                self.adv_ctx_spin.setValue(self.config.context_size if self.config.context_size > 0 else 0)
                self.adv_ctx_spin.blockSignals(False)
            self.ctx_spin.blockSignals(False)
            if hasattr(self, 'adv_predict_spin'):
                self.adv_predict_spin.setValue(self.config.predict if self.config.predict != -1 else -1)
            if hasattr(self, 'adv_batch_spin'):
                self.adv_batch_spin.setValue(self.config.logical_batch_size)
            if hasattr(self, 'adv_ubatch_spin'):
                self.adv_ubatch_spin.setValue(self.config.ubatch_size)
            if hasattr(self, 'adv_keep_spin'):
                self.adv_keep_spin.setValue(self.config.keep)
            if hasattr(self, 'adv_rope_scaling_combo'):
                self.adv_rope_scaling_combo.setCurrentText(self.config.rope_scaling)
            if hasattr(self, 'adv_rope_scale_spin'):
                self.adv_rope_scale_spin.setValue(self.config.rope_scale)
            if hasattr(self, 'adv_rope_freq_base_spin'):
                self.adv_rope_freq_base_spin.setValue(self.config.rope_freq_base if self.config.rope_freq_base > 0 else 0)
            if hasattr(self, 'adv_rope_freq_scale_spin'):
                self.adv_rope_freq_scale_spin.setValue(self.config.rope_freq_scale if self.config.rope_freq_scale > 0 else 0)
            if hasattr(self, 'adv_cache_type_k_combo'):
                self.adv_cache_type_k_combo.setCurrentText(self.config.cache_type_k)
            if hasattr(self, 'adv_cache_type_v_combo'):
                self.adv_cache_type_v_combo.setCurrentText(self.config.cache_type_v)
            if hasattr(self, 'cache_type_k_combo'):
                self.cache_type_k_combo.setCurrentText(self.config.cache_type_k)
            if hasattr(self, 'cache_type_v_combo'):
                self.cache_type_v_combo.setCurrentText(self.config.cache_type_v)
            if hasattr(self, 'adv_mmap_check'):
                self.adv_mmap_check.setChecked(self.config.use_mmap)
            if hasattr(self, 'adv_mlock_check'):
                self.adv_mlock_check.setChecked(self.config.mlock)
            if hasattr(self, 'adv_gpu_layers_spin'):
                self.adv_gpu_layers_spin.setValue(self.config.gpu_layers if self.config.gpu_layers != -1 else -1)
            if hasattr(self, 'adv_split_mode_combo'):
                self.adv_split_mode_combo.setCurrentText(self.config.split_mode)
            if hasattr(self, 'adv_tensor_split_input'):
                self.adv_tensor_split_input.setText(self.config.tensor_split or "")
            if hasattr(self, 'adv_main_gpu_spin'):
                self.adv_main_gpu_spin.setValue(self.config.main_gpu)
            if hasattr(self, 'adv_temp_spin'):
                self.adv_temp_spin.setValue(self.config.temp)
            if hasattr(self, 'adv_top_k_spin'):
                self.adv_top_k_spin.setValue(self.config.top_k)
            if hasattr(self, 'adv_top_p_spin'):
                self.adv_top_p_spin.setValue(self.config.top_p)
            if hasattr(self, 'adv_min_p_spin'):
                self.adv_min_p_spin.setValue(self.config.min_p)
            if hasattr(self, 'adv_repeat_last_n_spin'):
                self.adv_repeat_last_n_spin.setValue(self.config.repeat_last_n)
            if hasattr(self, 'adv_repeat_penalty_spin'):
                self.adv_repeat_penalty_spin.setValue(self.config.repeat_penalty)
            if hasattr(self, 'adv_presence_penalty_spin'):
                self.adv_presence_penalty_spin.setValue(self.config.presence_penalty)
            if hasattr(self, 'adv_frequency_penalty_spin'):
                self.adv_frequency_penalty_spin.setValue(self.config.frequency_penalty)
            if hasattr(self, 'adv_draft_n_max_spin'):
                self.adv_draft_n_max_spin.setValue(self.config.draft_n_max)
            if hasattr(self, 'adv_draft_n_min_spin'):
                self.adv_draft_n_min_spin.setValue(self.config.draft_n_min)

            # Mirostat
            if hasattr(self, 'adv_mirostat_spin'):
                self.adv_mirostat_spin.setValue(self.config.mirostat)
            if hasattr(self, 'adv_mirostat_lr_spin'):
                self.adv_mirostat_lr_spin.setValue(self.config.mirostat_lr)
            if hasattr(self, 'adv_mirostat_ent_spin'):
                self.adv_mirostat_ent_spin.setValue(self.config.mirostat_ent)

            # DRY
            if hasattr(self, 'adv_dry_multiplier_spin'):
                self.adv_dry_multiplier_spin.setValue(self.config.dry_multiplier)
            if hasattr(self, 'adv_dry_base_spin'):
                self.adv_dry_base_spin.setValue(self.config.dry_base)
            if hasattr(self, 'adv_dry_allowed_length_spin'):
                self.adv_dry_allowed_length_spin.setValue(self.config.dry_allowed_length)
            if hasattr(self, 'adv_dry_penalty_last_n_spin'):
                self.adv_dry_penalty_last_n_spin.setValue(self.config.dry_penalty_last_n)

            # Sampling extras
            if hasattr(self, 'adv_ignore_eos_check'):
                self.adv_ignore_eos_check.setChecked(self.config.ignore_eos)
            if hasattr(self, 'adv_backend_sampling_check'):
                self.adv_backend_sampling_check.setChecked(self.config.backend_sampling)

            # Spec draft extras
            if hasattr(self, 'adv_spec_draft_threads_spin'):
                self.adv_spec_draft_threads_spin.setValue(self.config.spec_draft_threads)
            if hasattr(self, 'adv_spec_draft_threads_batch_spin'):
                self.adv_spec_draft_threads_batch_spin.setValue(self.config.spec_draft_threads_batch)

            # Server options
            if hasattr(self, 'adv_parallel_spin'):
                self.adv_parallel_spin.setValue(self.config.parallel)
            if hasattr(self, 'adv_backend_combo'):
                self.adv_backend_combo.setCurrentText(self.config.backend)
            if hasattr(self, 'adv_jinja_check'):
                self.adv_jinja_check.setChecked(self.config.jinja)
            if hasattr(self, 'adv_warmup_check'):
                self.adv_warmup_check.setChecked(self.config.warmup)
            if hasattr(self, 'adv_embeddings_check'):
                self.adv_embeddings_check.setChecked(self.config.embeddings)
            if hasattr(self, 'adv_reranking_check'):
                self.adv_reranking_check.setChecked(self.config.reranking)
            if hasattr(self, 'adv_ui_check'):
                self.adv_ui_check.setChecked(self.config.ui)
            if hasattr(self, 'adv_agent_check'):
                self.adv_agent_check.setChecked(self.config.agent)
            if hasattr(self, 'adv_cont_batching_check'):
                self.adv_cont_batching_check.setChecked(self.config.cont_batching)

            # Generation & cache extras
            if hasattr(self, 'adv_pooling_combo'):
                self.adv_pooling_combo.setCurrentText(self.config.pooling)
            if hasattr(self, 'adv_kv_unified_check'):
                self.adv_kv_unified_check.setChecked(self.config.kv_unified)

            # Extra sampling
            if hasattr(self, 'adv_seed_spin'):
                self.adv_seed_spin.setValue(self.config.seed)
            if hasattr(self, 'adv_top_n_sigma_spin'):
                self.adv_top_n_sigma_spin.setValue(self.config.top_n_sigma)
            if hasattr(self, 'adv_typical_p_spin'):
                self.adv_typical_p_spin.setValue(self.config.typical_p)
            if hasattr(self, 'adv_xtc_probability_spin'):
                self.adv_xtc_probability_spin.setValue(self.config.xtc_probability)
            if hasattr(self, 'adv_xtc_threshold_spin'):
                self.adv_xtc_threshold_spin.setValue(self.config.xtc_threshold)
            # Adaptive & DynaTemp
            if hasattr(self, 'adv_adaptive_target_spin'):
                self.adv_adaptive_target_spin.setValue(self.config.adaptive_target)
            if hasattr(self, 'adv_adaptive_decay_spin'):
                self.adv_adaptive_decay_spin.setValue(self.config.adaptive_decay)
            if hasattr(self, 'adv_dynatemp_range_spin'):
                self.adv_dynatemp_range_spin.setValue(self.config.dynatemp_range)
            if hasattr(self, 'adv_dynatemp_exp_spin'):
                self.adv_dynatemp_exp_spin.setValue(self.config.dynatemp_exp)
            # Spec draft extras
            if hasattr(self, 'adv_spec_draft_n_max_spin'):
                self.adv_spec_draft_n_max_spin.setValue(self.config.spec_draft_n_max)
            if hasattr(self, 'adv_spec_draft_n_min_spin'):
                self.adv_spec_draft_n_min_spin.setValue(self.config.spec_draft_n_min)
            if hasattr(self, 'adv_spec_draft_p_split_spin'):
                self.adv_spec_draft_p_split_spin.setValue(self.config.spec_draft_p_split)
            if hasattr(self, 'adv_spec_draft_p_min_spin'):
                self.adv_spec_draft_p_min_spin.setValue(self.config.spec_draft_p_min)
            if hasattr(self, 'adv_spec_draft_backend_sampling_check'):
                self.adv_spec_draft_backend_sampling_check.setChecked(self.config.spec_draft_backend_sampling)
            if hasattr(self, 'adv_spec_draft_ngl_spin'):
                self.adv_spec_draft_ngl_spin.setValue(self.config.spec_draft_ngl)
            # Spec draft CPU affinity
            if hasattr(self, 'adv_spec_draft_cpu_strict_spin'):
                self.adv_spec_draft_cpu_strict_spin.setValue(self.config.spec_draft_cpu_strict)
            if hasattr(self, 'adv_spec_draft_prio_spin'):
                self.adv_spec_draft_prio_spin.setValue(self.config.spec_draft_prio)
            if hasattr(self, 'adv_spec_draft_poll_spin'):
                self.adv_spec_draft_poll_spin.setValue(self.config.spec_draft_poll)
            if hasattr(self, 'adv_spec_draft_cpu_strict_batch_spin'):
                self.adv_spec_draft_cpu_strict_batch_spin.setValue(self.config.spec_draft_cpu_strict_batch)
            if hasattr(self, 'adv_spec_draft_prio_batch_spin'):
                self.adv_spec_draft_prio_batch_spin.setValue(self.config.spec_draft_prio_batch)
            if hasattr(self, 'adv_spec_draft_poll_batch_spin'):
                self.adv_spec_draft_poll_batch_spin.setValue(self.config.spec_draft_poll_batch)
            if hasattr(self, 'adv_spec_draft_cpu_moe_check'):
                self.adv_spec_draft_cpu_moe_check.setChecked(self.config.spec_draft_cpu_moe)
            if hasattr(self, 'adv_spec_draft_n_cpu_moe_spin'):
                self.adv_spec_draft_n_cpu_moe_spin.setValue(self.config.spec_draft_n_cpu_moe)
            # Spec N-gram
            if hasattr(self, 'adv_spec_ngram_mod_n_min_spin'):
                self.adv_spec_ngram_mod_n_min_spin.setValue(self.config.spec_ngram_mod_n_min)
            if hasattr(self, 'adv_spec_ngram_mod_n_max_spin'):
                self.adv_spec_ngram_mod_n_max_spin.setValue(self.config.spec_ngram_mod_n_max)
            if hasattr(self, 'adv_spec_ngram_mod_n_match_spin'):
                self.adv_spec_ngram_mod_n_match_spin.setValue(self.config.spec_ngram_mod_n_match)
            if hasattr(self, 'adv_spec_ngram_simple_size_n_spin'):
                self.adv_spec_ngram_simple_size_n_spin.setValue(self.config.spec_ngram_simple_size_n)
            if hasattr(self, 'adv_spec_ngram_simple_size_m_spin'):
                self.adv_spec_ngram_simple_size_m_spin.setValue(self.config.spec_ngram_simple_size_m)
            if hasattr(self, 'adv_spec_ngram_simple_min_hits_spin'):
                self.adv_spec_ngram_simple_min_hits_spin.setValue(self.config.spec_ngram_simple_min_hits)
            if hasattr(self, 'adv_spec_ngram_map_k_size_n_spin'):
                self.adv_spec_ngram_map_k_size_n_spin.setValue(self.config.spec_ngram_map_k_size_n)
            if hasattr(self, 'adv_spec_ngram_map_k_size_m_spin'):
                self.adv_spec_ngram_map_k_size_m_spin.setValue(self.config.spec_ngram_map_k_size_m)
            if hasattr(self, 'adv_spec_ngram_map_k_min_hits_spin'):
                self.adv_spec_ngram_map_k_min_hits_spin.setValue(self.config.spec_ngram_map_k_min_hits)
            if hasattr(self, 'adv_spec_ngram_map_k4v_size_n_spin'):
                self.adv_spec_ngram_map_k4v_size_n_spin.setValue(self.config.spec_ngram_map_k4v_size_n)
            if hasattr(self, 'adv_spec_ngram_map_k4v_size_m_spin'):
                self.adv_spec_ngram_map_k4v_size_m_spin.setValue(self.config.spec_ngram_map_k4v_size_m)
            if hasattr(self, 'adv_spec_ngram_map_k4v_min_hits_spin'):
                self.adv_spec_ngram_map_k4v_min_hits_spin.setValue(self.config.spec_ngram_map_k4v_min_hits)
            # Extra cache
            if hasattr(self, 'adv_ctx_checkpoints_spin'):
                self.adv_ctx_checkpoints_spin.setValue(self.config.ctx_checkpoints)
            if hasattr(self, 'adv_checkpoint_min_step_spin'):
                self.adv_checkpoint_min_step_spin.setValue(self.config.checkpoint_min_step)
            if hasattr(self, 'adv_cache_ram_spin'):
                self.adv_cache_ram_spin.setValue(self.config.cache_ram)
            if hasattr(self, 'adv_cache_idle_slots_check'):
                self.adv_cache_idle_slots_check.setChecked(self.config.cache_idle_slots)
            if hasattr(self, 'adv_context_shift_check'):
                self.adv_context_shift_check.setChecked(self.config.context_shift)
            # Generation extras
            if hasattr(self, 'adv_special_check'):
                self.adv_special_check.setChecked(self.config.special)
            if hasattr(self, 'adv_spm_infill_check'):
                self.adv_spm_infill_check.setChecked(self.config.spm_infill)
            # Multimodal
            if hasattr(self, 'adv_mmproj_auto_check'):
                self.adv_mmproj_auto_check.setChecked(self.config.mmproj_auto)
            if hasattr(self, 'adv_mmproj_offload_check'):
                self.adv_mmproj_offload_check.setChecked(self.config.mmproj_offload)
            if hasattr(self, 'adv_image_min_tokens_spin'):
                self.adv_image_min_tokens_spin.setValue(self.config.image_min_tokens)
            if hasattr(self, 'adv_image_max_tokens_spin'):
                self.adv_image_max_tokens_spin.setValue(self.config.image_max_tokens)
            if hasattr(self, 'adv_mtmd_batch_max_tokens_spin'):
                self.adv_mtmd_batch_max_tokens_spin.setValue(self.config.mtmd_batch_max_tokens)
            # Server extras
            if hasattr(self, 'adv_embd_normalize_spin'):
                self.adv_embd_normalize_spin.setValue(self.config.embd_normalize)
            if hasattr(self, 'adv_reuse_port_check'):
                self.adv_reuse_port_check.setChecked(self.config.reuse_port)
            if hasattr(self, 'adv_ui_mcp_proxy_check'):
                self.adv_ui_mcp_proxy_check.setChecked(self.config.ui_mcp_proxy)

            # Set or clear string input fields based on preset values
            if hasattr(self, 'adv_cpu_mask_input'):
                self.adv_cpu_mask_input.setText(preset_config.cpu_mask or "")
            if hasattr(self, 'adv_cpu_range_input'):
                self.adv_cpu_range_input.setText(preset_config.cpu_range or "")
            if hasattr(self, 'adv_cpu_mask_batch_input'):
                self.adv_cpu_mask_batch_input.setText(preset_config.cpu_mask_batch or "")
            if hasattr(self, 'adv_cpu_range_batch_input'):
                self.adv_cpu_range_batch_input.setText(preset_config.cpu_range_batch or "")
            if hasattr(self, 'adv_numa_input'):
                self.adv_numa_input.setText(preset_config.numa or "")
            if hasattr(self, 'adv_device_input'):
                self.adv_device_input.setText(preset_config.device or "")
            if hasattr(self, 'adv_override_tensor_input'):
                self.adv_override_tensor_input.setText(preset_config.override_tensor or "")
            if hasattr(self, 'adv_fit_target_input'):
                self.adv_fit_target_input.setText(preset_config.fit_target or "")
            if hasattr(self, 'adv_override_kv_input'):
                self.adv_override_kv_input.setText(preset_config.override_kv or "")
            if hasattr(self, 'adv_lora_input'):
                self.adv_lora_input.setText(preset_config.lora or "")
            if hasattr(self, 'adv_lora_scaled_input'):
                self.adv_lora_scaled_input.setText(preset_config.lora_scaled or "")
            if hasattr(self, 'adv_control_vector_input'):
                self.adv_control_vector_input.setText(preset_config.control_vector or "")
            if hasattr(self, 'adv_control_vector_scaled_input'):
                self.adv_control_vector_scaled_input.setText(preset_config.control_vector_scaled or "")
            if hasattr(self, 'adv_control_vector_layer_range_input'):
                self.adv_control_vector_layer_range_input.setText(preset_config.control_vector_layer_range or "")
            if hasattr(self, 'adv_model_url_input'):
                self.adv_model_url_input.setText(preset_config.model_url or "")
            if hasattr(self, 'adv_docker_repo_input'):
                self.adv_docker_repo_input.setText(preset_config.docker_repo or "")
            if hasattr(self, 'adv_hf_repo_input'):
                self.adv_hf_repo_input.setText(preset_config.hf_repo or "")
            if hasattr(self, 'adv_hf_file_input'):
                self.adv_hf_file_input.setText(preset_config.hf_file or "")
            if hasattr(self, 'adv_hf_repo_v_input'):
                self.adv_hf_repo_v_input.setText(preset_config.hf_repo_v or "")
            if hasattr(self, 'adv_hf_file_v_input'):
                self.adv_hf_file_v_input.setText(preset_config.hf_file_v or "")
            if hasattr(self, 'adv_hf_token_input'):
                self.adv_hf_token_input.setText(preset_config.hf_token or "")
            if hasattr(self, 'adv_log_file_input'):
                self.adv_log_file_input.setText(preset_config.log_file or "")
            if hasattr(self, 'adv_grammar_input'):
                self.adv_grammar_input.setText(preset_config.grammar or "")
            if hasattr(self, 'adv_grammar_file_input'):
                self.adv_grammar_file_input.setText(preset_config.grammar_file or "")
            if hasattr(self, 'adv_json_schema_input'):
                self.adv_json_schema_input.setText(preset_config.json_schema or "")
            if hasattr(self, 'adv_json_schema_file_input'):
                self.adv_json_schema_file_input.setText(preset_config.json_schema_file or "")
            if hasattr(self, 'adv_spec_draft_hf_input'):
                self.adv_spec_draft_hf_input.setText(preset_config.spec_draft_hf or "")
            if hasattr(self, 'adv_spec_draft_override_tensor_input'):
                self.adv_spec_draft_override_tensor_input.setText(preset_config.spec_draft_override_tensor or "")
            if hasattr(self, 'adv_spec_draft_device_input'):
                self.adv_spec_draft_device_input.setText(preset_config.spec_draft_device or "")
            if hasattr(self, 'adv_spec_draft_model_input'):
                self.adv_spec_draft_model_input.setText(preset_config.spec_draft_model or "")
            if hasattr(self, 'adv_lookup_cache_static_input'):
                self.adv_lookup_cache_static_input.setText(preset_config.lookup_cache_static or "")
            if hasattr(self, 'adv_lookup_cache_dynamic_input'):
                self.adv_lookup_cache_dynamic_input.setText(preset_config.lookup_cache_dynamic or "")
            if hasattr(self, 'adv_reverse_prompt_input'):
                self.adv_reverse_prompt_input.setText(preset_config.reverse_prompt or "")
            if hasattr(self, 'adv_dry_sequence_breaker_input'):
                self.adv_dry_sequence_breaker_input.setText(preset_config.dry_sequence_breaker or "")
            if hasattr(self, 'adv_api_prefix_input'):
                self.adv_api_prefix_input.setText(preset_config.api_prefix or "")
            if hasattr(self, 'adv_ui_config_input'):
                self.adv_ui_config_input.setText(preset_config.ui_config or "")
            if hasattr(self, 'adv_ui_config_file_input'):
                self.adv_ui_config_file_input.setText(preset_config.ui_config_file or "")
            if hasattr(self, 'adv_tools_input'):
                self.adv_tools_input.setText(preset_config.tools or "")
            # Extra sampling string fields
            if hasattr(self, 'adv_samplers_input'):
                self.adv_samplers_input.setText(preset_config.samplers or "")
            if hasattr(self, 'adv_sampler_seq_input'):
                self.adv_sampler_seq_input.setText(preset_config.sampler_seq or "")
            if hasattr(self, 'adv_logit_bias_input'):
                self.adv_logit_bias_input.setText(preset_config.logit_bias or "")
            # Spec draft CPU affinity string fields
            if hasattr(self, 'adv_spec_draft_cpu_mask_input'):
                self.adv_spec_draft_cpu_mask_input.setText(preset_config.spec_draft_cpu_mask or "")
            if hasattr(self, 'adv_spec_draft_cpu_range_input'):
                self.adv_spec_draft_cpu_range_input.setText(preset_config.spec_draft_cpu_range or "")
            if hasattr(self, 'adv_spec_draft_cpu_mask_batch_input'):
                self.adv_spec_draft_cpu_mask_batch_input.setText(preset_config.spec_draft_cpu_mask_batch or "")
            # Multimodal string fields
            if hasattr(self, 'adv_mmproj_input'):
                self.adv_mmproj_input.setText(preset_config.mmproj or "")
            if hasattr(self, 'adv_mmproj_url_input'):
                self.adv_mmproj_url_input.setText(preset_config.mmproj_url or "")
            # Server extras string fields
            if hasattr(self, 'adv_alias_input'):
                self.adv_alias_input.setText(preset_config.alias or "")
            if hasattr(self, 'adv_tags_input'):
                self.adv_tags_input.setText(preset_config.tags or "")
            if hasattr(self, 'adv_path_input'):
                self.adv_path_input.setText(preset_config.path or "")

            self.update_command_preview()
        except Exception as e:
            error_msg = str(e)
            if hasattr(self, 'command_text'):
                self.command_text.setText(f"Preset Load Error: {error_msg}")
            QMessageBox.critical(self, "Preset Error", f"Failed to load preset '{preset_name}': {error_msg}")
            
    def update_command_preview(self):
        """Update the command preview based on current config."""
        # Sync all UI state (basic + advanced tabs) into self.config
        self.update_config_from_ui()

        # Build and display command
        try:
            validate_config(self.config)
            cmd = build_command(self.config)
            self.command_text.setText(cmd)
        except Exception as e:
            self.command_text.setText(f"Validation Error: {e}")

        # Update resource estimate (may set context default from training context)
        self._update_resource_estimate()

    def _update_resource_estimate(self):
        """Update the resource usage panel with current configuration estimate."""
        try:
            model_size_mb = get_file_size_mb(self.config.model_path)

            # No model selected -- reset panel gracefully
            if model_size_mb <= 0:
                self.resource_panel.update_estimate(None)
                return

            draft_size_mb = 0.0
            if self.config.draft_model_path:
                draft_size_mb = get_file_size_mb(self.config.draft_model_path)
            mmproj_size_mb = 0.0
            if self.config.mmproj_model_path:
                mmproj_size_mb = get_file_size_mb(self.config.mmproj_model_path)

            # Infer quant type from model path
            quant_type = ""
            model_family = "unknown"
            name_lower = self.config.model_path.lower()
            if "llama" in name_lower or "llm" in name_lower:
                model_family = "llama"
            elif "mistral" in name_lower:
                model_family = "mistral"
            elif "qwen" in name_lower or "qwen2" in name_lower or "qwq" in name_lower:
                model_family = "qwen"
            elif "gemma" in name_lower:
                model_family = "gemma"
            elif "phi" in name_lower:
                model_family = "phi"
            elif "codellama" in name_lower:
                model_family = "codellama"

            for pattern in ["q8_0", "q6_k", "q5_k_m", "q5_0", "q5_1",
                            "q4_k_m", "q4_0", "q4_1", "q3_k_m", "q3_k_s", "q2_k"]:
                if pattern in name_lower:
                    quant_type = pattern.upper().replace("_", "_")
                    break

            estimate = self.resource_estimator.calculate(
                config=self.config,
                gpu_info=self.gpu_info,
                model_size_mb=model_size_mb,
                model_family=model_family,
                quant_type=quant_type,
                model_path=self.config.model_path,
                draft_model_size_mb=draft_size_mb,
                mmproj_model_size_mb=mmproj_size_mb,
            )
            self.resource_panel.update_estimate(estimate, selected_ctx=self.config.context_size)

            # Auto-set context default from training context on first model load
            if (self._ctx_auto_set and
                estimate.training_context_length > 0 and
                self.config.context_size < estimate.training_context_length):
                self._set_ctx_from_model(estimate.training_context_length)
                self._ctx_auto_set = False
        except Exception as e:
            import sys, traceback
            print(f"ERROR _update_resource_estimate: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            
    def launch_server(self):
        """Launch llama-server."""
        try:
            # Sync all UI state (basic + advanced tabs) into self.config before launch
            self.update_config_from_ui()
            validate_config(self.config)
            cmd = build_command(self.config)
            
            # Clear previous logs and show command
            self.command_text.clear()
            self.command_text.setText(f"Launching llama-server...\nCommand:\n{cmd}\n\n--- Logs ---\n")
            
            # Kill existing process if any
            if hasattr(self, 'server_process') and self.server_process is not None:
                if self.server_process.state() == QProcess.ProcessState.Running:
                    self.server_process.terminate()
                    self.server_process.waitForFinished(2000)
                
            # Create QProcess
            self.server_process = QProcess(self)
            self.server_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.server_process.readyRead.connect(self.on_server_output)
            self.server_process.finished.connect(self.on_server_finished)
            
            # Parse command for QProcess
            import shlex
            args = shlex.split(cmd)
            if not args:
                raise Exception("Empty command")
                
            executable = args[0]
            arguments = args[1:]
            
            self.server_process.start(executable, arguments)
            if not self.server_process.waitForStarted(3000):
                raise Exception("Failed to start llama-server process")
                
        except Exception as e:
            self.command_text.append(f"\nLaunch Error: {e}")

    def on_server_output(self):
        """Handle server output."""
        if hasattr(self, 'server_process') and self.server_process is not None:
            if self.server_process.state() == QProcess.ProcessState.Running:
                output_bytes = bytes(self.server_process.readAll())
                output = output_bytes.decode('utf-8', errors='ignore')
                if output:
                    # Append to command_text
                    self.command_text.insertPlainText(output)
                    # Scroll to bottom
                    scrollbar = self.command_text.verticalScrollBar()
                    if scrollbar:
                        scrollbar.setValue(scrollbar.maximum())

    def on_server_finished(self, exit_code, exit_status=None):
        """Handle server process finished."""
        self.command_text.append(f"\n--- Server process finished with exit code: {exit_code} ---")

    def update_config_from_ui(self):
        """Update config from UI elements."""
        self.config.model_path = self.model_combo.currentText() or ""
        self.config.target_server = self.server_combo.currentText() or None
        mmproj_path = self.mmproj_combo.currentText().strip() if self.mmproj_combo.currentText() else None
        self.config.mmproj_model_path = mmproj_path if mmproj_path else None
        draft_path = self.draft_combo.currentText().strip() if self.draft_combo.currentText() else None
        self.config.draft_model_path = draft_path if draft_path else None
        self.config.context_size = self.ctx_spin.value()
        self.config.gpu_layers = self.gpu_spin.value()
        self.config.threads = self.threads_spin.value()

        # Advanced parameters from advanced tab
        if hasattr(self, 'adv_threads_spin'):
            val = self.adv_threads_spin.value()
            self.config.threads = val if val != -1 else 0

        if hasattr(self, 'adv_threads_batch_spin'):
            val = self.adv_threads_batch_spin.value()
            self.config.threads_batch = val if val != -1 else -1

        # Use advanced ctx only if explicitly set (non-zero), otherwise basic tab wins
        if hasattr(self, 'adv_ctx_spin'):
            val = self.adv_ctx_spin.value()
            if val > 0:
                self.config.context_size = val
                
        if hasattr(self, 'adv_predict_spin'):
            val = self.adv_predict_spin.value()
            self.config.predict = val if val != -1 else -1
            
        if hasattr(self, 'adv_batch_spin'):
            self.config.logical_batch_size = self.adv_batch_spin.value()
            
        if hasattr(self, 'adv_ubatch_spin'):
            self.config.ubatch_size = self.adv_ubatch_spin.value()
            
        if hasattr(self, 'adv_keep_spin'):
            val = self.adv_keep_spin.value()
            self.config.keep = val if val != 0 else 0
            
        if hasattr(self, 'adv_rope_scaling_combo'):
            self.config.rope_scaling = self.adv_rope_scaling_combo.currentText()
            
        if hasattr(self, 'adv_rope_scale_spin'):
            self.config.rope_scale = self.adv_rope_scale_spin.value()
            
        if hasattr(self, 'adv_rope_freq_base_spin'):
            val = self.adv_rope_freq_base_spin.value()
            if val != 0:
                self.config.rope_freq_base = val
                
        if hasattr(self, 'adv_rope_freq_scale_spin'):
            val = self.adv_rope_freq_scale_spin.value()
            if val != 0:
                self.config.rope_freq_scale = val
                
        if hasattr(self, 'cache_type_k_combo'):
            self.config.cache_type_k = self.cache_type_k_combo.currentText()
        elif hasattr(self, 'adv_cache_type_k_combo'):
            self.config.cache_type_k = self.adv_cache_type_k_combo.currentText()
            
        if hasattr(self, 'cache_type_v_combo'):
            self.config.cache_type_v = self.cache_type_v_combo.currentText()
        elif hasattr(self, 'adv_cache_type_v_combo'):
            self.config.cache_type_v = self.adv_cache_type_v_combo.currentText()
            
        if hasattr(self, 'adv_mmap_check'):
            self.config.use_mmap = self.adv_mmap_check.isChecked()
            
        if hasattr(self, 'adv_mlock_check'):
            self.config.mlock = self.adv_mlock_check.isChecked()
            
        if hasattr(self, 'adv_gpu_layers_spin'):
            val = self.adv_gpu_layers_spin.value()
            self.config.gpu_layers = val if val != -1 else -1
            
        if hasattr(self, 'adv_split_mode_combo'):
            self.config.split_mode = self.adv_split_mode_combo.currentText()
            
        if hasattr(self, 'adv_tensor_split_input'):
            text = self.adv_tensor_split_input.text().strip()
            self.config.tensor_split = text if text else None
            
        if hasattr(self, 'adv_main_gpu_spin'):
            self.config.main_gpu = self.adv_main_gpu_spin.value()
            
        if hasattr(self, 'adv_temp_spin'):
            self.config.temp = self.adv_temp_spin.value()
            
        if hasattr(self, 'adv_top_k_spin'):
            self.config.top_k = self.adv_top_k_spin.value()
            
        if hasattr(self, 'adv_top_p_spin'):
            self.config.top_p = self.adv_top_p_spin.value()
            
        if hasattr(self, 'adv_min_p_spin'):
            self.config.min_p = self.adv_min_p_spin.value()
            
        if hasattr(self, 'adv_repeat_last_n_spin'):
            val = self.adv_repeat_last_n_spin.value()
            self.config.repeat_last_n = val if val != 64 else 64
            
        if hasattr(self, 'adv_repeat_penalty_spin'):
            self.config.repeat_penalty = self.adv_repeat_penalty_spin.value()
            
        if hasattr(self, 'adv_presence_penalty_spin'):
            self.config.presence_penalty = self.adv_presence_penalty_spin.value()
            
        if hasattr(self, 'adv_frequency_penalty_spin'):
            self.config.frequency_penalty = self.adv_frequency_penalty_spin.value()
            
        if hasattr(self, 'adv_draft_n_max_spin'):
            self.config.draft_n_max = self.adv_draft_n_max_spin.value()
            
        if hasattr(self, 'adv_draft_n_min_spin'):
            self.config.draft_n_min = self.adv_draft_n_min_spin.value()
            
        self.config.port = self.port_spin.value()
        self.config.host = self.host_input.text()
        
        # kv_cache_type from basic or advanced tab
        if hasattr(self, 'cache_combo'):
            self.config.kv_cache_type = self.cache_combo.currentText()
        elif hasattr(self, 'adv_cache_type_k_combo') and hasattr(self, 'adv_cache_type_v_combo'):
            # Use cache type k as the main kv_cache_type for backward compatibility
            self.config.kv_cache_type = self.adv_cache_type_k_combo.currentText()
            
        if hasattr(self, 'batch_spin'):
            self.config.batch_size = self.batch_spin.value()
        elif hasattr(self, 'adv_ubatch_spin'):
            self.config.batch_size = self.adv_ubatch_spin.value()
            
        # String input fields from advanced tab
        if hasattr(self, 'adv_cpu_mask_input'):
            text = self.adv_cpu_mask_input.text().strip()
            self.config.cpu_mask = text if text else None
            
        if hasattr(self, 'adv_cpu_range_input'):
            text = self.adv_cpu_range_input.text().strip()
            self.config.cpu_range = text if text else None
            
        if hasattr(self, 'adv_cpu_mask_batch_input'):
            text = self.adv_cpu_mask_batch_input.text().strip()
            self.config.cpu_mask_batch = text if text else None
            
        if hasattr(self, 'adv_cpu_range_batch_input'):
            text = self.adv_cpu_range_batch_input.text().strip()
            self.config.cpu_range_batch = text if text else None
            
        if hasattr(self, 'adv_numa_input'):
            text = self.adv_numa_input.text().strip()
            self.config.numa = text if text else None
            
        if hasattr(self, 'adv_device_input'):
            text = self.adv_device_input.text().strip()
            self.config.device = text if text else None
            
        if hasattr(self, 'adv_override_tensor_input'):
            text = self.adv_override_tensor_input.text().strip()
            self.config.override_tensor = text if text else None
            
        if hasattr(self, 'adv_fit_target_input'):
            text = self.adv_fit_target_input.text().strip()
            self.config.fit_target = text if text else None
            
        if hasattr(self, 'adv_override_kv_input'):
            text = self.adv_override_kv_input.text().strip()
            self.config.override_kv = text if text else None
            
        if hasattr(self, 'adv_lora_input'):
            text = self.adv_lora_input.text().strip()
            self.config.lora = text if text else None
            
        if hasattr(self, 'adv_lora_scaled_input'):
            text = self.adv_lora_scaled_input.text().strip()
            self.config.lora_scaled = text if text else None
            
        if hasattr(self, 'adv_control_vector_input'):
            text = self.adv_control_vector_input.text().strip()
            self.config.control_vector = text if text else None
            
        if hasattr(self, 'adv_control_vector_scaled_input'):
            text = self.adv_control_vector_scaled_input.text().strip()
            self.config.control_vector_scaled = text if text else None
            
        if hasattr(self, 'adv_control_vector_layer_range_input'):
            text = self.adv_control_vector_layer_range_input.text().strip()
            self.config.control_vector_layer_range = text if text else None
            
        if hasattr(self, 'adv_model_url_input'):
            text = self.adv_model_url_input.text().strip()
            self.config.model_url = text if text else None
            
        if hasattr(self, 'adv_docker_repo_input'):
            text = self.adv_docker_repo_input.text().strip()
            self.config.docker_repo = text if text else None
            
        if hasattr(self, 'adv_hf_repo_input'):
            text = self.adv_hf_repo_input.text().strip()
            self.config.hf_repo = text if text else None
            
        if hasattr(self, 'adv_hf_file_input'):
            text = self.adv_hf_file_input.text().strip()
            self.config.hf_file = text if text else None
            
        if hasattr(self, 'adv_hf_repo_v_input'):
            text = self.adv_hf_repo_v_input.text().strip()
            self.config.hf_repo_v = text if text else None
            
        if hasattr(self, 'adv_hf_file_v_input'):
            text = self.adv_hf_file_v_input.text().strip()
            self.config.hf_file_v = text if text else None
            
        if hasattr(self, 'adv_hf_token_input'):
            text = self.adv_hf_token_input.text().strip()
            self.config.hf_token = text if text else None
            
        if hasattr(self, 'adv_log_file_input'):
            text = self.adv_log_file_input.text().strip()
            self.config.log_file = text if text else None
            
        if hasattr(self, 'adv_grammar_input'):
            text = self.adv_grammar_input.text().strip()
            self.config.grammar = text if text else None
            
        if hasattr(self, 'adv_grammar_file_input'):
            text = self.adv_grammar_file_input.text().strip()
            self.config.grammar_file = text if text else None
            
        if hasattr(self, 'adv_json_schema_input'):
            text = self.adv_json_schema_input.text().strip()
            self.config.json_schema = text if text else None
            
        if hasattr(self, 'adv_json_schema_file_input'):
            text = self.adv_json_schema_file_input.text().strip()
            self.config.json_schema_file = text if text else None
            
        if hasattr(self, 'adv_spec_draft_hf_input'):
            text = self.adv_spec_draft_hf_input.text().strip()
            self.config.spec_draft_hf = text if text else None
            
        if hasattr(self, 'adv_spec_draft_override_tensor_input'):
            text = self.adv_spec_draft_override_tensor_input.text().strip()
            self.config.spec_draft_override_tensor = text if text else None
            
        if hasattr(self, 'adv_spec_draft_device_input'):
            text = self.adv_spec_draft_device_input.text().strip()
            self.config.spec_draft_device = text if text else None

        # --- Missing advanced fields (context/predict) ---
        if hasattr(self, 'adv_swa_full_check'):
            self.config.swa_full = self.adv_swa_full_check.isChecked()
        if hasattr(self, 'adv_flash_attn_combo'):
            self.config.flash_attn = self.adv_flash_attn_combo.currentText()
        if hasattr(self, 'adv_perf_check'):
            self.config.perf = self.adv_perf_check.isChecked()
        if hasattr(self, 'adv_escape_check'):
            self.config.escape = self.adv_escape_check.isChecked()

        # --- YaRN fields ---
        if hasattr(self, 'adv_yarn_orig_ctx_spin'):
            self.config.yarn_orig_ctx = self.adv_yarn_orig_ctx_spin.value()
        if hasattr(self, 'adv_yarn_ext_factor_spin'):
            self.config.yarn_ext_factor = self.adv_yarn_ext_factor_spin.value()
        if hasattr(self, 'adv_yarn_attn_factor_spin'):
            self.config.yarn_attn_factor = self.adv_yarn_attn_factor_spin.value()
        if hasattr(self, 'adv_yarn_beta_slow_spin'):
            self.config.yarn_beta_slow = self.adv_yarn_beta_slow_spin.value()
        if hasattr(self, 'adv_yarn_beta_fast_spin'):
            self.config.yarn_beta_fast = self.adv_yarn_beta_fast_spin.value()

        # --- Cache & memory fields ---
        if hasattr(self, 'adv_kv_offload_check'):
            self.config.kv_offload = self.adv_kv_offload_check.isChecked()
        if hasattr(self, 'adv_repack_check'):
            self.config.repack = self.adv_repack_check.isChecked()
        if hasattr(self, 'adv_no_host_check'):
            self.config.no_host = self.adv_no_host_check.isChecked()
        if hasattr(self, 'adv_defrag_thold_spin'):
            self.config.defrag_thold = self.adv_defrag_thold_spin.value()
        if hasattr(self, 'adv_direct_io_check'):
            self.config.direct_io = self.adv_direct_io_check.isChecked()

        # --- GPU & NUMA fields ---
        if hasattr(self, 'adv_cpu_moe_check'):
            self.config.cpu_moe = self.adv_cpu_moe_check.isChecked()
        if hasattr(self, 'adv_n_cpu_moe_spin'):
            self.config.n_cpu_moe = self.adv_n_cpu_moe_spin.value()
        if hasattr(self, 'adv_fit_combo'):
            self.config.fit = self.adv_fit_combo.currentText()
        if hasattr(self, 'adv_fit_ctx_spin'):
            self.config.fit_ctx = self.adv_fit_ctx_spin.value()
        if hasattr(self, 'adv_check_tensors_check'):
            self.config.check_tensors = self.adv_check_tensors_check.isChecked()
        if hasattr(self, 'adv_op_offload_check'):
            self.config.op_offload = self.adv_op_offload_check.isChecked()

        # --- Threads & batching fields ---
        if hasattr(self, 'adv_cpu_strict_spin'):
            self.config.cpu_strict = self.adv_cpu_strict_spin.value()
        if hasattr(self, 'adv_prio_spin'):
            self.config.prio = self.adv_prio_spin.value()
        if hasattr(self, 'adv_poll_spin'):
            self.config.poll = self.adv_poll_spin.value()
        if hasattr(self, 'adv_cpu_strict_batch_spin'):
            self.config.cpu_strict_batch = self.adv_cpu_strict_batch_spin.value()
        if hasattr(self, 'adv_prio_batch_spin'):
            self.config.prio_batch = self.adv_prio_batch_spin.value()
        if hasattr(self, 'adv_poll_batch_spin'):
            self.config.poll_batch = self.adv_poll_batch_spin.value()

        # --- Speculative decoding fields ---
        if hasattr(self, 'adv_spec_type_combo'):
            self.config.spec_type = self.adv_spec_type_combo.currentText()
        if hasattr(self, 'adv_spec_draft_type_k_combo'):
            self.config.spec_draft_type_k = self.adv_spec_draft_type_k_combo.currentText()
        if hasattr(self, 'adv_spec_draft_type_v_combo'):
            self.config.spec_draft_type_v = self.adv_spec_draft_type_v_combo.currentText()

        # --- Logging fields ---
        if hasattr(self, 'adv_log_disable_check'):
            self.config.log_disable = self.adv_log_disable_check.isChecked()
        if hasattr(self, 'adv_verbose_check'):
            self.config.verbose = self.adv_verbose_check.isChecked()
        if hasattr(self, 'adv_offline_check'):
            self.config.offline = self.adv_offline_check.isChecked()
        if hasattr(self, 'adv_verbosity_spin'):
            self.config.verbosity = self.adv_verbosity_spin.value()
        if hasattr(self, 'adv_log_prefix_check'):
            self.config.log_prefix = self.adv_log_prefix_check.isChecked()
        if hasattr(self, 'adv_log_timestamps_check'):
            self.config.log_timestamps = self.adv_log_timestamps_check.isChecked()
        if hasattr(self, 'adv_log_colors_combo'):
            self.config.log_colors = self.adv_log_colors_combo.currentText()

        # --- Remaining string input fields ---
        if hasattr(self, 'adv_reverse_prompt_input'):
            text = self.adv_reverse_prompt_input.text().strip()
            self.config.reverse_prompt = text if text else ""
        if hasattr(self, 'adv_dry_sequence_breaker_input'):
            text = self.adv_dry_sequence_breaker_input.text().strip()
            self.config.dry_sequence_breaker = text if text else ""
        if hasattr(self, 'adv_api_prefix_input'):
            text = self.adv_api_prefix_input.text().strip()
            self.config.api_prefix = text if text else ""
        if hasattr(self, 'adv_ui_config_input'):
            text = self.adv_ui_config_input.text().strip()
            self.config.ui_config = text if text else ""
        if hasattr(self, 'adv_ui_config_file_input'):
            text = self.adv_ui_config_file_input.text().strip()
            self.config.ui_config_file = text if text else ""
        if hasattr(self, 'adv_tools_input'):
            text = self.adv_tools_input.text().strip()
            self.config.tools = text if text else ""
        if hasattr(self, 'adv_lookup_cache_static_input'):
            text = self.adv_lookup_cache_static_input.text().strip()
            self.config.lookup_cache_static = text if text else ""
        if hasattr(self, 'adv_lookup_cache_dynamic_input'):
            text = self.adv_lookup_cache_dynamic_input.text().strip()
            self.config.lookup_cache_dynamic = text if text else ""

        # --- Mirostat fields ---
        if hasattr(self, 'adv_mirostat_spin'):
            self.config.mirostat = self.adv_mirostat_spin.value()
        if hasattr(self, 'adv_mirostat_lr_spin'):
            self.config.mirostat_lr = self.adv_mirostat_lr_spin.value()
        if hasattr(self, 'adv_mirostat_ent_spin'):
            self.config.mirostat_ent = self.adv_mirostat_ent_spin.value()

        # --- DRY fields ---
        if hasattr(self, 'adv_dry_multiplier_spin'):
            self.config.dry_multiplier = self.adv_dry_multiplier_spin.value()
        if hasattr(self, 'adv_dry_base_spin'):
            self.config.dry_base = self.adv_dry_base_spin.value()
        if hasattr(self, 'adv_dry_allowed_length_spin'):
            self.config.dry_allowed_length = self.adv_dry_allowed_length_spin.value()
        if hasattr(self, 'adv_dry_penalty_last_n_spin'):
            self.config.dry_penalty_last_n = self.adv_dry_penalty_last_n_spin.value()

        # --- Sampling extras ---
        if hasattr(self, 'adv_ignore_eos_check'):
            self.config.ignore_eos = self.adv_ignore_eos_check.isChecked()
        if hasattr(self, 'adv_backend_sampling_check'):
            self.config.backend_sampling = self.adv_backend_sampling_check.isChecked()

        # --- Spec draft extras ---
        if hasattr(self, 'adv_spec_draft_model_input'):
            text = self.adv_spec_draft_model_input.text().strip()
            self.config.spec_draft_model = text if text else ""
        if hasattr(self, 'adv_spec_draft_threads_spin'):
            self.config.spec_draft_threads = self.adv_spec_draft_threads_spin.value()
        if hasattr(self, 'adv_spec_draft_threads_batch_spin'):
            self.config.spec_draft_threads_batch = self.adv_spec_draft_threads_batch_spin.value()

        # --- Server options ---
        if hasattr(self, 'adv_parallel_spin'):
            val = self.adv_parallel_spin.value()
            self.config.parallel = val if val != -1 else -1
        if hasattr(self, 'adv_backend_combo'):
            self.config.backend = self.adv_backend_combo.currentText()
        if hasattr(self, 'adv_jinja_check'):
            self.config.jinja = self.adv_jinja_check.isChecked()
        if hasattr(self, 'adv_warmup_check'):
            self.config.warmup = self.adv_warmup_check.isChecked()
        if hasattr(self, 'adv_embeddings_check'):
            self.config.embeddings = self.adv_embeddings_check.isChecked()
        if hasattr(self, 'adv_reranking_check'):
            self.config.reranking = self.adv_reranking_check.isChecked()
        if hasattr(self, 'adv_ui_check'):
            self.config.ui = self.adv_ui_check.isChecked()
        if hasattr(self, 'adv_agent_check'):
            self.config.agent = self.adv_agent_check.isChecked()
        if hasattr(self, 'adv_cont_batching_check'):
            self.config.cont_batching = self.adv_cont_batching_check.isChecked()

        # --- Generation & cache extras ---
        if hasattr(self, 'adv_pooling_combo'):
            self.config.pooling = self.adv_pooling_combo.currentText()
        if hasattr(self, 'adv_kv_unified_check'):
            self.config.kv_unified = self.adv_kv_unified_check.isChecked()

        # --- Extra sampling ---
        if hasattr(self, 'adv_samplers_input'):
            text = self.adv_samplers_input.text().strip()
            self.config.samplers = text if text else "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
        if hasattr(self, 'adv_sampler_seq_input'):
            text = self.adv_sampler_seq_input.text().strip()
            self.config.sampler_seq = text if text else "edskypmxt"
        if hasattr(self, 'adv_seed_spin'):
            self.config.seed = self.adv_seed_spin.value()
        if hasattr(self, 'adv_top_n_sigma_spin'):
            self.config.top_n_sigma = self.adv_top_n_sigma_spin.value()
        if hasattr(self, 'adv_typical_p_spin'):
            self.config.typical_p = self.adv_typical_p_spin.value()
        if hasattr(self, 'adv_xtc_probability_spin'):
            self.config.xtc_probability = self.adv_xtc_probability_spin.value()
        if hasattr(self, 'adv_xtc_threshold_spin'):
            self.config.xtc_threshold = self.adv_xtc_threshold_spin.value()
        if hasattr(self, 'adv_logit_bias_input'):
            text = self.adv_logit_bias_input.text().strip()
            self.config.logit_bias = text if text else ""

        # --- Adaptive & DynaTemp ---
        if hasattr(self, 'adv_adaptive_target_spin'):
            self.config.adaptive_target = self.adv_adaptive_target_spin.value()
        if hasattr(self, 'adv_adaptive_decay_spin'):
            self.config.adaptive_decay = self.adv_adaptive_decay_spin.value()
        if hasattr(self, 'adv_dynatemp_range_spin'):
            self.config.dynatemp_range = self.adv_dynatemp_range_spin.value()
        if hasattr(self, 'adv_dynatemp_exp_spin'):
            self.config.dynatemp_exp = self.adv_dynatemp_exp_spin.value()

        # --- Spec draft extras ---
        if hasattr(self, 'adv_spec_draft_n_max_spin'):
            self.config.spec_draft_n_max = self.adv_spec_draft_n_max_spin.value()
        if hasattr(self, 'adv_spec_draft_n_min_spin'):
            self.config.spec_draft_n_min = self.adv_spec_draft_n_min_spin.value()
        if hasattr(self, 'adv_spec_draft_p_split_spin'):
            self.config.spec_draft_p_split = self.adv_spec_draft_p_split_spin.value()
        if hasattr(self, 'adv_spec_draft_p_min_spin'):
            self.config.spec_draft_p_min = self.adv_spec_draft_p_min_spin.value()
        if hasattr(self, 'adv_spec_draft_backend_sampling_check'):
            self.config.spec_draft_backend_sampling = self.adv_spec_draft_backend_sampling_check.isChecked()
        if hasattr(self, 'adv_spec_draft_ngl_spin'):
            self.config.spec_draft_ngl = self.adv_spec_draft_ngl_spin.value()

        # --- Spec draft CPU affinity ---
        if hasattr(self, 'adv_spec_draft_cpu_mask_input'):
            text = self.adv_spec_draft_cpu_mask_input.text().strip()
            self.config.spec_draft_cpu_mask = text if text else ""
        if hasattr(self, 'adv_spec_draft_cpu_range_input'):
            text = self.adv_spec_draft_cpu_range_input.text().strip()
            self.config.spec_draft_cpu_range = text if text else ""
        if hasattr(self, 'adv_spec_draft_cpu_strict_spin'):
            self.config.spec_draft_cpu_strict = self.adv_spec_draft_cpu_strict_spin.value()
        if hasattr(self, 'adv_spec_draft_prio_spin'):
            self.config.spec_draft_prio = self.adv_spec_draft_prio_spin.value()
        if hasattr(self, 'adv_spec_draft_poll_spin'):
            self.config.spec_draft_poll = self.adv_spec_draft_poll_spin.value()
        if hasattr(self, 'adv_spec_draft_cpu_mask_batch_input'):
            text = self.adv_spec_draft_cpu_mask_batch_input.text().strip()
            self.config.spec_draft_cpu_mask_batch = text if text else ""
        if hasattr(self, 'adv_spec_draft_cpu_strict_batch_spin'):
            self.config.spec_draft_cpu_strict_batch = self.adv_spec_draft_cpu_strict_batch_spin.value()
        if hasattr(self, 'adv_spec_draft_prio_batch_spin'):
            self.config.spec_draft_prio_batch = self.adv_spec_draft_prio_batch_spin.value()
        if hasattr(self, 'adv_spec_draft_poll_batch_spin'):
            self.config.spec_draft_poll_batch = self.adv_spec_draft_poll_batch_spin.value()
        if hasattr(self, 'adv_spec_draft_cpu_moe_check'):
            self.config.spec_draft_cpu_moe = self.adv_spec_draft_cpu_moe_check.isChecked()
        if hasattr(self, 'adv_spec_draft_n_cpu_moe_spin'):
            self.config.spec_draft_n_cpu_moe = self.adv_spec_draft_n_cpu_moe_spin.value()

        # --- Spec N-gram / mod params ---
        if hasattr(self, 'adv_spec_ngram_mod_n_min_spin'):
            self.config.spec_ngram_mod_n_min = self.adv_spec_ngram_mod_n_min_spin.value()
        if hasattr(self, 'adv_spec_ngram_mod_n_max_spin'):
            self.config.spec_ngram_mod_n_max = self.adv_spec_ngram_mod_n_max_spin.value()
        if hasattr(self, 'adv_spec_ngram_mod_n_match_spin'):
            self.config.spec_ngram_mod_n_match = self.adv_spec_ngram_mod_n_match_spin.value()
        if hasattr(self, 'adv_spec_ngram_simple_size_n_spin'):
            self.config.spec_ngram_simple_size_n = self.adv_spec_ngram_simple_size_n_spin.value()
        if hasattr(self, 'adv_spec_ngram_simple_size_m_spin'):
            self.config.spec_ngram_simple_size_m = self.adv_spec_ngram_simple_size_m_spin.value()
        if hasattr(self, 'adv_spec_ngram_simple_min_hits_spin'):
            self.config.spec_ngram_simple_min_hits = self.adv_spec_ngram_simple_min_hits_spin.value()
        if hasattr(self, 'adv_spec_ngram_map_k_size_n_spin'):
            self.config.spec_ngram_map_k_size_n = self.adv_spec_ngram_map_k_size_n_spin.value()
        if hasattr(self, 'adv_spec_ngram_map_k_size_m_spin'):
            self.config.spec_ngram_map_k_size_m = self.adv_spec_ngram_map_k_size_m_spin.value()
        if hasattr(self, 'adv_spec_ngram_map_k_min_hits_spin'):
            self.config.spec_ngram_map_k_min_hits = self.adv_spec_ngram_map_k_min_hits_spin.value()
        if hasattr(self, 'adv_spec_ngram_map_k4v_size_n_spin'):
            self.config.spec_ngram_map_k4v_size_n = self.adv_spec_ngram_map_k4v_size_n_spin.value()
        if hasattr(self, 'adv_spec_ngram_map_k4v_size_m_spin'):
            self.config.spec_ngram_map_k4v_size_m = self.adv_spec_ngram_map_k4v_size_m_spin.value()
        if hasattr(self, 'adv_spec_ngram_map_k4v_min_hits_spin'):
            self.config.spec_ngram_map_k4v_min_hits = self.adv_spec_ngram_map_k4v_min_hits_spin.value()

        # --- Extra cache ---
        if hasattr(self, 'adv_ctx_checkpoints_spin'):
            self.config.ctx_checkpoints = self.adv_ctx_checkpoints_spin.value()
        if hasattr(self, 'adv_checkpoint_min_step_spin'):
            self.config.checkpoint_min_step = self.adv_checkpoint_min_step_spin.value()
        if hasattr(self, 'adv_cache_ram_spin'):
            self.config.cache_ram = self.adv_cache_ram_spin.value()
        if hasattr(self, 'adv_cache_idle_slots_check'):
            self.config.cache_idle_slots = self.adv_cache_idle_slots_check.isChecked()
        if hasattr(self, 'adv_context_shift_check'):
            self.config.context_shift = self.adv_context_shift_check.isChecked()

        # --- Generation extras ---
        if hasattr(self, 'adv_special_check'):
            self.config.special = self.adv_special_check.isChecked()
        if hasattr(self, 'adv_spm_infill_check'):
            self.config.spm_infill = self.adv_spm_infill_check.isChecked()

        # --- Multimodal ---
        if hasattr(self, 'adv_mmproj_input'):
            text = self.adv_mmproj_input.text().strip()
            self.config.mmproj = text if text else ""
        if hasattr(self, 'adv_mmproj_url_input'):
            text = self.adv_mmproj_url_input.text().strip()
            self.config.mmproj_url = text if text else ""
        if hasattr(self, 'adv_mmproj_auto_check'):
            self.config.mmproj_auto = self.adv_mmproj_auto_check.isChecked()
        if hasattr(self, 'adv_mmproj_offload_check'):
            self.config.mmproj_offload = self.adv_mmproj_offload_check.isChecked()
        if hasattr(self, 'adv_image_min_tokens_spin'):
            self.config.image_min_tokens = self.adv_image_min_tokens_spin.value()
        if hasattr(self, 'adv_image_max_tokens_spin'):
            self.config.image_max_tokens = self.adv_image_max_tokens_spin.value()
        if hasattr(self, 'adv_mtmd_batch_max_tokens_spin'):
            self.config.mtmd_batch_max_tokens = self.adv_mtmd_batch_max_tokens_spin.value()

        # --- Server extras ---
        if hasattr(self, 'adv_alias_input'):
            text = self.adv_alias_input.text().strip()
            self.config.alias = text if text else ""
        if hasattr(self, 'adv_tags_input'):
            text = self.adv_tags_input.text().strip()
            self.config.tags = text if text else ""
        if hasattr(self, 'adv_embd_normalize_spin'):
            self.config.embd_normalize = self.adv_embd_normalize_spin.value()
        if hasattr(self, 'adv_path_input'):
            text = self.adv_path_input.text().strip()
            self.config.path = text if text else ""
        if hasattr(self, 'adv_reuse_port_check'):
            self.config.reuse_port = self.adv_reuse_port_check.isChecked()
        if hasattr(self, 'adv_ui_mcp_proxy_check'):
            self.config.ui_mcp_proxy = self.adv_ui_mcp_proxy_check.isChecked()

    def on_context_editing_finished(self):
        """Handle context size editing finished and round to binary."""
        value = self.ctx_spin.value()
        rounded = self.ctx_spin.round_to_binary_context(value)
        if rounded != value:
            self.ctx_spin.blockSignals(True)
            self.ctx_spin.setValue(rounded)
            self.ctx_spin.blockSignals(False)

    def _sync_ctx_to_adv(self, value):
        """Sync basic context spin to advanced context spin."""
        self._ctx_auto_set = False
        if hasattr(self, 'adv_ctx_spin') and value > 0:
            self.adv_ctx_spin.blockSignals(True)
            self.adv_ctx_spin.setValue(value)
            self.adv_ctx_spin.blockSignals(False)

    def _sync_adv_ctx_to_basic(self, value):
        """Sync advanced context spin to basic context spin."""
        self._ctx_auto_set = False
        if hasattr(self, 'ctx_spin') and value > 0:
            self.ctx_spin.blockSignals(True)
            self.ctx_spin.setValue(value)
            self.ctx_spin.blockSignals(False)

    def _sync_cache_to_adv(self, value):
        """Sync basic cache type combo to advanced cache type combos and config."""
        if hasattr(self, 'adv_cache_type_k_combo') and hasattr(self, 'adv_cache_type_v_combo'):
            self.adv_cache_type_k_combo.blockSignals(True)
            self.adv_cache_type_v_combo.blockSignals(True)
            self.adv_cache_type_k_combo.setCurrentText(value)
            self.adv_cache_type_v_combo.setCurrentText(value)
            self.adv_cache_type_k_combo.blockSignals(False)
            self.adv_cache_type_v_combo.blockSignals(False)
        # Also update the config values the estimator uses
        self.config.cache_type_k = value
        self.config.cache_type_v = value

    def _set_ctx_from_model(self, training_ctx: int):
        """Set context slider to model's training context."""
        self.ctx_spin.blockSignals(True)
        self.ctx_spin.setValue(training_ctx)
        self.ctx_spin.blockSignals(False)
        self.config.context_size = training_ctx

        if hasattr(self, 'adv_ctx_spin'):
            self.adv_ctx_spin.blockSignals(True)
            self.adv_ctx_spin.setValue(training_ctx)
            self.adv_ctx_spin.blockSignals(False)

    def open_preset_manager(self):
        """Open preset manager dialog for CRUD operations."""
        dialog = PresetManagerDialog(self)
        if dialog.exec() == 1:
            # Refresh preset combo if presets were added/modified
            self.preset_combo.clear()
            self.preset_combo.addItems(PresetManager.get_all_presets())
            self.adv_preset_combo.clear()
            self.adv_preset_combo.addItems(PresetManager.get_all_presets())

    def save_current_preset(self):
        """Save current config to the currently selected preset."""
        # Get preset name from whichever preset combo is active or has a selection
        preset_name = None
        if self.tab_widget.currentIndex() == 1:  # Advanced Config tab is active
            preset_name = self.adv_preset_combo.currentText()
        else:
            preset_name = self.preset_combo.currentText()
            
        if not preset_name:
            return
            
        # Update config from UI
        self.update_config_from_ui()
        
        # Save as custom preset (overwrites if exists)
        PresetManager.save_custom_preset(preset_name, self.config)
        QMessageBox.information(self, "Success", f"Preset '{preset_name}' saved successfully.")

    def parse_launch_command(self, command_str):
        """Parse a llama-server command string and return a dict of parameters and unrecognized flags."""
        try:
            args = shlex.split(command_str)
        except ValueError:
            args = command_str.split()
        
        params = {}
        unrecognized_flags = []
        i = 0
        
        flag_map = {
            # Model paths
            '--model': 'model_path',
            '-m': 'model_path',
            '--mmproj': 'mmproj_model_path',
            '--mmproj-model': 'mmproj_model_path',
            '--draft-model': 'draft_model_path',
            '--draft': 'draft_model_path',
            # Common params - threads and batching
            '--threads': 'threads',
            '-t': 'threads',
            '--threads-batch': 'threads_batch',
            '-tb': 'threads_batch',
            '--cpu-mask': 'cpu_mask',
            '--cpu-range': 'cpu_range',
            '--cpu-strict': 'cpu_strict',
            '--prio': 'prio',
            '--poll': 'poll',
            '--cpu-mask-batch': 'cpu_mask_batch',
            '--cpu-range-batch': 'cpu_range_batch',
            '--cpu-strict-batch': 'cpu_strict_batch',
            '--prio-batch': 'prio_batch',
            '--poll-batch': 'poll_batch',
            # Common params - context and predict
            '--ctx-size': 'context_size',
            '-c': 'context_size',
            '--predict': 'predict',
            '-n': 'predict',
            '--n-predict': 'predict',
            '--batch-size': 'batch_size',
            '-b': 'batch_size',
            '--ubatch-size': 'ubatch_size',
            '-ub': 'ubatch_size',
            '--keep': 'keep',
            '--swa-full': 'swa_full',
            '--no-swa-full': 'swa_full',
            '--flash-attn': 'flash_attn',
            '-fa': 'flash_attn',
            '--perf': 'perf',
            '--no-perf': 'perf',
            '--escape': 'escape',
            '--no-escape': 'escape',
            # Common params - RoPE and YaRN
            '--rope-scaling': 'rope_scaling',
            '--rope-scale': 'rope_scale',
            '--rope-freq-base': 'rope_freq_base',
            '--rope-freq-scale': 'rope_freq_scale',
            '--yarn-orig-ctx': 'yarn_orig_ctx',
            '--yarn-ext-factor': 'yarn_ext_factor',
            '--yarn-attn-factor': 'yarn_attn_factor',
            '--yarn-beta-slow': 'yarn_beta_slow',
            '--yarn-beta-fast': 'yarn_beta_fast',
            # Common params - cache and memory
            '--kv-offload': 'kv_offload',
            '--no-kv-offload': 'kv_offload',
            '--repack': 'repack',
            '--no-repack': 'repack',
            '--no-host': 'no_host',
            '--host': 'no_host',
            '--cache-type-k': 'cache_type_k',
            '-ctk': 'cache_type_k',
            '--cache-type-v': 'cache_type_v',
            '-ctv': 'cache_type_v',
            '--kv-cache-type': 'kv_cache_type',
            '--defrag-thold': 'defrag_thold',
            '--mlock': 'mlock',
            '--no-mlock': 'mlock',
            '--mmap': 'use_mmap',
            '--no-mmap': 'use_mmap',
            '--direct-io': 'direct_io',
            '--no-direct-io': 'direct_io',
            # Common params - GPU and NUMA
            '--numa': 'numa',
            '--device': 'device',
            '--override-tensor': 'override_tensor',
            '--cpu-moe': 'cpu_moe',
            '--no-cpu-moe': 'cpu_moe',
            '--n-cpu-moe': 'n_cpu_moe',
            '--gpu-layers': 'gpu_layers',
            '-ngl': 'gpu_layers',
            '--n-gpu-layers': 'gpu_layers',
            '--split-mode': 'split_mode',
            '-sm': 'split_mode',
            '--tensor-split': 'tensor_split',
            '-ts': 'tensor_split',
            '--main-gpu': 'main_gpu',
            '-mg': 'main_gpu',
            '--fit': 'fit',
            '--fit-target': 'fit_target',
            '--fit-ctx': 'fit_ctx',
            '--check-tensors': 'check_tensors',
            '--no-check-tensors': 'check_tensors',
            '--override-kv': 'override_kv',
            '--op-offload': 'op_offload',
            # LoRA and control vectors
            '--lora': 'lora',
            '--lora-scaled': 'lora_scaled',
            '--control-vector': 'control_vector',
            '--control-vector-scaled': 'control_vector_scaled',
            '--control-vector-layer-range': 'control_vector_layer_range',
            # Model sources
            '--model-url': 'model_url',
            '--docker-repo': 'docker_repo',
            '--hf-repo': 'hf_repo',
            '--hf-file': 'hf_file',
            '--hf-repo-v': 'hf_repo_v',
            '--hf-file-v': 'hf_file_v',
            '--hf-token': 'hf_token',
            # Logging
            '--log-disable': 'log_disable',
            '--log-file': 'log_file',
            '--log-colors': 'log_colors',
            '--verbose': 'verbose',
            '--offline': 'offline',
            '--verbosity': 'verbosity',
            '--log-prefix': 'log_prefix',
            '--log-timestamps': 'log_timestamps',
            # Spec draft cache types
            '--spec-draft-type-k': 'spec_draft_type_k',
            '--spec-draft-type-v': 'spec_draft_type_v',
            # Sampling params
            '--samplers': 'samplers',
            '--seed': 'seed',
            '--sampler-seq': 'sampler_seq',
            '--ignore-eos': 'ignore_eos',
            '--no-ignore-eos': 'ignore_eos',
            '--temp': 'temp',
            '--temperature': 'temp',
            '--top-k': 'top_k',
            '--top-p': 'top_p',
            '--min-p': 'min_p',
            '--top-n-sigma': 'top_n_sigma',
            '--xtc-probability': 'xtc_probability',
            '--xtc-threshold': 'xtc_threshold',
            '--typical-p': 'typical_p',
            '--repeat-last-n': 'repeat_last_n',
            '--repeat-penalty': 'repeat_penalty',
            '--presence-penalty': 'presence_penalty',
            '--frequency-penalty': 'frequency_penalty',
            '--dry-multiplier': 'dry_multiplier',
            '--dry-base': 'dry_base',
            '--dry-allowed-length': 'dry_allowed_length',
            '--dry-penalty-last-n': 'dry_penalty_last_n',
            '--dry-sequence-breaker': 'dry_sequence_breaker',
            '--adaptive-target': 'adaptive_target',
            '--adaptive-decay': 'adaptive_decay',
            '--dynatemp-range': 'dynatemp_range',
            '--dynatemp-exp': 'dynatemp_exp',
            '--mirostat': 'mirostat',
            '--mirostat-lr': 'mirostat_lr',
            '--mirostat-ent': 'mirostat_ent',
            '--logit-bias': 'logit_bias',
            '--grammar': 'grammar',
            '--grammar-file': 'grammar_file',
            '--json-schema': 'json_schema',
            '--json-schema-file': 'json_schema_file',
            '--backend-sampling': 'backend_sampling',
            # Speculative params
            '--spec-draft-hf': 'spec_draft_hf',
            '--spec-draft-threads': 'spec_draft_threads',
            '--spec-draft-threads-batch': 'spec_draft_threads_batch',
            '--spec-draft-cpu-mask': 'spec_draft_cpu_mask',
            '--spec-draft-cpu-range': 'spec_draft_cpu_range',
            '--spec-draft-cpu-strict': 'spec_draft_cpu_strict',
            '--spec-draft-prio': 'spec_draft_prio',
            '--spec-draft-poll': 'spec_draft_poll',
            '--spec-draft-cpu-mask-batch': 'spec_draft_cpu_mask_batch',
            '--spec-draft-cpu-strict-batch': 'spec_draft_cpu_strict_batch',
            '--spec-draft-prio-batch': 'spec_draft_prio_batch',
            '--spec-draft-poll-batch': 'spec_draft_poll_batch',
            '--spec-draft-override-tensor': 'spec_draft_override_tensor',
            '--spec-draft-cpu-moe': 'spec_draft_cpu_moe',
            '--spec-draft-n-cpu-moe': 'spec_draft_n_cpu_moe',
            '--spec-draft-n-max': 'draft_n_max',
            '--spec-draft-n-min': 'draft_n_min',
            '--draft-n-max': 'draft_n_max',
            '--draft-n-min': 'draft_n_min',
            '--spec-draft-p-split': 'spec_draft_p_split',
            '--spec-draft-p-min': 'spec_draft_p_min',
            '--spec-draft-backend-sampling': 'spec_draft_backend_sampling',
            '--spec-draft-device': 'spec_draft_device',
            '--spec-draft-ngl': 'spec_draft_ngl',
            '--spec-draft-model': 'spec_draft_model',
            '--spec-type': 'spec_type',
            '--spec-ngram-mod-n-min': 'spec_ngram_mod_n_min',
            '--spec-ngram-mod-n-max': 'spec_ngram_mod_n_max',
            '--spec-ngram-mod-n-match': 'spec_ngram_mod_n_match',
            '--spec-ngram-simple-size-n': 'spec_ngram_simple_size_n',
            '--spec-ngram-simple-size-m': 'spec_ngram_simple_size_m',
            '--spec-ngram-simple-min-hits': 'spec_ngram_simple_min_hits',
            '--spec-ngram-map-k-size-n': 'spec_ngram_map_k_size_n',
            '--spec-ngram-map-k-size-m': 'spec_ngram_map_k_size_m',
            '--spec-ngram-map-k-min-hits': 'spec_ngram_map_k_min_hits',
            '--spec-ngram-map-k4v-size-n': 'spec_ngram_map_k4v_size_n',
            '--spec-ngram-map-k4v-size-m': 'spec_ngram_map_k4v_size_m',
            '--spec-ngram-map-k4v-min-hits': 'spec_ngram_map_k4v_min_hits',
            # Cache and context
            '--lookup-cache-static': 'lookup_cache_static',
            '--lookup-cache-dynamic': 'lookup_cache_dynamic',
            '--ctx-checkpoints': 'ctx_checkpoints',
            '--checkpoint-min-step': 'checkpoint_min_step',
            '--cache-ram': 'cache_ram',
            '--kv-unified': 'kv_unified',
            '--no-kv-unified': 'kv_unified',
            '--cache-idle-slots': 'cache_idle_slots',
            '--context-shift': 'context_shift',
            # Generation and server
            '--reverse-prompt': 'reverse_prompt',
            '--special': 'special',
            '--warmup': 'warmup',
            '--spm-infill': 'spm_infill',
            '--pooling': 'pooling',
            '--parallel': 'parallel',
            '--cont-batching': 'cont_batching',
            '--no-cont-batching': 'cont_batching',
            # Multimodal
            '--mmproj': 'mmproj_model_path',
            '--mmproj-url': 'mmproj_url',
            '--mmproj-auto': 'mmproj_auto',
            '--mmproj-offload': 'mmproj_offload',
            '--image-min-tokens': 'image_min_tokens',
            '--image-max-tokens': 'image_max_tokens',
            '--mtmd-batch-max-tokens': 'mtmd_batch_max_tokens',
            # Server and UI
            '--alias': 'alias',
            '--tags': 'tags',
            '--embd-normalize': 'embd_normalize',
            '--port': 'port',
            '-p': 'port',
            '--host': 'host',
            '-h': 'host',
            '--reuse-port': 'reuse_port',
            '--path': 'path',
            '--api-prefix': 'api_prefix',
            '--ui-config': 'ui_config',
            '--ui-config-file': 'ui_config_file',
            '--ui-mcp-proxy': 'ui_mcp_proxy',
            '--tools': 'tools',
            '--agent': 'agent',
            '--ui': 'ui',
            '--no-ui': 'ui',
            '--embeddings': 'embeddings',
            '--reranking': 'reranking',
            # Backend and jinja
            '--backend': 'backend',
            '--jinja': 'jinja',
            '--no-jinja': 'jinja',
        }
        
        while i < len(args):
            arg = args[i]
            
            # Handle --flag=value or -f=value format
            if '=' in arg and (arg.startswith('--') or arg.startswith('-')):
                key, value = arg.split('=', 1)
                config_key = flag_map.get(key)
                if config_key:
                    if config_key in ['context_size', 'gpu_layers', 'threads', 'threads_batch', 'port', 'batch_size', 'ubatch_size', 'top_k', 'repeat_last_n', 'rope_scale', 'rope_freq_base', 'predict', 'keep', 'defrag_thold', 'yarn_orig_ctx', 'fit_ctx']:
                        try:
                            params[config_key] = int(value)
                        except ValueError:
                            params[config_key] = value
                    elif config_key in ['temp', 'top_p', 'min_p', 'repeat_penalty', 'presence_penalty', 'frequency_penalty', 'yarn_ext_factor', 'yarn_attn_factor', 'yarn_beta_slow', 'yarn_beta_fast']:
                        try:
                            params[config_key] = float(value)
                        except ValueError:
                            params[config_key] = value
                    elif config_key in ['use_mmap', 'mlock', 'kv_offload', 'swa_full', 'perf', 'escape', 'repack', 'no_host', 'direct_io', 'cpu_moe', 'check_tensors', 'flash_attn', 'jinja', 'parallel', 'cont_batching', 'embeddings', 'reranking', 'ui', 'log_disable', 'offline', 'verbose']:
                        params[config_key] = value.lower() != 'false' and value.lower() != '0' and value.lower() != 'off'
                    else:
                        params[config_key] = value
                else:
                    unrecognized_flags.append(arg)
                i += 1
                continue
            
            if arg.startswith('--') or arg.startswith('-'):
                config_key = flag_map.get(arg)
                if config_key:
                    if i + 1 < len(args) and not args[i+1].startswith('-'):
                        value = args[i+1]
                        i += 2
                        
                        if config_key in ['context_size', 'gpu_layers', 'threads', 'threads_batch', 'port', 'batch_size', 'ubatch_size', 'top_k', 'repeat_last_n', 'rope_scale', 'rope_freq_base', 'predict', 'keep', 'defrag_thold', 'yarn_orig_ctx', 'fit_ctx']:
                            try:
                                params[config_key] = int(value)
                            except ValueError:
                                params[config_key] = value
                        elif config_key in ['temp', 'top_p', 'min_p', 'repeat_penalty', 'presence_penalty', 'frequency_penalty', 'yarn_ext_factor', 'yarn_attn_factor', 'yarn_beta_slow', 'yarn_beta_fast']:
                            try:
                                params[config_key] = float(value)
                            except ValueError:
                                params[config_key] = value
                        elif config_key in ['use_mmap', 'mlock', 'kv_offload', 'swa_full', 'perf', 'escape', 'repack', 'no_host', 'direct_io', 'cpu_moe', 'check_tensors', 'flash_attn', 'jinja', 'parallel', 'cont_batching', 'embeddings', 'reranking', 'ui', 'log_disable', 'offline', 'verbose']:
                            if arg.startswith('--no-'):
                                params[config_key] = False
                            else:
                                params[config_key] = True
                        else:
                            params[config_key] = value
                    else:
                        if config_key in ['use_mmap', 'mlock', 'kv_offload', 'swa_full', 'perf', 'escape', 'repack', 'no_host', 'direct_io', 'cpu_moe', 'check_tensors', 'flash_attn', 'jinja', 'parallel', 'cont_batching', 'embeddings', 'reranking', 'ui', 'log_disable', 'offline', 'verbose']:
                            if arg.startswith('--no-'):
                                params[config_key] = False
                            else:
                                params[config_key] = True
                        i += 1
                else:
                    unrecognized_flags.append(arg)
                    i += 1
            else:
                if 'model_path' not in params and not arg.startswith('-'):
                    if arg.endswith(('.gguf', '.bin', '.safetensors')) or '/' in arg or '\\' in arg:
                        params['model_path'] = arg
                i += 1
                
        return params, unrecognized_flags

    def create_server_management_tab(self):
        """Create server management tab for local and remote servers."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        self.server_management_widget = ServerManagementWidget(
            self.server_manager, self
        )
        self.server_management_widget.list_widget.servers_changed.connect(
            self._on_servers_changed
        )
        layout.addWidget(self.server_management_widget)

        self.tab_widget.addTab(tab, "Server Management")

    def create_import_launch_command_tab(self):
        """Create import launch command tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Command input
        command_group = QGroupBox("Launch Command")
        command_layout = QVBoxLayout()
        
        self.import_command_text = QTextEdit()
        self.import_command_text.setPlaceholderText("Paste llama-server command here, e.g.:\nllama-server --model /path/to/model.gguf --ctx-size 4096 --gpu-layers 33 --threads 8 --port 8080 --host 127.0.0.1")
        self.import_command_text.setMinimumHeight(150)
        
        command_layout.addWidget(self.import_command_text)
        command_group.setLayout(command_layout)
        
        # Parse and import button
        self.import_parse_btn = QPushButton("Parse Command and Create Preset")
        self.import_parse_btn.clicked.connect(self.import_launch_command)
        
        layout.addWidget(command_group)
        layout.addWidget(self.import_parse_btn)
        
        # Result display
        result_group = QGroupBox("Parsed Parameters")
        result_layout = QVBoxLayout()
        
        self.import_result_text = QTextEdit()
        self.import_result_text.setReadOnly(True)
        self.import_result_text.setMinimumHeight(150)
        
        result_layout.addWidget(self.import_result_text)
        result_group.setLayout(result_layout)
        
        layout.addWidget(result_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Import Launch Command")

    def import_launch_command(self):
        """Parse launch command and create preset."""
        command_str = self.import_command_text.toPlainText().strip()
        if not command_str:
            QMessageBox.warning(self, "Error", "Please enter a launch command.")
            return
        
        params, unrecognized_flags = self.parse_launch_command(command_str)
        
        if not params and not unrecognized_flags:
            QMessageBox.warning(self, "Error", "No parameters detected in the command.")
            return
        
        if unrecognized_flags:
            error_msg = "Unrecognized flags found:\n" + "\n".join(unrecognized_flags) + "\n\nPlease ensure all flags are valid llama-server parameters."
            QMessageBox.warning(self, "Import Error", error_msg)
            return
        
        # Display parsed parameters
        result_lines = ["Parsed Parameters:"]
        for key, value in params.items():
            result_lines.append(f"  {key}: {value}")
        
        self.import_result_text.setText('\n'.join(result_lines))
        
        # Prompt for preset name
        name, ok = QInputDialog.getText(self, "Create Preset", "Preset Name:")
        if ok and name:
            if name in PresetManager.get_all_presets():
                QMessageBox.warning(self, "Error", f"Preset '{name}' already exists.")
                return
            
            # Create config from parsed parameters with defaults
            config = LlamaConfig(
                model_path=params.get('model_path', ''),
                mmproj_model_path=params.get('mmproj_model_path', None),
                draft_model_path=params.get('draft_model_path', None),
                context_size=params.get('context_size', 4096),
                gpu_layers=params.get('gpu_layers', -1),
                threads=params.get('threads', 0),
                threads_batch=params.get('threads_batch', -1),
                port=params.get('port', 8080),
                host=params.get('host', '127.0.0.1'),
                backend=params.get('backend', 'auto'),
                use_mmap=params.get('use_mmap', True),
                kv_cache_type=params.get('kv_cache_type', params.get('cache_type_k', 'f16')),
                cache_type_k=params.get('cache_type_k', 'f16'),
                cache_type_v=params.get('cache_type_v', 'f16'),
                batch_size=params.get('batch_size', 2048),
                ubatch_size=params.get('ubatch_size', 512),
                temp=params.get('temp', 0.8),
                top_p=params.get('top_p', 0.95),
                top_k=params.get('top_k', 40),
                repeat_penalty=params.get('repeat_penalty', 1.0),
                flash_attn=params.get('flash_attn', 'auto'),
                parallel=params.get('parallel', -1),
                jinja=params.get('jinja', False),
                name=name,
                description=f"Imported from command: {command_str[:50]}..."
            )
            
            PresetManager.save_custom_preset(name, config)
            QMessageBox.information(self, "Success", f"Preset '{name}' created successfully.")
            self.refresh_preset_list_if_needed()

    def refresh_preset_list_if_needed(self):
        """Refresh preset lists if they exist."""
        if hasattr(self, 'preset_combo'):
            self.preset_combo.clear()
            self.preset_combo.addItems(PresetManager.get_all_presets())
        if hasattr(self, 'adv_preset_combo'):
            self.adv_preset_combo.clear()
            self.adv_preset_combo.addItems(PresetManager.get_all_presets())


class PresetManagerDialog(QDialog):
    """Dialog for managing presets (CRUD operations)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Presets")
        self.setGeometry(300, 300, 500, 400)
        
        self.layout = QVBoxLayout(self)
        
        # Preset list
        self.preset_list = QListWidget()
        self.refresh_preset_list()
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add New")
        self.delete_btn = QPushButton("Delete Selected")
        self.save_btn = QPushButton("Save Current Config as Preset")
        
        self.add_btn.clicked.connect(self.add_preset)
        self.delete_btn.clicked.connect(self.delete_preset)
        self.save_btn.clicked.connect(self.save_current_as_preset)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        
        self.layout.addWidget(QLabel("Existing Presets:"))
        self.layout.addWidget(self.preset_list)
        self.layout.addLayout(btn_layout)
        
    def refresh_preset_list(self):
        """Refresh the preset list."""
        self.preset_list.clear()
        for preset_name in PresetManager.get_all_presets():
            self.preset_list.addItem(preset_name)
            
    def add_preset(self):
        """Add a new preset."""
        name, ok = QInputDialog.getText(self, "New Preset", "Preset Name:")
        if ok and name:
            if name in PresetManager.get_all_presets():
                QMessageBox.warning(self, "Error", f"Preset '{name}' already exists.")
                return
            
            # Create default config for new preset
            new_config = LlamaConfig(
                model_path="",
                context_size=4096,
                gpu_layers=-1,
                threads=0,
                port=8080,
                host="127.0.0.1",
                backend="auto",
                use_mmap=True,
                kv_cache_type="f16",
                batch_size=512,
                name=name,
                description=f"Custom preset: {name}"
            )
            PresetManager.save_custom_preset(name, new_config)
            self.refresh_preset_list()
            
    def delete_preset(self):
        """Delete selected preset."""
        current_items = self.preset_list.selectedItems()
        if not current_items:
            QMessageBox.warning(self, "Error", "Please select a preset to delete.")
            return
            
        preset_name = current_items[0].text()
        
        # Check if it's a custom override or a built-in preset
        custom_presets = PresetManager._load_custom_presets()
        is_builtin_override = preset_name in custom_presets and preset_name in PresetManager.BUILTIN_PRESETS
        
        if not is_builtin_override and preset_name in PresetManager.BUILTIN_PRESETS:
            QMessageBox.warning(self, "Error", "Built-in presets cannot be deleted.")
            return
            
        reply = QMessageBox.question(self, "Confirm Delete", f"Delete preset '{preset_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if is_builtin_override or preset_name not in PresetManager.BUILTIN_PRESETS:
                PresetManager.delete_custom_preset(preset_name)
            self.refresh_preset_list()

    def save_current_as_preset(self):
        """Save the current window configuration as a new preset."""
        name, ok = QInputDialog.getText(self, "Save as Preset", "Preset Name:")
        if not ok or not name:
            return

        if name in PresetManager.get_all_presets():
            QMessageBox.warning(self, "Error", f"Preset '{name}' already exists.")
            return

        parent = self.parent()
        if parent:
            parent.update_config_from_ui()
            config = parent.config
            config.name = name
            config.description = f"Custom preset: {name}"
            PresetManager.save_custom_preset(name, config)
            self.refresh_preset_list()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    window = LlamaLauncherGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_gui()
