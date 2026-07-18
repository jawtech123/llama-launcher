"""PySide6 GUI Main Window."""
import dataclasses
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QTabWidget, QPushButton, QTextEdit,
                               QLineEdit, QLabel, QComboBox, QFileDialog, QGroupBox,
                               QListWidget, QListWidgetItem, QSpinBox, QCheckBox,
                               QDialog, QFormLayout, QInputDialog, QMessageBox)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QProcess, QTimer, QThread, Signal

from llama_launcher.core.config import LlamaConfig, migrate_config_dict
from llama_launcher.core.fields import CACHE_TYPES
from llama_launcher.core.validator import validate_config
from llama_launcher.core.builder import build_command
from llama_launcher.core.parser import parse_launch_command as _parse_launch_command
from llama_launcher.core.presets import PresetManager
from llama_launcher.gpu.detector import detect_gpu
from llama_launcher.gpu.types import GPUInfo
from llama_launcher.models.scanner import scan_models
from llama_launcher.models.storage import load_models as load_cached_models, save_models as save_cached_models
from llama_launcher.remote.manager import ServerManager
from llama_launcher.remote.types import ServerConfig, ConnectionType, ConnectionState
from llama_launcher.gui.advanced_tab import AdvancedConfigTab
from llama_launcher.gui.config_controller import ConfigController
from llama_launcher.gui.server_management import ServerManagementWidget
from llama_launcher.gui.resource_usage_panel import ResourceUsagePanel
from llama_launcher.gui.widgets import BinaryContextSpinBox, info_label
from llama_launcher.estimator import ResourceEstimator
from llama_launcher.utils.helpers import get_file_size_mb
from llama_launcher.utils.paths import config_file

logger = logging.getLogger(__name__)


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
                    logger.debug("Model scan failed for %s on %s", dir_path, self.server_name, exc_info=True)
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
                    logger.debug("nvidia-smi extended info unavailable", exc_info=True)
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
            logger.debug("nvidia-smi stats poll failed", exc_info=True)
        self.stats_done.emit("")


class LlamaLauncherGUI(QMainWindow):
    """Main GUI Window for llama-launcher."""

    def __init__(self):
        super().__init__()
        self.config = LlamaConfig()
        self.controller = ConfigController(self.config, self)
        self.models = []
        self.server_manager = ServerManager()
        self.gpu_info: GPUInfo = GPUInfo(backend="cpu", device_name="Detecting...")
        self.resource_estimator = ResourceEstimator()
        self._ctx_auto_set = True  # Allow auto-set on first model load
        self._active_workers = []
        self._closing = False
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

        # Push initial config values into all bound widgets
        self.controller.sync_ui()

        # Defer GPU detection and model scan to after UI is shown
        QTimer.singleShot(200, self._deferred_init)
        
        # Load first preset at launch time
        presets = PresetManager.get_all_presets()
        if presets:
            QTimer.singleShot(100, lambda: self.load_preset(presets[0]))

    def closeEvent(self, event):
        """Clean up timers, workers, and server connections on exit."""
        # Prevent deferred timers/signals from starting new work.
        self._closing = True
        for timer_name in ('_preset_style_timer', '_gpu_stats_timer'):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        mgmt = getattr(self, 'server_management_widget', None)
        if mgmt is not None:
            mgmt.shutdown()
        for w in list(self._active_workers):
            if w.isRunning():
                w.wait(1000)
        self.server_manager.disconnect_all()
        event.accept()

    def _start_worker(self, worker: QThread) -> None:
        """Start a worker thread with proper lifecycle management.

        Workers delete themselves on finish and are pruned from the tracking
        list, so ``_active_workers`` only ever holds running threads.
        """
        worker.setParent(self)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda: self._prune_worker(worker))
        self._active_workers.append(worker)
        worker.start()

    def _prune_worker(self, worker: QThread) -> None:
        try:
            self._active_workers.remove(worker)
        except ValueError:
            logger.debug("Suppressed exception", exc_info=True)

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
        self.refresh_gpu_btn.clicked.connect(lambda: self._detect_gpu_async(force=True))
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
        config_form.addRow(info_label("Main Model:", "Path to the main model file (.gguf, .bin, .safetensors)."), self.model_combo)
        
        self.mmproj_combo = QComboBox()
        self.mmproj_combo.setEditable(True)
        self.mmproj_combo.setPlaceholderText("Select or enter mmproj model path (.gguf) - optional for vision models")
        self.mmproj_combo.setMinimumWidth(300)
        self.mmproj_browse_btn = QPushButton("Browse...")
        self.mmproj_browse_btn.clicked.connect(self.browse_mmproj_model)
        config_form.addRow(info_label("mmproj Model:", "Path to mmproj model for vision-language models."), self.mmproj_combo)
        
        self.draft_combo = QComboBox()
        self.draft_combo.setEditable(True)
        self.draft_combo.setPlaceholderText("Select or enter draft model path (.gguf) - optional for speculative decoding")
        self.draft_combo.setMinimumWidth(300)
        self.draft_browse_btn = QPushButton("Browse...")
        self.draft_browse_btn.clicked.connect(self.browse_draft_model)
        config_form.addRow(info_label("Draft Model:", "Path to draft model for speculative decoding."), self.draft_combo)
        
        # Context Size
        self.ctx_spin = BinaryContextSpinBox()
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setToolTip("Size of the prompt context in tokens. Default: 4096.")
        self.ctx_spin.editingFinished.connect(self.on_context_editing_finished)
        self.ctx_spin.valueChanged.connect(lambda *_: setattr(self, '_ctx_auto_set', False))
        self.ctx_spin.setMinimumWidth(120)
        config_form.addRow(info_label("Context Size:", "Size of the prompt context in tokens."), self.ctx_spin)
        
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
        config_form.addRow(info_label("Threads:", "Number of CPU threads (0=auto)."), threads_layout)
        
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
        config_form.addRow(info_label("GPU Layers:", "Max layers to VRAM (-1=all)."), gpu_layers_layout)
        
        # Batch Size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 65536)
        self.batch_spin.setValue(2048)
        self.batch_spin.setToolTip("Logical maximum batch size (-b). Default: 2048.")
        self.batch_spin.setMinimumWidth(120)
        config_form.addRow(info_label("Batch Size:", "Physical max batch size."), self.batch_spin)
        
        # Port
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8080)
        self.port_spin.setToolTip("Port for the llama-server to listen on. Default: 8080.")
        self.port_spin.setMinimumWidth(120)
        config_form.addRow(info_label("Port:", "Server listen port."), self.port_spin)
        
        # Host
        self.host_input = QLineEdit()
        self.host_input.setText("127.0.0.1")
        self.host_input.setToolTip("Host address for the llama-server to bind to. Default: 127.0.0.1.")
        self.host_input.setMinimumWidth(150)
        config_form.addRow(info_label("Host:", "Server bind address."), self.host_input)
        
        # KV Cache Type
        self.cache_combo = QComboBox()
        self.cache_combo.addItems(list(CACHE_TYPES))
        self.cache_combo.setCurrentText("f16")
        self.cache_combo.setToolTip("KV cache data type for K and V. Default: f16.")
        self.cache_combo.setMinimumWidth(120)
        config_form.addRow(info_label("KV Cache Type:", "KV cache data type."), self.cache_combo)
        
        # mmap checkbox
        self.mmap_check = QCheckBox("--mmap / --no-mmap")
        self.mmap_check.setChecked(True)
        self.mmap_check.setToolTip("Whether to memory-map model. Default: True (enabled). If disabled, slower load but may reduce pageouts if not using mlock.")
        config_form.addRow(info_label("--mmap / --no-mmap", "Whether to memory-map model. Default: True (enabled). If disabled, slower load but may reduce pageouts if not using mlock."), self.mmap_check)
        
        config_layout.addLayout(config_form)
        config_group.setLayout(config_layout)

        layout.addWidget(config_group)

        # Bind all basic-tab widgets to the live config via the controller.
        self.controller.bind("model_path", self.model_combo.currentText,
                             self.model_combo.setCurrentText, self.model_combo.currentTextChanged)
        self.controller.bind("mmproj_model_path", self.mmproj_combo.currentText,
                             self.mmproj_combo.setCurrentText, self.mmproj_combo.currentTextChanged)
        self.controller.bind("draft_model_path", self.draft_combo.currentText,
                             self.draft_combo.setCurrentText, self.draft_combo.currentTextChanged)
        self.controller.bind("context_size", self.ctx_spin.value,
                             self.ctx_spin.setValue, self.ctx_spin.valueChanged)
        self.controller.bind("threads", self.threads_spin.value,
                             self.threads_spin.setValue, self.threads_spin.valueChanged)
        self.controller.bind("gpu_layers", self.gpu_spin.value,
                             self.gpu_spin.setValue, self.gpu_spin.valueChanged)
        self.controller.bind("batch_size", self.batch_spin.value,
                             self.batch_spin.setValue, self.batch_spin.valueChanged)
        self.controller.bind("port", self.port_spin.value,
                             self.port_spin.setValue, self.port_spin.valueChanged)
        self.controller.bind("host", self.host_input.text,
                             self.host_input.setText, self.host_input.textChanged)
        self.controller.bind("use_mmap", self.mmap_check.isChecked,
                             self.mmap_check.setChecked, self.mmap_check.stateChanged)
        # The basic KV cache combo drives both K and V cache types.
        self.controller.bind("cache_type_k", self.cache_combo.currentText,
                             self.cache_combo.setCurrentText, self.cache_combo.currentTextChanged)
        self.cache_combo.currentTextChanged.connect(
            lambda v: self.controller.set("cache_type_v", v))

        self.tab_widget.addTab(tab, "Basic Config")
        
    def create_advanced_config_tab(self):
        """Create the advanced config tab generated from the field registry."""
        self.advanced_tab = AdvancedConfigTab(
            self.controller,
            preset_names=PresetManager.get_all_presets(),
            on_preset_selected=self.load_preset,
            on_save_preset=self.save_current_preset,
            on_manage_presets=self.open_preset_manager,
        )
        self.adv_preset_combo = self.advanced_tab.preset_combo
        self.tab_widget.addTab(self.advanced_tab, "Advanced Config")

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
        
        # Rebuild the preview (debounced) whenever any bound field changes.
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(250)
        self._preview_debounce.timeout.connect(self.update_command_preview)
        self.controller.field_changed.connect(self._schedule_preview_update)
        self.update_command_preview()

    def _schedule_preview_update(self, *_args):
        """Restart the preview debounce timer after a field change."""
        if hasattr(self, '_preview_debounce'):
            self._preview_debounce.start()

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
        """Load saved folders from the config directory."""
        folders_file = config_file("folders.json")
        if not folders_file.exists():
            return
        try:
            with open(folders_file, 'r') as f:
                folders = json.load(f)
            for folder in folders:
                if not self.folders_list.findItems(folder, Qt.MatchExactly):
                    self.folders_list.addItem(folder)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load folders file %s: %s", folders_file, e)

    def _post_init_scan(self):
        """Run model scan after window is shown (non-blocking)."""
        if self.folders_list.count() > 0:
            self.scan_models()
                
    def save_folders(self):
        """Save folders to the config directory."""
        folders = []
        for i in range(self.folders_list.count()):
            folders.append(self.folders_list.item(i).text())
        try:
            with open(config_file("folders.json"), 'w') as f:
                json.dump(folders, f, indent=2)
        except OSError:
            logger.exception("Could not save folders file")
            
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
        if self._closing:
            return
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
        existing = getattr(self, '_scan_worker', None)
        if existing is not None and existing.isRunning():
            logger.debug("Model scan already in progress; ignoring request")
            return
        self._scan_worker = _ModelScanWorker(
            self.server_manager, server_name, dirs_to_scan, is_local_only
        )
        self._scan_worker.scan_done.connect(self._on_scan_done)
        self._scan_worker.finished.connect(lambda: setattr(self, '_scan_worker', None))
        self._start_worker(self._scan_worker)

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
            # Bound widgets propagate these into the config automatically.
            if cfg.hostname and cfg.hostname != "localhost":
                self.host_input.setText(cfg.hostname)
            if cfg.server_port:
                self.port_spin.setValue(cfg.server_port)
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
        if self._closing:
            return
        preset_names = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
        server_names = self.server_manager.get_all_names()
        worker = _PresetStyleWorker(preset_names, server_names, self.server_manager)
        worker.style_done.connect(self._on_preset_style_done)
        self._start_worker(worker)

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

    def _deferred_init(self):
        """Run blocking init tasks after UI is shown."""
        if self._closing:
            return
        self.scan_models()
        self._preset_style_timer = QTimer(self)
        self._preset_style_timer.timeout.connect(self._refresh_preset_combo_styles)
        self._preset_style_timer.start(15000)
        QTimer.singleShot(300, self._detect_gpu_async)

    def _detect_gpu_async(self, force: bool = False):
        """Run GPU detection in a background thread.

        Args:
            force: Re-detect even if detection already succeeded (used by the
                "Refresh GPU Info" button).
        """
        if not force and (self.gpu_info.backend != "cpu" or self.gpu_info.device_name != "Detecting..."):
            return
        if self._closing:
            return
        worker = _GpuDetectWorker()
        worker.detect_done.connect(self._on_gpu_detect_done)
        self._start_worker(worker)

    def _on_gpu_detect_done(self, info_text, gpu_info_dict):
        self.gpu_info_label.setText(info_text)
        if gpu_info_dict:
            self.gpu_info = GPUInfo(**gpu_info_dict)
            if gpu_info_dict.get('backend') == 'cuda' and getattr(self, '_gpu_stats_timer', None) is None:
                self._gpu_stats_timer = QTimer(self)
                self._gpu_stats_timer.timeout.connect(self._gpu_poll_stats)
                self._gpu_stats_timer.start(30000)
        self._update_resource_estimate()

    def _gpu_poll_stats(self):
        if self._closing:
            return
        worker = _GpuStatsWorker()
        worker.stats_done.connect(self._on_gpu_stats_done)
        self._start_worker(worker)

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

    def load_preset(self, preset_name):
        """Load a preset configuration into the live config."""
        if not preset_name:
            return
        try:
            for combo in (getattr(self, 'preset_combo', None),
                          getattr(self, 'adv_preset_combo', None)):
                if combo is not None and combo.currentText() != preset_name:
                    combo.blockSignals(True)
                    combo.setCurrentText(preset_name)
                    combo.blockSignals(False)

            preset_config = PresetManager.get_preset(preset_name)
            self.controller.set_config(preset_config)

            # Server selection glue
            target = self.config.target_server
            if target and target in self.server_manager.get_all_names():
                self._sync_server_combos(target)
                self._reload_server_models()

            self.update_command_preview()
        except Exception as e:
            logger.exception("Failed to load preset '%s'", preset_name)
            if hasattr(self, 'command_text'):
                self.command_text.setText(f"Preset Load Error: {e}")
            QMessageBox.critical(self, "Preset Error", f"Failed to load preset '{preset_name}': {e}")

    def update_command_preview(self):
        """Update the command preview based on current config."""
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
        except Exception:
            logger.exception("Failed to update resource estimate")
            
    def launch_server(self):
        """Launch llama-server."""
        try:
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

    def on_context_editing_finished(self):
        """Snap a hand-typed context size to the nearest binary size.

        setValue() emits valueChanged, which propagates to the advanced tab and
        the command preview via the existing signal connections.
        """
        value = self.ctx_spin.value()
        rounded = self.ctx_spin.round_to_binary_context(value)
        if rounded != value:
            self.ctx_spin.setValue(rounded)

    def _set_ctx_from_model(self, training_ctx: int):
        """Set context size from the model's training context."""
        self.controller.set("context_size", training_ctx)

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

        # Save as custom preset (overwrites if exists)
        PresetManager.save_custom_preset(preset_name, self.config)
        QMessageBox.information(self, "Success", f"Preset '{preset_name}' saved successfully.")

    def parse_launch_command(self, command_str):
        """Parse a llama-server command string into (params, unrecognized_flags).

        Delegates to :func:`llama_launcher.core.parser.parse_launch_command`.
        """
        return _parse_launch_command(command_str)

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
        self.server_management_widget.set_command_provider(self._build_launch_command)
        layout.addWidget(self.server_management_widget)

        self.tab_widget.addTab(tab, "Server Management")

    def _build_launch_command(self) -> str:
        """Build the llama-server command for the current configuration.

        Used by the Server Management tab so Start/Restart runs the same
        command shown in the Command Preview.
        """
        return build_command(self.config)

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

            # Build config from all parsed parameters that map to config fields
            migrated = migrate_config_dict(params)
            valid_fields = {f.name for f in dataclasses.fields(LlamaConfig)}
            filtered = {k: v for k, v in migrated.items() if k in valid_fields}
            dropped = sorted(set(migrated) - valid_fields)
            if dropped:
                logger.warning("Import: ignoring unknown parameters: %s", ", ".join(dropped))

            config = LlamaConfig(
                **filtered,
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
