"""PySide6 GUI Main Window: shell + wiring between tabs and services."""
import logging
import shlex
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QSettings, QThread, QTimer
from PySide6.QtWidgets import (
    QInputDialog, QMainWindow, QMessageBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)
from PySide6.QtGui import QFont

from llama_launcher.core.builder import build_command
from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.presets import PresetManager
from llama_launcher.core.validator import validate_config
from llama_launcher.estimator import ResourceEstimator
from llama_launcher.gpu.types import GPUInfo
from llama_launcher.gui.advanced_tab import AdvancedConfigTab
from llama_launcher.gui.config_controller import ConfigController
from llama_launcher.gui.dialogs.preset_dialog import PresetManagerDialog
from llama_launcher.gui.resource_usage_panel import ResourceUsagePanel
from llama_launcher.gui.server_management import ServerManagementWidget
from llama_launcher.gui.tabs.basic_tab import BasicTab
from llama_launcher.gui.tabs.gpu_tab import GpuTab
from llama_launcher.gui.tabs.import_tab import ImportTab
from llama_launcher.gui.tabs.models_tab import ModelsTab
from llama_launcher.gui.tabs.preview_tab import PreviewTab
from llama_launcher.gui import theme
from llama_launcher.gui.workers import (
    EstimateWorker, GpuDetectWorker, GpuStatsWorker, ModelScanWorker, PresetStyleWorker,
)
from llama_launcher.models.storage import (
    load_models as load_cached_models,
    save_models as save_cached_models,
)
from llama_launcher.remote.manager import ServerManager
from llama_launcher.remote.types import ConnectionState, ConnectionType, ServerConfig

logger = logging.getLogger(__name__)


class LlamaLauncherGUI(QMainWindow):
    """Main GUI window: assembles tabs and wires cross-tab interactions."""

    def __init__(self):
        super().__init__()
        self.config = LlamaConfig()
        self.controller = ConfigController(self.config, self)
        self.models = []
        self.server_manager = ServerManager()
        self.gpu_info: GPUInfo = GPUInfo(backend="cpu", device_name="Detecting...")
        self.resource_estimator = ResourceEstimator()
        self.server_process = None
        self._ctx_auto_set = True  # Allow auto-set on first model load
        self._active_workers = []
        self._closing = False
        self._settings = QSettings()
        self._ensure_local_server()
        self.init_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _ensure_local_server(self):
        """Ensure a default local server exists with a resolvable binary."""
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
                llama_server_path=self._find_local_llama_server(),
                working_directory="~",
                enabled=True,
            )
            self.server_manager.add_server(local_config)
            return

        # Repair an existing local config whose binary is not resolvable.
        for c in configs:
            if c.friendly_name == "Local Machine" and c.connection_type == ConnectionType.LOCAL:
                binary = c.llama_server_path or "llama-server"
                resolved = shutil.which(binary) or (
                    binary if Path(binary).is_file() else None
                )
                if resolved is None:
                    found = self._find_local_llama_server()
                    if found != binary:
                        c.llama_server_path = found
                        self.server_manager.add_server(c)  # persists update
                break

    @staticmethod
    def _find_local_llama_server() -> str:
        """Locate a llama-server binary: PATH first, then common build locations."""
        found = shutil.which("llama-server")
        if found:
            return found
        for candidate in (
            "/workspace/llama.cpp/build/bin/llama-server",
            "/usr/local/bin/llama-server",
            str(Path.home() / "llama.cpp/build/bin/llama-server"),
        ):
            if Path(candidate).is_file():
                return candidate
        return "llama-server"

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("llama-launcher")
        # Guarantee standard title-bar buttons on every platform/WM.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(1100, 750)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Menu bar with a guaranteed way out (Ctrl+Q), independent of
        # window-manager decorations.
        file_menu = self.menuBar().addMenu("&File")
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        self.tab_widget = QTabWidget()

        # Vertical splitter: tabs on top, resizable resource panel at bottom
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tab_widget)
        self.resource_panel = ResourceUsagePanel()
        self.resource_panel.setMinimumHeight(110)
        splitter.addWidget(self.resource_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([580, 170])
        main_layout.addWidget(splitter)

        self._create_tabs()
        self._wire_signals()

        # Push initial config values into all bound widgets
        self.controller.sync_ui()

        # Restore persisted window/preset state
        self._restore_state()

        # Defer blocking init tasks to after the UI is shown
        QTimer.singleShot(200, self._deferred_init)

    def _create_tabs(self):
        self.basic_tab = BasicTab(self.controller)
        self.basic_tab.preset_combo.addItems(PresetManager.get_all_presets())
        self._refresh_server_combo()
        self.tab_widget.addTab(self.basic_tab, "Basic Config")

        self.advanced_tab = AdvancedConfigTab(
            self.controller,
            preset_names=PresetManager.get_all_presets(),
            on_preset_selected=self.load_preset,
            on_save_preset=self.save_current_preset,
            on_manage_presets=self.open_preset_manager,
        )
        self.tab_widget.addTab(self.advanced_tab, "Advanced Config")

        self.gpu_tab = GpuTab()
        self.tab_widget.addTab(self.gpu_tab, "GPU Info")

        self.models_tab = ModelsTab()
        self._refresh_browser_server_combo()
        self.tab_widget.addTab(self.models_tab, "Model Browser")

        self.preview_tab = PreviewTab()
        self.tab_widget.addTab(self.preview_tab, "Command Preview & Launch")

        self.server_management_widget = ServerManagementWidget(self.server_manager, self)
        self.server_management_widget.list_widget.servers_changed.connect(
            self._on_servers_changed)
        self.server_management_widget.set_command_provider(self._build_launch_command)
        self.tab_widget.addTab(self.server_management_widget, "Server Management")

        self.import_tab = ImportTab()
        self.tab_widget.addTab(self.import_tab, "Import Launch Command")

    def _wire_signals(self):
        # Basic tab
        self.basic_tab.preset_combo.currentTextChanged.connect(self.load_preset)
        self.basic_tab.save_preset_btn.clicked.connect(self.save_current_preset)
        self.basic_tab.manage_presets_btn.clicked.connect(self.open_preset_manager)
        self.basic_tab.server_combo.currentTextChanged.connect(self.on_server_changed)
        self.basic_tab.ctx_spin.valueChanged.connect(
            lambda *_: setattr(self, "_ctx_auto_set", False))

        # Models tab
        self.models_tab.scan_requested.connect(self.scan_models)
        self.models_tab.model_selected.connect(self._on_model_selected)
        self.models_tab.browser_server_changed.connect(self.on_browser_server_changed)

        # GPU tab
        self.gpu_tab.refresh_requested.connect(lambda: self._detect_gpu_async(force=True))

        # Preview tab
        self.preview_tab.launch_requested.connect(self.launch_server)
        self.preview_tab.stop_requested.connect(self.stop_server)

        # Import tab
        self.import_tab.preset_created.connect(self.refresh_preset_list_if_needed)

        # Command preview (debounced) on any config change
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(250)
        self._preview_debounce.timeout.connect(self.update_command_preview)
        self.controller.field_changed.connect(self._schedule_preview_update)
        self.update_command_preview()

    def _restore_state(self):
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        tab_index = self._settings.value("tab_index", 0, type=int)
        if 0 <= tab_index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(tab_index)

        presets = PresetManager.get_all_presets()
        last = self._settings.value("last_preset", "")
        chosen = last if last in presets else (presets[0] if presets else None)
        if chosen:
            QTimer.singleShot(100, lambda: self.load_preset(chosen))

    def _save_state(self):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("tab_index", self.tab_widget.currentIndex())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Persist state, then clean up timers, workers, and connections."""
        self._closing = True
        self._save_state()
        # Stop a locally launched llama-server first so it can't hold us open.
        self.stop_server()
        for timer_name in ("_preset_style_timer", "_gpu_stats_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        mgmt = getattr(self, "server_management_widget", None)
        if mgmt is not None:
            mgmt.shutdown()
        for w in list(self._active_workers):
            if w.isRunning():
                w.wait(1000)
        self.server_manager.disconnect_all()
        event.accept()

    def _start_worker(self, worker: QThread) -> None:
        """Start a worker with proper lifecycle (self-delete + tracking)."""
        worker.setParent(self)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda: self._prune_worker(worker))
        self._active_workers.append(worker)
        worker.start()

    def _prune_worker(self, worker: QThread) -> None:
        try:
            self._active_workers.remove(worker)
        except ValueError:
            pass

    def _deferred_init(self):
        """Run blocking init tasks after UI is shown."""
        if self._closing:
            return
        self.scan_models()
        self._preset_style_timer = QTimer(self)
        self._preset_style_timer.timeout.connect(self._refresh_preset_combo_styles)
        self._preset_style_timer.start(15000)
        QTimer.singleShot(300, self._detect_gpu_async)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def load_preset(self, preset_name):
        """Load a preset configuration into the live config."""
        if not preset_name or self._closing:
            return
        try:
            for combo in (self.basic_tab.preset_combo, self.advanced_tab.preset_combo):
                if combo.currentText() != preset_name:
                    combo.blockSignals(True)
                    combo.setCurrentText(preset_name)
                    combo.blockSignals(False)

            preset_config = PresetManager.get_preset(preset_name)
            self.controller.set_config(preset_config)

            target = self.config.target_server
            if target and target in self.server_manager.get_all_names():
                self._sync_server_combos(target)
                self._reload_server_models()

            self._settings.setValue("last_preset", preset_name)
            self.update_command_preview()
        except Exception as e:
            logger.exception("Failed to load preset '%s'", preset_name)
            QMessageBox.critical(
                self, "Preset Error", f"Failed to load preset '{preset_name}': {e}")

    def save_current_preset(self):
        """Save the live config as a preset (prompts for a name, Save-As style)."""
        current = (
            self.advanced_tab.preset_combo.currentText()
            if self.tab_widget.currentWidget() is self.advanced_tab
            else self.basic_tab.preset_combo.currentText()
        )
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:", text=current)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in PresetManager.BUILTIN_PRESETS:
            reply = QMessageBox.question(
                self, "Override Built-in Preset",
                f"'{name}' is a built-in preset. Save a custom override?\n"
                f"(The built-in can be restored by deleting the override.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        PresetManager.save_custom_preset(name, self.config)
        self.refresh_preset_list_if_needed()
        for combo in (self.basic_tab.preset_combo, self.advanced_tab.preset_combo):
            combo.blockSignals(True)
            combo.setCurrentText(name)
            combo.blockSignals(False)
        self._settings.setValue("last_preset", name)
        QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully.")

    def open_preset_manager(self):
        """Open preset manager dialog for CRUD operations."""
        dialog = PresetManagerDialog(lambda: self.config, self)
        dialog.exec()
        # Refresh regardless of how the dialog was closed (X included).
        self.refresh_preset_list_if_needed()

    def refresh_preset_list_if_needed(self):
        """Refresh preset combo contents after preset CRUD."""
        presets = PresetManager.get_all_presets()
        for combo in (self.basic_tab.preset_combo, self.advanced_tab.preset_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(presets)
            if current in presets:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _refresh_preset_combo_styles(self):
        if self._closing:
            return
        preset_names = [
            self.basic_tab.preset_combo.itemText(i)
            for i in range(self.basic_tab.preset_combo.count())
        ]
        worker = PresetStyleWorker(preset_names, self.server_manager.get_all_names(),
                                   self.server_manager)
        worker.style_done.connect(self._on_preset_style_done)
        self._start_worker(worker)

    def _on_preset_style_done(self, results):
        offline_set = {name for name, offline in results if offline}
        for combo in (self.basic_tab.preset_combo, self.advanced_tab.preset_combo):
            for i in range(combo.count()):
                name = combo.itemText(i)
                offline = name in offline_set
                font = QFont()
                if offline:
                    font.setItalic(True)
                    combo.setItemData(i, theme.FG_MUTED, Qt.ItemDataRole.ForegroundRole)
                else:
                    combo.setItemData(i, None, Qt.ItemDataRole.ForegroundRole)
                combo.setFont(font)

    # ------------------------------------------------------------------
    # Server selection glue
    # ------------------------------------------------------------------

    def _refresh_server_combo(self):
        combo = self.basic_tab.server_combo
        names = self.server_manager.get_all_names()
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name)
        if current and current in names:
            combo.setCurrentText(current)
        elif names:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _refresh_browser_server_combo(self):
        combo = self.models_tab.browser_server_combo
        names = self.server_manager.get_all_names()
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name)
        if current and current in names:
            combo.setCurrentText(current)
        elif names:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _sync_server_combos(self, source_text: str):
        for combo in (self.basic_tab.server_combo, self.models_tab.browser_server_combo):
            if combo.currentText() != source_text:
                combo.blockSignals(True)
                combo.setCurrentText(source_text)
                combo.blockSignals(False)

    def _get_current_server(self) -> str:
        text = self.basic_tab.server_combo.currentText()
        if text:
            return text
        names = self.server_manager.get_all_names()
        return names[0] if names else ""

    def _get_browser_server(self) -> str:
        return self.models_tab.browser_server_combo.currentText() or self._get_current_server()

    def on_server_changed(self, server_name: str):
        self.config.target_server = server_name or None
        self._sync_server_combos(server_name)
        self._on_server_selected(server_name)

    def on_browser_server_changed(self, server_name: str):
        self.config.target_server = server_name or None
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
                self.basic_tab.host_input.setText(cfg.hostname)
            if cfg.server_port:
                self.basic_tab.port_spin.setValue(cfg.server_port)
        self._reload_server_models()

    def _on_servers_changed(self):
        self._refresh_server_combo()
        self._refresh_browser_server_combo()
        current = self.config.target_server
        names = self.server_manager.get_all_names()
        if current and current not in names:
            self.config.target_server = names[0] if names else None
        self._refresh_preset_combo_styles()

    # ------------------------------------------------------------------
    # Model scanning
    # ------------------------------------------------------------------

    def scan_models(self):
        """Scan models in a background thread for the selected server."""
        if self._closing:
            return
        server_name = self._get_browser_server()
        dirs_to_scan = self.models_tab.get_folders()

        if server_name:
            for d in self.server_manager.get_server_model_directories(server_name):
                if d not in dirs_to_scan:
                    dirs_to_scan.append(d)

        if not dirs_to_scan:
            return

        existing = getattr(self, "_scan_worker", None)
        if existing is not None and existing.isRunning():
            logger.debug("Model scan already in progress; ignoring request")
            return

        is_local_only = not server_name
        self.models_tab.set_scanning(True)
        self._scan_worker = ModelScanWorker(
            self.server_manager, server_name, dirs_to_scan, is_local_only)
        self._scan_worker.scan_done.connect(self._on_scan_done)
        self._scan_worker.finished.connect(lambda: setattr(self, "_scan_worker", None))
        self._start_worker(self._scan_worker)

    def _on_scan_done(self, scan_server_name, models):
        self.models_tab.set_scanning(False)
        if scan_server_name != self._get_browser_server():
            return
        self.models = models
        if scan_server_name:
            save_cached_models(scan_server_name, models)
        self._populate_model_lists()

    def _populate_model_lists(self):
        self.models_tab.populate(self.models)

        # Refresh the model path combo without clobbering current text.
        combo = self.basic_tab.model_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for model in self.models:
            combo.addItem(model.file_path, model.file_path)
        combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_model_selected(self, file_path: str):
        """Model picked in the browser: set it as the active model path."""
        self.controller.set("model_path", file_path)

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

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _detect_gpu_async(self, force: bool = False):
        """Run GPU detection in a background thread."""
        if self._closing:
            return
        if not force and (self.gpu_info.backend != "cpu"
                          or self.gpu_info.device_name != "Detecting..."):
            return
        worker = GpuDetectWorker()
        worker.detect_done.connect(self._on_gpu_detect_done)
        self._start_worker(worker)

    def _on_gpu_detect_done(self, info_text, gpu_info_dict):
        self.gpu_tab.set_info_text(info_text)
        if gpu_info_dict:
            self.gpu_info = GPUInfo(**gpu_info_dict)
            if gpu_info_dict.get("backend") == "cuda" and getattr(
                    self, "_gpu_stats_timer", None) is None:
                self._gpu_stats_timer = QTimer(self)
                self._gpu_stats_timer.timeout.connect(self._gpu_poll_stats)
                self._gpu_stats_timer.start(30000)
        self._update_resource_estimate()

    def _gpu_poll_stats(self):
        if self._closing:
            return
        worker = GpuStatsWorker()
        worker.stats_done.connect(self._on_gpu_stats_done)
        self._start_worker(worker)

    def _on_gpu_stats_done(self, stats_text):
        if not stats_text:
            return
        existing = self.gpu_tab.info_text()
        idx = existing.find("\n--- Live Stats ---")
        existing = existing[:idx] + stats_text if idx >= 0 else existing + stats_text
        self.gpu_tab.set_info_text(existing)

    # ------------------------------------------------------------------
    # Command preview + resource estimate
    # ------------------------------------------------------------------

    def _schedule_preview_update(self, *_args):
        if hasattr(self, "_preview_debounce"):
            self._preview_debounce.start()

    def update_command_preview(self):
        """Rebuild and display the command from the live config."""
        try:
            validate_config(self.config)
            self.preview_tab.set_command(build_command(self.config))
            self._mark_model_invalid(False)
        except Exception as e:
            self.preview_tab.set_command(f"Validation Error: {e}")
            self._mark_model_invalid("model_path" in str(e))

        self._update_resource_estimate()

    def _mark_model_invalid(self, invalid: bool):
        """Toggle error styling on the model path combo."""
        combo = self.basic_tab.model_combo
        combo.setProperty("invalid", invalid)
        combo.style().unpolish(combo)
        combo.style().polish(combo)

    def _update_resource_estimate(self):
        """Schedule a debounced background estimate (never blocks the UI)."""
        if not hasattr(self, "_estimate_debounce"):
            self._estimate_debounce = QTimer(self)
            self._estimate_debounce.setSingleShot(True)
            self._estimate_debounce.setInterval(300)
            self._estimate_debounce.timeout.connect(self._start_estimate_worker)
        self._estimate_debounce.start()

    def _start_estimate_worker(self):
        if self._closing:
            return
        existing = getattr(self, "_estimate_worker", None)
        if existing is not None and existing.isRunning():
            self._estimate_rerun = True  # rerun with latest config when done
            return
        import dataclasses

        snapshot = dataclasses.replace(self.config)
        worker = EstimateWorker(
            self.resource_estimator, snapshot, self.gpu_info,
            self.config.model_path, self.config.draft_model_path,
            self.config.mmproj_model_path,
        )
        worker.estimate_done.connect(self._on_estimate_done)
        worker.finished.connect(lambda: setattr(self, "_estimate_worker", None))
        self._estimate_worker = worker
        self._start_worker(worker)

    def _on_estimate_done(self, estimate):
        self.resource_panel.update_estimate(estimate,
                                            selected_ctx=self.config.context_size)

        # Auto-set context default from training context on first model load
        if (estimate is not None
                and self._ctx_auto_set
                and estimate.training_context_length > 0
                and self.config.context_size < estimate.training_context_length):
            self.controller.set("context_size", estimate.training_context_length)
            self._ctx_auto_set = False

        if getattr(self, "_estimate_rerun", False):
            self._estimate_rerun = False
            self._update_resource_estimate()

    # ------------------------------------------------------------------
    # Local launch
    # ------------------------------------------------------------------

    def launch_server(self):
        """Launch llama-server locally with the current configuration."""
        try:
            validate_config(self.config)
            cmd = self._resolve_server_binary(build_command(self.config))
        except Exception as e:
            self._mark_model_invalid("model_path" in str(e))
            QMessageBox.warning(self, "Cannot Launch", str(e))
            return
        self._mark_model_invalid(False)

        if self.server_process is not None and self.server_process.state() == (
                QProcess.ProcessState.Running):
            logger.info("llama-server already running; launch ignored")
            return

        self.preview_tab.clear_log()
        self.preview_tab.append_log(f"Launching: {cmd}\n\n")

        self.server_process = QProcess(self)
        self.server_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.server_process.readyRead.connect(self.on_server_output)
        self.server_process.finished.connect(self.on_server_finished)
        self.server_process.errorOccurred.connect(self.on_server_error)

        try:
            args = shlex.split(cmd)
            self.server_process.start(args[0], args[1:])
            if not self.server_process.waitForStarted(3000):
                raise RuntimeError(
                    f"Could not start '{args[0]}'.\n\n"
                    f"Make sure the llama-server binary exists and is executable. "
                    f"Set the correct path in Server Management > Edit Server > Server Exec."
                )
            self.preview_tab.set_running(True)
        except Exception as e:
            # Non-modal: show in the log pane so nothing can trap the window.
            logger.exception("Launch failed")
            self.preview_tab.append_log(f"\nLaunch Error: {e}\n")

    def on_server_error(self, error):
        """Surface QProcess errors (e.g. binary not found) in the log."""
        proc = self.server_process
        detail = proc.errorString() if proc is not None else str(error)
        logger.error("llama-server process error: %s", detail)
        self.preview_tab.append_log(f"\nProcess Error: {detail}\n")

    def _resolve_server_binary(self, cmd: str) -> str:
        """Substitute argv[0] with the selected server's configured binary path."""
        server = self._get_current_server()
        conn = self.server_manager.get_connection(server) if server else None
        binary = conn.config.llama_server_path if conn else ""
        if binary and binary != "llama-server":
            parts = cmd.split(" ", 1)
            return binary + (" " + parts[1] if len(parts) > 1 else "")
        return cmd

    def stop_server(self):
        """Stop the locally launched llama-server process."""
        proc = self.server_process
        if proc is not None and proc.state() == QProcess.ProcessState.Running:
            proc.terminate()
            if not proc.waitForFinished(3000):
                proc.kill()

    def on_server_output(self):
        """Handle server output."""
        proc = self.server_process
        if proc is not None and proc.state() == QProcess.ProcessState.Running:
            output = bytes(proc.readAll()).decode("utf-8", errors="ignore")
            if output:
                self.preview_tab.append_log(output)

    def on_server_finished(self, exit_code, exit_status=None):
        """Handle server process finished."""
        self.preview_tab.append_log(f"\n--- Server process finished (exit code {exit_code}) ---\n")
        self.preview_tab.set_running(False)

    def _build_launch_command(self) -> str:
        """Build the llama-server command for the current configuration.

        Used by the Server Management tab so Start/Restart runs the same
        command shown in the Command Preview.
        """
        return build_command(self.config)


def run_gui():
    """Run the GUI application (convenience for ``python -m`` style launches)."""
    import sys
    from PySide6.QtWidgets import QApplication

    from llama_launcher.gui.theme import apply_dark_theme

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = LlamaLauncherGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
