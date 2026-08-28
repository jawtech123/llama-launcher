"""PySide6 GUI Main Window: shell + wiring between tabs and services."""
import dataclasses
import logging
import shutil
import time
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QThread, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QInputDialog, QMainWindow, QMessageBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from llama_launcher import __version__
from llama_launcher.core.builder import build_command
from llama_launcher.utils.helpers import format_age, get_icons_dir

from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.presets import PresetManager
from llama_launcher.core.validator import validate_config
from llama_launcher.benchmark import config_fingerprint, latest_session_for_preset
from llama_launcher.estimator import ResourceEstimator
from llama_launcher.gpu.types import GPUInfo
from llama_launcher.gui.advanced_tab import AdvancedConfigTab
from llama_launcher.gui.config_controller import ConfigController
from llama_launcher.gui.dialogs.preset_dialog import PresetManagerDialog
from llama_launcher.gui.resource_usage_panel import ResourceUsagePanel
from llama_launcher.gui.server_management import ServerManagementWidget
from llama_launcher.gui.dialogs.path_picker import (
    RemotePathDialog, posix_parent, wait_for_active_workers,
)
from llama_launcher.gui.tabs.basic_tab import BasicTab
from llama_launcher.gui.tabs.benchmark_tab import BenchmarkTab
from llama_launcher.gui.tabs.gpu_tab import GpuTab
from llama_launcher.gui.tabs.import_tab import ImportTab
from llama_launcher.gui.tabs.models_tab import ModelsTab
from llama_launcher.gui.tabs.preview_tab import PreviewTab
from llama_launcher.gui import theme
from llama_launcher.gui.widgets import format_gpu_info
from llama_launcher.remote.host_info import (
    load_profile as load_host_profile, save_profile as save_host_profile,
)
from llama_launcher.gui.workers import (
    EstimateWorker, HostConnectWorker, ModelScanWorker, PresetStyleWorker,
    TaskWorker, UsageSampleWorker,
)
from llama_launcher.models.scanner import is_mmproj_file
from llama_launcher.models.storage import (
    load_models as load_cached_models,
    save_models as save_cached_models,
)
from llama_launcher.remote.console import LogFollowWorker
from llama_launcher.remote.manager import ServerManager
from llama_launcher.remote.types import (
    LOCAL_SERVER_NAME, ConnectionState, ServerConfig, ServerProcessState,
)

logger = logging.getLogger(__name__)

#: Live-usage poll intervals. A remote sample costs an SSH round trip plus
#: nvidia-smi on that host, so it is paced more gently than our own machine.
LOCAL_STATS_INTERVAL_MS = 2000
REMOTE_STATS_INTERVAL_MS = 5000


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
        #: Live log stream mirroring whatever the Preview tab's selected
        #: server is running -- attached whether launched from here or from
        #: Server Management.
        self._preview_log_worker = None
        #: True only when Preview itself started the currently-attached
        #: session (gates whether a benchmark session gets recorded, since
        #: an externally-started server's exact config isn't known here).
        self._preview_launched_by_us = False
        #: Server name the Preview tab's log/status is currently mirroring.
        self._preview_watch_server = None
        #: Grace window after our own launch during which the dedicated
        #: startup-failure check (not the generic reconciliation poll) owns
        #: reporting an immediate crash -- avoids both racing to detach first.
        self._preview_launch_started_at = 0.0
        self._ctx_auto_set = True  # Allow auto-set on first model load
        self._active_workers = []
        self._closing = False
        #: Last connection failure per server, shown on the server bars.
        self._connect_errors: dict = {}
        #: Total RAM of the selected server, from its host profile.
        self._host_ram_mb: float = -1.0
        self._last_usage_sample = None
        self._last_estimate = None
        self._settings = QSettings()
        self._ensure_local_server()
        self.init_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _ensure_local_server(self):
        """Ensure the permanent local server entry exists with a usable binary.

        Loads the persisted servers *first*: without that, the in-memory config
        list is always empty on a fresh process and we would overwrite the saved
        Local Machine entry (working dir, model dirs, port) on every launch.
        """
        self.server_manager.load_saved_servers()

        configs = self.server_manager.get_all_configs()
        has_local = any(c.friendly_name == LOCAL_SERVER_NAME for c in configs)
        if not has_local:
            local_config = ServerConfig(
                friendly_name=LOCAL_SERVER_NAME,
                llama_server_path=self._find_local_llama_server(),
                working_directory="~",
                enabled=True,
            )
            self.server_manager.add_server(local_config)

        # Repair an existing local config whose binary is not resolvable.
        for c in self.server_manager.get_all_configs():
            if c.friendly_name == LOCAL_SERVER_NAME:
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
        self.setWindowTitle(f"llama-launcher v{__version__} — Created by Jay Walker")

        # Guarantee standard title-bar buttons on every platform/WM.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )

        icon_dir = get_icons_dir()
        icon = QIcon()
        from PySide6.QtCore import QSize
        for size in [16, 32, 48, 64, 128, 256]:
            icon_path = icon_dir / f"ll-{size}.png"
            if icon_path.is_file():
                icon.addFile(str(icon_path), QSize(size, size))
        if not icon.isNull():
            self.setWindowIcon(icon)

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

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("&About llama-launcher")
        about_action.triggered.connect(self._show_about)

        self.tab_widget = QTabWidget()

        # Vertical splitter: tabs on top, resizable resource panel at bottom
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tab_widget)
        self.resource_panel = ResourceUsagePanel()
        self.resource_panel.setMinimumHeight(150)
        splitter.addWidget(self.resource_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([560, 190])
        main_layout.addWidget(splitter)

        self._create_tabs()
        self._wire_signals()

        # Push initial config values into all bound widgets
        self.controller.sync_ui()

        # Restore persisted window/preset state
        self._restore_state()

        # Defer blocking init tasks to after the UI is shown
        QTimer.singleShot(200, self._deferred_init)

    def _show_about(self):
        QMessageBox.about(
            self, "About llama-launcher",
            f"<b>llama-launcher</b> v{__version__}<br><br>"
            "Generates and manages llama-server launch configurations.<br><br>"
            "Created by Jay Walker.<br><br>"
            "© 2026 Jay Walker. All rights reserved.<br>"
            "Built with PySide6 (Qt), licensed under LGPLv3 — "
            "see THIRD-PARTY-NOTICES.md.")

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

        self.preview_tab = PreviewTab()
        # "&&" escapes the Qt mnemonic marker so a literal '&' is shown.
        self.tab_widget.addTab(self.preview_tab, "Preview && Launch")

        self.server_management_widget = ServerManagementWidget(self.server_manager, self)
        self.server_management_widget.list_widget.servers_changed.connect(
            self._on_servers_changed)
        self.server_management_widget.set_command_provider(self._build_launch_command)
        self.tab_widget.addTab(self.server_management_widget, "Server Management")

        self.models_tab = ModelsTab()
        self._refresh_browser_server_combo()
        self.tab_widget.addTab(self.models_tab, "Model Browser")

        self.gpu_tab = GpuTab()
        self.tab_widget.addTab(self.gpu_tab, "GPU Info")

        self.benchmark_tab = BenchmarkTab(self.server_manager, self._get_current_server)
        self.tab_widget.addTab(self.benchmark_tab, "Benchmark")

        self.import_tab = ImportTab()
        self.tab_widget.addTab(self.import_tab, "Import Launch Command")

        # preview_tab's ServerBar didn't exist for the earlier refreshes above,
        # so it never got the server list -- its combo would stay empty,
        # `current_server()` would stay '', and every live-stats poll would
        # read that as "selection changed" and tear down Preview's Stop state.
        self._refresh_server_combo()

    def _wire_signals(self):
        # Basic tab
        # Lets the model Browse buttons target the selected server's filesystem.
        self.basic_tab.set_browse_context(self.server_manager, self._get_current_server)
        self.basic_tab.preset_combo.currentTextChanged.connect(self.load_preset)
        self.basic_tab.save_preset_btn.clicked.connect(self.save_current_preset)
        self.basic_tab.manage_presets_btn.clicked.connect(self.open_preset_manager)
        self.basic_tab.benchmark_info_btn.clicked.connect(
            self._show_preset_benchmark_info)
        # Every ServerBar drives the same selection and the same connect action.
        for bar in self._server_bars():
            bar.server_changed.connect(self.on_server_changed)
            bar.connect_requested.connect(self._on_connect_requested)
            bar.disconnect_requested.connect(self._on_disconnect_requested)
        # Server Management routes through the same flow, so both agree on what
        # is connected regardless of where the button was pressed.
        self.server_management_widget.connect_requested.connect(
            self._on_connect_requested)
        self.server_management_widget.disconnect_requested.connect(
            self._on_disconnect_requested)
        # Selecting a row there is just as much "picking a server" as using a
        # ServerBar's combo -- without this, a server connected/scanned from
        # Server Management stays invisible in the model dropdowns until the
        # user re-selects it elsewhere.
        self.server_management_widget.server_selected.connect(
            self.on_server_changed)
        self.basic_tab.ctx_spin.valueChanged.connect(
            lambda *_: setattr(self, "_ctx_auto_set", False))

        # Models tab
        self.models_tab.scan_requested.connect(self.scan_models)
        self.models_tab.model_selected.connect(self._on_model_selected)
        # Server changes arrive via the shared ServerBar wiring above, so
        # browser_server_changed is not connected again here.

        # GPU tab: refresh re-reads the selected server, not always this machine.
        self.gpu_tab.refresh_requested.connect(self._refresh_selected_host)

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
        self.controller.field_changed.connect(self._maybe_auto_select_mmproj)
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
        # Stop a Preview-launched llama-server first so it can't hold us open.
        # Done synchronously (not via the async stop_server()/TaskWorker path
        # used for the Stop button) because the Qt event loop that would
        # deliver the worker's completion signal isn't guaranteed to keep
        # pumping during shutdown.
        self._shutdown_preview_server()
        for timer_name in ("_preset_style_timer", "_live_stats_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        # Close connections BEFORE waiting on workers so threads that are
        # blocked in I/O calls error out and can be cleaned up.
        self.server_manager.disconnect_all()
        mgmt = getattr(self, "server_management_widget", None)
        if mgmt is not None:
            mgmt.shutdown()
        for w in list(self._active_workers):
            if w.isRunning():
                w.wait(1500)
        # Listing/probe threads are not parented to any widget (a dialog can be
        # dismissed while one is in flight), so they are drained explicitly.
        wait_for_active_workers()
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
        self._preset_style_timer = QTimer(self)
        self._preset_style_timer.timeout.connect(self._refresh_preset_combo_styles)
        self._preset_style_timer.start(15000)
        self._live_stats_timer = QTimer(self)
        self._live_stats_timer.timeout.connect(self._poll_live_stats)
        self._live_stats_timer.start(LOCAL_STATS_INTERVAL_MS)
        # The local entry involves no network, so connecting it is free and
        # nothing would be listed until it happened. Remote servers stay
        # untouched until the user presses Connect.
        QTimer.singleShot(100, lambda: self._on_connect_requested(LOCAL_SERVER_NAME))

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
    # Preset benchmark info
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_benchmark_lines(session) -> list:
        from datetime import datetime
        started = session.get("started", 0)
        when = (datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M")
                if started else "unknown")
        prompt_ms = float(session.get("prompt_ms", 0))
        gen_ms = float(session.get("gen_ms", 0))
        p_tok = int(session.get("prompt_tokens", 0))
        g_tok = int(session.get("gen_tokens", 0))
        speed_in = p_tok / (prompt_ms / 1000.0) if prompt_ms > 0 else 0.0
        speed_out = g_tok / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
        lines = [
            f"Recorded:  {when}",
            f"Requests:  {int(session.get('requests', 0))}",
            f"Speed in:  {speed_in:.1f} tok/s   ({p_tok:,} tokens)",
            f"Speed out: {speed_out:.1f} tok/s   ({g_tok:,} tokens)",
        ]
        ttft_p50 = session.get("ttft_ms_p50", 0)
        if ttft_p50:
            lines.append(
                f"TTFT:      avg {session.get('ttft_ms_avg', 0):.0f} ms   "
                f"p50 {ttft_p50:.0f} ms   p95 {session.get('ttft_ms_p95', 0):.0f} ms")
        return lines

    def _show_preset_benchmark_info(self):
        """Show the benchmark recorded for the current preset.

        The info is only valid for the exact parameters it was measured
        with: if any parameter changed since the recording (fingerprint
        mismatch), the info is reported as reset.
        """
        preset = self.basic_tab.preset_combo.currentText().strip()
        if not preset:
            return
        session = latest_session_for_preset(preset)
        if session is None:
            QMessageBox.information(
                self, "Preset Benchmark",
                f"No benchmark data recorded for preset '{preset}' yet.\n\n"
                f"Launch a server with this preset and run some requests — "
                f"the throughput stats will appear here and in the Benchmark tab.")
            return

        lines = self._fmt_benchmark_lines(session)
        fp_now = config_fingerprint(self.config)
        recorded_fp = session.get("fingerprint") or ""
        if recorded_fp != fp_now:
            QMessageBox.information(
                self, "Preset Benchmark",
                f"Benchmark info for '{preset}' was RESET because the "
                f"parameters changed since it was recorded.\n\n"
                f"Last recorded values (stale):\n" + "\n".join(lines) +
                "\n\nLaunch the server to benchmark the current parameters.")
            return

        QMessageBox.information(
            self, "Preset Benchmark",
            f"Benchmark for preset '{preset}' (current parameters):\n\n"
            + "\n".join(lines)
            + "\n\nFull history is in the Benchmark tab.")

    # ------------------------------------------------------------------
    # Server selection glue
    # ------------------------------------------------------------------

    def _server_bars(self):
        """Every ServerBar in the app; they all show the same selection.

        Tolerates tabs that do not exist yet: server refreshes happen while the
        UI is still being assembled.
        """
        bars = []
        for tab_name in ("basic_tab", "models_tab", "gpu_tab", "preview_tab"):
            tab = getattr(self, tab_name, None)
            bar = getattr(tab, "server_bar", None) if tab is not None else None
            if bar is not None:
                bars.append(bar)
        return bars

    def _refresh_server_combo(self):
        names = self.server_manager.get_all_names()
        current = self._get_current_server()
        for bar in self._server_bars():
            bar.set_servers(names, current)
        self._refresh_server_state_display()

    def _refresh_browser_server_combo(self):
        # Both selectors are driven together now; kept for the existing callers.
        self._refresh_server_combo()

    def _sync_server_combos(self, source_text: str):
        for bar in self._server_bars():
            bar.set_current_server(source_text)
        self._refresh_server_state_display()

    def _refresh_server_state_display(self):
        """Push the selected server's connection state onto every bar.

        Reads cached state only -- nothing here may touch the network.
        """
        name = self._get_current_server()
        state = ConnectionState.DISCONNECTED
        detail = ""
        if name:
            state = self.server_manager.get_connection_state(name)
            detail = self._connect_errors.get(name, "")
        for bar in self._server_bars():
            bar.set_state(state, detail)

    def _get_current_server(self) -> str:
        text = self.basic_tab.server_combo.currentText()
        if text:
            return text
        names = self.server_manager.get_all_names()
        return names[0] if names else ""

    def _get_browser_server(self) -> str:
        return self.models_tab.browser_server_combo.currentText() or self._get_current_server()

    def on_server_changed(self, server_name: str):
        self._sync_server_combos(server_name)
        self._on_server_selected(server_name)

    def on_browser_server_changed(self, server_name: str):
        self._sync_server_combos(server_name)
        self._on_server_selected(server_name)

    def _on_server_selected(self, server_name: str):
        if not server_name:
            return
        conn = self.server_manager.get_connection(server_name)
        if conn:
            cfg = conn.config
            # Only the port carries over. --host is llama-server's *bind*
            # address on the machine it runs on; the SSH hostname is how we
            # reach that machine. Copying it across made the launch command bind
            # to a single interface (and it is meaningless to the remote host).
            if cfg.server_port:
                self.basic_tab.port_spin.setValue(cfg.server_port)
            # A loopback bind on a remote host is unreachable from here, so it
            # is never what was meant. Any other value is left as chosen.
            if (self.server_manager.is_remote(server_name)
                    and self.config.host in ("127.0.0.1", "localhost", "::1")):
                self.basic_tab.host_input.setText("0.0.0.0")
        self._refresh_server_state_display()
        # GPU tab, estimate and live bars follow the selection immediately,
        # from cache -- selecting a server never reaches out over the network.
        self._apply_host_profile(server_name)
        self._restart_live_stats_timer()
        self._reload_server_models()

    def _on_servers_changed(self):
        self._refresh_server_combo()
        self._refresh_preset_combo_styles()

    # ------------------------------------------------------------------
    # Connecting: the moment we learn about a host
    # ------------------------------------------------------------------

    def _on_connect_requested(self, server_name: str):
        """Connect, then read the host's GPU, RAM and models and cache them."""
        if not server_name or self._closing:
            return
        for bar in self._server_bars():
            bar.set_busy(True, f"Connecting to {server_name}…")

        worker = HostConnectWorker(self.server_manager, server_name)
        worker.connect_done.connect(self._on_connect_done)
        self._start_worker(worker)

    def _on_connect_done(self, server_name, ok, message, profile, models,
                         binary_status=None):
        for bar in self._server_bars():
            bar.set_busy(False)
        self._connect_errors[server_name] = "" if ok else message

        if not ok:
            self._refresh_server_state_display()
            # Re-enables Server Management's button too, whichever tab asked.
            self._notify_connection_changed(server_name)
            QMessageBox.warning(
                self, "Connection Failed",
                f"Could not connect to '{server_name}'.\n\n{message}")
            return

        if profile is not None:
            save_host_profile(server_name, profile)
        if models is not None:
            save_cached_models(server_name, models)
            if server_name == self._get_browser_server():
                self.models = models
                self._populate_model_lists()
                if not models:
                    self.models_tab.show_placeholder(
                        f"No models found in the directories configured for "
                        f"'{server_name}'.")

        self._refresh_server_state_display()
        # GPU, estimate and live bars all describe the selected server now.
        self._apply_host_profile(server_name)
        self._restart_live_stats_timer()
        self._notify_connection_changed(server_name)
        self._offer_binary_fix(server_name, binary_status)

    def _offer_binary_fix(self, server_name: str, binary_status):
        """Point out a llama-server path that does not exist on this host.

        Connecting is the first moment we can check, and a wrong path otherwise
        only surfaces as a failed launch with the reason buried in a log.
        """
        if not binary_status:
            return
        runnable, found = binary_status
        if runnable:
            return

        conn = self.server_manager.get_connection(server_name)
        if conn is None:
            return
        configured = conn.config.llama_server_path or "(not set)"

        if not found:
            QMessageBox.warning(
                self, "llama-server Not Found",
                f"'{configured}' is not runnable on '{server_name}', and no "
                f"llama-server was found in the usual build locations.\n\n"
                f"Set the correct path in Server Management > Edit Server > "
                f"Server Exec — its Browse button searches that machine.")
            return

        box = QMessageBox(self)
        box.setWindowTitle("llama-server Not Found")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"'{configured}' is not runnable on '{server_name}'.")
        box.setInformativeText(f"Found instead:\n{found}\n\nUse this path?")
        use_btn = box.addButton("Use This", QMessageBox.AcceptRole)
        browse_btn = box.addButton("Browse…", QMessageBox.ActionRole)
        box.addButton("Ignore", QMessageBox.RejectRole)
        box.exec()

        chosen = ""
        if box.clickedButton() is use_btn:
            chosen = found
        elif box.clickedButton() is browse_btn:
            chosen = RemotePathDialog.get_path(
                self, conn, mode="file", start_path=posix_parent(found),
                title=f"Select llama-server on {server_name}") or ""
        if chosen:
            conn.config.llama_server_path = chosen
            self.server_manager.add_server(conn.config)   # persists
            self._on_servers_changed()

    def _on_disconnect_requested(self, server_name: str):
        if not server_name:
            return
        self.server_manager.disconnect(server_name)
        self._connect_errors.pop(server_name, None)
        self._refresh_server_state_display()
        self._apply_host_profile(server_name)
        self._restart_live_stats_timer()
        self._notify_connection_changed(server_name)

    def _notify_connection_changed(self, server_name: str):
        """Tell Server Management a server's connection state changed."""
        mgmt = getattr(self, "server_management_widget", None)
        if mgmt is not None:
            mgmt.on_connection_changed(server_name)

    def _refresh_selected_host(self):
        """Re-read the selected server's hardware (the GPU tab's Refresh)."""
        name = self._get_current_server()
        if not name:
            return
        if self.server_manager.get_connection_state(name) != ConnectionState.CONNECTED:
            QMessageBox.information(
                self, "Not Connected",
                f"Connect to '{name}' first to read its GPU information.")
            return
        self._on_connect_requested(name)

    def _apply_host_profile(self, server_name: str):
        """Point the GPU tab and the estimator at this server's hardware."""
        if not server_name:
            return
        profile = load_host_profile(server_name)
        connected = (self.server_manager.get_connection_state(server_name)
                     == ConnectionState.CONNECTED)

        if profile is None:
            self.gpu_tab.set_staleness(
                f"'{server_name}' has not been connected yet — connect to read its GPU.")
            self.gpu_tab.set_info_text("No GPU information for this server.")
            self._host_ram_mb = -1.0
            return

        self.gpu_info = profile.gpu or GPUInfo(backend="cpu", device_name="Unknown")
        self._host_ram_mb = profile.total_ram_mb
        self.gpu_tab.set_info_text(format_gpu_info(self.gpu_info, profile.total_ram_mb))
        self.gpu_tab.set_staleness(
            "" if connected else
            f"Not connected — showing what '{server_name}' reported "
            f"{format_age(profile.age_seconds)} ago.")
        self._update_resource_estimate()

    # ------------------------------------------------------------------
    # Model scanning
    # ------------------------------------------------------------------

    def scan_models(self):
        """Scan models in a background thread for the selected server."""
        if self._closing:
            return
        server_name = self._get_browser_server()
        dirs_to_scan = self.models_tab.get_folders()

        is_remote = bool(server_name) and self.server_manager.is_remote(server_name)
        if server_name:
            server_dirs = self.server_manager.get_server_model_directories(server_name)
            if is_remote:
                # The Models-tab folder list describes this machine, so it says
                # nothing about what exists on a remote host.
                dirs_to_scan = [d for d in server_dirs if d]
            else:
                for d in server_dirs:
                    if d and d not in dirs_to_scan:
                        dirs_to_scan.append(d)

        if not dirs_to_scan:
            # Returning quietly left the previous server's models on screen,
            # which read as "the browser is still showing local models".
            self.models = []
            self._populate_model_lists()
            if is_remote:
                self.models_tab.show_placeholder(
                    f"No model directories configured for '{server_name}'. "
                    "Add them in Server Management > Edit Server > Model Dirs.")
            else:
                self.models_tab.show_placeholder(
                    "No model folders to scan. Add one with 'Add Folder...'.")
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
        # The main combo lists mmproj files too, since picking one there is
        # handled specially (see _maybe_auto_select_mmproj) -- it redirects
        # to the paired base model instead of trying to load it as one.
        # Draft excludes them: an mmproj is never a valid speculative-decoding
        # draft, and there's no equivalent redirect for that field.
        mmproj_models = [m for m in self.models if is_mmproj_file(m.file_name)]
        non_mmproj_models = [m for m in self.models if not is_mmproj_file(m.file_name)]
        self._refresh_model_combo(self.basic_tab.model_combo, self.models)
        self._refresh_model_combo(self.basic_tab.mmproj_combo, mmproj_models)
        self._refresh_model_combo(self.basic_tab.draft_combo, non_mmproj_models)

    @staticmethod
    def _refresh_model_combo(combo, models):
        """Refresh a model-path combo's dropdown items without clobbering current text."""
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for model in models:
            combo.addItem(model.file_path, model.file_path)
        combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_model_selected(self, file_path: str):
        """Model picked in the browser: set it as the active model path."""
        self.controller.set("model_path", file_path)

    def _maybe_auto_select_mmproj(self, field_name: str):
        """Keep mmproj in sync with whichever model is selected.

        Vision models are almost always distributed with their mmproj file
        in the same directory, so a same-directory match is auto-selected.
        Switching to a model with no such match clears mmproj too -- leaving
        a stale mmproj from a previous, unrelated model attached would
        otherwise launch it against the wrong model.

        Picking an mmproj file itself from the main model combo (it's listed
        there so it's easy to find) is redirected to the base model it
        belongs with, since an mmproj file can't be loaded as the model --
        that recursive ``set()`` re-enters this method with the real model
        path, which then runs the normal same-directory match below.
        """
        if field_name != "model_path":
            return
        model_path = self.config.model_path
        if model_path and is_mmproj_file(Path(model_path).name):
            model_dir = str(Path(model_path).parent)
            siblings = sorted(
                (m for m in self.models
                 if not is_mmproj_file(m.file_name) and str(Path(m.file_path).parent) == model_dir),
                key=lambda m: m.file_path,
            )
            if siblings:
                self.controller.set("model_path", siblings[0].file_path)
            else:
                # No base model alongside it to redirect to -- at least
                # register it as the mmproj so that part isn't lost.
                self.controller.set("mmproj_model_path", model_path)
            return

        candidates = []
        if model_path:
            model_dir = str(Path(model_path).parent)
            candidates = sorted(
                (m for m in self.models
                 if is_mmproj_file(m.file_name) and str(Path(m.file_path).parent) == model_dir),
                key=lambda m: m.file_path,
            )
        self.controller.set("mmproj_model_path", candidates[0].file_path if candidates else "")

    def _reload_server_models(self):
        server_name = self._get_current_server()
        if not server_name:
            return
        cached = load_cached_models(server_name)
        if cached:
            self.models = cached
            self._populate_model_lists()
            return

        # Drop the previous server's models, so a slow scan never leaves
        # another machine's files on screen in the meantime.
        self.models = []
        self._populate_model_lists()

        connected = (self.server_manager.get_connection_state(server_name)
                     == ConnectionState.CONNECTED)
        if not connected and self.server_manager.is_remote(server_name):
            # Scanning would reach out over the network, and connecting is the
            # user's call. Connecting is also what fills this list.
            self.models_tab.show_placeholder(
                f"Not connected to '{server_name}'. Connect to list its models.")
            return
        self.models_tab.show_placeholder(f"Scanning models on '{server_name}'…")
        self.scan_models()

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    # GPU information now comes from the selected server's host profile, read
    # when that server connects (_on_connect_requested -> _apply_host_profile).
    # The old local-only detect/poll pair was removed: it described this machine
    # regardless of which server was selected, and its 30s poller appended local
    # live stats to a tab that may be showing a remote host.

    # ------------------------------------------------------------------
    # Live CPU/GPU usage bars
    # ------------------------------------------------------------------

    def _restart_live_stats_timer(self):
        """Re-pace polling for the selected server: 2s local, 5s remote.

        A remote sample is an SSH round trip plus nvidia-smi on that host, so it
        is deliberately gentler than polling our own machine.
        """
        name = self._get_current_server()
        interval = LOCAL_STATS_INTERVAL_MS
        if name and self.server_manager.is_remote(name):
            interval = REMOTE_STATS_INTERVAL_MS
        timer = getattr(self, "_live_stats_timer", None)
        if timer is not None and timer.interval() != interval:
            timer.start(interval)

    def _poll_live_stats(self):
        """Sample the selected server, but only while it is connected."""
        if self._closing:
            return
        self._poll_preview_status()
        existing = getattr(self, "_rt_worker", None)
        if existing is not None and existing.isRunning():
            return

        name = self._get_current_server()
        if not name or (self.server_manager.get_connection_state(name)
                        != ConnectionState.CONNECTED):
            # Nothing is polled behind the user's back; a disconnected server
            # simply shows no live figures.
            self.resource_panel.clear_live_usage(name)
            return

        self._rt_worker = UsageSampleWorker(self.server_manager, name)
        self._rt_worker.sample_ready.connect(self._on_usage_sample)
        self._rt_worker.finished.connect(lambda: setattr(self, "_rt_worker", None))
        self._start_worker(self._rt_worker)

    def _on_usage_sample(self, server_name, sample):
        if self._closing or sample is None:
            return
        # A late sample from a server the user has since switched away from
        # must not be shown as the current one.
        if server_name != self._get_current_server():
            return
        self.resource_panel.update_live_usage(sample, server_name)
        self._last_usage_sample = sample
        if self.benchmark_tab.is_recording():
            self.benchmark_tab.record_usage_sample(sample)

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
            own_pids=self._own_server_pids(),
            vram_override=self._remote_vram_override(),
            system_ram_mb=self._host_ram_mb,
            # A model on a remote host cannot be stat'ed here; the scan already
            # recorded its size.
            known_sizes={m.file_path: m.file_size_mb for m in self.models},
        )
        worker.estimate_done.connect(self._on_estimate_done)
        worker.finished.connect(lambda: setattr(self, "_estimate_worker", None))
        self._estimate_worker = worker
        self._start_worker(worker)

    def _remote_vram_override(self):
        """Live VRAM figures for a remote server, or None to sample locally.

        Returns ``(total_mb, available_mb)`` where available counts this
        server's own llama-server allocation as reclaimable -- relaunching it
        frees that memory first, exactly as for a local server.
        """
        name = self._get_current_server()
        if not name or not self.server_manager.is_remote(name):
            return None
        sample = self._last_usage_sample
        if sample is None or sample.gpu_mem_total_mb <= 0:
            return None
        free_mb = max(0.0, sample.gpu_mem_total_mb - sample.gpu_mem_used_mb)
        return (sample.gpu_mem_total_mb, free_mb + sample.own_vram_mb)

    def _own_server_pids(self) -> list:
        """PIDs of llama-servers this launcher started on this machine.

        Their VRAM counts as available to a new estimate: relaunching a server
        frees what it currently holds. VRAM held by anything else does not.
        """
        pids = []
        server = self._get_current_server()
        if server and not self.server_manager.is_remote(server):
            try:
                pid = self.server_manager.get_server_pid(server)
            except Exception:
                logger.debug("Could not read server PID for %s", server, exc_info=True)
                pid = None
            if pid:
                pids.append(int(pid))
        return pids

    def _on_estimate_done(self, estimate):
        self._last_estimate = estimate
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
    # Launch (targets whichever server the Preview tab has selected; goes
    # through the same ServerManager mechanism Server Management uses, so
    # either screen can start/stop what the other started)
    # ------------------------------------------------------------------

    def launch_server(self):
        """Launch llama-server on the Preview tab's selected server."""
        try:
            validate_config(self.config)
            cmd = self._resolve_server_binary(build_command(self.config))
        except Exception as e:
            self._mark_model_invalid("model_path" in str(e))
            QMessageBox.warning(self, "Cannot Launch", str(e))
            return
        self._mark_model_invalid(False)

        target = self.preview_tab.server_bar.current_server() or self._get_current_server()
        if not target:
            QMessageBox.warning(self, "Cannot Launch", "No server selected.")
            return

        conflict = self.server_manager.check_port_conflict(target)
        if conflict:
            QMessageBox.warning(self, "Cannot Launch", conflict)
            return

        self.preview_tab.clear_log()
        self.preview_tab.append_log(f"Launching: {cmd}\n\n")

        worker = TaskWorker(lambda: self.server_manager.start_server(target, cmd))
        worker.done.connect(lambda ok: self._on_launch_done(target, ok))
        worker.failed.connect(self._on_launch_failed)
        self._start_worker(worker)

    def _on_launch_done(self, target: str, ok: bool):
        if not ok:
            self.preview_tab.append_log(f"\nLaunch Error: could not start '{target}'.\n")
            return
        self._attach_preview_stream(target, own_launch=True)
        self._preview_launch_started_at = time.time()
        QTimer.singleShot(2500, lambda: self._verify_preview_launch(target))

    def _on_launch_failed(self, message: str):
        logger.error("Launch failed: %s", message)
        self.preview_tab.append_log(f"\nLaunch Error: {message}\n")

    def _verify_preview_launch(self, target: str):
        """Confirm a just-launched process didn't die immediately.

        Mirrors Server Management's own post-start check
        (``ServerStatusWidget._verify_started``); owning this for the first
        few seconds after our own launch keeps it from racing the generic
        reconciliation poll in ``_poll_live_stats`` for the same event.
        """
        if self._preview_watch_server != target or not self._preview_launched_by_us:
            return  # detached or superseded already
        if self.server_manager.tracked_pid_alive(target):
            return
        log = (self.server_manager.tail_log(target, 40) or "").strip()
        detail = log[-1500:] if log else "(no log output)"
        self._detach_preview_stream()
        QMessageBox.warning(
            self, "Server Failed to Start",
            f"llama-server exited right after launch on '{target}'.\n\n"
            f"Log output:\n{detail}")

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
        """Stop the server currently selected on the Preview tab.

        Mirrors Server Management's own Stop button exactly: it acts on
        whichever server is selected there, not on the internal "watching"
        state (which can lag a poll cycle behind), so Stop always targets
        what the user is actually looking at.
        """
        target = self.preview_tab.server_bar.current_server() or self._get_current_server()
        if not target:
            return
        self.preview_tab.stop_btn.setEnabled(False)
        worker = TaskWorker(lambda: self.server_manager.stop_server(target))
        worker.done.connect(lambda _ok: self._on_preview_stop_done(target))
        worker.failed.connect(lambda msg: (
            logger.error("Stop failed: %s", msg), self._on_preview_stop_done(target)))
        self._start_worker(worker)

    def _on_preview_stop_done(self, target: str):
        if self._preview_watch_server == target:
            self._detach_preview_stream()
        else:
            self.preview_tab.stop_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Preview <-> live server mirroring
    # ------------------------------------------------------------------

    def _attach_preview_stream(self, target: str, own_launch: bool):
        """Start mirroring ``target``'s live log/status into the Preview tab.

        Called both for launches Preview itself initiated and for servers
        the reconciliation poll (``_poll_live_stats``) discovers are already
        running (e.g. started from Server Management). Only an
        ``own_launch`` attach records a benchmark session, since only then
        do we actually know the config that produced it.
        """
        if self._preview_watch_server == target:
            return
        if self._preview_watch_server is not None:
            self._detach_preview_stream()

        self._preview_watch_server = target
        self._preview_launched_by_us = own_launch
        self.preview_tab.set_running(True)

        self._preview_log_worker = LogFollowWorker(self.server_manager, target)
        self._preview_log_worker.line_received.connect(
            lambda line: self._on_preview_log_line(line))
        self._preview_log_worker.start()

        if own_launch:
            self.benchmark_tab.begin_session(
                preset=self.basic_tab.preset_combo.currentText().strip(),
                fingerprint=config_fingerprint(self.config),
                config=dataclasses.replace(self.config),
                estimate=self._last_estimate,
            )

    def _on_preview_log_line(self, line: str):
        self.preview_tab.append_log(line + "\n")
        if self._preview_launched_by_us:
            self.benchmark_tab.feed_log(line + "\n")

    def _detach_preview_stream(self):
        """Stop mirroring whatever server Preview was watching."""
        worker = self._preview_log_worker
        self._preview_log_worker = None
        if worker is not None:
            worker.line_received.disconnect()
            worker.stop()

        self.preview_tab.set_running(False)
        if self._preview_launched_by_us:
            self.benchmark_tab.end_session()

        self._preview_watch_server = None
        self._preview_launched_by_us = False

    def _shutdown_preview_server(self):
        """Synchronously stop whatever Preview is watching, for app close."""
        target = self._preview_watch_server
        if target:
            try:
                self.server_manager.stop_server(target)
            except Exception:
                logger.debug("Stop on close failed for %s", target, exc_info=True)
        if self._preview_log_worker is not None:
            self._preview_log_worker.line_received.disconnect()
            self._preview_log_worker.stop()
            self._preview_log_worker = None
        self._preview_watch_server = None
        self._preview_launched_by_us = False

    def _poll_preview_status(self):
        """Background-check the Preview tab's selected server's run state.

        ``get_status`` is a real round trip for SSH servers, so this runs on
        a background ``TaskWorker`` (same pattern as ``UsageSampleWorker``
        just above) rather than blocking the GUI thread on every poll tick.
        """
        existing = getattr(self, "_status_worker", None)
        if existing is not None and existing.isRunning():
            return

        selected = self.preview_tab.server_bar.current_server()
        watched = self._preview_watch_server
        if watched and watched != selected:
            self._detach_preview_stream()

        if not selected or (self.server_manager.get_connection_state(selected)
                            != ConnectionState.CONNECTED):
            return

        self._status_worker = TaskWorker(lambda: self.server_manager.get_status(selected))
        self._status_worker.done.connect(lambda status: self._reconcile_preview_stream(selected, status))
        self._status_worker.finished.connect(lambda: setattr(self, "_status_worker", None))
        self._start_worker(self._status_worker)

    def _reconcile_preview_stream(self, selected: str, status):
        """Keep Preview's Launch/Stop/log state matching reality.

        Runs on the same cadence as the live resource poll so either
        screen's Start/Stop is reflected on the other without the user
        having to do anything: a server started from Server Management
        shows up here as Running with live log output (but no fabricated
        benchmark session, since we don't know what config produced it),
        and a stop issued from Server Management (or a crash) clears
        Preview's Running state here too.
        """
        if self._closing or selected != self.preview_tab.server_bar.current_server():
            return  # stale result -- user already moved on

        watched = self._preview_watch_server
        running = status is not None and status.state == ServerProcessState.RUNNING

        if running and watched is None:
            self._attach_preview_stream(selected, own_launch=False)
        elif not running and watched == selected:
            # Give our own just-launched process a grace window so
            # `_verify_preview_launch` (not this generic check) is the one
            # that reports an immediate crash, with its tail-log detail.
            if (self._preview_launched_by_us
                    and time.time() - self._preview_launch_started_at < 3.0):
                return
            self._detach_preview_stream()

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
