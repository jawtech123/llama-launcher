"""Benchmark tab: tokens/second and token usage from llama-server logs.

Live sessions are fed log output by the main window (begin_session /
feed_log / end_session). Remote servers started via Server Management log
to their remote file instead; "Import Remote Log" pulls and parses that.
"""
import logging
import time
from datetime import datetime

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from llama_launcher.benchmark import (
    BenchmarkRunConfig,
    ResourceAccumulator,
    SessionStats,
    aggregate,
    append_session,
    build_session_label,
    build_session_tooltip,
    clear_sessions,
    config_snapshot_fields,
    estimate_summary,
    load_sessions,
    parse_log_line,
    parse_log_text,
    replace_sessions_for_source,
    run_benchmark,
    wait_for_server_ready,
)
from llama_launcher.gui import theme
from llama_launcher.gui.dialogs.benchmark_compare_dialog import SessionCompareDialog
from llama_launcher.models.scanner import infer_model_hints


logger = logging.getLogger(__name__)

_REMOTE_TAIL_LINES = 4000


class _LogFetchWorker(QThread):
    """Fetch a server's log tail in the background."""

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, manager, server_name, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._server_name = server_name

    def run(self):
        try:
            log = self._manager.tail_log(self._server_name, _REMOTE_TAIL_LINES)
            self.done.emit(log or "")
        except Exception as e:
            self.failed.emit(str(e))


class _BenchmarkRunWorker(QThread):
    """Run a controlled benchmark against the currently-launched local server."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, base_url: str, run_cfg: BenchmarkRunConfig, parent=None):
        super().__init__(parent)
        self._base_url = base_url
        self._run_cfg = run_cfg

    def run(self):
        try:
            if not wait_for_server_ready(self._base_url, timeout_s=60.0):
                self.failed.emit(
                    f"Server did not respond on {self._base_url}/health within 60s.")
                return
            run_benchmark(self._base_url, self._run_cfg)
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


def _fmt_count(n: float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n:,}"
    return str(n)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


class BenchmarkTab(QWidget):
    """Throughput statistics per session and in total, parsed from server logs."""

    def __init__(self, server_manager=None, server_name_provider=None, parent=None):
        super().__init__(parent)
        self._manager = server_manager
        self._server_name_provider = server_name_provider
        self._stats = SessionStats()
        self._running = False
        self._started = 0.0
        self._line_buffer = ""
        self._session_preset = ""
        self._session_fingerprint = ""
        self._session_config = None
        self._session_estimate = None
        self._resource = ResourceAccumulator()
        self._current_sessions = []
        self._fetch_worker = None
        self._bench_worker = None
        self._bench_snapshot = None
        self._bench_run_cfg = None
        self._build_ui()
        self._reload_history()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(1000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        session_group = QGroupBox("Current Session")
        grid = QGridLayout(session_group)

        self._status_label = self._value_label("Idle")
        self._requests_label = self._value_label("0")
        self._tokens_in_label = self._value_label("0")
        self._tokens_out_label = self._value_label("0")
        self._speed_in_label = self._value_label("--")
        self._speed_out_label = self._value_label("--")
        self._elapsed_label = self._value_label("--")

        for i, text in enumerate(("Tokens In:", "Tokens Out:", "Requests:")):
            grid.addWidget(self._head_label(text), 0, i)
        grid.addWidget(self._tokens_in_label, 1, 0)
        grid.addWidget(self._tokens_out_label, 1, 1)
        grid.addWidget(self._requests_label, 1, 2)

        for i, text in enumerate(("Speed In (t/s):", "Speed Out (t/s):",
                                  "Elapsed:", "Status:")):
            grid.addWidget(self._head_label(text), 2, i)
        grid.addWidget(self._speed_in_label, 3, 0)
        grid.addWidget(self._speed_out_label, 3, 1)
        grid.addWidget(self._elapsed_label, 3, 2)
        grid.addWidget(self._status_label, 3, 3)

        self._gpu_label = self._value_label("--")
        self._vram_label = self._value_label("--")
        for i, text in enumerate(("GPU Util:", "VRAM Used:")):
            grid.addWidget(self._head_label(text), 4, i)
        grid.addWidget(self._gpu_label, 5, 0)
        grid.addWidget(self._vram_label, 5, 1)

        layout.addWidget(session_group)

        buttons = QHBoxLayout()
        self.run_bench_btn = QPushButton("Run Benchmark")
        self.run_bench_btn.setEnabled(False)
        self.run_bench_btn.setToolTip(
            "Send a fixed prompt/generation-length request to the running\n"
            "local server for a controlled, repeatable throughput number\n"
            "(rather than relying on passive real usage).")
        self.run_bench_btn.clicked.connect(self._on_run_benchmark)
        buttons.addWidget(self.run_bench_btn)

        self.import_btn = QPushButton("Import Remote Log")
        self.import_btn.setToolTip(
            "Fetch the selected server's llama-server log (via Server Management\n"
            "connection) and import its throughput stats as a session.\n"
            "Re-importing the same server replaces its previous import.")
        self.import_btn.clicked.connect(self._on_import_remote)
        buttons.addWidget(self.import_btn)

        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self._on_clear_history)
        buttons.addWidget(self.clear_btn)

        self.compare_btn = QPushButton("Compare Selected")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip("Select 2 or more history rows (Ctrl/Shift+click) to compare.")
        self.compare_btn.clicked.connect(self._on_compare_selected)
        buttons.addWidget(self.compare_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        layout.addWidget(QLabel("Session History:"))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Started", "Source", "Requests", "Tokens In", "Tokens Out",
            "Speed In (t/s)", "Speed Out (t/s)", "Duration",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        self._totals_label = QLabel("")
        self._totals_label.setStyleSheet(
            theme.monospace_style(theme.FG_TEXT, 11) + " font-weight: bold; padding: 4px;")
        layout.addWidget(self._totals_label)

    @staticmethod
    def _head_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {theme.FG_DIM}; font-size: 10px;")
        return lbl

    @staticmethod
    def _value_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(theme.monospace_style(theme.FG_TEXT, 12))
        return lbl

    # ------------------------------------------------------------------
    # Live session API (called by the main window)
    # ------------------------------------------------------------------

    def begin_session(self, preset: str = "", fingerprint: str = "",
                      config=None, estimate=None):
        """Start accumulating stats for a freshly launched server.

        ``preset``/``fingerprint`` link the session to the preset and the
        exact parameters in use; the preset info button uses them to show
        (and invalidate) benchmark info. ``config``/``estimate`` are used
        to build a descriptive label and detail tooltip once the session
        is persisted.
        """
        self._stats = SessionStats()
        self._line_buffer = ""
        self._session_preset = preset or ""
        self._session_fingerprint = fingerprint or ""
        self._session_config = config
        self._session_estimate = estimate
        self._resource = ResourceAccumulator()
        self._running = True
        self._started = time.time()
        self._status_label.setText("Running")
        self._gpu_label.setText("--")
        self._vram_label.setText("--")
        self.run_bench_btn.setEnabled(True)
        self._refresh_live_labels()

    def is_recording(self) -> bool:
        """Whether a live session is currently accumulating stats."""
        return self._running

    def record_usage_sample(self, sample) -> None:
        """Feed one live resource sample (GPU%/VRAM/RAM) into the running session.

        Called by the main window's existing usage poller; only accumulates
        while a session is actually running.
        """
        if not self._running or sample is None:
            return
        self._resource.add(sample)
        if sample.gpu_percent >= 0:
            self._gpu_label.setText(f"{sample.gpu_percent:.0f}%")
        if sample.gpu_mem_used_mb >= 0:
            self._vram_label.setText(f"{sample.gpu_mem_used_mb:.0f} MB")

    def feed_log(self, text: str):
        """Parse a chunk of server log output (handles split lines)."""
        if not text:
            return
        self._line_buffer += text
        lines = self._line_buffer.split("\n")
        self._line_buffer = lines.pop()  # trailing partial line
        changed = self._consume_lines(lines)
        if changed and self._running:
            self._refresh_live_labels()

    def end_session(self):
        """Stop accumulating; persist the session if anything was measured."""
        if self._line_buffer:
            self._consume_lines([self._line_buffer])
            self._line_buffer = ""
        self._running = False
        self._status_label.setText("Idle")
        self.run_bench_btn.setEnabled(False)
        if self._stats.requests <= 0 and self._stats.gen_tokens <= 0:
            self._refresh_live_labels()
            return
        self._refresh_live_labels()
        model_path = getattr(self._session_config, "model_path", "") or ""
        context_size = getattr(self._session_config, "context_size", None)
        label = build_session_label(
            model_path, self._session_preset, kind="live", source="local",
            context_size=context_size)
        session = self._session_dict(
            source="local", label=label,
            started=self._started, duration_s=time.time() - self._started,
            kind="live", model_path=model_path,
            config=self._session_config, estimate=self._session_estimate)
        session["preset"] = self._session_preset
        session["fingerprint"] = self._session_fingerprint
        append_session(session)
        self._reload_history()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _consume_lines(self, lines) -> bool:
        changed = False
        for line in lines:
            rec = parse_log_line(line)
            if rec is not None:
                self._stats.add(rec)
                changed = True
        return changed

    def _refresh_live_labels(self):
        s = self._stats
        self._tokens_in_label.setText(_fmt_count(s.prompt_tokens))
        self._tokens_out_label.setText(_fmt_count(s.gen_tokens))
        self._requests_label.setText(str(s.requests))
        self._speed_in_label.setText(f"{s.prompt_tps:.1f}" if s.prompt_ms > 0 else "--")
        self._speed_out_label.setText(f"{s.gen_tps:.1f}" if s.gen_ms > 0 else "--")

    def _update_elapsed(self):
        if self._running and self._started > 0:
            self._elapsed_label.setText(_fmt_duration(time.time() - self._started))
        else:
            self._elapsed_label.setText("--")

    def _session_dict(self, source: str, label: str,
                      started: float, duration_s: float,
                      stats: SessionStats = None, kind: str = "live",
                      model_path: str = "", config=None, estimate=None) -> dict:
        s = stats if stats is not None else self._stats
        family, quant = infer_model_hints(model_path) if model_path else ("unknown", None)
        # Resource sampling only accumulates for the live-session path (see
        # record_usage_sample); an imported/benchmark-delta stats object has
        # no corresponding sample window, so resource stats are omitted for it.
        resource = self._resource.summary() if (kind == "live" and self._resource.count) else {}
        return {
            "source": source,
            "label": label,
            "started": started,
            "duration_s": duration_s,
            "requests": s.requests,
            "prompt_tokens": s.prompt_tokens,
            "prompt_ms": s.prompt_ms,
            "gen_tokens": s.gen_tokens,
            "gen_ms": s.gen_ms,
            "kind": kind,
            "model_path": model_path,
            "model_family": family if family != "unknown" else "",
            "model_quant": quant or "",
            "config_snapshot": config_snapshot_fields(config) if config is not None else {},
            "estimate": estimate_summary(estimate),
            "resource": resource,
            **s.percentiles(),
        }

    # ------------------------------------------------------------------
    # History / totals
    # ------------------------------------------------------------------

    def _reload_history(self):
        sessions = load_sessions()
        self._current_sessions = list(reversed(sessions))  # row index -> session, newest on top
        self.table.setRowCount(len(sessions))
        for row, s in enumerate(self._current_sessions):
            started = s.get("started", 0)
            started_text = (datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M")
                            if started else "--")
            prompt_ms = float(s.get("prompt_ms", 0))
            gen_ms = float(s.get("gen_ms", 0))
            p_tok = int(s.get("prompt_tokens", 0))
            g_tok = int(s.get("gen_tokens", 0))
            speed_in = p_tok / (prompt_ms / 1000.0) if prompt_ms > 0 else 0.0
            speed_out = g_tok / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            values = [
                started_text,
                str(s.get("label") or s.get("source") or "--"),
                str(int(s.get("requests", 0))),
                _fmt_count(p_tok),
                _fmt_count(g_tok),
                f"{speed_in:.1f}" if prompt_ms > 0 else "--",
                f"{speed_out:.1f}" if gen_ms > 0 else "--",
                _fmt_duration(float(s.get("duration_s", 0))),
            ]
            tooltip = build_session_tooltip(s)
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(tooltip)
                self.table.setItem(row, col, item)
        self._update_totals(sessions)

    def _on_row_double_clicked(self, row: int, _column: int):
        if 0 <= row < len(self._current_sessions):
            session = self._current_sessions[row]
            QMessageBox.information(
                self, "Session Detail", build_session_tooltip(session))

    def _on_selection_changed(self):
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        self.compare_btn.setEnabled(len(rows) >= 2)

    def _on_compare_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        sessions = [self._current_sessions[r] for r in rows
                   if 0 <= r < len(self._current_sessions)]
        if len(sessions) < 2:
            return
        dialog = SessionCompareDialog(sessions, self)
        dialog.exec()

    def _update_totals(self, sessions):
        total = SessionStats()
        for s in sessions:
            total.requests += int(s.get("requests", 0))
            total.prompt_tokens += int(s.get("prompt_tokens", 0))
            total.prompt_ms += float(s.get("prompt_ms", 0))
            total.gen_tokens += int(s.get("gen_tokens", 0))
            total.gen_ms += float(s.get("gen_ms", 0))
        self._totals_label.setText(
            f"Total ({len(sessions)} sessions, {total.requests} requests):   "
            f"In: {_fmt_count(total.prompt_tokens)} tokens "
            f"(avg {total.prompt_tps:.1f} t/s)   |   "
            f"Out: {_fmt_count(total.gen_tokens)} tokens "
            f"(avg {total.gen_tps:.1f} t/s)"
        )

    # ------------------------------------------------------------------
    # Remote import + clear
    # ------------------------------------------------------------------

    def _on_import_remote(self):
        if self._manager is None or self._server_name_provider is None:
            return
        server_name = self._server_name_provider()
        if not server_name:
            QMessageBox.information(self, "Import Remote Log",
                                    "No server selected.")
            return
        self.import_btn.setEnabled(False)
        self.import_btn.setText("Fetching...")
        self._fetch_worker = _LogFetchWorker(self._manager, server_name, self)
        self._fetch_worker.done.connect(
            lambda log: self._on_log_fetched(server_name, log, ""))
        self._fetch_worker.failed.connect(
            lambda msg: self._on_log_fetched(server_name, "", msg))
        self._fetch_worker.finished.connect(self._fetch_worker.deleteLater)
        self._fetch_worker.start()

    def _on_log_fetched(self, server_name: str, log: str, error: str):
        self.import_btn.setEnabled(True)
        self.import_btn.setText("Import Remote Log")
        if error:
            QMessageBox.warning(self, "Import Remote Log",
                                f"Could not fetch log from '{server_name}': {error}")
            return
        stats = aggregate(parse_log_text(log))
        if stats.requests <= 0 and stats.gen_tokens <= 0:
            QMessageBox.information(
                self, "Import Remote Log",
                f"No timing data found in the last {_REMOTE_TAIL_LINES} log lines "
                f"of '{server_name}'.")
            return
        # Build the imported session from the parsed stats directly so an
        # in-progress live session's accumulators are never clobbered.
        label = build_session_label("", "", kind="remote_import", source=server_name)
        session = self._session_dict(
            source=server_name, label=label,
            started=time.time(), duration_s=stats.busy_seconds, stats=stats,
            kind="remote_import")
        replace_sessions_for_source(server_name, session)
        self._reload_history()

    def _on_clear_history(self):
        reply = QMessageBox.question(
            self, "Clear History",
            "Delete all recorded benchmark sessions?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            clear_sessions()
            self._reload_history()

    # ------------------------------------------------------------------
    # Built-in benchmark runner
    #
    # v1 targets only the ad hoc "Launch" flow on this machine (the running
    # live session already wired to feed_log). The benchmark's completions
    # flow through that same live log pipeline as any other traffic, so the
    # live SessionStats keeps counting them as genuine server activity too
    # -- this just also slices out a separately-labeled "benchmark" session
    # from the same underlying samples.
    # ------------------------------------------------------------------

    def _on_run_benchmark(self):
        if not self._running or self._bench_worker is not None:
            return
        host = getattr(self._session_config, "host", "") or "127.0.0.1"
        if host in ("0.0.0.0", ""):
            host = "127.0.0.1"
        port = getattr(self._session_config, "port", None) or 8080
        base_url = f"http://{host}:{port}"

        self._bench_snapshot = (len(self._stats.prompt_ms_samples),
                                len(self._stats.gen_tps_samples))
        self._bench_run_cfg = BenchmarkRunConfig()
        self.run_bench_btn.setEnabled(False)
        self.run_bench_btn.setText("Running Benchmark...")

        self._bench_worker = _BenchmarkRunWorker(base_url, self._bench_run_cfg, self)
        self._bench_worker.done.connect(self._on_benchmark_http_done)
        self._bench_worker.failed.connect(self._on_benchmark_failed)
        self._bench_worker.finished.connect(self._on_benchmark_worker_finished)
        self._bench_worker.start()

    def _on_benchmark_worker_finished(self):
        if self._bench_worker is not None:
            self._bench_worker.deleteLater()
        self._bench_worker = None

    def _on_benchmark_http_done(self):
        # Give the log stream a brief moment to flush the timing lines for
        # the just-completed HTTP requests before slicing them out.
        QTimer.singleShot(500, self._capture_benchmark_result)

    def _capture_benchmark_result(self):
        self.run_bench_btn.setText("Run Benchmark")
        self.run_bench_btn.setEnabled(self._running)
        if self._bench_snapshot is None:
            return
        prompt_start, gen_start = self._bench_snapshot
        self._bench_snapshot = None
        cfg = self._bench_run_cfg

        sliced = self._stats.slice_from(prompt_start, gen_start)
        if cfg.warmup:
            sliced = sliced.slice_from(1, 1) if sliced.requests > 1 else SessionStats()

        if sliced.requests <= 0:
            QMessageBox.information(
                self, "Run Benchmark",
                "No timing data was captured for the benchmark run.")
            return

        model_path = getattr(self._session_config, "model_path", "") or ""
        context_size = getattr(self._session_config, "context_size", None)
        label = build_session_label(
            model_path, self._session_preset, kind="benchmark", source="local",
            context_size=context_size)
        session = self._session_dict(
            source="local", label=label,
            started=time.time() - sliced.busy_seconds, duration_s=sliced.busy_seconds,
            stats=sliced, kind="benchmark", model_path=model_path,
            config=self._session_config, estimate=self._session_estimate)
        session["benchmark_params"] = {
            "n_predict": cfg.n_predict, "iterations": cfg.iterations, "warmup": cfg.warmup,
        }
        append_session(session)
        self._reload_history()

    def _on_benchmark_failed(self, msg: str):
        self.run_bench_btn.setText("Run Benchmark")
        self.run_bench_btn.setEnabled(self._running)
        self._bench_snapshot = None
        QMessageBox.warning(self, "Run Benchmark", f"Benchmark run failed: {msg}")
