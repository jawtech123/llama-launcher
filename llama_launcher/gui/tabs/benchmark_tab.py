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
    SessionStats,
    aggregate,
    append_session,
    clear_sessions,
    load_sessions,
    parse_log_line,
    parse_log_text,
    replace_sessions_for_source,
)
from llama_launcher.gui import theme
from llama_launcher.remote.types import ConnectionType

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
        self._fetch_worker = None
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

        layout.addWidget(session_group)

        buttons = QHBoxLayout()
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

    def begin_session(self, preset: str = "", fingerprint: str = ""):
        """Start accumulating stats for a freshly launched server.

        ``preset``/``fingerprint`` link the session to the preset and the
        exact parameters in use; the preset info button uses them to show
        (and invalidate) benchmark info.
        """
        self._stats = SessionStats()
        self._line_buffer = ""
        self._session_preset = preset or ""
        self._session_fingerprint = fingerprint or ""
        self._running = True
        self._started = time.time()
        self._status_label.setText("Running")
        self._refresh_live_labels()

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
        if self._stats.requests <= 0 and self._stats.gen_tokens <= 0:
            self._refresh_live_labels()
            return
        self._refresh_live_labels()
        label = self._session_preset or "Local session"
        session = self._session_dict(
            source="local", label=label,
            started=self._started, duration_s=time.time() - self._started)
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
                      stats: SessionStats = None) -> dict:
        s = stats if stats is not None else self._stats
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
        }

    # ------------------------------------------------------------------
    # History / totals
    # ------------------------------------------------------------------

    def _reload_history(self):
        sessions = load_sessions()
        self.table.setRowCount(len(sessions))
        for row, s in enumerate(reversed(sessions)):  # newest on top
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
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
        self._update_totals(sessions)

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
        conn = self._manager.get_connection(server_name)
        if conn is None:
            QMessageBox.warning(self, "Import Remote Log",
                                f"No connection for '{server_name}'.")
            return
        if (conn.config.connection_type == ConnectionType.SSH
                and conn.connection_state.value == "disconnected"):
            reply = QMessageBox.question(
                self, "Import Remote Log",
                f"'{server_name}' is not connected. Connect now and fetch the log?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
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
        session = self._session_dict(
            source=server_name, label=f"Remote: {server_name}",
            started=time.time(), duration_s=stats.busy_seconds, stats=stats)
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
