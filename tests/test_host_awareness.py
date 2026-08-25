"""Tests for describing and monitoring whichever server is selected.

The launcher runs on one machine and the model runs on another, so GPU, RAM and
utilization must all describe the selected server rather than this box.
"""
import pytest

from llama_launcher.gpu.detector import parse_nvidia_smi_gpus
from llama_launcher.remote import procinfo
from llama_launcher.remote.host_info import (
    HostProfile, load_profile, remove_profile, save_profile,
)
from llama_launcher.remote.local import LocalServerConnection
from llama_launcher.remote.ssh import SSHServerConnection
from llama_launcher.remote.types import ConnectionState, ServerConfig


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


def _remote_config(**kwargs):
    defaults = dict(friendly_name="rem", hostname="10.0.0.5", username="bob")
    defaults.update(kwargs)
    return ServerConfig(**defaults)


class TestNvidiaSmiParsing:
    """One parser serves the local card and a host reached over SSH."""

    def test_single_gpu(self):
        gpu = parse_nvidia_smi_gpus(
            "NVIDIA GeForce RTX 5090, 32607 MiB, 1618 MiB, 595.84")
        assert gpu.backend == "cuda"
        assert gpu.device_name == "NVIDIA GeForce RTX 5090"
        assert gpu.vram_mb == 32607
        assert gpu.available_vram_mb == 1618
        assert gpu.driver_version == "595.84"
        assert gpu.gpu_count == 1

    def test_multi_gpu_totals_are_summed(self):
        gpu = parse_nvidia_smi_gpus(
            "NVIDIA A100, 40000 MiB, 30000 MiB, 550.1\n"
            "NVIDIA A100, 40000 MiB, 10000 MiB, 550.1")
        assert gpu.gpu_count == 2
        assert gpu.vram_mb == 80000
        assert gpu.available_vram_mb == 40000
        assert len(gpu.gpu_infos) == 2

    def test_driver_mismatch_text_is_not_a_gpu(self):
        # Seen on a real host: nvidia-smi prints this instead of a table.
        assert parse_nvidia_smi_gpus(
            "Failed to initialize NVML: Driver/library version mismatch") is None

    def test_empty_output(self):
        assert parse_nvidia_smi_gpus("") is None
        assert parse_nvidia_smi_gpus("   \n  ") is None

    def test_partial_row_is_skipped(self):
        assert parse_nvidia_smi_gpus("NVIDIA Thing, not-a-number, x, y") is None


class TestRemoteGpuReporting:
    def _conn(self, output, ok=True, stderr=""):
        conn = SSHServerConnection(_remote_config())
        conn._last_stderr = stderr
        conn._run_remote = lambda cmd, capture=False, timeout=30: (ok, output)
        return conn

    def test_reports_the_remote_card(self):
        gpu = self._conn("NVIDIA L40S, 46068 MiB, 46000 MiB, 550.2").detect_gpu()
        assert gpu.device_name == "NVIDIA L40S"
        assert gpu.vram_mb == 46068

    def test_driver_mismatch_is_surfaced_not_hidden(self):
        # nvidia-smi exits non-zero here, so the exit code must not gate this.
        gpu = self._conn(
            "Failed to initialize NVML: Driver/library version mismatch",
            ok=False).detect_gpu()
        assert gpu.supported is False
        assert "NVML" in gpu.device_name

    def test_host_without_nvidia_smi_says_so_plainly(self):
        gpu = self._conn("", ok=False,
                         stderr="bash: nvidia-smi: command not found").detect_gpu()
        assert gpu.supported is True
        assert "No NVIDIA GPU" in gpu.device_name


class TestUsageProbe:
    PROBE = (
        "@@LL@@\ncpu  100 0 100 800 0 0 0 0\n"
        "@@LL@@\ncpu  120 0 120 860 0 0 0 0\n"
        "@@LL@@\nMemTotal:       16000000 kB\nMemAvailable:    8000000 kB\n"
        "@@LL@@\n30, 4096, 24576\n"
        "@@LL@@\n4242, 2048\n"
    )

    def test_parses_every_section(self):
        sample = procinfo.parse_usage_probe(self.PROBE, own_pids=[4242])
        # 40 busy of 100 total jiffies between the two readings.
        assert sample.cpu_percent == pytest.approx(40.0, abs=0.1)
        assert sample.ram_total_mb == pytest.approx(15625.0, abs=1)
        assert sample.ram_used_mb == pytest.approx(7812.5, abs=1)
        assert sample.gpu_percent == 30.0
        assert sample.gpu_mem_used_mb == 4096
        assert sample.gpu_mem_total_mb == 24576
        assert sample.own_vram_mb == 2048

    def test_other_processes_vram_is_not_counted_as_ours(self):
        sample = procinfo.parse_usage_probe(self.PROBE, own_pids=[999])
        assert sample.own_vram_mb == 0

    def test_cpu_and_ram_survive_a_broken_nvidia_smi(self):
        # The probe's exit status is its last command's; a GPU-less host must
        # still report CPU and RAM.
        broken = (
            "@@LL@@\ncpu  100 0 100 800 0 0 0 0\n"
            "@@LL@@\ncpu  120 0 120 860 0 0 0 0\n"
            "@@LL@@\nMemTotal:       16000000 kB\nMemAvailable:    8000000 kB\n"
            "@@LL@@\n\n@@LL@@\n"
        )
        sample = procinfo.parse_usage_probe(broken)
        assert sample.cpu_percent > 0
        assert sample.ram_total_mb > 0
        assert sample.gpu_percent == -1.0      # "unavailable", not "idle"

    def test_empty_output_reports_unavailable_not_zero(self):
        sample = procinfo.parse_usage_probe("")
        assert sample.cpu_percent == -1.0
        assert sample.ram_total_mb == -1.0
        assert sample.gpu_percent == -1.0

    def test_identical_samples_do_not_divide_by_zero(self):
        line = "cpu  100 0 100 800 0 0 0 0"
        assert procinfo.cpu_percent_from_samples(line, line) == -1.0

    def test_meminfo_uses_available_not_free(self):
        used, total = procinfo.parse_meminfo(
            "MemTotal:       1048576 kB\nMemFree:          10240 kB\n"
            "MemAvailable:    524288 kB\n")
        # Cache is reclaimable, so used is measured against MemAvailable.
        assert total == pytest.approx(1024.0)
        assert used == pytest.approx(512.0)

    def test_probe_asks_for_two_cpu_readings_in_one_command(self):
        probe = procinfo.build_usage_probe()
        assert probe.count("/proc/stat") == 2
        assert "sleep" in probe


class TestHostProfileCache:
    def test_round_trip(self):
        from llama_launcher.gpu.types import GPUInfo
        gpu = GPUInfo(backend="cuda", vram_mb=32607, available_vram_mb=1618,
                      device_name="RTX 5090", driver_version="595.84")
        save_profile("box-a", HostProfile(gpu=gpu, total_ram_mb=30718.0))
        loaded = load_profile("box-a")
        assert loaded.gpu.device_name == "RTX 5090"
        assert loaded.gpu.vram_mb == 32607
        assert loaded.total_ram_mb == pytest.approx(30718.0)
        assert loaded.age_seconds < 60

    def test_multi_gpu_breakdown_survives(self):
        from llama_launcher.gpu.types import GPUInfo
        gpu = GPUInfo(backend="cuda", vram_mb=80000, available_vram_mb=40000,
                      gpu_count=2, gpu_infos=[
                          GPUInfo(backend="cuda", vram_mb=40000, device_name="A100-0"),
                          GPUInfo(backend="cuda", vram_mb=40000, device_name="A100-1")])
        save_profile("box-multi", HostProfile(gpu=gpu, total_ram_mb=1.0))
        loaded = load_profile("box-multi")
        assert [g.device_name for g in loaded.gpu.gpu_infos] == ["A100-0", "A100-1"]

    def test_never_connected_server_has_no_profile(self):
        assert load_profile("never-seen") is None

    def test_remove(self):
        save_profile("box-b", HostProfile(total_ram_mb=1.0))
        assert remove_profile("box-b") is True
        assert load_profile("box-b") is None
        assert remove_profile("box-b") is False

    def test_profile_without_a_gpu_round_trips(self):
        save_profile("cpu-box", HostProfile(gpu=None, total_ram_mb=4096.0))
        loaded = load_profile("cpu-box")
        assert loaded.gpu is None
        assert loaded.total_ram_mb == 4096.0


class TestConnectionStateIsPassive:
    """Reading state must never open a connection."""

    def test_ssh_state_does_not_probe(self):
        conn = SSHServerConnection(_remote_config(hostname="192.0.2.1"))
        issued = []
        conn._run_remote = lambda *a, **k: issued.append(a) or (False, "")
        assert conn.connection_state == ConnectionState.DISCONNECTED
        assert issued == [], "reading the state reached out over the network"

    def test_state_reflects_an_explicit_connect(self):
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "")
        conn.connect()
        assert conn.connection_state == ConnectionState.CONNECTED

    def test_disconnect_actually_clears_the_state(self):
        # It used to keep CONNECTED so the old auto-probe would not re-run,
        # which now means Disconnect appears to do nothing.
        conn = SSHServerConnection(_remote_config())
        conn._run_remote = lambda cmd, capture=False, timeout=30: (True, "")
        conn.connect()
        conn.shutdown = lambda: None
        conn.disconnect()
        assert conn.connection_state == ConnectionState.DISCONNECTED

    def test_local_disconnect_clears_the_state(self):
        conn = LocalServerConnection(ServerConfig(friendly_name="l",
                                                  llama_server_path="echo"))
        conn.connect()
        assert conn.connection_state == ConnectionState.CONNECTED
        conn.disconnect()
        assert conn.connection_state == ConnectionState.DISCONNECTED


class TestServerBar:
    def _bar(self, app):
        from llama_launcher.gui.server_bar import ServerBar
        return ServerBar()

    def test_connected_is_the_only_coloured_state(self, app):
        from llama_launcher.gui import theme
        bar = self._bar(app)
        bar.set_state(ConnectionState.CONNECTED)
        assert theme.SUCCESS in bar.status_label.styleSheet()

        bar.set_state(ConnectionState.DISCONNECTED)
        # An idle server must not look like a warning.
        assert bar.status_label.styleSheet() == ""

    def test_error_state_is_red(self, app):
        from llama_launcher.gui import theme
        bar = self._bar(app)
        bar.set_state(ConnectionState.ERROR, "No route to host")
        assert theme.ERROR in bar.status_label.styleSheet()
        assert bar.status_label.toolTip() == "No route to host"

    def test_button_swaps_with_state(self, app):
        bar = self._bar(app)
        bar.set_state(ConnectionState.DISCONNECTED)
        assert bar.connect_btn.text() == "Connect"
        bar.set_state(ConnectionState.CONNECTED)
        assert bar.connect_btn.text() == "Disconnect"

    def test_button_emits_the_right_signal(self, app):
        bar = self._bar(app)
        bar.set_servers(["alpha"])
        seen = []
        bar.connect_requested.connect(lambda n: seen.append(("connect", n)))
        bar.disconnect_requested.connect(lambda n: seen.append(("disconnect", n)))

        bar.set_state(ConnectionState.DISCONNECTED)
        bar.connect_btn.click()
        bar.set_state(ConnectionState.CONNECTED)
        bar.connect_btn.click()
        assert seen == [("connect", "alpha"), ("disconnect", "alpha")]

    def test_busy_blocks_interaction(self, app):
        bar = self._bar(app)
        bar.set_busy(True)
        assert not bar.connect_btn.isEnabled()
        assert not bar.combo.isEnabled()
        bar.set_busy(False)
        bar.set_state(ConnectionState.DISCONNECTED)
        assert bar.connect_btn.isEnabled()

    def test_setting_the_server_does_not_re_emit(self, app):
        bar = self._bar(app)
        bar.set_servers(["alpha", "beta"])
        seen = []
        bar.server_changed.connect(seen.append)
        bar.set_current_server("beta")
        assert seen == [], "programmatic sync must not look like a user change"


class TestEstimatorUsesSuppliedHost:
    def test_supplied_ram_replaces_this_machines(self):
        from llama_launcher.core.config import LlamaConfig
        from llama_launcher.estimator.estimator import ResourceEstimator
        from llama_launcher.gpu.types import GPUInfo
        est = ResourceEstimator().calculate(
            config=LlamaConfig(model_path="/m/x.gguf", context_size=4096),
            gpu_info=GPUInfo(backend="cuda", vram_mb=24000, available_vram_mb=24000),
            model_size_mb=4000.0, system_ram_mb=15420.0)
        assert est.system_total_ram_mb == pytest.approx(15420.0)

    def test_falls_back_to_local_ram(self):
        from llama_launcher.core.config import LlamaConfig
        from llama_launcher.estimator.estimator import ResourceEstimator
        estimator = ResourceEstimator()
        est = estimator.calculate(
            config=LlamaConfig(model_path="/m/x.gguf", context_size=4096),
            model_size_mb=4000.0)
        assert est.system_total_ram_mb == pytest.approx(estimator._system_ram_mb)


class TestConnectionStaysInSyncAcrossTabs:
    """Connecting anywhere must be visible everywhere.

    There used to be two independent connect paths -- the server bars and the
    Server Management tab -- so each was blind to the other's state.
    """

    @pytest.fixture
    def window(self, app, monkeypatch):
        from llama_launcher.gui import main_window as mw
        monkeypatch.setattr(mw.LlamaLauncherGUI, "_deferred_init", lambda self: None)
        win = mw.LlamaLauncherGUI()
        win.server_manager.add_server(
            ServerConfig(friendly_name="box", llama_server_path="echo"), persist=False)
        win._refresh_server_combo()
        yield win
        win._closing = True
        win.deleteLater()

    def test_management_tab_does_not_connect_on_its_own(self, window):
        # It must delegate, or the rest of the app never learns about the host.
        status = window.server_management_widget.status_widget
        requested = []
        status.connect_requested.connect(requested.append)
        status._selected_server = "box"
        status._on_connect()
        assert requested == ["box"]
        # And it must not have connected behind the app's back.
        assert window.server_manager.get_connection_state("box") != ConnectionState.CONNECTED

    def test_management_disconnect_delegates_too(self, window):
        status = window.server_management_widget.status_widget
        requested = []
        status.disconnect_requested.connect(requested.append)
        status._selected_server = "box"
        status._on_disconnect()
        assert requested == ["box"]

    def test_connecting_refreshes_the_management_tab(self, window, monkeypatch):
        seen = []
        monkeypatch.setattr(window.server_management_widget, "on_connection_changed",
                            seen.append)
        window._on_disconnect_requested("box")
        assert seen == ["box"]

    def test_a_failed_connect_still_re_enables_the_button(self, window, monkeypatch):
        from llama_launcher.gui import main_window as mw
        monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
        seen = []
        monkeypatch.setattr(window.server_management_widget, "on_connection_changed",
                            seen.append)
        window._on_connect_done("box", False, "No route to host", None, None)
        # Without this the management tab's Connect stays greyed out forever.
        assert seen == ["box"]

    def test_management_widget_forwards_status_widget_signals(self, window):
        forwarded = []
        window.server_management_widget.connect_requested.connect(forwarded.append)
        window.server_management_widget.status_widget.connect_requested.emit("box")
        assert forwarded == ["box"]


class TestServerBinaryDetection:
    """A wrong llama-server path should surface on connect, not as a failed launch."""

    def test_local_resolves_a_name_on_path(self):
        conn = LocalServerConnection(ServerConfig(friendly_name="l"))
        assert conn.resolve_executable("echo").endswith("/echo")

    def test_local_rejects_a_missing_path(self):
        conn = LocalServerConnection(ServerConfig(friendly_name="l"))
        assert conn.resolve_executable("/definitely/not/here") == ""
        assert conn.resolve_executable("") == ""

    def test_local_rejects_a_non_executable_file(self, tmp_path):
        plain = tmp_path / "not-a-binary"
        plain.write_text("x")
        conn = LocalServerConnection(ServerConfig(friendly_name="l"))
        assert conn.resolve_executable(str(plain)) == ""

    def test_remote_bare_name_uses_command_v(self):
        conn = SSHServerConnection(_remote_config())
        issued = []

        def fake(cmd, capture=False, timeout=30):
            issued.append(cmd)
            return (True, "")

        conn._run_remote = fake
        conn.resolve_executable("llama-server")
        assert "command -v" in issued[0]

    def test_remote_path_is_tested_for_executability(self):
        conn = SSHServerConnection(_remote_config())
        issued = []

        def fake(cmd, capture=False, timeout=30):
            issued.append(cmd)
            return (True, "/opt/llama-server")

        conn._run_remote = fake
        assert conn.resolve_executable("/opt/llama-server") == "/opt/llama-server"
        assert "-x" in issued[0]

    def test_remote_search_covers_the_usual_locations(self):
        from llama_launcher.remote.types import LLAMA_SERVER_CANDIDATES
        conn = SSHServerConnection(_remote_config())
        issued = []

        def fake(cmd, capture=False, timeout=30):
            issued.append(cmd)
            return (True, "/workspace/llama.cpp/build/bin/llama-server")

        conn._run_remote = fake
        found = conn.find_server_binary()
        assert found == "/workspace/llama.cpp/build/bin/llama-server"
        # One round trip, not one per candidate.
        assert len(issued) == 1
        for candidate in LLAMA_SERVER_CANDIDATES:
            assert candidate in issued[0]

    def test_manager_returns_blank_for_unknown_server(self):
        from llama_launcher.remote.manager import ServerManager
        mgr = ServerManager()
        assert mgr.resolve_executable("nope", "llama-server") == ""
        assert mgr.find_server_binary("nope") == ""


class TestBinaryFixPrompt:
    @pytest.fixture
    def window(self, app, monkeypatch):
        from llama_launcher.gui import main_window as mw
        monkeypatch.setattr(mw.LlamaLauncherGUI, "_deferred_init", lambda self: None)
        win = mw.LlamaLauncherGUI()
        win.server_manager.add_server(
            ServerConfig(friendly_name="box", llama_server_path="llama-server"),
            persist=False)
        win._refresh_server_combo()
        yield win
        win._closing = True
        win.deleteLater()

    def test_no_prompt_when_the_path_is_fine(self, window, monkeypatch):
        from llama_launcher.gui import main_window as mw
        shown = []
        monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: shown.append(a))
        window._offer_binary_fix("box", (True, ""))
        assert shown == []

    def test_warns_when_nothing_can_be_found(self, window, monkeypatch):
        from llama_launcher.gui import main_window as mw
        shown = []
        monkeypatch.setattr(mw.QMessageBox, "warning",
                            lambda *a, **k: shown.append(a[2]))
        window._offer_binary_fix("box", (False, ""))
        assert shown and "not runnable" in shown[0]
        assert "Server Exec" in shown[0]

    def _click(self, monkeypatch, label):
        """Make the suggestion dialog answer with the named button."""
        from llama_launcher.gui import main_window as mw
        from PySide6.QtWidgets import QMessageBox

        recorded = {}

        def fake_exec(self):
            recorded["text"] = self.text()
            recorded["informative"] = self.informativeText()
            for button in self.buttons():
                if button.text().rstrip("…") == label.rstrip("…"):
                    recorded["button"] = button
            return 0

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        monkeypatch.setattr(QMessageBox, "clickedButton",
                            lambda self: recorded.get("button"))
        return recorded

    def test_accepting_the_suggestion_updates_and_persists(self, window, monkeypatch):
        from llama_launcher.remote.storage import load_servers
        recorded = self._click(monkeypatch, "Use This")

        window._offer_binary_fix("box", (False, "/opt/llama.cpp/bin/llama-server"))

        assert "not runnable" in recorded["text"]
        assert "/opt/llama.cpp/bin/llama-server" in recorded["informative"]
        cfg = window.server_manager.get_connection("box").config
        assert cfg.llama_server_path == "/opt/llama.cpp/bin/llama-server"
        stored = {s.friendly_name: s for s in load_servers()}
        assert stored["box"].llama_server_path == "/opt/llama.cpp/bin/llama-server"

    def test_ignoring_leaves_the_config_alone(self, window, monkeypatch):
        self._click(monkeypatch, "Ignore")
        window._offer_binary_fix("box", (False, "/opt/llama.cpp/bin/llama-server"))
        cfg = window.server_manager.get_connection("box").config
        assert cfg.llama_server_path == "llama-server"

    def test_missing_status_is_ignored(self, window):
        # An older/failed connect emits None; must not raise.
        window._offer_binary_fix("box", None)
