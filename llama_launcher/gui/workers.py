"""Background worker threads for the GUI.

All blocking operations (model scanning, GPU detection, preset style checks)
run in these QThread workers so the UI never freezes. Workers are started via
``LlamaLauncherGUI._start_worker`` which parents them and frees them on finish.
"""
import logging
import platform
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from dataclasses import replace

from llama_launcher.core.presets import PresetManager
from llama_launcher.gpu.detector import detect_gpu, sample_vram
from llama_launcher.models.scanner import infer_model_hints, scan_models
from llama_launcher.remote.types import ConnectionState
from llama_launcher.utils.helpers import get_file_size_mb

logger = logging.getLogger(__name__)


class TaskWorker(QThread):
    """Run an arbitrary callable in a background thread.

    Shared by everything that touches a connection from the UI -- directory
    listings, connection tests, file operations -- because an SSH round trip
    must never run on the GUI thread.
    """

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:
            logger.debug("Background task failed: %s", e, exc_info=True)
            self.failed.emit(str(e))


class TransferWorker(TaskWorker):
    """A task worker with a progress signal for file transfers."""

    progress = Signal(int)


class HostConnectWorker(QThread):
    """Connect to a server and read everything that describes its machine.

    Connecting is the one moment we are entitled to reach out, so the GPU, the
    RAM and the model list are all fetched in this single pass rather than
    trickling out over later round trips.
    """

    # server_name, ok, message, HostProfile|None, models|None, binary_status
    # binary_status is (configured_is_runnable, path_found_by_search).
    connect_done = Signal(str, bool, str, object, object, object)

    def __init__(self, manager, server_name):
        super().__init__()
        self.manager = manager
        self.server_name = server_name

    def run(self):
        from llama_launcher.remote.host_info import HostProfile

        try:
            ok, message = self.manager.test_connection(self.server_name)
        except Exception as e:
            logger.debug("Connect failed for %s", self.server_name, exc_info=True)
            self.connect_done.emit(self.server_name, False, str(e), None, None, None)
            return
        if not ok:
            self.connect_done.emit(self.server_name, False, message, None, None, None)
            return

        profile = None
        try:
            profile = HostProfile(
                gpu=self.manager.detect_gpu(self.server_name),
                total_ram_mb=self.manager.total_ram_mb(self.server_name),
            )
        except Exception:
            logger.debug("Could not read host profile for %s",
                         self.server_name, exc_info=True)

        models = []
        try:
            for directory in self.manager.get_server_model_directories(self.server_name):
                models.extend(
                    self.manager.scan_server_models(self.server_name, directory))
        except Exception:
            logger.debug("Model scan failed for %s", self.server_name, exc_info=True)
            models = None

        # Connecting is also when we can finally tell whether the configured
        # llama-server actually exists on this machine.
        binary_status = None
        try:
            conn = self.manager.get_connection(self.server_name)
            configured = conn.config.llama_server_path if conn else ""
            runnable = bool(self.manager.resolve_executable(self.server_name, configured))
            found = "" if runnable else self.manager.find_server_binary(self.server_name)
            binary_status = (runnable, found)
        except Exception:
            logger.debug("Binary check failed for %s", self.server_name, exc_info=True)

        self.connect_done.emit(self.server_name, True, message, profile, models,
                               binary_status)


class UsageSampleWorker(QThread):
    """Sample live utilization from one server."""

    sample_ready = Signal(str, object)

    def __init__(self, manager, server_name):
        super().__init__()
        self.manager = manager
        self.server_name = server_name

    def run(self):
        try:
            sample = self.manager.sample_usage(self.server_name)
        except Exception:
            logger.debug("Usage sample failed for %s", self.server_name, exc_info=True)
            sample = None
        self.sample_ready.emit(self.server_name, sample)


class ModelScanWorker(QThread):
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
                    logger.debug(
                        "Model scan failed for %s on %s", dir_path, self.server_name,
                        exc_info=True,
                    )
        self.scan_done.emit(self.server_name, results)


class PresetStyleWorker(QThread):
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
            results.append((name, False))
        self.style_done.emit(results)


class GpuDetectWorker(QThread):
    """Detect GPU info in a background thread."""

    detect_done = Signal(str, object)

    def run(self):
        import traceback

        gpu_info_dict = {}
        try:
            gpu_info = detect_gpu()
            gpu_info_dict = {
                "backend": gpu_info.backend,
                "device_name": gpu_info.device_name,
                "vram_mb": gpu_info.vram_mb,
                "available_vram_mb": gpu_info.available_vram_mb,
                "driver_version": gpu_info.driver_version,
                "supported": gpu_info.supported,
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
                        ["nvidia-smi",
                         "--query-gpu=driver_version,name,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu,utilization.memory,power.draw,power.limit",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if nvidia_sm.returncode == 0 and nvidia_sm.stdout.strip():
                        parts = [p.strip() for p in nvidia_sm.stdout.strip().split(",")[0].replace("\n", ",").split(",")]
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


class GpuStatsWorker(QThread):
    """Lightweight worker to fetch live GPU stats (temp, util, power) only."""

    stats_done = Signal(str)

    def run(self):
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")[0].replace("\n", ",").split(",")]
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


def _read_cpu_times() -> tuple:
    """Return (total, idle) jiffies from /proc/stat."""
    with open("/proc/stat", "r") as f:
        parts = f.readline().split()
    vals = [float(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
    return sum(vals), idle


def _sample_cpu_percent(interval: float = 0.25) -> float:
    """Sample aggregate CPU utilization over ``interval`` seconds. -1 if N/A."""
    try:
        import psutil  # optional dependency
        return float(psutil.cpu_percent(interval=interval))
    except ImportError:
        pass
    except Exception:
        return -1.0
    if platform.system() == "Linux":
        try:
            total1, idle1 = _read_cpu_times()
            time.sleep(interval)
            total2, idle2 = _read_cpu_times()
            d_total, d_idle = total2 - total1, idle2 - idle1
            if d_total > 0:
                return max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0))
        except Exception:
            logger.debug("CPU sample failed", exc_info=True)
    return -1.0


def _sample_gpu() -> tuple:
    """Sample GPU utilization and memory via nvidia-smi.

    Returns (util_pct, mem_used_mb, mem_total_mb); all -1 when unavailable.
    Multi-GPU: util is the max across GPUs, memory is summed.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            utils, useds, totals = [], [], []
            for line in result.stdout.strip().split("\n"):
                vals = [v.strip() for v in line.split(",")]
                if len(vals) >= 3:
                    try:
                        utils.append(float(vals[0]))
                        useds.append(float(vals[1]))
                        totals.append(float(vals[2]))
                    except ValueError:
                        continue
            if utils:
                return max(utils), sum(useds), sum(totals)
    except Exception:
        logger.debug("nvidia-smi live sample failed", exc_info=True)
    return -1.0, -1.0, -1.0


class RealtimeStatsWorker(QThread):
    """Samples local CPU% and GPU utilization/memory for the live usage bars.

    Runs briefly (~0.3s for the CPU sampling interval) and is re-launched
    on a timer by the main window.
    """

    stats_ready = Signal(float, float, float, float)

    def run(self):
        cpu_pct = _sample_cpu_percent()
        gpu_pct, gpu_used_mb, gpu_total_mb = _sample_gpu()
        self.stats_ready.emit(cpu_pct, gpu_pct, gpu_used_mb, gpu_total_mb)


class EstimateWorker(QThread):
    """Compute a resource estimate in a background thread.

    Reads GGUF metadata (cached) and file sizes off the GUI thread so typing
    in any field never stalls the UI.
    """

    estimate_done = Signal(object)

    def __init__(self, estimator, config_snapshot, gpu_info,
                 model_path, draft_path, mmproj_path, own_pids=(),
                 vram_override=None, system_ram_mb: float = -1.0,
                 known_sizes=None):
        super().__init__()
        self._estimator = estimator
        self._config = config_snapshot
        self._gpu_info = gpu_info
        self._model_path = model_path
        self._draft_path = draft_path
        self._mmproj_path = mmproj_path
        self._own_pids = tuple(own_pids)
        #: (total_mb, available_mb) measured on a remote host, if that is where
        #: the model will run. None means "sample this machine".
        self._vram_override = vram_override
        self._system_ram_mb = system_ram_mb
        #: path -> size in MB, from the model scan. A model on a remote host
        #: cannot be stat'ed here, so without this there is no estimate at all.
        self._known_sizes = dict(known_sizes or {})

    def _size_mb(self, path: str) -> float:
        if not path:
            return 0.0
        local_size = get_file_size_mb(path)
        if local_size > 0:
            return local_size
        return float(self._known_sizes.get(path, 0.0))

    def _live_gpu_info(self):
        """gpu_info refreshed with a current VRAM reading.

        Detection happens once per connect, so its free-memory figure goes stale
        as soon as anything loads a model. For a remote server the caller passes
        figures from that host; otherwise this machine is sampled here (already
        off the GUI thread).
        """
        gpu_info = self._gpu_info
        if gpu_info is None or not gpu_info.vram_mb:
            return gpu_info

        if self._vram_override is not None:
            total_mb, available_mb = self._vram_override
            total = int(total_mb) or gpu_info.vram_mb
        else:
            sample = sample_vram(self._own_pids)
            if not sample.valid:
                return gpu_info
            total = int(sample.total_mb) or gpu_info.vram_mb
            available_mb = sample.available_mb

        refreshed = replace(
            gpu_info,
            vram_mb=total,
            available_vram_mb=int(available_mb),
        )
        # Per-GPU breakdowns come from detection; leave them as they are.
        refreshed.gpu_infos = gpu_info.gpu_infos
        return refreshed

    def run(self):
        try:
            model_size_mb = self._size_mb(self._model_path)
            if model_size_mb <= 0:
                self.estimate_done.emit(None)
                return

            draft_size_mb = self._size_mb(self._draft_path)
            mmproj_size_mb = self._size_mb(self._mmproj_path)
            family, quant = infer_model_hints(self._model_path)
            quant = quant or ""

            estimate = self._estimator.calculate(
                config=self._config,
                gpu_info=self._live_gpu_info(),
                model_size_mb=model_size_mb,
                model_family=family,
                quant_type=quant,
                model_path=self._model_path,
                draft_model_size_mb=draft_size_mb,
                mmproj_model_size_mb=mmproj_size_mb,
                system_ram_mb=self._system_ram_mb,
            )
            self.estimate_done.emit(estimate)
        except Exception:
            logger.exception("Estimate worker failed")
            self.estimate_done.emit(None)
