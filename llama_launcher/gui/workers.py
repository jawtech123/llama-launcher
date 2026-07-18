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

from llama_launcher.core.presets import PresetManager
from llama_launcher.gpu.detector import detect_gpu
from llama_launcher.models.scanner import scan_models
from llama_launcher.remote.types import ConnectionState
from llama_launcher.utils.helpers import get_file_size_mb

logger = logging.getLogger(__name__)


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
            target = getattr(preset_config, "target_server", None)
            offline = False
            if target:
                state = self.manager.get_connection_state(target)
                offline = state in (ConnectionState.DISCONNECTED, ConnectionState.ERROR)
            results.append((name, offline))
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


_QUANT_PATTERNS = (
    ("q8_0", "Q8_0"), ("q6_k", "Q6_K"), ("q5_k_m", "Q5_K_M"),
    ("q5_0", "Q5_0"), ("q5_1", "Q5_1"), ("q4_k_m", "Q4_K_M"),
    ("q4_0", "Q4_0"), ("q4_1", "Q4_1"), ("q3_k_m", "Q3_K_M"),
    ("q3_k_s", "Q3_K_S"), ("q2_k", "Q2_K"),
)

_FAMILY_HINTS = (
    ("llama", "llama"), ("llm", "llama"), ("mistral", "mistral"),
    ("qwen", "qwen"), ("qwq", "qwen"), ("gemma", "gemma"),
    ("phi", "phi"), ("codellama", "codellama"),
)


def infer_model_hints(model_path: str) -> tuple:
    """Infer (family, quant_type) from a model file name."""
    name_lower = model_path.lower()
    family = "unknown"
    for hint, family_name in _FAMILY_HINTS:
        if hint in name_lower:
            family = family_name
            break
    quant = ""
    for pattern, quant_name in _QUANT_PATTERNS:
        if pattern in name_lower:
            quant = quant_name
            break
    return family, quant


class EstimateWorker(QThread):
    """Compute a resource estimate in a background thread.

    Reads GGUF metadata (cached) and file sizes off the GUI thread so typing
    in any field never stalls the UI.
    """

    estimate_done = Signal(object)

    def __init__(self, estimator, config_snapshot, gpu_info,
                 model_path, draft_path, mmproj_path):
        super().__init__()
        self._estimator = estimator
        self._config = config_snapshot
        self._gpu_info = gpu_info
        self._model_path = model_path
        self._draft_path = draft_path
        self._mmproj_path = mmproj_path

    def run(self):
        try:
            model_size_mb = get_file_size_mb(self._model_path)
            if model_size_mb <= 0:
                self.estimate_done.emit(None)
                return

            draft_size_mb = get_file_size_mb(self._draft_path) if self._draft_path else 0.0
            mmproj_size_mb = get_file_size_mb(self._mmproj_path) if self._mmproj_path else 0.0
            family, quant = infer_model_hints(self._model_path)

            estimate = self._estimator.calculate(
                config=self._config,
                gpu_info=self._gpu_info,
                model_size_mb=model_size_mb,
                model_family=family,
                quant_type=quant,
                model_path=self._model_path,
                draft_model_size_mb=draft_size_mb,
                mmproj_model_size_mb=mmproj_size_mb,
            )
            self.estimate_done.emit(estimate)
        except Exception:
            logger.exception("Estimate worker failed")
            self.estimate_done.emit(None)
