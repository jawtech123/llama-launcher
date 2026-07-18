"""Background worker threads for the GUI.

All blocking operations (model scanning, GPU detection, preset style checks)
run in these QThread workers so the UI never freezes. Workers are started via
``LlamaLauncherGUI._start_worker`` which parents them and frees them on finish.
"""
import logging
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from llama_launcher.core.presets import PresetManager
from llama_launcher.gpu.detector import detect_gpu
from llama_launcher.models.scanner import scan_models
from llama_launcher.remote.types import ConnectionState

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
