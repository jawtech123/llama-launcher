"""Background worker threads for the GUI.

All blocking operations (connecting, model scanning, usage sampling, resource
estimation) run in these QThread workers so the UI never freezes. Workers are started via
``LlamaLauncherGUI._start_worker`` which parents them and frees them on finish.
"""
import logging
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from llama_launcher.gpu.detector import sample_vram
from llama_launcher.models.scanner import infer_model_hints, scan_models
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
                draft_model_path=self._draft_path,
                mmproj_model_path=self._mmproj_path,
            )
            self.estimate_done.emit(estimate)
        except Exception:
            logger.exception("Estimate worker failed")
            self.estimate_done.emit(None)
