"""GPU Auto-Detection Module."""
import logging
from dataclasses import dataclass
import subprocess
import sys
import re
from typing import Optional
from .probe import build_full_probe, build_live_probe, parse_gpu_probe
from .types import GPUInfo

logger = logging.getLogger(__name__)


def detect_gpu() -> GPUInfo:
    """Detect the GPU(s) of this machine and the llama.cpp backend for them.

    On POSIX hosts one shell probe covers NVIDIA (nvidia-smi), AMD (amd-smi,
    rocm-smi, or the amdgpu sysfs files when ROCm is not installed) and
    Vulkan-only devices; it is the same probe that describes remote hosts.
    Windows falls back to calling nvidia-smi directly.
    """
    if sys.platform == 'darwin':
        metal_info = _detect_metal()
        if metal_info:
            return metal_info

    if sys.platform != 'win32':
        info = run_gpu_probe()
        if info:
            return info
    else:
        cuda_info = _detect_cuda()
        if cuda_info:
            return cuda_info
        vulkan_info = _detect_vulkan()
        if vulkan_info:
            return vulkan_info

    return GPUInfo(backend="cpu", vram_mb=0, available_vram_mb=0, device_name="CPU", supported=True)


def run_gpu_probe(timeout: float = 20.0) -> Optional[GPUInfo]:
    """Run the full GPU probe locally; None when nothing usable was found."""
    try:
        result = subprocess.run(["sh", "-c", build_full_probe()],
                                capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        logger.debug("GPU probe failed", exc_info=True)
        return None
    return parse_gpu_probe(result.stdout, result.stderr)


#: The nvidia-smi query whose output parse_nvidia_smi_gpus() understands. Shared
#: so the local and remote detectors ask the same question.
NVIDIA_SMI_QUERY = [
    'nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
    '--format=csv,noheader',
]


def parse_nvidia_smi_gpus(csv_text: str) -> Optional[GPUInfo]:
    """Parse ``NVIDIA_SMI_QUERY`` output into an aggregate GPUInfo.

    Pure text in, GPUInfo out: the same output parses identically whether it came
    from this machine or from a remote host over SSH. Returns None when there is
    nothing usable -- including nvidia-smi's error text, which it prints on
    stdout on a driver/library mismatch.
    """
    text = (csv_text or "").strip()
    if not text:
        return None

    gpu_infos = []
    total_vram = 0
    total_available = 0

    for i, line in enumerate(text.split('\n')):
        parts = [p.strip() for p in line.split(',')]
        # An error line ("Failed to initialize NVML: ...") has no comma-separated
        # memory figures; skip anything that does not look like a GPU row.
        if len(parts) < 2:
            continue

        device_name = parts[0] or f"NVIDIA GPU {i}"
        vram_mb = 0
        available_vram_mb = 0
        driver_version = None

        try:
            vram_mb = int(parts[1].replace('MiB', '').strip())
        except ValueError:
            continue        # not a GPU row after all

        if len(parts) >= 3:
            try:
                available_vram_mb = int(parts[2].replace('MiB', '').strip())
            except ValueError:
                logger.debug("Unparseable free memory in %r", line)

        if len(parts) >= 4:
            driver_version = parts[3].strip()

        gpu_infos.append(GPUInfo(
            backend="cuda",
            vram_mb=vram_mb,
            available_vram_mb=available_vram_mb,
            device_name=device_name,
            driver_version=driver_version,
            supported=True,
        ))
        total_vram += vram_mb
        total_available += available_vram_mb

    if not gpu_infos:
        return None

    first_gpu = gpu_infos[0]
    return GPUInfo(
        backend="cuda",
        vram_mb=total_vram,
        available_vram_mb=total_available,
        device_name=first_gpu.device_name,
        driver_version=first_gpu.driver_version,
        supported=True,
        gpu_count=len(gpu_infos),
        gpu_infos=gpu_infos,
    )


def _detect_cuda() -> Optional[GPUInfo]:
    """Detect NVIDIA CUDA GPU(s) via nvidia-smi."""
    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return parse_nvidia_smi_gpus(result.stdout)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.debug("Suppressed exception", exc_info=True)
    return None


def _detect_metal() -> Optional[GPUInfo]:
    """Detect Apple Metal GPU."""
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if 'metal' in output or 'gpu' in output:
                device_name = "Apple GPU"
                for line in result.stdout.split('\n'):
                    if 'Chipset Model:' in line:
                        device_name = line.split(':')[-1].strip()
                        break

                # Estimate VRAM (unified memory)
                vram_mb = 8192
                available_vram_mb = vram_mb // 2
                try:
                    mem_result = subprocess.run(
                        ['sysctl', 'hw.memsize'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if mem_result.returncode == 0:
                        mem_bytes = int(mem_result.stdout.split(':')[1].strip())
                        vram_mb = mem_bytes // (1024 * 1024)
                        available_vram_mb = int(vram_mb * 0.6)
                except Exception:
                    logger.debug("Suppressed exception", exc_info=True)

                return GPUInfo(
                    backend="metal",
                    vram_mb=vram_mb,
                    available_vram_mb=available_vram_mb,
                    device_name=device_name,
                    supported=True,
                )
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.debug("Suppressed exception", exc_info=True)
    return None


def _detect_vulkan() -> Optional[GPUInfo]:
    """Detect Vulkan GPU(s)."""
    try:
        result = subprocess.run(
            ['vulkaninfo', '--summary'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse device names from vulkaninfo output
            device_names = []
            for line in result.stdout.split('\n'):
                if 'deviceName' in line or 'device name:' in line:
                    name = line.split(':')[-1].strip()
                    if name:
                        device_names.append(name)

            if not device_names:
                device_names = ["Vulkan GPU"]

            gpu_count = len(device_names)
            if gpu_count == 0:
                gpu_count = 1

            # Try to get VRAM info via nvidia-smi if available (common on Linux)
            vram_mb = 0
            available_vram_mb = 0
            try:
                nv_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total,memory.free',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=3
                )
                if nv_result.returncode == 0:
                    parts = [p.strip() for p in nv_result.stdout.strip().split(',')[0:2]]
                    if len(parts) >= 1:
                        try:
                            vram_mb = int(parts[0].replace('MiB', '').strip())
                        except ValueError:
                            logger.debug("Suppressed exception", exc_info=True)
                    if len(parts) >= 2:
                        try:
                            available_vram_mb = int(parts[1].replace('MiB', '').strip())
                        except ValueError:
                            logger.debug("Suppressed exception", exc_info=True)
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)

            return GPUInfo(
                backend="vulkan",
                vram_mb=vram_mb,
                available_vram_mb=available_vram_mb or vram_mb,
                device_name=device_names[0] if device_names else "Vulkan GPU",
                supported=True,
                gpu_count=max(gpu_count, 1),
            )
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.debug("Suppressed exception", exc_info=True)

    # Fallback: check for Vulkan device node
    try:
        if sys.platform == 'linux':
            import os
            if os.path.exists('/dev/dri/renderD128'):
                return GPUInfo(
                    backend="vulkan",
                    vram_mb=0,
                    available_vram_mb=0,
                    device_name="Linux Vulkan GPU",
                    supported=True,
                )
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)

    return None


@dataclass
class VramSample:
    """A live VRAM reading, unlike the snapshot taken at detection time."""

    total_mb: float = 0.0
    free_mb: float = 0.0
    #: VRAM currently held by the launcher's own llama-server instance(s).
    own_used_mb: float = 0.0
    valid: bool = False

    @property
    def available_mb(self) -> float:
        """VRAM a new configuration could actually use.

        Free memory *plus* whatever our own server already holds: reconfiguring
        a running server replaces that allocation rather than adding to it, so
        counting it as unavailable would make the model look like it no longer
        fits the card it is currently loaded on.
        """
        return self.free_mb + self.own_used_mb


def sample_vram(own_pids=()) -> VramSample:
    """Read current GPU memory, attributing what our own processes hold.

    Returns an invalid sample when there is no usable GPU, so callers can fall
    back to the totals from detection.
    """
    try:
        result = subprocess.run(["sh", "-c", build_live_probe(own_pids)],
                                capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        logger.debug("VRAM sample failed", exc_info=True)
        return VramSample()
    return vram_sample_from_gpu(parse_gpu_probe(result.stdout), own_pids)


def vram_sample_from_gpu(gpu: Optional[GPUInfo], own_pids=()) -> VramSample:
    """Derive a VramSample from a live GPU reading."""
    if gpu is None or gpu.vram_mb <= 0 or gpu.used_vram_mb < 0:
        return VramSample()
    wanted = {int(p) for p in own_pids if p}
    own = sum(p.vram_mb for p in gpu.processes if p.pid in wanted)
    return VramSample(total_mb=float(gpu.vram_mb), free_mb=float(gpu.vram_mb - gpu.used_vram_mb),
                      own_used_mb=own, valid=True)
