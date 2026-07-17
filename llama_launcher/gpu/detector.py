"""GPU Auto-Detection Module."""
import logging
import subprocess
import sys
import re
from typing import Optional
from .types import GPUInfo

logger = logging.getLogger(__name__)


def detect_gpu() -> GPUInfo:
    """Detect available GPU and backend.

    For CUDA, detects all GPUs and returns a GPUInfo with nested gpu_infos.
    """
    # Check for CUDA (NVIDIA)
    cuda_info = _detect_cuda()
    if cuda_info:
        return cuda_info

    # Check for Metal (Apple macOS)
    if sys.platform == 'darwin':
        metal_info = _detect_metal()
        if metal_info:
            return metal_info

    # Check for Vulkan
    vulkan_info = _detect_vulkan()
    if vulkan_info:
        return vulkan_info

    # Fallback to CPU
    return GPUInfo(backend="cpu", vram_mb=0, available_vram_mb=0, device_name="CPU", supported=True)


def _detect_cuda() -> Optional[GPUInfo]:
    """Detect NVIDIA CUDA GPU(s) via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            gpu_count = len(lines)

            # Parse all GPUs
            gpu_infos = []
            total_vram = 0
            total_available = 0

            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]

                device_name = parts[0] if len(parts) > 0 else f"NVIDIA GPU {i}"
                vram_mb = 0
                available_vram_mb = 0
                driver_version = None

                if len(parts) >= 2:
                    try:
                        vram_str = parts[1].replace('MiB', '').strip()
                        vram_mb = int(vram_str)
                    except ValueError:
                        logger.debug("Suppressed exception", exc_info=True)

                if len(parts) >= 3:
                    try:
                        free_str = parts[2].replace('MiB', '').strip()
                        available_vram_mb = int(free_str)
                    except ValueError:
                        logger.debug("Suppressed exception", exc_info=True)

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

            # Return aggregate GPUInfo with nested details
            first_gpu = gpu_infos[0]
            return GPUInfo(
                backend="cuda",
                vram_mb=total_vram,
                available_vram_mb=total_available,
                device_name=first_gpu.device_name,
                driver_version=first_gpu.driver_version,
                supported=True,
                gpu_count=gpu_count,
                gpu_infos=gpu_infos,
            )
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
