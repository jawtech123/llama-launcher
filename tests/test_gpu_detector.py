"""Tests for GPU auto-detection."""
import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from llama_launcher.gpu.detector import detect_gpu, _detect_cuda, _detect_metal, _detect_vulkan
from llama_launcher.gpu.types import GPUInfo


def _make_smi_output(lines):
    """Create nvidia-smi style CSV output."""
    return "\n".join(lines) + "\n"


class TestDetectCuda:
    """Test CUDA detection."""

    def test_detect_cuda_single_gpu(self):
        output = _make_smi_output(["NVIDIA GeForce RTX 4090, 24576 MiB, 20000 MiB, 535.129"])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            info = _detect_cuda()
        assert info is not None
        assert info.backend == "cuda"
        assert info.vram_mb == 24576
        assert info.available_vram_mb == 20000
        assert info.device_name == "NVIDIA GeForce RTX 4090"
        assert info.driver_version == "535.129"
        assert info.gpu_count == 1

    def test_detect_cuda_multi_gpu(self):
        output = _make_smi_output([
            "GPU 0, 12000 MiB, 10000 MiB, 535.129",
            "GPU 1, 12000 MiB, 11000 MiB, 535.129",
        ])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            info = _detect_cuda()
        assert info is not None
        assert info.backend == "cuda"
        assert info.vram_mb == 24000  # total
        assert info.available_vram_mb == 21000  # total available
        assert info.gpu_count == 2
        assert len(info.gpu_infos) == 2

    def test_detect_cuda_no_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            info = _detect_cuda()
        assert info is None

    def test_detect_cuda_failed_command(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            info = _detect_cuda()
        assert info is None

    def test_detect_cuda_subprocess_error(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.SubprocessError("boom")
            info = _detect_cuda()
        assert info is None

    def test_detect_cuda_file_not_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            info = _detect_cuda()
        assert info is None


class TestDetectMetal:
    """Test Metal GPU detection (macOS only)."""

    def test_detect_metal_on_darwin(self):
        output = "Chipset Model: Apple M2\nMetal Support: yes"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            with patch("sys.platform", "darwin"):
                info = _detect_metal()
        assert info is not None
        assert info.backend == "metal"
        assert "M2" in info.device_name
        assert info.vram_mb > 0

    def test_detect_metal_not_on_linux(self):
        with patch("subprocess.run") as mock_run:
            with patch("sys.platform", "linux"):
                info = _detect_metal()
        assert info is None

    def test_detect_metal_subprocess_error(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.SubprocessError()
            with patch("sys.platform", "darwin"):
                info = _detect_metal()
        assert info is None


class TestDetectVulkan:
    """Test Vulkan GPU detection."""

    def test_detect_vulkan_success(self):
        output = "deviceName: NVIDIA RTX 3080\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            info = _detect_vulkan()
        assert info is not None
        assert info.backend == "vulkan"
        assert "RTX 3080" in info.device_name

    def test_detect_vulkan_no_devices(self):
        output = ""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            info = _detect_vulkan()
        assert info is not None
        assert info.backend == "vulkan"
        assert info.device_name == "Vulkan GPU"

    def test_detect_vulkan_subprocess_error(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.SubprocessError()
            info = _detect_vulkan()
        # Should fall through to renderD128 check or return None
        assert info is None or info.backend == "vulkan"


class TestDetectGpuFallback:
    """Test GPU detection fallback chain."""

    def test_detect_gpu_cuda_priority(self):
        with patch("llama_launcher.gpu.detector._detect_cuda") as mock_cuda:
            mock_cuda.return_value = GPUInfo(backend="cuda", vram_mb=12000)
            with patch("llama_launcher.gpu.detector._detect_metal") as mock_metal:
                with patch("llama_launcher.gpu.detector._detect_vulkan") as mock_vulkan:
                    info = detect_gpu()
            assert info.backend == "cuda"
            mock_metal.assert_not_called()
            mock_vulkan.assert_not_called()

    def test_detect_gpu_fallback_to_cpu(self):
        with patch("llama_launcher.gpu.detector._detect_cuda") as mock_cuda, \
                patch("llama_launcher.gpu.detector._detect_metal") as mock_metal, \
                patch("llama_launcher.gpu.detector._detect_vulkan") as mock_vulkan, \
                patch("llama_launcher.gpu.detector.run_gpu_probe") as mock_probe:
                    mock_cuda.return_value = None
                    mock_metal.return_value = None
                    mock_vulkan.return_value = None
                    mock_probe.return_value = None
                    info = detect_gpu()
        assert info.backend == "cpu"
        assert info.vram_mb == 0
        assert info.device_name == "CPU"
        assert info.supported is True


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_gpu_info_default_available_vram(self):
        info = GPUInfo(backend="cuda", vram_mb=12000)
        assert info.available_vram_mb == 12000  # defaults to total when 0

    def test_gpu_info_custom_available_vram(self):
        info = GPUInfo(backend="cuda", vram_mb=12000, available_vram_mb=8000)
        assert info.available_vram_mb == 8000

    def test_gpu_info_all_fields(self):
        info = GPUInfo(
            backend="cuda",
            vram_mb=24000,
            available_vram_mb=20000,
            device_name="RTX 4090",
            driver_version="535.129",
            supported=True,
            gpu_count=1,
        )
        assert info.backend == "cuda"
        assert info.vram_mb == 24000
        assert info.available_vram_mb == 20000
        assert info.device_name == "RTX 4090"
        assert info.driver_version == "535.129"
        assert info.supported is True
        assert info.gpu_count == 1

    def test_gpu_info_with_nested_gpus(self):
        gpu1 = GPUInfo(backend="cuda", vram_mb=12000, device_name="GPU 0")
        gpu2 = GPUInfo(backend="cuda", vram_mb=12000, device_name="GPU 1")
        aggregate = GPUInfo(
            backend="cuda", vram_mb=24000, available_vram_mb=20000,
            gpu_count=2, gpu_infos=[gpu1, gpu2],
        )
        assert aggregate.gpu_count == 2
        assert len(aggregate.gpu_infos) == 2


class TestAllBackends:
    """Test all backend types."""

    def test_all_backends_valid(self):
        for backend in ("cuda", "metal", "vulkan", "cpu"):
            info = GPUInfo(backend=backend, vram_mb=1000)
            assert info.backend == backend
