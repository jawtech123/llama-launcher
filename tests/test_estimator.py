"""Tests for the GGUF metadata cache."""
import os

from llama_launcher.estimator.estimator import _GgufReader, _METADATA_CACHE


def _make_fake_gguf(path) -> str:
    """Write a minimal file that starts with the GGUF magic + version."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write((3).to_bytes(4, "little"))
        f.write(b"\x00" * 64)
    return str(path)


def test_metadata_cache_returns_same_object_on_repeat(tmp_path):
    model = _make_fake_gguf(tmp_path / "model-a.gguf")
    m1 = _GgufReader.read_full_metadata(model)
    m2 = _GgufReader.read_full_metadata(model)
    assert m1 is m2  # second call served from cache


def test_metadata_cache_invalidates_on_file_change(tmp_path):
    model = _make_fake_gguf(tmp_path / "model-b.gguf")
    m1 = _GgufReader.read_full_metadata(model)
    # Rewrite with different content/size -> cache key changes
    with open(model, "ab") as f:
        f.write(b"\x00" * 128)
    m2 = _GgufReader.read_full_metadata(model)
    assert m1 is not m2


def test_metadata_cache_handles_missing_file(tmp_path):
    meta = _GgufReader.read_full_metadata(str(tmp_path / "nope.gguf"))
    assert meta.arch == "unknown"
    assert str(tmp_path / "nope.gguf") not in [
        k[0] for k in _METADATA_CACHE.keys()
    ]


# ---------------------------------------------------------------------------
# VRAM accounting
# ---------------------------------------------------------------------------

from llama_launcher.core.config import LlamaConfig
from llama_launcher.estimator import ResourceEstimator
from llama_launcher.gpu.types import GPUInfo


def _cuda_estimate(flash_attn="auto", available_mb=None):
    gpu = GPUInfo(
        backend="cuda",
        vram_mb=32607,
        available_vram_mb=available_mb if available_mb is not None else 32607,
        device_name="Test GPU",
    )
    cfg = LlamaConfig(
        model_path="",  # no GGUF read: architecture falls back to size estimate
        context_size=4096,
        gpu_layers=-1,
        kv_offload=True,
        flash_attn=flash_attn,
        batch_size=2048,
        ubatch_size=512,
    )
    est = ResourceEstimator().calculate(
        config=cfg, gpu_info=gpu, model_size_mb=10000.0,
    )
    return est


def test_backend_overhead_not_double_counted():
    """CUDA runtime + flash workspace must appear exactly once in the total."""
    est = _cuda_estimate(flash_attn="auto")
    flash_workspace_mb = (512 * 5120 * 4.0) / (1024 * 1024)
    expected = (
        est.gpu_weights_mb
        + est.kv_cache_gpu_mb
        + est.compute_buffers_gpu_mb
        + ResourceEstimator.CUDA_RUNTIME_MB
        + flash_workspace_mb
    )
    assert abs(est.total_vram_mb - expected) < 0.01


def test_no_flash_workspace_when_flash_attn_off():
    est = _cuda_estimate(flash_attn="off")
    expected = (
        est.gpu_weights_mb
        + est.kv_cache_gpu_mb
        + est.compute_buffers_gpu_mb
        + ResourceEstimator.CUDA_RUNTIME_MB
    )
    assert abs(est.total_vram_mb - expected) < 0.01


def test_remaining_vram_is_total_minus_used():
    """Remaining must match the used/total bar, not a stale free snapshot."""
    est = _cuda_estimate()
    assert abs(est.gpu_remaining_vram_mb
               - (est.gpu_total_vram_mb - est.total_vram_mb)) < 0.01


def test_remaining_vram_ignores_stale_available_snapshot():
    """A low free-memory snapshot (e.g. a server already running) must not
    leak into the Remaining figure."""
    est = _cuda_estimate(available_mb=9600)
    assert abs(est.gpu_remaining_vram_mb
               - (est.gpu_total_vram_mb - est.total_vram_mb)) < 0.01
