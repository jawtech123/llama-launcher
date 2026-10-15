"""Tests for the GGUF metadata cache and VRAM estimation."""
import os
from unittest.mock import patch, MagicMock

from llama_launcher.estimator.estimator import _GgufReader, _METADATA_CACHE
from llama_launcher.core.config import LlamaConfig
from llama_launcher.estimator import ResourceEstimator
from llama_launcher.gpu.types import GPUInfo


def _make_fake_gguf(path) -> str:
    """Write a minimal file that starts with the GGUF magic + version."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write((3).to_bytes(4, "little"))
        f.write(b"\x00" * 64)
    return str(path)


class TestMetadataCache:
    """Test GGUF metadata caching."""

    def test_metadata_cache_returns_same_object_on_repeat(self, tmp_path):
        model = _make_fake_gguf(tmp_path / "model-a.gguf")
        m1 = _GgufReader.read_full_metadata(model)
        m2 = _GgufReader.read_full_metadata(model)
        assert m1 is m2

    def test_metadata_cache_invalidates_on_file_change(self, tmp_path):
        model = _make_fake_gguf(tmp_path / "model-b.gguf")
        m1 = _GgufReader.read_full_metadata(model)
        with open(model, "ab") as f:
            f.write(b"\x00" * 128)
        m2 = _GgufReader.read_full_metadata(model)
        assert m1 is not m2

    def test_metadata_cache_handles_missing_file(self, tmp_path):
        meta = _GgufReader.read_full_metadata(str(tmp_path / "nope.gguf"))
        assert meta.arch == "unknown"
        assert str(tmp_path / "nope.gguf") not in [
            k[0] for k in _METADATA_CACHE.keys()
        ]

    def test_metadata_cache_isolated_per_file(self, tmp_path):
        m1 = _make_fake_gguf(tmp_path / "file1.gguf")
        m2 = _make_fake_gguf(tmp_path / "file2.gguf")
        meta1 = _GgufReader.read_full_metadata(m1)
        meta2 = _GgufReader.read_full_metadata(m2)
        assert meta1 is not meta2
        # Cache should have both
        cache_keys = [k[0] for k in _METADATA_CACHE.keys()]
        assert m1 in cache_keys or len(_METADATA_CACHE) >= 2

    def test_read_full_metadata_returns_dict_like(self, tmp_path):
        model = _make_fake_gguf(tmp_path / "meta.gguf")
        meta = _GgufReader.read_full_metadata(model)
        assert hasattr(meta, "arch")
        assert meta.arch in ("unknown", "llama", "gpt2", "mistral", "qwen2")


class TestResourceEstimator:
    """Test VRAM resource estimation."""

    def _cuda_estimate(self, flash_attn="auto", available_mb=None, model_size_mb=10000.0):
        gpu = GPUInfo(
            backend="cuda",
            vram_mb=32607,
            available_vram_mb=available_mb if available_mb is not None else 32607,
            device_name="Test GPU",
        )
        cfg = LlamaConfig(
            model_path="",
            context_size=4096,
            gpu_layers=-1,
            kv_offload=True,
            flash_attn=flash_attn,
            batch_size=2048,
            ubatch_size=512,
        )
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=model_size_mb,
        )
        return est

    def test_total_vram_is_sum_of_breakdown(self):
        est = self._cuda_estimate(flash_attn="auto")
        bd = est.vram_breakdown
        expected = (bd.gpu_weights_mb + bd.kv_cache_mb + bd.inference_buffers_mb
                    + bd.cuda_runtime_mb + bd.temp_allocations_mb + bd.draft_model_mb
                    + bd.mmproj_model_mb + bd.backend_overhead_mb + bd.safety_margin_mb)
        assert abs(est.total_vram_mb - expected) < 0.01
        assert bd.cuda_runtime_mb == ResourceEstimator.CUDA_RUNTIME_MB

    def test_flash_attention_off_needs_much_bigger_compute_buffer(self):
        # Without flash attention the KQ score matrix (ubatch x ctx x heads, f32)
        # is materialised in the compute buffer.
        on = self._cuda_estimate(flash_attn="on")
        off = self._cuda_estimate(flash_attn="off")
        assert off.compute_buffers_gpu_mb > on.compute_buffers_gpu_mb * 3
        assert off.gpu_weights_mb == on.gpu_weights_mb

    def test_remaining_vram_is_available_minus_used(self):
        # On an idle card available == total, so this is also total - used.
        est = self._cuda_estimate()
        assert abs(est.gpu_remaining_vram_mb
                   - (est.gpu_available_vram_mb - est.total_vram_mb)) < 0.01

    def test_remaining_vram_accounts_for_memory_held_by_others(self):
        # Measuring against total would claim room that cannot be allocated.
        est = self._cuda_estimate(available_mb=9600)
        assert abs(est.gpu_remaining_vram_mb - (9600 - est.total_vram_mb)) < 0.01
        assert est.gpu_remaining_vram_mb < est.gpu_total_vram_mb - est.total_vram_mb

    def test_estimator_has_cuda_runtime_constant(self):
        assert hasattr(ResourceEstimator, "CUDA_RUNTIME_MB")
        assert ResourceEstimator.CUDA_RUNTIME_MB > 0

    def test_estimator_fields_on_result(self):
        est = self._cuda_estimate()
        assert hasattr(est, "gpu_total_vram_mb")
        assert hasattr(est, "gpu_weights_mb")
        assert hasattr(est, "kv_cache_gpu_mb")
        assert hasattr(est, "compute_buffers_gpu_mb")
        assert hasattr(est, "kv_cache_cpu_mb")
        assert hasattr(est, "total_vram_mb")
        assert hasattr(est, "gpu_remaining_vram_mb")

    def test_small_model_estimate(self):
        est = self._cuda_estimate(model_size_mb=2000.0)
        assert est.gpu_weights_mb > 0
        assert est.total_vram_mb > 0
        assert est.gpu_remaining_vram_mb > 0

    def test_large_model_estimate(self):
        est = self._cuda_estimate(model_size_mb=50000.0)
        assert est.gpu_weights_mb > 0
        assert est.total_vram_mb > 0
        # Large model may exceed VRAM
        assert est.gpu_remaining_vram_mb <= est.gpu_total_vram_mb

    def test_context_size_affects_kv_cache(self):
        est_small = self._cuda_estimate(model_size_mb=1000.0)
        cfg = LlamaConfig(
            model_path="", context_size=8192, gpu_layers=-1,
            kv_offload=True, flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        gpu = GPUInfo(backend="cuda", vram_mb=32607, available_vram_mb=32607)
        est_large_ctx = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=1000.0,
        )
        # Larger context should use more VRAM
        assert est_large_ctx.kv_cache_gpu_mb >= est_small.kv_cache_gpu_mb

    def test_no_kv_offload_moves_cache_to_cpu(self):
        cfg = LlamaConfig(
            model_path="", context_size=4096, gpu_layers=-1,
            kv_offload=False,  # disable KV offload
            flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        gpu = GPUInfo(backend="cuda", vram_mb=32607, available_vram_mb=32607)
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=1000.0,
        )
        assert est.kv_cache_cpu_mb > 0

    def test_gpu_layers_zero_all_cpu(self):
        cfg = LlamaConfig(
            model_path="", context_size=4096, gpu_layers=0,
            kv_offload=True, flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        gpu = GPUInfo(backend="cuda", vram_mb=32607, available_vram_mb=32607)
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=1000.0,
        )
        # gpu_layers=0 means no GPU layers -> all weights on CPU
        assert est.gpu_weights_mb == 0
        assert est.cpu_weights_mb > 0

    def test_gpu_layers_all_negative(self):
        cfg = LlamaConfig(
            model_path="", context_size=4096, gpu_layers=-1,  # all layers to GPU
            kv_offload=True, flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        gpu = GPUInfo(backend="cuda", vram_mb=32607, available_vram_mb=32607)
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=1000.0,
        )
        assert est.gpu_weights_mb > 0


class TestEstimatorEdgeCases:
    """Test estimator with edge case inputs."""

    def test_zero_model_size(self):
        cfg = LlamaConfig(
            model_path="", context_size=4096, gpu_layers=-1,
            kv_offload=True, flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        gpu = GPUInfo(backend="cuda", vram_mb=32607, available_vram_mb=32607)
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=0.0,
        )
        assert est.total_vram_mb >= 0

    def test_zero_vram_gpu(self):
        gpu = GPUInfo(backend="cuda", vram_mb=0, available_vram_mb=0)
        cfg = LlamaConfig(
            model_path="", context_size=4096, gpu_layers=-1,
            kv_offload=True, flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=1000.0,
        )
        assert est.total_vram_mb >= 0

    def test_cpu_backend_estimate(self):
        gpu = GPUInfo(backend="cpu", vram_mb=0, available_vram_mb=0)
        cfg = LlamaConfig(
            model_path="", context_size=4096, gpu_layers=-1,
            kv_offload=True, flash_attn="auto",
            batch_size=2048, ubatch_size=512,
        )
        est = ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=1000.0,
        )
        # CPU backend should have minimal GPU estimates
        assert est.gpu_total_vram_mb == 0


class TestCoHostedVramWarning:
    """Warn when a model fits the card but not the VRAM currently free.

    This is the normal situation when running a second llama-server on a GPU
    that already has a model loaded.
    """

    def _estimate(self, total_mb, available_mb):
        gpu = GPUInfo(backend="cuda", device_name="Test GPU", vram_mb=total_mb,
                      available_vram_mb=available_mb, supported=True)
        cfg = LlamaConfig(model_path="/models/m.gguf", context_size=8192,
                          gpu_layers=-1)
        return ResourceEstimator().calculate(
            config=cfg, gpu_info=gpu, model_size_mb=22000.0,
            model_family="qwen", quant_type="Q6_K")

    def _contention_warnings(self, est):
        return [w for w in est.warnings if "held by other processes" in w]

    def test_warns_when_another_model_holds_the_vram(self):
        est = self._estimate(total_mb=32000, available_mb=8000)
        assert self._contention_warnings(est), est.warnings
        # It fits the card, but not right now -- the allocation would fail, so
        # this is flagged as over, not merely a caution.
        assert est.vram_exceeded
        assert est.gpu_remaining_vram_mb < 0

    def test_contention_message_differs_from_does_not_fit_at_all(self):
        contended = self._estimate(total_mb=32000, available_mb=8000)
        too_big = self._estimate(total_mb=16000, available_mb=15800)
        assert "exceeds total VRAM" in " ".join(too_big.warnings)
        assert "exceeds total VRAM" not in " ".join(contended.warnings)

    def test_silent_on_an_idle_gpu(self):
        est = self._estimate(total_mb=32000, available_mb=31500)
        assert not self._contention_warnings(est)

    def test_not_repeated_when_the_model_cannot_fit_at_all(self):
        # The "exceeds total VRAM" warning already covers this case.
        est = self._estimate(total_mb=16000, available_mb=15800)
        assert est.vram_exceeded
        assert not self._contention_warnings(est)

    def test_desktop_compositor_usage_does_not_trigger_it(self):
        # A few hundred MB of resident VRAM is normal, not a loaded model.
        est = self._estimate(total_mb=32000, available_mb=31700)
        assert not self._contention_warnings(est)

    def test_exceeds_message_says_total_not_available(self):
        est = self._estimate(total_mb=16000, available_mb=15800)
        exceeded = [w for w in est.warnings if "exceeds total VRAM" in w]
        assert exceeded, est.warnings
