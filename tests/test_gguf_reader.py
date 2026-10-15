"""Tests for the GGUF header reader and the metadata the estimator derives from it.

A small GGUF file is synthesised in memory: real headers are the same bytes,
only with multi-GB tensor payloads behind them.
"""
import struct

import pytest

from llama_launcher.core.config import LlamaConfig
from llama_launcher.estimator.estimator import _GgufReader, _place_layers, ResourceEstimator
from llama_launcher.estimator.gguf import GGML_TYPES, GgufFormatError, TensorInfo, read_gguf
from llama_launcher.gpu.types import GPUInfo

_T_UINT32, _T_FLOAT32, _T_STRING, _T_ARRAY, _T_UINT64 = 4, 6, 8, 9, 10


def _string(s: str) -> bytes:
    data = s.encode()
    return struct.pack("<Q", len(data)) + data


def _kv(key, value):
    out = _string(key)
    if isinstance(value, str):
        return out + struct.pack("<I", _T_STRING) + _string(value)
    if isinstance(value, float):
        return out + struct.pack("<If", _T_FLOAT32, value)
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            body = b"".join(_string(v) for v in value)
            return out + struct.pack("<IIQ", _T_ARRAY, _T_STRING, len(value)) + body
        body = b"".join(struct.pack("<I", v) for v in value)
        return out + struct.pack("<IIQ", _T_ARRAY, _T_UINT32, len(value)) + body
    return out + struct.pack("<II", _T_UINT32, value)


def _tensor(name, shape, ggml_type, offset=0):
    out = _string(name) + struct.pack("<I", len(shape))
    out += b"".join(struct.pack("<Q", d) for d in shape)
    return out + struct.pack("<IQ", ggml_type, offset)


def write_gguf(path, metadata: dict, tensors: list) -> str:
    blob = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        blob += _kv(k, v)
    for t in tensors:
        blob += _tensor(*t)
    path.write_bytes(blob)
    return str(path)


Q4_K, Q6_K, F32 = 12, 14, 0


@pytest.fixture
def small_llama(tmp_path):
    """A 2-layer llama-style model with a tied embedding and one MoE-free FFN."""
    meta = {
        "general.architecture": "llama",
        "general.file_type": 15,
        "llama.block_count": 2,
        "llama.context_length": 8192,
        "llama.embedding_length": 64,
        "llama.feed_forward_length": 128,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
        "tokenizer.ggml.tokens": ["a", "b", "c"] * 400,  # long string array must be skipped correctly
        "tokenizer.chat_template": "{{ messages }}",
    }
    tensors = [("token_embd.weight", (64, 1000), Q6_K)]
    for i in range(2):
        tensors += [
            (f"blk.{i}.attn_q.weight", (64, 64), Q4_K),
            (f"blk.{i}.ffn_up.weight", (64, 128), Q4_K),
            (f"blk.{i}.attn_norm.weight", (64,), F32),
        ]
    tensors.append(("output_norm.weight", (64,), F32))
    return write_gguf(tmp_path / "small.gguf", meta, tensors)


def test_tensor_bytes_follow_ggml_block_layout():
    name, block_elems, block_bytes = GGML_TYPES[Q4_K]
    assert name == "Q4_K"
    t = TensorInfo("x", (256 * 3, 10), Q4_K, 0)
    assert t.n_bytes == 10 * 3 * block_bytes
    assert TensorInfo("y", (7, 2), F32, 0).n_bytes == 7 * 2 * 4
    assert TensorInfo("y", (7, 2), F32, 0).n_elements == 14


def test_read_gguf_parses_metadata_and_tensors(small_llama):
    g = read_gguf(small_llama)
    assert g.architecture == "llama"
    assert g.get("block_count") == 2
    assert g.get("tokenizer.chat_template") == "{{ messages }}"
    assert len(g.metadata["tokenizer.ggml.tokens"]) == 512  # prefix kept, rest skipped
    assert [t.name for t in g.tensors][0] == "token_embd.weight"
    assert len(g.tensors) == 8


def test_read_gguf_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GgufFormatError):
        read_gguf(str(bad))


def test_metadata_geometry_and_sizes(small_llama):
    meta = _GgufReader.read_full_metadata(small_llama)
    assert meta.arch == "llama"
    assert meta.layers == 2 and meta.heads == 4 and meta.kv_heads == 2
    assert meta.head_dim == 16 and meta.key_length == 16 and meta.value_length == 16
    assert meta.context_length == 8192
    assert meta.file_type == "Q4_K_M"
    assert meta.chat_template == "{{ messages }}"
    assert meta.vocab_size == 1000
    assert meta.has_tensor_sizes
    assert len(meta.layer_bytes) == 2 and meta.layer_bytes[0] == meta.layer_bytes[1] > 0
    assert meta.embd_bytes > 0 and meta.output_bytes == 0  # tied embedding
    assert meta.total_bytes == meta.embd_bytes + sum(meta.layer_bytes) + meta.other_bytes
    assert meta.total_params == 64 * 1000 + 2 * (64 * 64 + 64 * 128 + 64) + 64
    assert meta.quant_types == {"Q6_K": 1, "Q4_K": 4, "F32": 3}


def test_placement_offloads_output_layer_first():
    p = _place_layers(36, 10, True)
    assert p.offloaded == 10 and p.offloadable == 37
    assert p.output_on_gpu and p.gpu_layers == list(range(27, 36))
    assert _place_layers(36, -1, True).offloaded == 37
    assert _place_layers(36, 0, True).offloaded == 0
    assert _place_layers(36, 99, True).gpu_layers == list(range(36))
    assert _place_layers(36, -1, False).offloaded == 0


def _estimate(path, **cfg):
    gpu = GPUInfo(backend="cuda", vram_mb=24000, available_vram_mb=24000)
    config = LlamaConfig(model_path=path, **cfg)
    return ResourceEstimator().calculate(config, gpu, model_size_mb=1.0, model_path=path)


def test_weights_split_uses_tensor_sizes(small_llama):
    meta = _GgufReader.read_full_metadata(small_llama)
    mib = 1024 * 1024
    full = _estimate(small_llama, gpu_layers=-1)
    # Tied output is copied to the GPU; the embedding also stays on the host.
    assert full.gpu_weights_mb * mib == pytest.approx(sum(meta.layer_bytes) + meta.embd_bytes + meta.other_bytes)
    assert full.cpu_weights_mb * mib == pytest.approx(meta.embd_bytes)
    assert full.offloaded_layers == 3 and full.offloadable_layers == 3

    partial = _estimate(small_llama, gpu_layers=2)  # output + last layer
    assert partial.gpu_weights_mb * mib == pytest.approx(meta.layer_bytes[1] + meta.embd_bytes + meta.other_bytes)
    none = _estimate(small_llama, gpu_layers=0)
    assert none.gpu_weights_mb == 0
    assert none.cpu_weights_mb * mib == pytest.approx(meta.total_bytes)


def test_kv_cache_matches_llama_cpp_formula(small_llama):
    est = _estimate(small_llama, context_size=4096, cache_type_k="f16", cache_type_v="f16")
    expected = 2 * 2 * (16 + 16) * 2 * 4096 / (1024 * 1024)  # layers*kv_heads*(k+v)*bytes*ctx
    assert est.kv_cache_gpu_mb == pytest.approx(expected)
    assert est.kv_cache_cpu_mb == 0
    q8 = _estimate(small_llama, context_size=4096, cache_type_k="q8_0", cache_type_v="q8_0")
    assert q8.kv_cache_gpu_mb == pytest.approx(expected * (34 / 32) / 2)


def test_kv_cache_follows_layer_placement(small_llama):
    est = _estimate(small_llama, context_size=4096, gpu_layers=2)  # one layer on GPU
    assert est.kv_cache_gpu_mb == pytest.approx(est.kv_cache_cpu_mb)
    off = _estimate(small_llama, context_size=4096, kv_offload=False)
    assert off.kv_cache_gpu_mb == 0 and off.kv_cache_cpu_mb > 0


def test_compute_buffer_scales_with_ubatch_and_context(small_llama):
    a = _estimate(small_llama, context_size=4096, ubatch_size=512)
    b = _estimate(small_llama, context_size=4096, ubatch_size=1024)
    c = _estimate(small_llama, context_size=32768, ubatch_size=512)
    assert b.compute_buffers_gpu_mb > a.compute_buffers_gpu_mb
    assert c.compute_buffers_gpu_mb > a.compute_buffers_gpu_mb


def test_hybrid_model_only_caches_attention_layers(tmp_path):
    meta = {
        "general.architecture": "qwen35", "qwen35.block_count": 8,
        "qwen35.embedding_length": 64, "qwen35.attention.head_count": 4,
        "qwen35.attention.head_count_kv": 2, "qwen35.attention.key_length": 16,
        "qwen35.attention.value_length": 16, "qwen35.full_attention_interval": 4,
        "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.inner_size": 32,
        "qwen35.ssm.state_size": 8, "qwen35.ssm.group_count": 2,
    }
    tensors = [(f"blk.{i}.x.weight", (64, 64), Q4_K) for i in range(8)]
    path = write_gguf(tmp_path / "hybrid.gguf", meta, tensors)
    est = _estimate(path, context_size=4096, parallel=1)
    # Two of eight layers are attention layers.
    assert est.kv_cache_gpu_mb == pytest.approx(2 * 2 * 32 * 2 * 4096 / (1024 * 1024))
    # The other six keep a recurrent state: conv + ssm state, f32, per slot.
    conv = 3 * (32 + 2 * 2 * 8)
    state = 32 * 8
    assert est.temp_allocations_mb == pytest.approx(6 * (conv + state) * 4 / (1024 * 1024))
    four_slots = _estimate(path, context_size=4096, parallel=-1)
    assert four_slots.temp_allocations_mb == pytest.approx(est.temp_allocations_mb * 4)


def test_moe_expert_offload_moves_expert_weights_to_cpu(tmp_path):
    meta = {
        "general.architecture": "qwen3moe", "qwen3moe.block_count": 4,
        "qwen3moe.embedding_length": 64, "qwen3moe.attention.head_count": 4,
        "qwen3moe.attention.head_count_kv": 2, "qwen3moe.expert_count": 8,
        "qwen3moe.expert_used_count": 2, "qwen3moe.expert_feed_forward_length": 32,
    }
    tensors = []
    for i in range(4):
        tensors += [(f"blk.{i}.attn_q.weight", (64, 64), Q4_K),
                    (f"blk.{i}.ffn_up_exps.weight", (64, 32, 8), Q4_K)]
    path = write_gguf(tmp_path / "moe.gguf", meta, tensors)
    m = _GgufReader.read_full_metadata(path)
    assert m.is_moe and all(b > 0 for b in m.expert_bytes)
    mib = 1024 * 1024
    all_gpu = _estimate(path, context_size=1024)
    half = _estimate(path, context_size=1024, n_cpu_moe=2)
    cpu = _estimate(path, context_size=1024, cpu_moe=True)
    assert all_gpu.gpu_weights_mb * mib == pytest.approx(m.total_bytes)
    assert (all_gpu.gpu_weights_mb - half.gpu_weights_mb) * mib == pytest.approx(sum(m.expert_bytes[:2]))
    assert (all_gpu.gpu_weights_mb - cpu.gpu_weights_mb) * mib == pytest.approx(sum(m.expert_bytes))
    assert cpu.compute_buffers_gpu_mb > all_gpu.compute_buffers_gpu_mb


def test_partial_offload_warns(small_llama):
    est = _estimate(small_llama, gpu_layers=2)
    assert any("layers on the GPU" in w for w in est.warnings)
    assert not any("layers on the GPU" in w for w in _estimate(small_llama).warnings)
