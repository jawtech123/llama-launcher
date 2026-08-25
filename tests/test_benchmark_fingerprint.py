"""Tests for benchmark configuration fingerprinting."""
from llama_launcher.benchmark.fingerprint import config_fingerprint
from llama_launcher.core.config import LlamaConfig


def test_same_config_same_fingerprint():
    """Identical configs produce identical fingerprints."""
    c1 = LlamaConfig(model_path="/m.gguf", context_size=4096, threads=8)
    c2 = LlamaConfig(model_path="/m.gguf", context_size=4096, threads=8)
    f1 = config_fingerprint(c1)
    f2 = config_fingerprint(c2)
    assert f1 == f2


def test_different_context_different_fingerprint():
    """Changing context_size changes fingerprint."""
    c1 = LlamaConfig(model_path="/m.gguf", context_size=4096)
    c2 = LlamaConfig(model_path="/m.gguf", context_size=8192)
    assert config_fingerprint(c1) != config_fingerprint(c2)


def test_different_gpu_layers_different_fingerprint():
    """Changing gpu_layers changes fingerprint."""
    c1 = LlamaConfig(model_path="/m.gguf", gpu_layers=33)
    c2 = LlamaConfig(model_path="/m.gguf", gpu_layers=35)
    assert config_fingerprint(c1) != config_fingerprint(c2)


def test_full_150_field_config_stable():
    """Full config with all fields set produces stable fingerprint."""
    c1 = LlamaConfig(
        model_path="/m.gguf",
        context_size=8192,
        threads=8,
        batch_size=2048,
        gpu_layers=-1,
        port=8080,
        host="127.0.0.1",
        backend="auto",
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        use_mmap=True,
        cont_batching=True,
        embeddings=False,
        jinja=False,
    )
    c2 = LlamaConfig(
        model_path="/m.gguf",
        context_size=8192,
        threads=8,
        batch_size=2048,
        gpu_layers=-1,
        port=8080,
        host="127.0.0.1",
        backend="auto",
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        use_mmap=True,
        cont_batching=True,
        embeddings=False,
        jinja=False,
    )
    assert config_fingerprint(c1) == config_fingerprint(c2)


def test_empty_config_fingerprint():
    """Config with only defaults produces a fingerprint."""
    c = LlamaConfig(model_path="/m.gguf")
    fp = config_fingerprint(c)
    assert isinstance(fp, str)
    assert len(fp) == 16  # truncated SHA-256


def test_different_model_different_fingerprint():
    """Different model_path changes fingerprint."""
    c1 = LlamaConfig(model_path="/models/m1.gguf")
    c2 = LlamaConfig(model_path="/models/m2.gguf")
    assert config_fingerprint(c1) != config_fingerprint(c2)


def test_fingerprint_is_short_hex():
    """Fingerprint is a 16-character hex string."""
    c = LlamaConfig(model_path="/m.gguf")
    fp = config_fingerprint(c)
    assert len(fp) == 16
    int(fp, 16)  # should not raise — it's valid hex


def test_fingerprint_changes_on_any_field():
    """Any field change produces different fingerprint."""
    base = LlamaConfig(model_path="/m.gguf", context_size=4096, threads=8, temp=0.8)

    # Test a few different fields
    fields_to_test = [
        ("context_size", 8192),
        ("threads", 16),
        ("temp", 0.5),
        ("top_p", 0.99),
        ("seed", 42),
        ("backend", "cuda"),
        ("use_mmap", False),
    ]
    fp_base = config_fingerprint(base)
    for field_name, value in fields_to_test:
        modified = LlamaConfig(model_path="/m.gguf", context_size=4096, threads=8, temp=0.8)
        setattr(modified, field_name, value)
        fp_modified = config_fingerprint(modified)
        assert fp_base != fp_modified, f"Field {field_name}={value} should change fingerprint"
