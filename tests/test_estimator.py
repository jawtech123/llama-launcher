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
