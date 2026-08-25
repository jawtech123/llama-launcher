"""Tests for per-server model cache persistence."""
import json
import tempfile
import pytest

from llama_launcher.models.types import ModelInfo
from llama_launcher.models.storage import (
    load_models,
    save_models,
    get_cached_server_names,
    _cache_file,
)


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    """Point the config dir at a temp location."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    yield


def _make_model(**kwargs):
    return ModelInfo(
        file_path=kwargs.get("file_path", "/models/m.gguf"),
        file_name=kwargs.get("file_name", "m.gguf"),
        file_size_mb=kwargs.get("file_size_mb", 5000.0),
        format=kwargs.get("format", "gguf"),
        family=kwargs.get("family", "llama"),
        quant_type=kwargs.get("quant_type", "Q4_0"),
    )


class TestSaveLoadModels:
    """Test model cache save and load."""

    def test_save_and_load_single_model(self):
        models = [_make_model()]
        save_models("test-server", models)
        loaded = load_models("test-server")
        assert len(loaded) == 1
        assert loaded[0].file_name == "m.gguf"
        assert loaded[0].format == "gguf"

    def test_save_and_load_multiple_models(self):
        models = [
            _make_model(file_path="/m1.gguf", file_name="m1.gguf"),
            _make_model(file_path="/m2.gguf", file_name="m2.gguf"),
        ]
        save_models("test-server", models)
        loaded = load_models("test-server")
        assert len(loaded) == 2
        names = [m.file_name for m in loaded]
        assert "m1.gguf" in names
        assert "m2.gguf" in names

    def test_save_empty_models(self):
        save_models("test-server", [])
        loaded = load_models("test-server")
        assert loaded == []

    def test_load_nonexistent_server(self):
        loaded = load_models("nope")
        assert loaded == []

    def test_multiple_servers_cached_independently(self):
        save_models("srv1", [_make_model(file_path="/srv1.gguf")])
        save_models("srv2", [_make_model(file_path="/srv2.gguf")])
        srv1 = load_models("srv1")
        srv2 = load_models("srv2")
        assert srv1[0].file_path == "/srv1.gguf"
        assert srv2[0].file_path == "/srv2.gguf"


class TestGetCachedServerNames:
    """Test getting cached server names."""

    def test_empty_cache(self):
        names = get_cached_server_names()
        assert names == []

    def test_cache_with_entries(self):
        save_models("srv1", [_make_model()])
        save_models("srv2", [_make_model()])
        names = get_cached_server_names()
        assert "srv1" in names
        assert "srv2" in names
        assert len(names) == 2

    def test_names_are_strings(self):
        save_models("srv1", [_make_model()])
        names = get_cached_server_names()
        for n in names:
            assert isinstance(n, str)


class TestJsonFormat:
    """Test JSON file format."""

    def test_cache_file_structure(self):
        save_models("test", [_make_model()])
        from llama_launcher.models.storage import _cache_file
        fpath = _cache_file()
        assert fpath.exists()
        with open(fpath) as f:
            data = json.load(f)
        assert "test" in data

    def test_model_entry_fields(self):
        save_models("test", [_make_model(
            file_path="/path/m.gguf",
            file_name="m.gguf",
            file_size_mb=1234.5,
            format="gguf",
            family="llama",
            quant_type="Q4_0",
        )])
        from llama_launcher.models.storage import _cache_file
        fpath = _cache_file()
        with open(fpath) as f:
            data = json.load(f)
        entry = data["test"][0]
        assert entry["file_path"] == "/path/m.gguf"
        assert entry["file_size_mb"] == 1234.5
        assert entry["format"] == "gguf"
        assert entry["family"] == "llama"
        assert entry["quant_type"] == "Q4_0"


class TestCorruptedCache:
    """Test handling of corrupted cache files."""

    def test_corrupted_json(self):
        from llama_launcher.models.storage import _cache_file
        fpath = _cache_file()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("{invalid json!!!")
        loaded = load_models("test")
        assert loaded == []

    def test_malformed_entry_skipped(self):
        """Malformed entries should be skipped without crashing."""
        from llama_launcher.models.storage import _cache_file, _save_cache
        fpath = _cache_file()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(json.dumps({
            "test": [
                {"file_path": "/good.gguf", "file_name": "good.gguf", "file_size_mb": 1.0, "format": "gguf"},
                {"bad_entry": True},  # missing required fields
            ]
        }))
        loaded = load_models("test")
        assert len(loaded) == 1
        assert loaded[0].file_name == "good.gguf"
