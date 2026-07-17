"""Tests for core.presets.PresetManager."""
import pytest

from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.presets import PresetManager


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    """Point the config dir at a temp location for every test here."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    yield


def test_builtin_preset_returns_independent_copy():
    a = PresetManager.get_preset("Small 8GB")
    a.context_size = 999999
    b = PresetManager.get_preset("Small 8GB")
    assert b.context_size != 999999


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        PresetManager.get_preset("does-not-exist")


def test_custom_preset_roundtrip():
    cfg = LlamaConfig(
        model_path="/models/m.gguf",
        context_size=32768,
        gpu_layers=99,
        temp=0.42,
        name="my-preset",
    )
    PresetManager.save_custom_preset("my-preset", cfg)

    assert "my-preset" in PresetManager.get_all_presets()
    loaded = PresetManager.get_custom_preset("my-preset")
    assert loaded.model_path == "/models/m.gguf"
    assert loaded.context_size == 32768
    assert loaded.gpu_layers == 99
    assert loaded.temp == 0.42

    # Also retrievable through the generic getter
    loaded2 = PresetManager.get_preset("my-preset")
    assert loaded2.context_size == 32768


def test_custom_preset_overwrites_builtin_shadow():
    cfg = LlamaConfig(model_path="/x.gguf", context_size=12345, name="Small 8GB")
    PresetManager.save_custom_preset("Small 8GB", cfg)
    loaded = PresetManager.get_preset("Small 8GB")
    assert loaded.context_size == 12345

    # Deleting the override restores the builtin
    PresetManager.delete_custom_preset("Small 8GB")
    restored = PresetManager.get_preset("Small 8GB")
    assert restored.context_size == 8192


def test_delete_unknown_preset_raises():
    with pytest.raises(ValueError):
        PresetManager.delete_custom_preset("nope")


def test_delete_pure_builtin_raises():
    with pytest.raises(ValueError, match="built-in"):
        PresetManager.delete_custom_preset("Medium 12GB")
