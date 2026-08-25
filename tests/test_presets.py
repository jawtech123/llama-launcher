"""Tests for core.presets.PresetManager."""
import json
import dataclasses
import pytest
import os

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


# ---- Additional preset tests ----

def test_builtin_presets_have_names():
    for name, cfg in PresetManager.BUILTIN_PRESETS.items():
        assert name, "preset name should not be empty"
        assert cfg.name == name, f"preset {name} has mismatched name"


def test_builtin_presets_have_context_sizes():
    for name, cfg in PresetManager.BUILTIN_PRESETS.items():
        assert cfg.context_size > 0, f"{name}: context_size should be positive"


def test_builtin_small_8gb_defaults():
    cfg = PresetManager.get_preset("Small 8GB")
    assert cfg.context_size == 8192
    assert cfg.cache_type_k == "q8_0"
    assert cfg.cache_type_v == "q8_0"


def test_builtin_medium_12gb_defaults():
    cfg = PresetManager.get_preset("Medium 12GB")
    assert cfg.context_size == 16384


def test_builtin_large_32gb_defaults():
    cfg = PresetManager.get_preset("Large 32GB")
    assert cfg.context_size == 32768


def test_get_all_presets_includes_builtins():
    all_presets = PresetManager.get_all_presets()
    for name in PresetManager.BUILTIN_PRESETS:
        assert name in all_presets


def test_get_all_presets_includes_custom():
    cfg = LlamaConfig(model_path="/m.gguf", name="custom1")
    PresetManager.save_custom_preset("custom1", cfg)
    all_presets = PresetManager.get_all_presets()
    assert "custom1" in all_presets


def test_get_custom_presets_returns_only_custom():
    custom = PresetManager.get_custom_presets()
    for name in custom:
        assert name not in PresetManager.BUILTIN_PRESETS


def test_get_custom_presets_empty_initially():
    custom = PresetManager.get_custom_presets()
    assert custom == []


def test_save_and_load_preset_with_description():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        name="desc-test",
        description="A test preset with description",
    )
    PresetManager.save_custom_preset("desc-test", cfg)
    loaded = PresetManager.get_custom_preset("desc-test")
    assert loaded.description == "A test preset with description"


def test_save_preset_auto_sets_description():
    cfg = LlamaConfig(model_path="/m.gguf", name="auto-desc")
    PresetManager.save_custom_preset("auto-desc", cfg)
    loaded = PresetManager.get_custom_preset("auto-desc")
    assert loaded.description == "Custom: auto-desc"


def test_delete_restores_builtin_after_custom_override():
    # Override builtin
    cfg = LlamaConfig(model_path="/override.gguf", context_size=99999, name="Medium 12GB")
    PresetManager.save_custom_preset("Medium 12GB", cfg)
    assert PresetManager.get_preset("Medium 12GB").context_size == 99999

    # Delete override
    PresetManager.delete_custom_preset("Medium 12GB")

    # Should be back to builtin
    assert PresetManager.get_preset("Medium 12GB").context_size == 16384


def test_get_preset_returns_copy_not_reference():
    cfg = PresetManager.get_preset("Small 8GB")
    cfg.context_size = 7777
    cfg2 = PresetManager.get_preset("Small 8GB")
    assert cfg2.context_size == 8192


def test_custom_preset_can_override_any_field():
    cfg = LlamaConfig(
        model_path="/deep.gguf",
        context_size=65536,
        gpu_layers=100,
        threads=32,
        port=9999,
        host="0.0.0.0",
        temp=0.1,
        name="deep-config",
    )
    PresetManager.save_custom_preset("deep-config", cfg)
    loaded = PresetManager.get_custom_preset("deep-config")
    assert loaded.model_path == "/deep.gguf"
    assert loaded.context_size == 65536
    assert loaded.gpu_layers == 100
    assert loaded.threads == 32
    assert loaded.port == 9999
    assert loaded.host == "0.0.0.0"
    assert loaded.temp == 0.1


def test_delete_custom_preset_removes_from_list():
    PresetManager.save_custom_preset("to-delete", LlamaConfig(model_path="/m.gguf", name="to-delete"))
    assert "to-delete" in PresetManager.get_all_presets()
    PresetManager.delete_custom_preset("to-delete")
    assert "to-delete" not in PresetManager.get_all_presets()


def test_presets_file_is_created_after_save(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = LlamaConfig(model_path="/m.gguf", name="file-test")
    PresetManager.save_custom_preset("file-test", cfg)
    presets_file = PresetManager._presets_file()
    assert presets_file.exists()


def test_presets_file_contains_valid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = LlamaConfig(model_path="/m.gguf", name="json-test")
    PresetManager.save_custom_preset("json-test", cfg)
    presets_file = PresetManager._presets_file()
    with open(presets_file, 'r') as f:
        data = json.load(f)
    assert "json-test" in data


def test_corrupted_presets_file_handled_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    presets_file = PresetManager._presets_file()
    presets_file.parent.mkdir(parents=True, exist_ok=True)
    presets_file.write_text("{invalid json")
    # Should not crash
    all_presets = PresetManager.get_all_presets()
    assert all_presets == list(PresetManager.BUILTIN_PRESETS.keys())


def test_get_preset_falls_back_to_builtin_when_no_custom():
    # No custom presets created
    for name in PresetManager.BUILTIN_PRESETS:
        cfg = PresetManager.get_preset(name)
        assert cfg is not None


def test_multiple_custom_presets_stored():
    cfg1 = LlamaConfig(model_path="/m1.gguf", name="preset-a")
    cfg2 = LlamaConfig(model_path="/m2.gguf", name="preset-b")
    PresetManager.save_custom_preset("preset-a", cfg1)
    PresetManager.save_custom_preset("preset-b", cfg2)
    all_presets = PresetManager.get_all_presets()
    assert "preset-a" in all_presets
    assert "preset-b" in all_presets
    assert PresetManager.get_custom_preset("preset-a").model_path == "/m1.gguf"
    assert PresetManager.get_custom_preset("preset-b").model_path == "/m2.gguf"


def test_overwrite_custom_preset():
    cfg1 = LlamaConfig(model_path="/first.gguf", name="overwrite-test")
    cfg2 = LlamaConfig(model_path="/second.gguf", name="overwrite-test")
    PresetManager.save_custom_preset("overwrite-test", cfg1)
    assert PresetManager.get_custom_preset("overwrite-test").model_path == "/first.gguf"
    PresetManager.save_custom_preset("overwrite-test", cfg2)
    assert PresetManager.get_custom_preset("overwrite-test").model_path == "/second.gguf"


def test_delete_custom_preset_not_builtin():
    PresetManager.save_custom_preset("temp", LlamaConfig(model_path="/m.gguf", name="temp"))
    PresetManager.delete_custom_preset("temp")
    # Should not raise
    custom = PresetManager.get_custom_presets()
    assert "temp" not in custom


def test_save_profile_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = LlamaConfig(model_path="/m.gguf", name="profile-test", description="A profile")
    profile_path = str(tmp_path / "profile.json")
    PresetManager.save_profile(cfg, profile_path)
    assert os.path.exists(profile_path)


def test_load_profile_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = LlamaConfig(model_path="/m.gguf", name="profile-test", description="A profile")
    profile_path = str(tmp_path / "profile.json")
    PresetManager.save_profile(cfg, profile_path)
    loaded = PresetManager.load_profile(profile_path)
    assert loaded.model_path == "/m.gguf"
    assert loaded.name == "profile-test"
    assert loaded.description == "A profile"


def test_profile_missing_config_key(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    profile_path = str(tmp_path / "empty-profile.json")
    with open(profile_path, 'w') as f:
        json.dump({"name": "empty"}, f)
    loaded = PresetManager.load_profile(profile_path)
    assert loaded.name == "empty"


def test_load_profile_with_legacy_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    profile_path = str(tmp_path / "legacy-profile.json")
    profile_data = {
        "name": "legacy",
        "config": {
            "model_path": "/m.gguf",
            "kv_cache_type": "q8_0",
            "mmproj": "/legacy.mmproj",
        }
    }
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f)
    loaded = PresetManager.load_profile(profile_path)
    assert loaded.cache_type_k == "q8_0"
    assert loaded.cache_type_v == "q8_0"
    assert loaded.mmproj_model_path == "/legacy.mmproj"
