"""Preset System for LlamaConfig."""
import dataclasses
import json
from pathlib import Path
from .config import LlamaConfig


class PresetManager:
    """Manages preset configurations for llama-server."""

    CUSTOM_PRESETS_FILE = Path.home() / ".llama_launcher_presets.json"

    BUILTIN_PRESETS = {
        "Small 8GB": LlamaConfig(
            name="Small 8GB",
            description="7B model, 8K context, q8_0 cache for 8GB VRAM",
            context_size=8192,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            kv_cache_type="q8_0",
        ),
        "Medium 12GB": LlamaConfig(
            name="Medium 12GB",
            description="13B model, 16K context, f16 cache for 12GB VRAM",
            context_size=16384,
        ),
        "Large 32GB": LlamaConfig(
            name="Large 32GB",
            description="30-70B model, 32K context, f16 cache for 32GB VRAM",
            context_size=32768,
        ),
    }

    @classmethod
    def get_preset(cls, preset_name: str) -> LlamaConfig:
        """Get a preset configuration by name."""
        if preset_name in cls._load_custom_presets():
            return cls.get_custom_preset(preset_name)

        if preset_name not in cls.BUILTIN_PRESETS:
            available = list(cls.BUILTIN_PRESETS.keys()) + [
                p for p in cls._load_custom_presets().keys()
                if p not in cls.BUILTIN_PRESETS
            ]
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")

        return dataclasses.replace(cls.BUILTIN_PRESETS[preset_name])

    @classmethod
    def save_profile(cls, config: LlamaConfig, path: str) -> None:
        """Save a configuration as a profile JSON file."""
        profile_data = {
            "name": config.name or "unnamed-profile",
            "description": config.description,
            "config": {k: v for k, v in dataclasses.asdict(config).items()
                       if k not in ("name", "description")},
        }
        with open(path, 'w') as f:
            json.dump(profile_data, f, indent=2)

    @classmethod
    def load_profile(cls, path: str) -> LlamaConfig:
        """Load a configuration from a profile JSON file."""
        with open(path, 'r') as f:
            profile_data = json.load(f)
        config_data = profile_data.get("config", {})
        valid_fields = {f.name for f in dataclasses.fields(LlamaConfig)}
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}
        return LlamaConfig(
            **filtered,
            name=profile_data.get("name"),
            description=profile_data.get("description"),
        )

    @classmethod
    def _load_custom_presets(cls) -> dict:
        """Load custom presets from file."""
        if cls.CUSTOM_PRESETS_FILE.exists():
            try:
                with open(cls.CUSTOM_PRESETS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @classmethod
    def _save_custom_presets(cls, presets: dict):
        """Save custom presets to file."""
        with open(cls.CUSTOM_PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)

    @classmethod
    def get_all_presets(cls) -> list:
        """Get all preset names (builtin + custom)."""
        all_p = list(cls.BUILTIN_PRESETS.keys())
        for p in cls._load_custom_presets():
            if p not in all_p:
                all_p.append(p)
        return all_p

    @classmethod
    def get_custom_presets(cls) -> list:
        """Get custom preset names."""
        return list(cls._load_custom_presets().keys())

    @classmethod
    def get_custom_preset(cls, preset_name: str) -> LlamaConfig:
        """Get a custom preset configuration by name."""
        custom_presets = cls._load_custom_presets()
        if preset_name not in custom_presets:
            raise ValueError(f"Custom preset '{preset_name}' not found.")
        config_data = custom_presets[preset_name]
        valid_fields = {f.name for f in dataclasses.fields(LlamaConfig)}
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}
        return LlamaConfig(**filtered)

    @classmethod
    def save_custom_preset(cls, preset_name: str, config: LlamaConfig):
        """Save a custom preset configuration."""
        custom_presets = cls._load_custom_presets()
        custom_presets[preset_name] = {
            k: v for k, v in dataclasses.asdict(config).items()
        }
        custom_presets[preset_name]["name"] = config.name or preset_name
        custom_presets[preset_name]["description"] = config.description or f"Custom: {preset_name}"
        cls._save_custom_presets(custom_presets)

    @classmethod
    def delete_custom_preset(cls, preset_name: str):
        """Delete a custom preset or an override of a built-in preset."""
        custom_presets = cls._load_custom_presets()
        if preset_name not in custom_presets:
            if preset_name in cls.BUILTIN_PRESETS:
                raise ValueError("Cannot delete built-in presets.")
            raise ValueError(f"Preset '{preset_name}' not found.")
        del custom_presets[preset_name]
        cls._save_custom_presets(custom_presets)
