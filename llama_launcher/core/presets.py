"""Preset System for LlamaConfig."""
import dataclasses
import json
import logging
from pathlib import Path

from llama_launcher.utils.paths import config_file, secure_file
from .config import LlamaConfig, config_to_dict, migrate_config_dict

logger = logging.getLogger(__name__)


class PresetManager:
    """Manages preset configurations for llama-server."""

    @classmethod
    def _presets_file(cls) -> Path:
        """Path to the custom presets store (may contain secrets like HF tokens)."""
        return config_file("presets.json")

    BUILTIN_PRESETS = {
        "Small 8GB": LlamaConfig(
            name="Small 8GB",
            description="7B model, 8K context, q8_0 cache for 8GB VRAM",
            context_size=8192,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
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
        """Get a preset configuration by name (custom overrides win)."""
        custom = cls._load_custom_presets()
        if preset_name in custom:
            return cls._from_dict(custom[preset_name])

        if preset_name not in cls.BUILTIN_PRESETS:
            available = list(cls.BUILTIN_PRESETS) + [
                p for p in custom if p not in cls.BUILTIN_PRESETS]
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")

        return dataclasses.replace(cls.BUILTIN_PRESETS[preset_name])

    @staticmethod
    def _from_dict(data: dict) -> LlamaConfig:
        """Build a config from a stored dict, migrating and dropping unknown keys."""
        config_data = migrate_config_dict(data)
        valid_fields = {f.name for f in dataclasses.fields(LlamaConfig)}
        return LlamaConfig(**{k: v for k, v in config_data.items() if k in valid_fields})

    @classmethod
    def save_profile(cls, config: LlamaConfig, path: str) -> None:
        """Save a configuration as a profile JSON file."""
        profile_data = {
            "name": config.name or "unnamed-profile",
            "description": config.description,
            "config": {k: v for k, v in config_to_dict(config).items()
                       if k not in ("name", "description")},
        }
        with open(path, 'w') as f:
            json.dump(profile_data, f, indent=2)

    @classmethod
    def load_profile(cls, path: str) -> LlamaConfig:
        """Load a configuration from a profile JSON file."""
        with open(path, 'r') as f:
            profile_data = json.load(f)
        config = cls._from_dict(profile_data.get("config", {}))
        config.name = profile_data.get("name")
        config.description = profile_data.get("description")
        return config

    @classmethod
    def _load_custom_presets(cls) -> dict:
        """Load custom presets from file."""
        presets_file = cls._presets_file()
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not read presets file %s: %s", presets_file, e)
        return {}

    @classmethod
    def _save_custom_presets(cls, presets: dict):
        """Save custom presets to file (user-private: may contain HF tokens)."""
        presets_file = cls._presets_file()
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        secure_file(presets_file)

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
        return cls._from_dict(custom_presets[preset_name])

    @classmethod
    def save_custom_preset(cls, preset_name: str, config: LlamaConfig):
        """Save a custom preset configuration."""
        custom_presets = cls._load_custom_presets()
        custom_presets[preset_name] = config_to_dict(config)
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
