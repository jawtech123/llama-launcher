"""Application config directory management.

All runtime state (presets, servers, model caches, folder lists, logs) lives in
a single config directory instead of loose files in the user's home directory.
Legacy ``~/.llama_launcher_*.json`` files are migrated automatically on first use.
"""
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

APP_DIR_NAME = "llama-launcher"

#: Legacy home-dir file name -> new name inside the config directory.
LEGACY_FILES = {
    ".llama_launcher_folders.json": "folders.json",
    ".llama_launcher_models.json": "models.json",
    ".llama_launcher_presets.json": "presets.json",
    ".llama_launcher_servers.json": "servers.json",
}

#: Files that may contain secrets (passwords, tokens) and must stay user-private.
PRIVATE_FILES = ("presets.json", "servers.json")


def config_dir(create: bool = True) -> Path:
    """Return the application config directory, migrating legacy files.

    Honors ``XDG_CONFIG_HOME``; defaults to ``~/.config/llama-launcher``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    cfg = base / APP_DIR_NAME
    if create:
        cfg.mkdir(parents=True, exist_ok=True)
        _migrate_legacy(cfg)
    return cfg


def config_file(name: str) -> Path:
    """Return the path of a named file inside the config directory."""
    return config_dir() / name


def secure_file(path: Path) -> None:
    """Restrict a file to user read/write only (for files holding secrets)."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("Could not set permissions 600 on %s", path)


def _migrate_legacy(cfg: Path) -> None:
    """Move legacy home-dir JSON files into the config directory once."""
    home = Path.home()
    for legacy_name, new_name in LEGACY_FILES.items():
        legacy = home / legacy_name
        target = cfg / new_name
        if legacy.exists() and not target.exists():
            try:
                shutil.move(str(legacy), str(target))
                logger.info("Migrated legacy config %s -> %s", legacy, target)
                if new_name in PRIVATE_FILES:
                    secure_file(target)
            except OSError:
                logger.exception("Failed to migrate legacy config %s", legacy)
