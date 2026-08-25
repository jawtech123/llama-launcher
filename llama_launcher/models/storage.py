"""Per-server model cache persistence."""
import json
import logging
from pathlib import Path
from typing import Dict, List

from llama_launcher.models.types import ModelInfo
from llama_launcher.utils.paths import config_file

logger = logging.getLogger(__name__)


def _cache_file() -> Path:
    return config_file("models.json")


def _load_cache() -> Dict[str, list]:
    cache_file = _cache_file()
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read model cache %s: %s", cache_file, e)
        return {}


def _save_cache(data: Dict[str, list]) -> None:
    cache_file = _cache_file()
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def load_models(server_name: str) -> List[ModelInfo]:
    """Return cached model scan results for a server, newest first not guaranteed."""
    cached = _load_cache()
    entries = cached.get(server_name, [])
    result = []
    for e in entries:
        try:
            result.append(ModelInfo(**e))
        except (TypeError, KeyError) as exc:
            logger.debug("Skipping malformed cached model entry: %s", exc)
    return result


def save_models(server_name: str, models: List[ModelInfo]) -> None:
    """Persist model scan results for a server."""
    cached = _load_cache()
    cached[server_name] = [
        {
            "file_path": m.file_path,
            "file_name": m.file_name,
            "file_size_mb": m.file_size_mb,
            "format": m.format,
            "family": m.family,
            "quant_type": m.quant_type,
        }
        for m in models
    ]
    _save_cache(cached)


def remove_models(server_name: str) -> bool:
    """Drop a server's cached model list. True if there was one to drop."""
    cached = _load_cache()
    if server_name not in cached:
        return False
    del cached[server_name]
    _save_cache(cached)
    return True


def get_cached_server_names() -> List[str]:
    """Return server names that have cached model lists."""
    return list(_load_cache().keys())
