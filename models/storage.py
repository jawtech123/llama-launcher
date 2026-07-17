"""Per-server model cache persistence."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from models.types import ModelInfo

MODELS_CACHE_FILE = Path.home() / ".llama_launcher_models.json"


def _load_cache() -> Dict[str, list]:
    if not MODELS_CACHE_FILE.exists():
        return {}
    try:
        with open(MODELS_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: Dict[str, list]) -> None:
    with open(MODELS_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_models(server_name: str) -> List[ModelInfo]:
    cached = _load_cache()
    entries = cached.get(server_name, [])
    result = []
    for e in entries:
        try:
            result.append(ModelInfo(**e))
        except Exception:
            pass
    return result


def save_models(server_name: str, models: List[ModelInfo]) -> None:
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


def get_cached_server_names() -> List[str]:
    return list(_load_cache().keys())
