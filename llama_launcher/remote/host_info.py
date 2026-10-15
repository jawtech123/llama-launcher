"""What connecting to a server taught us about that machine.

Connecting is the moment the app learns a host's hardware, so the result is
cached per server: a disconnected server can still show its last-known GPU and
RAM (marked stale) instead of falling back to describing the launcher's machine.

Mirrors :mod:`llama_launcher.models.storage`, which caches model lists the same
way and by the same key (the server's friendly name).
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from llama_launcher.gpu.types import GPUInfo, gpu_from_dict, gpu_to_dict
from llama_launcher.utils.paths import config_file

logger = logging.getLogger(__name__)


@dataclass
class HostProfile:
    """Hardware facts about one machine, as of ``detected_at``."""

    gpu: Optional[GPUInfo] = None
    total_ram_mb: float = -1.0
    detected_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.detected_at)


def _cache_file() -> Path:
    return config_file("hosts.json")


_gpu_to_dict = gpu_to_dict
_gpu_from_dict = gpu_from_dict


def _load_cache() -> Dict[str, Dict]:
    cache_file = _cache_file()
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Could not read host cache %s: %s", cache_file, e)
        return {}


def _save_cache(data: Dict[str, Dict]) -> None:
    try:
        with open(_cache_file(), "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        logger.warning("Could not write host cache", exc_info=True)


def load_profile(server_name: str) -> Optional[HostProfile]:
    """Return the cached profile for a server, or None if never connected."""
    entry = _load_cache().get(server_name)
    if not entry:
        return None
    return HostProfile(
        gpu=_gpu_from_dict(entry.get("gpu")),
        total_ram_mb=float(entry.get("total_ram_mb", -1.0)),
        detected_at=float(entry.get("detected_at", 0.0)),
    )


def save_profile(server_name: str, profile: HostProfile) -> None:
    """Persist what we learned about a server's machine."""
    cache = _load_cache()
    cache[server_name] = {
        "gpu": _gpu_to_dict(profile.gpu),
        "total_ram_mb": profile.total_ram_mb,
        "detected_at": profile.detected_at,
    }
    _save_cache(cache)


def remove_profile(server_name: str) -> bool:
    """Drop a server's cached profile. True if there was one."""
    cache = _load_cache()
    if server_name not in cache:
        return False
    del cache[server_name]
    _save_cache(cache)
    return True
