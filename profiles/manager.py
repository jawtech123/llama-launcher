"""Server Profile Manager."""
import json
from pathlib import Path
from core.config import LlamaConfig


def export_profile(config: LlamaConfig, path: str) -> None:
    """Export configuration to a profile JSON file."""
    profile_data = {
        "name": config.name or "unnamed-profile",
        "description": config.description,
        "config": {
            "model_path": config.model_path,
            "context_size": config.context_size,
            "gpu_layers": config.gpu_layers,
            "threads": config.threads,
            "port": config.port,
            "host": config.host,
            "backend": config.backend,
            "use_mmap": config.use_mmap,
            "kv_cache_type": config.kv_cache_type,
            "batch_size": config.batch_size,
            "extra_flags": config.extra_flags
        }
    }
    
    with open(path, 'w') as f:
        json.dump(profile_data, f, indent=2)


def import_profile(path: str) -> LlamaConfig:
    """Import configuration from a profile JSON file."""
    with open(path, 'r') as f:
        profile_data = json.load(f)
    
    config_data = profile_data.get("config", {})
    return LlamaConfig(
        model_path=config_data.get("model_path", ""),
        context_size=config_data.get("context_size", 4096),
        gpu_layers=config_data.get("gpu_layers", -1),
        threads=config_data.get("threads", 0),
        port=config_data.get("port", 8080),
        host=config_data.get("host", "127.0.0.1"),
        backend=config_data.get("backend", "auto"),
        use_mmap=config_data.get("use_mmap", True),
        kv_cache_type=config_data.get("kv_cache_type", "f16"),
        batch_size=config_data.get("batch_size", 512),
        extra_flags=config_data.get("extra_flags", []),
        name=profile_data.get("name"),
        description=profile_data.get("description")
    )
