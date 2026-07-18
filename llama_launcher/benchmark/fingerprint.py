"""Configuration fingerprinting for benchmark validity.

A benchmark result is only meaningful for the exact parameters it was
measured with. The fingerprint hashes the full launch configuration, so
changing any parameter invalidates (resets) the preset's benchmark info.
"""
import dataclasses
import hashlib
import json


def config_fingerprint(config) -> str:
    """Stable short hash of a LlamaConfig (any field change => new hash)."""
    data = dataclasses.asdict(config)
    blob = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
