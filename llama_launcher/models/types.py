"""Model Information Types."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Information about detected model files."""
    file_path: str
    file_name: str
    file_size_mb: float
    format: str  # gguf, bin, safetensors
    family: Optional[str] = None
    quant_type: Optional[str] = None
