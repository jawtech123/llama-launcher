"""Model Filesystem Scanner.

The filename heuristics here are the single source of truth: the local scan, the
remote (SSH) scan, and the resource estimator all classify models the same way.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
from .types import ModelInfo

logger = logging.getLogger(__name__)

#: Model file extensions llama-server can load, mapped to a format label.
MODEL_EXTENSIONS = {
    ".gguf": "gguf",
    ".bin": "bin",
    ".safetensors": "safetensors",
}

#: Filename fragment -> family. Ordered: more specific names must be tested
#: before the general ones they contain ("codellama" before "llama").
_FAMILY_HINTS = (
    ("codellama", "codellama"),
    ("tinyllama", "llama"),
    ("llama", "llama"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("qwen", "qwen"),
    ("qwq", "qwen"),
    ("gemma", "gemma"),
    ("phi", "phi"),
    ("deepseek", "deepseek"),
    ("llm", "llama"),
)

#: Filename fragment -> canonical quantization label. Longest variants first so
#: "q4_k_m" is not shortened to "q4_k".
_QUANT_PATTERNS = (
    ("q8_0", "Q8_0"),
    ("q6_k", "Q6_K"),
    ("q5_k_m", "Q5_K_M"),
    ("q5_k_s", "Q5_K_S"),
    ("q5_0", "Q5_0"),
    ("q5_1", "Q5_1"),
    ("q4_k_m", "Q4_K_M"),
    ("q4_k_s", "Q4_K_S"),
    ("q4_0", "Q4_0"),
    ("q4_1", "Q4_1"),
    ("q3_k_m", "Q3_K_M"),
    ("q3_k_s", "Q3_K_S"),
    ("q2_k", "Q2_K"),
)


def infer_model_hints(name: str) -> Tuple[str, Optional[str]]:
    """Infer ``(family, quant_type)`` from a model file name or path."""
    name_lower = (name or "").lower()
    family = "unknown"
    for hint, family_name in _FAMILY_HINTS:
        if hint in name_lower:
            family = family_name
            break
    quant_type = None
    for pattern, quant_name in _QUANT_PATTERNS:
        if pattern in name_lower:
            quant_type = quant_name
            break
    return family, quant_type


def is_mmproj_file(name: str) -> bool:
    """Whether a model file name looks like a multimodal projector.

    llama.cpp's convention (and every mmproj file in practice) puts
    "mmproj" directly in the file name, unlike quant/family tags which vary.
    """
    return "mmproj" in (name or "").lower()


def model_format(name: str) -> str:
    """Format label for a model file name."""
    return MODEL_EXTENSIONS.get(Path(name).suffix.lower(), "unknown")


def build_model_info(file_path: str, size_bytes: float) -> ModelInfo:
    """Build a ModelInfo from a path and size, without touching the filesystem.

    Shared with the remote scanner, which already has the size from its ``find``
    output and cannot stat the file locally.
    """
    file_name = Path(file_path).name
    family, quant_type = infer_model_hints(file_name)
    return ModelInfo(
        file_path=str(file_path),
        file_name=file_name,
        file_size_mb=round(size_bytes / (1024 * 1024), 2),
        format=model_format(file_name),
        family=family,
        quant_type=quant_type,
    )


def scan_models(directory: str) -> list[ModelInfo]:
    """Scan a directory tree for model files (.gguf, .bin, .safetensors)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    models = []
    # One traversal, filtering by suffix: globbing per extension walked the
    # whole tree once for each of them.
    try:
        entries = dir_path.rglob("*")
    except OSError:
        logger.debug("Could not scan %s", directory, exc_info=True)
        return []

    while True:
        try:
            file_path = next(entries, None)
        except OSError:
            # An unreadable subdirectory must not abort the whole scan.
            logger.debug("Skipping unreadable entry under %s", directory, exc_info=True)
            continue
        if file_path is None:
            break
        if file_path.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        model_info = _parse_model_file(file_path)
        if model_info:
            models.append(model_info)
    return models


def _parse_model_file(file_path: Path) -> Optional[ModelInfo]:
    """Parse a model file and extract metadata."""
    try:
        if not file_path.is_file():
            return None
        return build_model_info(str(file_path), file_path.stat().st_size)
    except (OSError, PermissionError, ValueError, TypeError):
        return None
    except Exception:
        logger.debug("Unexpected error parsing model file %s", file_path, exc_info=True)
        return None
