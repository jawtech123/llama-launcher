"""Minimal GGUF header reader: architecture metadata and per-tensor sizes.

Reads only the header (key/value metadata plus the tensor directory), never the
weights, so a multi-GB file costs a few milliseconds. Pure stdlib: the estimator
must work inside the frozen GUI build without numpy or gguf-py.

Tensor sizes use the same block layout table as ggml, so the per-layer figures
match what llama-server reports in its ``load_tensors`` log lines.
"""
from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types.
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32 = 0, 1, 2, 3, 4, 5
_T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_FORMATS = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<?",
    _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}

#: ggml type id -> (name, elements per block, bytes per block). From ggml.h /
#: gguf-py GGML_QUANT_SIZES; ids missing here were removed from ggml.
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
}

#: general.file_type -> quantization label (llama_ftype in llama.h).
FILE_TYPES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE",
}

#: Arrays longer than this keep only a prefix in memory (tokenizer vocabularies
#: run to hundreds of thousands of entries and are irrelevant for sizing).
_MAX_ARRAY_ITEMS = 512


class GgufFormatError(ValueError):
    """The file is not a GGUF v2/v3 file or its header is corrupt."""


@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        entry = GGML_TYPES.get(self.ggml_type)
        return entry[0] if entry else f"TYPE_{self.ggml_type}"

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        entry = GGML_TYPES.get(self.ggml_type)
        if entry is None:
            return self.n_elements * 2  # unknown type: assume 16-bit
        _, block_elems, block_bytes = entry
        # ggml rows are padded to whole blocks along the first dimension.
        ne0 = self.shape[0] if self.shape else 0
        rows = self.n_elements // ne0 if ne0 else 0
        blocks_per_row = -(-ne0 // block_elems) if block_elems else 0
        return rows * blocks_per_row * block_bytes


@dataclass
class GgufFile:
    """Parsed header of a GGUF file."""
    version: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: List[TensorInfo] = field(default_factory=list)

    @property
    def architecture(self) -> str:
        arch = self.metadata.get("general.architecture", "unknown")
        return arch if isinstance(arch, str) and arch else "unknown"

    def get(self, key: str, default: Any = None) -> Any:
        """Metadata lookup with the ``{arch}.`` prefix filled in.

        ``get("block_count")`` reads ``llama.block_count`` for a llama model and
        ``qwen2.block_count`` for a Qwen2 one; a key that does not exist with
        the prefix is looked up verbatim (``general.name``).
        """
        prefixed = f"{self.architecture}.{key}"
        if prefixed in self.metadata:
            return self.metadata[prefixed]
        return self.metadata.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        value = self.get(key)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return default

    def get_int_list(self, key: str) -> Optional[List[int]]:
        """A per-layer integer array, or None if the key is absent/scalar."""
        value = self.get(key)
        if isinstance(value, list) and value and all(isinstance(v, int) for v in value):
            return value
        return None


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GgufFormatError("Unexpected end of file in GGUF header")
    return data


def _read_scalar(f: BinaryIO, fmt: str):
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]


def _read_string(f: BinaryIO) -> str:
    length = _read_scalar(f, "<Q")
    if length > 64 * 1024 * 1024:
        raise GgufFormatError(f"Implausible string length {length}")
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _skip_string(f: BinaryIO) -> None:
    length = _read_scalar(f, "<Q")
    f.seek(length, os.SEEK_CUR)


def _read_value(f: BinaryIO, value_type: int) -> Any:
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is not None:
        return _read_scalar(f, fmt)
    if value_type == _T_STRING:
        return _read_string(f)
    if value_type == _T_ARRAY:
        item_type = _read_scalar(f, "<I")
        count = _read_scalar(f, "<Q")
        keep = min(count, _MAX_ARRAY_ITEMS)
        items = [_read_value(f, item_type) for _ in range(keep)]
        remaining = count - keep
        if remaining:
            item_fmt = _SCALAR_FORMATS.get(item_type)
            if item_fmt is not None:
                f.seek(remaining * struct.calcsize(item_fmt), os.SEEK_CUR)
            elif item_type == _T_STRING:
                for _ in range(remaining):
                    _skip_string(f)
            else:
                for _ in range(remaining):
                    _read_value(f, item_type)
        return items
    raise GgufFormatError(f"Unknown GGUF value type {value_type}")


def read_gguf(path: str) -> GgufFile:
    """Parse the header of a GGUF file (metadata + tensor directory).

    Raises:
        GgufFormatError: not a GGUF v2/v3 file, or a truncated header.
        OSError: the file cannot be opened.
    """
    result = GgufFile()
    with open(path, "rb") as f:
        if _read_exact(f, 4) != GGUF_MAGIC:
            raise GgufFormatError("Not a GGUF file")
        result.version = _read_scalar(f, "<I")
        if result.version not in (2, 3):
            raise GgufFormatError(f"Unsupported GGUF version {result.version}")
        tensor_count = _read_scalar(f, "<Q")
        kv_count = _read_scalar(f, "<Q")
        if tensor_count > 1_000_000 or kv_count > 1_000_000:
            raise GgufFormatError("Implausible GGUF header counts")

        for _ in range(kv_count):
            key = _read_string(f)
            value_type = _read_scalar(f, "<I")
            result.metadata[key] = _read_value(f, value_type)

        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims = _read_scalar(f, "<I")
            if n_dims > 4:
                raise GgufFormatError(f"Tensor {name!r} has {n_dims} dimensions")
            shape = tuple(_read_scalar(f, "<Q") for _ in range(n_dims))
            ggml_type = _read_scalar(f, "<I")
            offset = _read_scalar(f, "<Q")
            result.tensors.append(TensorInfo(name, shape, ggml_type, offset))
    return result
