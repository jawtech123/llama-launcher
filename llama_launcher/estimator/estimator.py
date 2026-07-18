"""Resource Estimator - predicts VRAM and RAM usage before launching llama-server.

All sizing logic lives here. The GUI never performs memory calculations.

Estimation methodology:
- Model weights: extracted from GGUF tensor information (quantization-aware)
- KV cache: computed from architecture parameters read from GGUF header
- Compute buffers: estimated from hidden size, batch size, and layer count
- Runtime overhead: fixed estimates based on backend type
- Multi-GPU: supports layer and row split modes across up to 8 GPUs
"""
import logging
import os
import struct
import platform
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from llama_launcher.core.config import LlamaConfig
from llama_launcher.gpu.types import GPUInfo
from .types import (
    ResourceEstimate,
    VramBreakdown,
    RamBreakdown,
    ModelMetadata,
    GpuBreakdown,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GGML type codes -> bytes per element mapping
# Used to compute actual tensor memory from GGUF tensor type + shape
# ---------------------------------------------------------------------------

_GGML_TYPE_BYTES: dict[int, float] = {
    0: 4,      # GGML_TYPE_F32
    1: 2,      # GGML_TYPE_F16
    2: 2,      # GGML_TYPE_Q4_0
    3: 2,      # GGML_TYPE_Q4_1
    6: 2,      # GGML_TYPE_Q5_0
    7: 2,      # GGML_TYPE_Q5_1
    8: 2,      # GGML_TYPE_Q8_0
    9: 2,      # GGML_TYPE_Q8_1
    10: 1.96875,  # GGML_TYPE_Q2_K
    11: 2.0625,   # GGML_TYPE_Q3_K
    12: 2.5625,   # GGML_TYPE_Q4_K
    13: 3.0,      # GGML_TYPE_Q5_K
    14: 3.875,    # GGML_TYPE_Q6_K
    15: 4,        # GGML_TYPE_IQ2_XXS
    16: 4.125,    # GGML_TYPE_IQ2_XS
    17: 1.75,     # GGML_TYPE_IQ3_XXS
    18: 4,        # GGML_TYPE_IQ1_S
    19: 4.25,     # GGML_TYPE_IQ4_NL
    20: 4.5625,   # GGML_TYPE_IQ3_S
    21: 4,        # GGML_TYPE_IQ2_S
    22: 4.6875,   # GGML_TYPE_IQ4_XS
    23: 4,        # GGML_TYPE_IQ1_M
    24: 2,        # GGML_TYPE_BF16
    25: 4,        # GGML_TYPE_Q4_0_4_4
    26: 4,        # GGML_TYPE_Q4_0_4_8
    27: 4,        # GGML_TYPE_Q4_0_8_8
    28: 4,        # GGML_TYPE_TQ1_0
    29: 4,        # GGML_TYPE_TQ2_0
    30: 2,        # GGML_TYPE_IQ3_M
    31: 4,        # GGML_TYPE_IQ3_MSS
    32: 4,        # GGML_TYPE_IQ1_MSS
    33: 4,        # GGML_TYPE_BF16S
    34: 4,        # GGML_TYPE_Q5_0_1_6
    35: 4,        # GGML_TYPE_Q6_0_1_6
    36: 4,        # GGML_TYPE_Q8_0_5_10
    37: 4,        # GGML_TYPE_Q8_0_4_8_8
    38: 4,        # GGML_TYPEIQ1_BLOCK
    39: 4,        # GGML_TYPEIQ4_NL_4_4
}

# Approximate bits-per-element for parameter counting
_GGML_TYPE_BITS: dict[int, float] = {
    0: 32,      # F32
    1: 16,      # F16
    24: 16,     # BF16
    2: 4.0,     # Q4_0
    3: 4.1,     # Q4_1
    6: 5.0,     # Q5_0
    7: 5.1,     # Q5_1
    8: 8.0,     # Q8_0
    9: 8.1,     # Q8_1
    10: 2.5,    # Q2_K
    11: 3.5,    # Q3_K
    12: 4.5,    # Q4_K
    13: 5.5,    # Q5_K
    14: 6.0,    # Q6_K
    19: 4.0,    # IQ4_NL
    25: 4.0,    # Q4_0_4_4
    26: 4.0,    # Q4_0_4_8
    27: 4.0,    # Q4_0_8_8
}


def _ggml_type_name(t: int) -> str:
    """Get human-readable GGML type name."""
    names = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
        15: "IQ2_XXS", 16: "IQ2_XS", 17: "IQ3_XXS", 18: "IQ1_S",
        19: "IQ4_NL", 20: "IQ3_S", 21: "IQ2_S", 22: "IQ4_XS", 23: "IQ1_M",
        24: "BF16", 25: "Q4_0_4_4", 26: "Q4_0_4_8", 27: "Q4_0_8_8",
        28: "TQ1_0", 29: "TQ2_0", 30: "IQ3_M", 31: "IQ3_MSS", 32: "IQ1_MSS",
        33: "BF16S",
    }
    return names.get(t, f"TYPE_{t}")


# ---------------------------------------------------------------------------
# GGUF header reader
# ---------------------------------------------------------------------------

#: Cache for GGUF metadata, keyed by (path, mtime_ns, size). Avoids re-parsing
#: multi-GB model files on every config change.
_METADATA_CACHE: dict = {}
_METADATA_CACHE_MAX = 64


def _metadata_cache_key(model_path: str):
    """Return a cache key for a model file, or None if it can't be stat'ed."""
    if not model_path:
        return None
    try:
        st = os.stat(model_path)
    except OSError:
        return None
    return (model_path, st.st_mtime_ns, st.st_size)


class _GgufReader:
    """GGUF v2/v3 header reader for architecture parameters and tensor information."""

    GGUF_MAGIC = b"GGUF"

    # Architecture-specific key mappings
    _ARCH_KEY = "general.architecture"

    _CONTEXT_KEYS = {
        "llama": "llama.context_length",
        "mistral": "llama.context_length",
        "gemma": "llama.context_length",
        "gemma2": "llama.context_length",
        "phi": "phi.context_length",
        "qwen2": "llama.context_length",
        "command-r": "llama.context_length",
        "stablelm": "llama.context_length",
        "falcon": "llama.context_length",
        "mpt": "mpt.context_length",
        "chatglm": "llama.context_length",
        "exaone": "llama.context_length",
        "granite": "llama.context_length",
        "internlm2": "llama.context_length",
        "minicpm": "llama.context_length",
        "olmo": "llama.context_length",
        "olmo2": "llama.context_length",
        "persimmon": "llama.context_length",
        "plamo": "llama.context_length",
        "refact": "llama.context_length",
        "dbrx": "dbrx.context_length",
        "deepseek2": "llama.context_length",
        "jamba": "jamba.context_length",
        "mamba": "mamba.context_length",
        "ark": "llama.context_length",
        "bert": "bert.max_position_embeddings",
    }

    _LAYER_KEYS = {
        "llama": "llama.block_count",
        "mistral": "llama.block_count",
        "gemma": "llama.block_count",
        "gemma2": "llama.block_count",
        "phi": "phi.block_count",
        "qwen2": "llama.block_count",
        "command-r": "llama.block_count",
        "stablelm": "llama.block_count",
        "falcon": "llama.block_count",
        "mpt": "llama.block_count",
        "chatglm": "llama.block_count",
        "exaone": "llama.block_count",
        "granite": "llama.block_count",
        "internlm2": "llama.block_count",
        "minicpm": "llama.block_count",
        "olmo": "llama.block_count",
        "olmo2": "llama.block_count",
        "persimmon": "llama.block_count",
        "plamo": "llama.block_count",
        "refact": "llama.block_count",
        "dbrx": "dbrx.block_count",
        "deepseek2": "llama.block_count",
        "jamba": "jamba.block_count",
        "mamba": "mamba.block_count",
        "ark": "llama.block_count",
        "bert": "bert.num_hidden_layers",
    }

    _HEAD_KEYS = {
        "llama": "llama.attention.head_count",
        "mistral": "llama.attention.head_count",
        "gemma": "llama.attention.head_count",
        "gemma2": "llama.attention.head_count",
        "phi": "phi.attention.head_count",
        "qwen2": "llama.attention.head_count",
        "command-r": "llama.attention.head_count",
        "stablelm": "llama.attention.head_count",
        "falcon": "llama.attention.head_count",
        "mpt": "mpt.attention.head_count",
        "chatglm": "llama.attention.head_count",
        "exaone": "llama.attention.head_count",
        "granite": "llama.attention.head_count",
        "internlm2": "llama.attention.head_count",
        "minicpm": "llama.attention.head_count",
        "olmo": "llama.attention.head_count",
        "olmo2": "llama.attention.head_count",
        "persimmon": "llama.attention.head_count",
        "plamo": "llama.attention.head_count",
        "refact": "llama.attention.head_count",
        "deepseek2": "llama.attention.head_count",
        "ark": "llama.attention.head_count",
    }

    _KV_HEAD_KEYS = {
        "llama": "llama.attention.head_count_kv",
        "mistral": "llama.attention.head_count_kv",
        "gemma": "llama.attention.head_count_kv",
        "gemma2": "llama.attention.head_count_kv",
        "phi": "phi.attention.head_count_kv",
        "qwen2": "llama.attention.head_count_kv",
        "command-r": "llama.attention.head_count_kv",
        "stablelm": "llama.attention.head_count_kv",
        "falcon": "llama.attention.head_count_kv",
        "mpt": "mpt.attention.head_count",
        "chatglm": "llama.attention.head_count_kv",
        "exaone": "llama.attention.head_count_kv",
        "granite": "llama.attention.head_count_kv",
        "internlm2": "llama.attention.head_count_kv",
        "minicpm": "llama.attention.head_count_kv",
        "olmo": "llama.attention.head_count_kv",
        "olmo2": "llama.attention.head_count_kv",
        "persimmon": "llama.attention.head_count_kv",
        "plamo": "llama.attention.head_count_kv",
        "refact": "llama.attention.head_count_kv",
        "deepseek2": "llama.attention.head_count_kv",
        "ark": "llama.attention.head_count_kv",
    }

    _EMBEDDING_KEYS = {
        "llama": "llama.embedding_length",
        "mistral": "llama.embedding_length",
        "gemma": "llama.embedding_length",
        "gemma2": "llama.embedding_length",
        "phi": "phi.embedding_length",
        "qwen2": "llama.embedding_length",
        "command-r": "llama.embedding_length",
        "stablelm": "llama.embedding_length",
        "falcon": "llama.embedding_length",
        "mpt": "mpt.embedding_length",
        "chatglm": "llama.embedding_length",
        "exaone": "llama.embedding_length",
        "granite": "llama.embedding_length",
        "internlm2": "llama.embedding_length",
        "minicpm": "llama.embedding_length",
        "olmo": "llama.embedding_length",
        "olmo2": "llama.embedding_length",
        "persimmon": "llama.embedding_length",
        "plamo": "llama.embedding_length",
        "refact": "llama.embedding_length",
        "deepseek2": "llama.embedding_length",
        "ark": "llama.embedding_length",
    }

    # Head dimension overrides (some architectures don't use embd/heads)
    _HEAD_DIM_KEYS = {
        "llama": "llama.attention.key_length",
        "mistral": "llama.attention.key_length",
        "gemma": "llama.attention.key_length",
        "gemma2": "llama.attention.key_length",
        "phi": "phi.attention.key_length",
        "qwen2": "llama.attention.key_length",
        "command-r": "llama.attention.key_length",
        "stablelm": "llama.attention.key_length",
        "falcon": "llama.attention.key_length",
        "chatglm": "llama.attention.key_length",
        "exaone": "llama.attention.key_length",
        "granite": "llama.attention.key_length",
        "internlm2": "llama.attention.key_length",
        "minicpm": "llama.attention.key_length",
        "olmo": "llama.attention.key_length",
        "olmo2": "llama.attention.key_length",
        "persimmon": "llama.attention.key_length",
        "plamo": "llama.attention.key_length",
        "refact": "llama.attention.key_length",
        "deepseek2": "llama.attention.key_length",
        "ark": "llama.attention.key_length",
    }

    # FF dimension keys (for Gemma KV cache V-dim computation)
    _FF_DIM_KEYS = {
        "gemma": "llama.feed_forward_length",
        "gemma2": "llama.feed_forward_length",
    }



    @staticmethod
    def read_full_metadata(model_path: str) -> ModelMetadata:
        """Read complete model metadata from GGUF file header.

        Returns ModelMetadata with architecture params, tensor info, and parameter count.
        Results are cached by (path, mtime, size) so repeated estimates for the
        same file are free.
        """
        cache_key = _metadata_cache_key(model_path)
        if cache_key is not None and cache_key in _METADATA_CACHE:
            return _METADATA_CACHE[cache_key]

        meta = ModelMetadata()

        if not model_path or not os.path.isfile(model_path):
            return meta

        try:
            os.path.getsize(model_path)
        except OSError:
            return meta

        _MAX_HEADER_READ = 32 * 1024 * 1024  # 32 MB: covers large tokenizer metadata sections
        _bytes_read = 0

        # Timeout: abort GGUF read if it takes more than 2 seconds
        # This prevents hanging on slow network files or locked files.
        # (SIGALRM only works in the main thread; worker threads skip this.)
        _read_timeout = 2
        _timeout_handler = None
        if platform.system() != "Windows" and threading.current_thread() is threading.main_thread():
            import signal
            def _handler(signum, frame):
                raise TimeoutError("GGUF read timeout")
            _timeout_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(_read_timeout)

        try:
            with open(model_path, "rb") as f:
                # Read magic
                magic = f.read(4)
                _bytes_read += 4
                if magic != _GgufReader.GGUF_MAGIC:
                    return meta

                # Read version
                version, = struct.unpack("<I", f.read(4))
                _bytes_read += 4
                if version not in (2, 3):
                    return meta

                # Read tensor count and KV count
                tensor_count, = struct.unpack("<Q", f.read(8))
                kv_count, = struct.unpack("<Q", f.read(8))
                _bytes_read += 16

                meta.tensor_count = tensor_count

                # Collect all keys we need from KV pairs
                needed_keys = {_GgufReader._ARCH_KEY}
                for keys_map in (
                    _GgufReader._CONTEXT_KEYS.values(),
                    _GgufReader._LAYER_KEYS.values(),
                    _GgufReader._HEAD_KEYS.values(),
                    _GgufReader._KV_HEAD_KEYS.values(),
                    _GgufReader._EMBEDDING_KEYS.values(),
                    _GgufReader._HEAD_DIM_KEYS.values(),
                    _GgufReader._FF_DIM_KEYS.values(),
                ):
                    needed_keys.update(keys_map)

                # Also look for general name/author for display
                needed_keys.add("general.name")

                # Read KV pairs
                kv_data = {}
                max_kv = min(kv_count, 2000)
                for _ in range(max_kv):
                    if _bytes_read >= _MAX_HEADER_READ:
                        break
                    try:
                        key_len, = struct.unpack("<Q", f.read(8))
                        _bytes_read += 8
                        if key_len > 2048:
                            break
                        key = f.read(key_len).decode("utf-8", errors="replace")
                        _bytes_read += key_len
                        # NOTE: GGUF metadata keys are NOT 8-byte padded (spec).

                        value_type, = struct.unpack("<i", f.read(4))
                        _bytes_read += 4

                        value = _GgufReader._read_value(f, value_type, version)
                        _bytes_read += _GgufReader._value_byte_size(value_type, value)

                        kv_data[key] = value
                    except (struct.error, OSError, UnicodeDecodeError):
                        # Truncated/unreadable tail (e.g. header sample) --
                        # keep everything parsed so far instead of losing it.
                        break

                    if needed_keys.issubset(kv_data.keys()):
                        break

                # Extract architecture string
                arch = kv_data.get(_GgufReader._ARCH_KEY, "unknown")
                if isinstance(arch, bytes):
                    arch = arch.decode("utf-8", errors="replace")
                meta.arch = arch

                # Extract scalar architecture parameters
                meta.layers = _GgufReader._get_int(kv_data, _GgufReader._LAYER_KEYS, arch)
                meta.heads = _GgufReader._get_int(kv_data, _GgufReader._HEAD_KEYS, arch)
                meta.kv_heads = _GgufReader._get_int(kv_data, _GgufReader._KV_HEAD_KEYS, arch)
                if meta.kv_heads == 0 and meta.heads > 0:
                    meta.kv_heads = meta.heads

                meta.embedding_length = _GgufReader._get_int(kv_data, _GgufReader._EMBEDDING_KEYS, arch)
                meta.context_length = _GgufReader._get_int(kv_data, _GgufReader._CONTEXT_KEYS, arch)

                # Head dimension: explicit or computed
                meta.head_dim = _GgufReader._get_int(kv_data, _GgufReader._HEAD_DIM_KEYS, arch)
                if meta.head_dim == 0 and meta.embedding_length > 0 and meta.heads > 0:
                    meta.head_dim = meta.embedding_length // meta.heads

                # FF dimension (for Gemma V-cache)
                meta.ff_dim = _GgufReader._get_int(kv_data, _GgufReader._FF_DIM_KEYS, arch)

                # Gemma uses ff_dim for V-cache dimension (kv_dim = ff_dim / n_heads)
                if arch in ("gemma", "gemma2") and meta.ff_dim > 0 and meta.heads > 0:
                    meta.kv_dim = meta.ff_dim // meta.heads

                # Hybrid SSM/attention architectures (e.g. qwen35): only every
                # Nth layer is full attention and needs KV cache.
                interval = kv_data.get(f"{arch}.full_attention_interval", 1)
                if isinstance(interval, int) and interval > 1:
                    meta.attn_interval = interval

                # ---- Read tensor information (metadata only, first few tensors) ----
                # We read only the first few tensors to get quantization type info.
                # Weight size comes from file size, NOT from tensor data, to avoid
                # seeking past multi-GB tensors which can be slow or unreliable.
                meta.total_params = 0
                _MAX_TENSORS = 10  # Only read first 10 tensors for quant type sampling

                for i in range(min(tensor_count, _MAX_TENSORS)):
                    if _bytes_read >= _MAX_HEADER_READ:
                        break

                    try:
                        # Tensor name
                        name_len, = struct.unpack("<Q", f.read(8))
                        _bytes_read += 8
                        if name_len > 2048:
                            break
                        name = f.read(name_len).decode("utf-8", errors="replace")
                        _bytes_read += name_len
                        # NOTE: GGUF tensor names are NOT 8-byte padded (spec).

                        # Number of dimensions
                        n_dims, = struct.unpack("<I", f.read(4))
                        _bytes_read += 4
                        if n_dims > 4:
                            break

                        # Shape
                        shape = []
                        for _ in range(n_dims):
                            dim, = struct.unpack("<Q", f.read(8))
                            _bytes_read += 8
                            shape.append(dim)

                        # GGML type
                        ggml_type, = struct.unpack("<I", f.read(4))
                        _bytes_read += 4

                        # Track quantization types (from sample of tensors)
                        type_name = _ggml_type_name(ggml_type)
                        meta.quant_types[type_name] = meta.quant_types.get(type_name, 0) + 1

                        # Parameter counting from sample
                        elem_count = 1
                        for d in shape:
                            elem_count *= d
                        type_bits = _GGML_TYPE_BITS.get(ggml_type, 16.0)
                        if type_bits > 0:
                            meta.total_params += int(elem_count * (32.0 / type_bits))

                        # DON'T seek past tensor data - it's too slow and unreliable
                        # Weight size will come from file size instead

                    except struct.error:
                        break

        except (Exception, TimeoutError):
            logger.debug("Suppressed exception", exc_info=True)
        finally:
            # Cancel the alarm
            if _timeout_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, _timeout_handler)

        if cache_key is not None:
            if len(_METADATA_CACHE) >= _METADATA_CACHE_MAX:
                # FIFO eviction of the oldest cached entry
                _METADATA_CACHE.pop(next(iter(_METADATA_CACHE)))
            _METADATA_CACHE[cache_key] = meta

        return meta

    @staticmethod
    def read_architecture(model_path: str) -> dict[str, int]:
        """Backward-compatible: read just architecture parameters."""
        meta = _GgufReader.read_full_metadata(model_path)
        return {
            "arch": meta.arch,
            "layers": meta.layers,
            "heads": meta.heads,
            "kv_heads": meta.kv_heads,
            "embedding_length": meta.embedding_length,
            "context_length": meta.context_length,
        }

    @staticmethod
    def _read_value(f, value_type: int, version: int):
        """Read a single GGUF value."""
        if value_type == 8:  # STRING
            strlen, = struct.unpack("<Q", f.read(8))
            strlen = min(strlen, 65536)
            # GGUF strings are length-prefixed with no NUL terminator (spec).
            return f.read(strlen)
        elif value_type == 9:  # ARRAY
            arr_type, = struct.unpack("<i", f.read(4))
            arr_count, = struct.unpack("<Q", f.read(8))
            # Consume every item to keep the stream in sync; keep the first
            # 500 for inspection (tokenizer arrays can have 100k+ entries).
            items = []
            for idx in range(arr_count):
                item = _GgufReader._read_value(f, arr_type, version)
                if idx < 500:
                    items.append(item)
            return items
        elif value_type in (0, 4, 10):  # UINT8, UINT32, UINT64
            sizes = {0: "<B", 4: "<I", 10: "<Q"}
            return struct.unpack(sizes[value_type], f.read(struct.calcsize(sizes[value_type])))[0]
        elif value_type in (1, 3, 5):  # INT8, INT16, INT32
            sizes = {1: "<b", 3: "<h", 5: "<i"}
            return struct.unpack(sizes[value_type], f.read(struct.calcsize(sizes[value_type])))[0]
        elif value_type == 2:  # UINT16
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == 6:  # FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == 12:  # FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        elif value_type == 7:  # BOOL
            return struct.unpack("<?", f.read(1))[0]
        elif value_type == 11:  # INT64
            return struct.unpack("<q", f.read(8))[0]
        else:
            return None

    @staticmethod
    def _value_byte_size(value_type: int, value) -> int:
        """Estimate bytes consumed by a GGUF value (for seeking past arrays)."""
        if value_type == 8:  # STRING
            if isinstance(value, bytes):
                return 8 + len(value)
            return 0
        elif value_type == 9:  # ARRAY
            return 0  # Handled recursively
        elif value_type in (0, 2, 4, 6, 7):  # UINT8, UINT16, UINT32, FLOAT32, BOOL
            return {0: 1, 2: 2, 4: 4, 6: 4, 7: 1}[value_type]
        elif value_type in (1, 3):  # INT8, INT16
            return {1: 1, 3: 2}[value_type]
        elif value_type in (5, 10, 11):  # INT32, UINT64, INT64
            return {5: 4, 10: 8, 11: 8}[value_type]
        elif value_type == 12:  # FLOAT64
            return 8
        elif value_type in (5, 11):  # INT32, INT64
            return {5: 4, 11: 8}[value_type]
        elif value_type == 10:  # UINT64
            return 8
        elif value_type == 12:  # FLOAT64
            return 8
        return 0

    @staticmethod
    def _estimate_tensor_data_size(shape: list[int], ggml_type: int) -> int:
        """Estimate the size of tensor data in bytes based on shape and type.

        This uses the block-size based calculation that GGUF uses.
        For standard types (F32, F16), it's just elem_count * bytes_per_elem.
        For quantized types, we need to account for block overhead.
        """
        if not shape:
            return 0

        elem_count = 1
        for d in shape:
            elem_count *= d

        # For most quantized types, the block size is 32
        # Block overhead: 2 bytes (d) + 2 bytes (b) per block for Q4/Q5/Q8
        # We use a simplified formula: the _GGML_TYPE_BYTES already accounts for this
        type_bytes = _GGML_TYPE_BYTES.get(ggml_type, 2.0)
        return int(elem_count * type_bytes)

    @staticmethod
    def _get_int(kv_data: dict, key_map: dict, arch: str) -> int:
        """Get an integer value from KV data using architecture-specific key."""
        key = key_map.get(arch)
        if key and key in kv_data:
            try:
                return int(kv_data[key])
            except (ValueError, TypeError):
                logger.debug("Suppressed exception", exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# Architecture defaults (fallback when GGUF can't be read)
# ---------------------------------------------------------------------------


# Modern Qwen architectures use their arch name as the GGUF key prefix
# (e.g. "qwen35.context_length"). Register them against the llama-style keys.
for _arch in ("qwen3", "qwen3moe", "qwen35", "qwen35moe"):
    _GgufReader._CONTEXT_KEYS[_arch] = f"{_arch}.context_length"
    _GgufReader._LAYER_KEYS[_arch] = f"{_arch}.block_count"
    _GgufReader._HEAD_KEYS[_arch] = f"{_arch}.attention.head_count"
    _GgufReader._KV_HEAD_KEYS[_arch] = f"{_arch}.attention.head_count_kv"
    _GgufReader._EMBEDDING_KEYS[_arch] = f"{_arch}.embedding_length"
    _GgufReader._HEAD_DIM_KEYS[_arch] = f"{_arch}.attention.key_length"
    _GgufReader._FF_DIM_KEYS[_arch] = f"{_arch}.feed_forward_length"
del _arch

@dataclass
class _ArchDefaults:
    """Fallback architecture parameters estimated from model size."""
    layers: int
    heads: int
    kv_heads: int
    embedding_length: int


def _estimate_arch_from_size(model_size_mb: float, quant_type: str = "") -> _ArchDefaults:
    """Estimate architecture parameters from model file size."""
    bits_per_param = 4.0
    if quant_type:
        qt = quant_type.lower().replace("_", "").replace("-", "")
        if "q8" in qt:
            bits_per_param = 8.0
        elif "q6" in qt:
            bits_per_param = 6.0
        elif "q5" in qt:
            bits_per_param = 5.0
        elif "q4" in qt:
            bits_per_param = 4.5
        elif "q3" in qt:
            bits_per_param = 3.5
        elif "q2" in qt:
            bits_per_param = 2.5

    total_params = (model_size_mb * 1024 * 1024 * 8) / bits_per_param
    total_params_b = total_params / 1e9

    if total_params_b >= 70:
        return _ArchDefaults(layers=80, heads=64, kv_heads=8, embedding_length=8192)
    elif total_params_b >= 34:
        return _ArchDefaults(layers=48, heads=64, kv_heads=8, embedding_length=8192)
    elif total_params_b >= 13:
        return _ArchDefaults(layers=40, heads=40, kv_heads=8, embedding_length=5120)
    elif total_params_b >= 7:
        return _ArchDefaults(layers=32, heads=32, kv_heads=8, embedding_length=4096)
    elif total_params_b >= 3:
        return _ArchDefaults(layers=28, heads=16, kv_heads=16, embedding_length=3072)
    elif total_params_b >= 1:
        return _ArchDefaults(layers=16, heads=8, kv_heads=8, embedding_length=2048)
    else:
        return _ArchDefaults(layers=8, heads=4, kv_heads=4, embedding_length=1024)


# ---------------------------------------------------------------------------
# Cache type byte sizes
# ---------------------------------------------------------------------------

_CACHE_TYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "q8_0": 1,
    "q5_0": 0.625,
    "q5_1": 0.625,
    "q4_0": 0.5,
    "q4_1": 0.5,
    "iq4_nl": 0.5,
}

_DEFAULT_CACHE_BYTES = 2  # f16


def _get_cache_type_bytes(cache_type: str) -> float:
    return _CACHE_TYPE_BYTES.get(cache_type.lower() if cache_type else "", _DEFAULT_CACHE_BYTES)


# ---------------------------------------------------------------------------
# System RAM detection
# ---------------------------------------------------------------------------

def _get_system_ram_mb() -> float:
    """Get total system RAM in MB."""
    try:
        if platform.system() == "Windows":
            import ctypes
            c_ulonglong = ctypes.c_ulonglong
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            buf = _MemoryStatusEx()
            buf.dwLength = ctypes.sizeof(buf)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(buf))
            return buf.ullTotalPhys / (1024 * 1024)
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 * 1024)
        else:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)
    return 16384.0


# ---------------------------------------------------------------------------
# ResourceEstimator
# ---------------------------------------------------------------------------

class ResourceEstimator:
    """Predicts VRAM and system RAM usage for a given launch configuration.

    Uses GGUF tensor information for accurate weight sizing, architecture metadata
    for KV cache computation, and configurable formulas for compute buffers.
    Supports multi-GPU with layer and row split modes.
    """

    # Fixed overhead estimates (MB) per GPU
    CUDA_RUNTIME_MB = 300
    HIP_RUNTIME_MB = 250
    VULKAN_RUNTIME_MB = 200
    METAL_RUNTIME_MB = 100
    CPU_RUNTIME_MB = 100

    # Flash attention workspace: ~ubatch * hidden * 4 bytes per layer (approximate)
    FLASH_ATTN_WORKSPACE_FACTOR = 4.0

    # Safety margin: percentage of total VRAM estimate
    DEFAULT_SAFETY_MARGIN = 0.07  # 7%

    # Max supported GPUs
    MAX_GPUS = 8

    def __init__(self, safety_margin: float = DEFAULT_SAFETY_MARGIN):
        self._system_ram_mb = _get_system_ram_mb()
        self._safety_margin = safety_margin

    # ---- Public API ----

    def calculate(
        self,
        config: LlamaConfig,
        gpu_info: Optional[GPUInfo] = None,
        model_size_mb: float = 0.0,
        model_family: str = "unknown",
        quant_type: str = "",
        model_path: str = "",
        draft_model_size_mb: float = 0.0,
        mmproj_model_size_mb: float = 0.0,
        total_layer_count: int = 0,
    ) -> ResourceEstimate:
        """Calculate resource estimates for the given configuration."""
        est = ResourceEstimate()

        # No model -- return empty estimate
        if model_size_mb <= 0:
            est.system_total_ram_mb = self._system_ram_mb
            if gpu_info:
                est.gpu_total_vram_mb = float(gpu_info.vram_mb)
                est.gpu_available_vram_mb = float(
                    getattr(gpu_info, "available_vram_mb", gpu_info.vram_mb) or gpu_info.vram_mb
                )
            return est

        est.model_file_size_mb = model_size_mb
        est.model_family = str(model_family) if model_family else "unknown"
        est.quant_type = str(quant_type) if quant_type else "unknown"
        est.system_total_ram_mb = self._system_ram_mb

        # ---- Read GGUF metadata ----
        meta = _GgufReader.read_full_metadata(model_path) if model_path else ModelMetadata()

        # Override with provided total_layer_count if available
        layers = total_layer_count or meta.layers
        heads = meta.heads
        kv_heads = meta.kv_heads
        embedding_length = meta.embedding_length
        head_dim = meta.head_dim
        training_ctx = meta.context_length

        # Fallback: estimate from file size if GGUF couldn't be read
        if not layers:
            defaults = _estimate_arch_from_size(model_size_mb, quant_type)
            layers = defaults.layers
            heads = defaults.heads
            kv_heads = defaults.kv_heads
            embedding_length = defaults.embedding_length

        if head_dim == 0 and embedding_length > 0 and heads > 0:
            head_dim = embedding_length // heads

        if kv_heads == 0 and heads > 0:
            kv_heads = heads

        # Gemma V-cache dimension
        kv_dim = meta.kv_dim
        if kv_dim == 0:
            kv_dim = head_dim

        # Store metadata
        est.model_metadata = meta
        est.layer_count = layers
        est.hidden_size = embedding_length
        est.attention_heads = heads
        est.kv_heads = kv_heads
        est.training_context_length = training_ctx

        # ---- 1. MODEL WEIGHTS ----
        # Use file size as the weight size (model is already quantized on disk)
        est.model_weights_mb = model_size_mb
        est.model_weights_from_gguf_mb = model_size_mb

        # ---- GPU info & multi-GPU setup ----
        gpu_list = self._resolve_gpu_list(gpu_info)
        num_gpus = len(gpu_list)

        # Determine GPU layer count
        gpu_layers = self._resolve_gpu_layers(config.gpu_layers, layers, gpu_info)
        est.split_mode = config.split_mode or "layer"

        if num_gpus > 1:
            est.multi_gpu = True

        # ---- 2. KV CACHE CALCULATION ----
        # Hybrid SSM/attention archs (qwen35): KV cache only exists on the
        # full-attention layers (every attn_interval-th layer).
        kv_layers = layers // meta.attn_interval if meta.attn_interval > 1 else layers
        kv_cache_bytes_per_token = self._compute_kv_cache_per_token(
            kv_layers, kv_heads, head_dim, kv_dim,
            config.cache_type_k, config.cache_type_v,
            meta.arch,
        )
        ctx = max(config.context_size, 256)
        parallel = max(config.parallel if config.parallel > 0 else 1, 1)

        kv_cache_total_bytes = kv_cache_bytes_per_token * ctx * parallel
        kv_cache_total_mb = kv_cache_total_bytes / (1024 * 1024)
        est.kv_cache_total_mb = kv_cache_total_mb

        # ---- 3. COMPUTE BUFFER ESTIMATION ----
        ubatch = max(config.ubatch_size, 1)
        batch = max(config.batch_size, ubatch)

        # Primary compute buffer: forward pass activations
        # Per layer: ~hidden * ubatch * 3 (for Q*K^T, A*V, and output) * sizeof(f16)
        # This is the dominant term and scales with ubatch (not batch)
        compute_buffer_mb = self._compute_inference_buffers(
            embedding_length, layers, ubatch, batch,
            config.flash_attn,
        )
        est.compute_buffers_mb = compute_buffer_mb
        est.inference_buffers_mb = compute_buffer_mb

        # ---- 4. BACKEND OVERHEAD ----
        backend_overhead_mb = 0.0
        if gpu_info and gpu_info.backend == "cuda":
            backend_overhead_mb = self.CUDA_RUNTIME_MB * num_gpus
        elif gpu_info and gpu_info.backend == "hip":
            backend_overhead_mb = self.HIP_RUNTIME_MB * num_gpus
        elif gpu_info and gpu_info.backend == "vulkan":
            backend_overhead_mb = self.VULKAN_RUNTIME_MB * num_gpus
        elif gpu_info and gpu_info.backend == "metal":
            backend_overhead_mb = self.METAL_RUNTIME_MB
        else:
            backend_overhead_mb = self.CPU_RUNTIME_MB

        # Flash attention workspace: llama.cpp reuses a small shared buffer
        # for flash attention; no per-layer workspace is needed.
        flash_attn_workspace_mb = 0.0
        if config.flash_attn in ("on", "auto") and gpu_info and gpu_info.backend in ("cuda", "hip"):
            flash_attn_workspace_mb = (
                ubatch * embedding_length * self.FLASH_ATTN_WORKSPACE_FACTOR
            ) / (1024 * 1024)
            backend_overhead_mb += flash_attn_workspace_mb

        est.backend_overhead_mb = backend_overhead_mb

        # ---- 5. SAFETY MARGIN ----
        est.safety_margin_mb = 0.0

        # ---- 6. GPU/CPU SPLIT ----
        # Split weights between GPU(s) and CPU
        total_weight_units = layers + 2  # +2 for embedding + output layers
        gpu_weight_ratio = gpu_layers / max(total_weight_units, 1)
        cpu_weight_ratio = (total_weight_units - gpu_layers) / max(total_weight_units, 1)

        est.gpu_weights_mb = est.model_weights_mb * gpu_weight_ratio
        est.cpu_weights_mb = est.model_weights_mb * cpu_weight_ratio

        # KV cache split
        if gpu_info and gpu_info.backend != "cpu" and config.kv_offload and gpu_layers > 0:
            est.kv_cache_gpu_mb = kv_cache_total_mb
            est.kv_cache_cpu_mb = 0.0
        else:
            est.kv_cache_gpu_mb = 0.0
            est.kv_cache_cpu_mb = kv_cache_total_mb

        # Compute buffer split (proportional to GPU layers)
        est.compute_buffers_gpu_mb = compute_buffer_mb * gpu_weight_ratio
        est.compute_buffers_cpu_mb = compute_buffer_mb * cpu_weight_ratio

        # ---- 7. MULTI-GPU DISTRIBUTION ----
        if num_gpus > 1 and gpu_layers > 0:
            est.gpu_breakdowns = self._distribute_across_gpus(
                gpu_list, gpu_layers, layers,
                est.gpu_weights_mb, est.kv_cache_gpu_mb,
                est.compute_buffers_gpu_mb,
                backend_overhead_mb, est.safety_margin_mb,
                config.split_mode, config.tensor_split,
                draft_model_size_mb, mmproj_model_size_mb,
            )
            est.safety_margin_mb = self._calculate_safety_margin(
                sum(gb.total_used_mb for gb in est.gpu_breakdowns)
            )
            # Update GPU breakdowns with safety margin
            for gb in est.gpu_breakdowns:
                gb.safety_margin_mb = est.safety_margin_mb / num_gpus
        elif num_gpus == 1 and gpu_list and gpu_layers > 0:
            # Single GPU
            gpu = gpu_list[0]
            gb = GpuBreakdown(
                gpu_index=0,
                gpu_name=gpu.device_name or f"GPU 0",
                total_vram_mb=gpu.vram_mb,
                available_vram_mb=gpu.available_vram_mb,
                model_weights_mb=est.gpu_weights_mb,
                kv_cache_mb=est.kv_cache_gpu_mb,
                compute_buffers_mb=est.compute_buffers_gpu_mb,
                backend_overhead_mb=backend_overhead_mb,
            )
            est.gpu_breakdowns = [gb]

        # ---- 8. AUXILIARY MODELS ----
        est.draft_model_mb = draft_model_size_mb
        est.mmproj_model_mb = mmproj_model_size_mb

        if gpu_layers > 0 and mmproj_model_size_mb > 0:
            est.gpu_weights_mb += mmproj_model_size_mb
            if est.gpu_breakdowns:
                est.gpu_breakdowns[0].model_weights_mb += mmproj_model_size_mb

        # ---- 9. TOTALS ----
        # Single-GPU / backward-compatible breakdown
        vram_bd = VramBreakdown(
            gpu_weights_mb=est.gpu_weights_mb,
            kv_cache_mb=est.kv_cache_gpu_mb,
            inference_buffers_mb=est.compute_buffers_gpu_mb,
            cuda_runtime_mb=backend_overhead_mb if gpu_info and gpu_info.backend in ("cuda", "hip") else 0,
            temp_allocations_mb=flash_attn_workspace_mb,
            backend_overhead_mb=backend_overhead_mb if config.flash_attn != "enable" else 0,
            safety_margin_mb=est.safety_margin_mb / max(num_gpus, 1),
            draft_model_mb=est.draft_model_mb if gpu_layers > 0 else 0.0,
            mmproj_model_mb=est.mmproj_model_mb if gpu_layers > 0 else 0.0,
        )

        ram_bd = RamBreakdown(
            cpu_weights_mb=est.cpu_weights_mb,
            kv_cache_mb=est.kv_cache_cpu_mb,
            inference_buffers_mb=est.compute_buffers_cpu_mb,
            mmap_overhead_mb=model_size_mb * 0.05 if config.use_mmap else 0.0,
            runtime_overhead_mb=self.CPU_RUNTIME_MB,
            draft_model_mb=est.draft_model_mb if gpu_layers <= 0 else 0.0,
            mmproj_model_mb=est.mmproj_model_mb if gpu_layers <= 0 else 0.0,
        )

        est.vram_breakdown = vram_bd
        est.ram_breakdown = ram_bd

        if num_gpus > 1:
            est.total_vram_mb = sum(gb.total_used_mb for gb in est.gpu_breakdowns)
        else:
            est.total_vram_mb = vram_bd.total_mb

        est.total_ram_mb = ram_bd.total_mb

        # Estimated VRAM if all layers on GPU (for display when no GPU)
        est.estimated_vram_mb = (
            est.model_weights_mb +
            kv_cache_total_mb +
            compute_buffer_mb +
            backend_overhead_mb +
            est.draft_model_mb +
            est.mmproj_model_mb
        )

        # ---- GPU capacity ----
        if gpu_info:
            est.gpu_total_vram_mb = float(gpu_info.vram_mb)
            est.gpu_available_vram_mb = float(
                getattr(gpu_info, "available_vram_mb", gpu_info.vram_mb) or gpu_info.vram_mb
            )

        if num_gpus > 1:
            est.gpu_total_vram_mb = sum(gb.total_vram_mb for gb in est.gpu_breakdowns)
            est.gpu_available_vram_mb = sum(gb.available_vram_mb for gb in est.gpu_breakdowns)

        if est.gpu_total_vram_mb > 0:
            est.gpu_remaining_vram_mb = est.gpu_available_vram_mb - est.total_vram_mb

        # ---- 10. WARNINGS ----
        est.warnings = []

        if est.gpu_total_vram_mb > 0 and est.total_vram_mb > est.gpu_total_vram_mb:
            est.vram_exceeded = True
            est.warnings.append(
                f"Estimated VRAM usage ({self._fmt_mb(est.total_vram_mb)}) exceeds "
                f"available VRAM ({self._fmt_mb(est.gpu_total_vram_mb)}). "
                f"Consider reducing gpu_layers, context size, or batch size."
            )
        elif est.gpu_total_vram_mb > 0 and est.vram_percentage > 90:
            est.warnings.append(
                f"VRAM usage is at {est.vram_percentage:.0f}%. "
                f"Configuration is near VRAM limit."
            )

        if est.total_ram_mb > est.system_total_ram_mb * 0.9:
            est.ram_warning = True
            est.warnings.append(
                f"Estimated RAM usage ({self._fmt_mb(est.total_ram_mb)}) is near "
                f"system limit ({self._fmt_mb(est.system_total_ram_mb)})."
            )

        # Multi-GPU: check if any single GPU is the bottleneck
        if est.gpu_breakdowns:
            for gb in est.gpu_breakdowns:
                if gb.usage_percentage > 100:
                    est.warnings.append(
                        f"GPU {gb.gpu_index} ({gb.gpu_name}) exceeds capacity: "
                        f"{self._fmt_mb(gb.total_used_mb)} / {self._fmt_mb(gb.available_vram_mb)}. "
                        f"Even with multiple GPUs, this GPU is the limiting factor."
                    )

        if config.embeddings:
            est.warnings.append(
                "Embeddings mode detected - KV cache will not be used. "
                "Actual VRAM usage will be lower."
            )

        if training_ctx > 0 and ctx > training_ctx:
            est.warnings.append(
                f"Selected context ({ctx} tokens) exceeds the model's trained context "
                f"({training_ctx} tokens). RoPE scaling will be used and quality may degrade."
            )

        return est

    # ---- Internal helpers ----

    def _resolve_gpu_list(self, gpu_info: Optional[GPUInfo]) -> list[GPUInfo]:
        """Resolve GPU info into a list of individual GPUs."""
        if not gpu_info or gpu_info.backend == "cpu":
            return []

        # Check if gpu_info contains nested gpu_infos (multi-GPU detection)
        gpu_infos = getattr(gpu_info, "gpu_infos", None)
        if gpu_infos and len(gpu_infos) > 1:
            return gpu_infos[:self.MAX_GPUS]

        # Single GPU or gpu_count > 1 but no detailed info
        gpu_count = getattr(gpu_info, "gpu_count", 1)
        if gpu_count > 1:
            # Create placeholder GPUs from the single detected GPU
            gpus = []
            for i in range(min(gpu_count, self.MAX_GPUS)):
                gpus.append(GPUInfo(
                    backend=gpu_info.backend,
                    vram_mb=gpu_info.vram_mb,
                    available_vram_mb=gpu_info.available_vram_mb,
                    device_name=f"{gpu_info.device_name} :{i}" if gpu_info.device_name else f"GPU {i}",
                    driver_version=gpu_info.driver_version,
                ))
            return gpus

        return [gpu_info]

    @staticmethod
    def _resolve_gpu_layers(gpu_layers_cfg: int, total_layers: int, gpu_info: Optional[GPUInfo]) -> int:
        """Resolve the actual number of GPU layers."""
        if not gpu_info or gpu_info.backend == "cpu" or gpu_info.vram_mb == 0:
            return 0
        if gpu_layers_cfg == -1:
            return total_layers
        if gpu_layers_cfg == 0:
            return 0
        return min(gpu_layers_cfg, total_layers)

    def _compute_kv_cache_per_token(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_size: int,
        kv_dim: int,
        cache_type_k: str,
        cache_type_v: str,
        arch: str,
    ) -> int:
        """Compute KV cache size in bytes per token per sequence.

        Formula (llama.cpp):
            For most architectures:
                K_cache_per_layer = n_kv_heads * head_size * sizeof(type_k)
                V_cache_per_layer = n_kv_heads * head_size * sizeof(type_v)
            For Gemma:
                V_cache_per_layer = n_kv_heads * (ff_dim / n_heads) * sizeof(type_v)

            Total = n_layers * (K_cache + V_cache)
        """
        bytes_k = _get_cache_type_bytes(cache_type_k)
        bytes_v = _get_cache_type_bytes(cache_type_v)

        k_cache = n_kv_heads * head_size * bytes_k

        # Gemma uses different V-cache dimension
        if arch in ("gemma", "gemma2") and kv_dim > 0:
            v_cache = n_kv_heads * kv_dim * bytes_v
        else:
            v_cache = n_kv_heads * head_size * bytes_v

        per_layer = k_cache + v_cache
        return int(n_layers * per_layer)

    def _compute_inference_buffers(
        self,
        n_embd: int,
        n_layers: int,
        ubatch: int,
        batch: int,
        flash_attn: str,
    ) -> float:
        """Compute inference buffer size in MB.

        The primary VRAM impact is ubatch because it controls the actual
        processing chunk size.

        Components:
        1. Layer activations: hidden * ubatch * 3 * sizeof(f16) * layers
           - 3 = forward pass (residual, gate, output)
        2. Batch buffer: hidden * batch * sizeof(f16)
           - Token embeddings buffer for the logical batch
        3. Attention workspace (if not flash attn):
           - Q*K^T: ubatch * ctx * sizeof(f32)
           - A*V:   ubatch * hidden * sizeof(f32)
        """
        if n_embd <= 0 or n_layers <= 0:
            return 0.0

        # Layer activations (dominant term, scales with ubatch)
        layer_activations = (n_embd * ubatch * 3 * 2 * n_layers) / (1024 * 1024)

        # Token embeddings buffer
        token_embeddings = (n_embd * batch * 2) / (1024 * 1024)

        # Attention workspace (non-flash attention)
        attn_workspace = 0.0
        if flash_attn != "on":
            # Conservative estimate: ubatch * ubatch * sizeof(f16) for QK^T intermediate
            attn_workspace = (ubatch * ubatch * 2 * n_layers) / (1024 * 1024)

        total_mb = layer_activations + token_embeddings + attn_workspace
        return total_mb * 1.25  # 25% headroom for temporary allocations

    def _calculate_safety_margin(self, total_vram_used_mb: float) -> float:
        """Calculate safety margin as percentage of total estimated VRAM."""
        return total_vram_used_mb * self._safety_margin

    def _distribute_across_gpus(
        self,
        gpu_list: list[GPUInfo],
        gpu_layers: int,
        total_layers: int,
        gpu_weights_mb: float,
        kv_cache_mb: float,
        compute_mb: float,
        overhead_mb: float,
        margin_mb: float,
        split_mode: str,
        tensor_split: Optional[str],
        draft_mb: float,
        mmproj_mb: float,
    ) -> list[GpuBreakdown]:
        """Distribute memory across multiple GPUs."""
        num_gpus = len(gpu_list)
        breakdowns = []

        if not tensor_split:
            # Auto-distribute based on split mode
            if split_mode == "layer":
                # Layer split: distribute layers proportionally to VRAM
                total_vram = sum(g.available_vram_mb for g in gpu_list)
                layers_per_gpu = []
                remaining_layers = gpu_layers
                for i, gpu in enumerate(gpu_list):
                    if i == num_gpus - 1:
                        layers_per_gpu.append(remaining_layers)
                    else:
                        share = max(1, int(gpu_layers * gpu.available_vram_mb / total_vram))
                        share = min(share, remaining_layers - (num_gpus - i - 1))
                        layers_per_gpu.append(share)
                        remaining_layers -= share

                total_weight_units = total_layers + 2
                for i, gpu in enumerate(gpu_list):
                    layer_share = layers_per_gpu[i]
                    weight_mb = gpu_weights_mb * (layer_share / max(total_weight_units, 1))
                    kv_mb = kv_cache_mb / num_gpus  # KV cache typically split evenly
                    comp_mb = compute_mb * (layer_share / max(gpu_layers, 1))
                    over_mb = overhead_mb / num_gpus

                    breakdowns.append(GpuBreakdown(
                        gpu_index=i,
                        gpu_name=gpu.device_name or f"GPU {i}",
                        total_vram_mb=gpu.vram_mb,
                        available_vram_mb=gpu.available_vram_mb,
                        model_weights_mb=weight_mb,
                        kv_cache_mb=kv_mb,
                        compute_buffers_mb=comp_mb,
                        backend_overhead_mb=over_mb,
                        safety_margin_mb=margin_mb / num_gpus,
                        auxiliary_mb=mmproj_mb if i == 0 else 0.0,
                    ))

            elif split_mode == "row":
                # Row split: weights split by rows across GPUs
                # All GPUs participate in each layer, weights are split by row
                for i, gpu in enumerate(gpu_list):
                    breakdowns.append(GpuBreakdown(
                        gpu_index=i,
                        gpu_name=gpu.device_name or f"GPU {i}",
                        total_vram_mb=gpu.vram_mb,
                        available_vram_mb=gpu.available_vram_mb,
                        model_weights_mb=gpu_weights_mb / num_gpus,
                        kv_cache_mb=kv_cache_mb / num_gpus,
                        compute_buffers_mb=compute_mb / num_gpus,
                        backend_overhead_mb=overhead_mb / num_gpus,
                        safety_margin_mb=margin_mb / num_gpus,
                        auxiliary_mb=mmproj_mb if i == 0 else 0.0,
                    ))
            else:
                # Default: distribute evenly
                return self._distribute_across_gpus(
                    gpu_list, gpu_layers, total_layers,
                    gpu_weights_mb, kv_cache_mb, compute_mb,
                    overhead_mb, margin_mb, "layer", tensor_split,
                    draft_mb, mmproj_mb,
                )
        else:
            # Explicit tensor split ratios (e.g., "3,1" means 75%/25%)
            try:
                ratios = [float(x.strip()) for x in tensor_split.split(",")]
                ratios = ratios[:num_gpus]
                while len(ratios) < num_gpus:
                    ratios.append(1.0)
                total_ratio = sum(ratios)

                for i, gpu in enumerate(gpu_list):
                    ratio = ratios[i] / total_ratio
                    breakdowns.append(GpuBreakdown(
                        gpu_index=i,
                        gpu_name=gpu.device_name or f"GPU {i}",
                        total_vram_mb=gpu.vram_mb,
                        available_vram_mb=gpu.available_vram_mb,
                        model_weights_mb=gpu_weights_mb * ratio,
                        kv_cache_mb=kv_cache_mb * ratio,
                        compute_buffers_mb=compute_mb * ratio,
                        backend_overhead_mb=overhead_mb * ratio,
                        safety_margin_mb=margin_mb * ratio,
                        auxiliary_mb=mmproj_mb if i == 0 else 0.0,
                    ))
            except (ValueError, IndexError):
                # Fallback to even distribution
                return self._distribute_across_gpus(
                    gpu_list, gpu_layers, total_layers,
                    gpu_weights_mb, kv_cache_mb, compute_mb,
                    overhead_mb, margin_mb, "layer", None,
                    draft_mb, mmproj_mb,
                )

        return breakdowns

    @staticmethod
    def _fmt_mb(mb: float) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"
