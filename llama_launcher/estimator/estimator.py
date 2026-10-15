"""Resource Estimator - predicts VRAM and RAM usage before launching llama-server.

All sizing logic lives here. The GUI never performs memory calculations.

The model follows what llama-server actually allocates, as read from its own
``load_tensors`` / ``llama_kv_cache`` / ``sched_reserve`` log lines on a real
GPU (see ``docs/ESTIMATOR.md``):

- Model weights: per-tensor sizes from the GGUF tensor directory. The output
  layer is offloaded first, then the last ``N-1`` repeating layers; the token
  embedding always stays in host memory.
- KV cache: per attention layer, ``ctx x kv_heads x (key_len + value_len)``
  at the cache type's bytes-per-element. Sliding-window layers are capped at
  the window, recurrent (SSM) layers have a small per-slot state instead.
  The cache lives wherever its layer lives.
- Compute buffer: proportional to the micro-batch, plus the attention mask
  (or, without flash attention, the full score matrix) over the context.
- Runtime: a fixed per-GPU allocation for the CUDA context.
"""
import logging
import os
import platform
from dataclasses import dataclass
from typing import List, Optional

from llama_launcher.core.config import LlamaConfig
from llama_launcher.gpu.types import GPUInfo
from .gguf import FILE_TYPES, GGML_TYPES, GgufFormatError, read_gguf
from .types import (
    ResourceEstimate,
    VramBreakdown,
    RamBreakdown,
    ModelMetadata,
    GpuBreakdown,
)

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: Cache for GGUF metadata, keyed by (path, mtime_ns, size). Avoids re-parsing
#: model headers on every config change.
_METADATA_CACHE: dict = {}
_METADATA_CACHE_MAX = 64

#: Ignore this much resident VRAM before assuming another model is loaded --
#: a desktop compositor alone accounts for a few hundred MB.
_VRAM_IN_USE_FLOOR_MB = 512

#: Sliding-window attention pattern (every Nth layer is full attention) for
#: architectures that do not store it in the GGUF.
_SWA_PATTERNS = {"gemma2": 2, "gemma3": 6, "gemma3n": 6, "llama4": 4, "cohere2": 4,
                 "gpt-oss": 2, "olmo3": 4}


def _metadata_cache_key(model_path: str):
    """Return a cache key for a model file, or None if it can't be stat'ed."""
    if not model_path:
        return None
    try:
        st = os.stat(model_path)
    except OSError:
        return None
    return (model_path, st.st_mtime_ns, st.st_size)


# ---------------------------------------------------------------------------
# GGUF -> ModelMetadata
# ---------------------------------------------------------------------------

class _GgufReader:
    """Builds :class:`ModelMetadata` from a GGUF header (cached per file)."""

    @staticmethod
    def read_full_metadata(model_path: str) -> ModelMetadata:
        """Read architecture parameters and tensor sizes from a GGUF file.

        Never raises: an unreadable or non-GGUF file yields an empty
        ``ModelMetadata`` (arch "unknown") so the caller falls back to
        size-based guesses.
        """
        cache_key = _metadata_cache_key(model_path)
        if cache_key is not None and cache_key in _METADATA_CACHE:
            return _METADATA_CACHE[cache_key]

        meta = ModelMetadata()
        if cache_key is None or not os.path.isfile(model_path):
            return meta

        try:
            gguf = read_gguf(model_path)
            meta = _GgufReader._from_gguf(gguf)
        except GgufFormatError as e:
            logger.debug("Not a usable GGUF file %s: %s", model_path, e)
        except OSError:
            logger.debug("Could not read %s", model_path, exc_info=True)
        except Exception:
            logger.exception("Unexpected error reading GGUF %s", model_path)

        if len(_METADATA_CACHE) >= _METADATA_CACHE_MAX:
            _METADATA_CACHE.pop(next(iter(_METADATA_CACHE)))
        _METADATA_CACHE[cache_key] = meta
        return meta

    @staticmethod
    def _from_gguf(gguf) -> ModelMetadata:
        meta = ModelMetadata(arch=gguf.architecture, tensor_count=len(gguf.tensors))
        meta.layers = gguf.get_int("block_count")
        meta.heads = gguf.get_int("attention.head_count")
        meta.embedding_length = gguf.get_int("embedding_length")
        meta.context_length = gguf.get_int("context_length")
        meta.ff_dim = gguf.get_int("feed_forward_length")
        meta.n_expert = gguf.get_int("expert_count")
        meta.n_expert_used = gguf.get_int("expert_used_count")
        meta.expert_ff_dim = gguf.get_int("expert_feed_forward_length")
        meta.shared_expert_ff_dim = gguf.get_int("expert_shared_feed_forward_length")
        meta.vocab_size = gguf.get_int("vocab_size")
        template = gguf.metadata.get("tokenizer.chat_template")
        if isinstance(template, str):
            meta.chat_template = template

        # Head counts may be per-layer arrays (e.g. models with pruned heads).
        kv_list = gguf.get_int_list("attention.head_count_kv")
        if kv_list:
            meta.kv_heads_per_layer = kv_list
            meta.kv_heads = max(kv_list)
        else:
            meta.kv_heads = gguf.get_int("attention.head_count_kv") or meta.heads
        head_list = gguf.get_int_list("attention.head_count")
        if head_list:
            meta.heads = max(head_list)

        if meta.embedding_length and meta.heads:
            meta.head_dim = meta.embedding_length // meta.heads
        meta.key_length = gguf.get_int("attention.key_length") or meta.head_dim
        meta.value_length = gguf.get_int("attention.value_length") or meta.key_length
        meta.kv_dim = meta.value_length
        if not meta.head_dim:
            meta.head_dim = meta.key_length

        interval = gguf.get_int("full_attention_interval")
        if interval > 1:
            meta.attn_interval = interval
        meta.swa_window = gguf.get_int("attention.sliding_window")
        if meta.swa_window:
            meta.swa_pattern = (gguf.get_int("attention.sliding_window_pattern")
                                or _SWA_PATTERNS.get(meta.arch, 0))

        meta.ssm_conv_kernel = gguf.get_int("ssm.conv_kernel")
        meta.ssm_inner_size = gguf.get_int("ssm.inner_size")
        meta.ssm_state_size = gguf.get_int("ssm.state_size")
        meta.ssm_group_count = gguf.get_int("ssm.group_count")

        file_type = gguf.metadata.get("general.file_type")
        if isinstance(file_type, int):
            meta.file_type = FILE_TYPES.get(file_type, "")

        # ---- tensor directory ----
        n_layer = max(meta.layers, 0)
        layer_bytes = [0] * n_layer
        expert_bytes = [0] * n_layer
        for t in gguf.tensors:
            nbytes = t.n_bytes
            meta.total_bytes += nbytes
            meta.total_params += t.n_elements
            meta.quant_types[t.type_name] = meta.quant_types.get(t.type_name, 0) + 1
            name = t.name
            if name.startswith("blk."):
                try:
                    idx = int(name.split(".", 2)[1])
                except (IndexError, ValueError):
                    meta.other_bytes += nbytes
                    continue
                if idx >= len(layer_bytes):
                    layer_bytes.extend([0] * (idx + 1 - len(layer_bytes)))
                    expert_bytes.extend([0] * (idx + 1 - len(expert_bytes)))
                layer_bytes[idx] += nbytes
                if "_exps." in name or name.endswith("_exps.weight"):
                    expert_bytes[idx] += nbytes
            elif name == "token_embd.weight":
                meta.embd_bytes += nbytes
            elif name.startswith("output.") :
                meta.output_bytes += nbytes
            else:
                meta.other_bytes += nbytes
        meta.layer_bytes = layer_bytes
        meta.expert_bytes = expert_bytes
        if not meta.layers and layer_bytes:
            meta.layers = len(layer_bytes)
        if not meta.vocab_size and meta.embedding_length:
            for t in gguf.tensors:
                if t.name == "token_embd.weight" and len(t.shape) == 2:
                    meta.vocab_size = t.shape[1]
                    break
        return meta


# ---------------------------------------------------------------------------
# Fallbacks when the GGUF cannot be read (remote models, non-GGUF files)
# ---------------------------------------------------------------------------

@dataclass
class _ArchDefaults:
    layers: int
    heads: int
    kv_heads: int
    embedding_length: int
    ff_dim: int
    vocab_size: int


def _estimate_arch_from_size(model_size_mb: float, quant_type: str = "") -> _ArchDefaults:
    """Guess architecture parameters from the file size and quant label."""
    bits_per_param = 4.5
    qt = (quant_type or "").lower()
    for marker, bits in (("q8", 8.5), ("q6", 6.6), ("q5", 5.7), ("q4", 4.8),
                         ("q3", 3.9), ("q2", 3.0), ("iq4", 4.5), ("iq3", 3.5),
                         ("iq2", 2.6), ("f16", 16), ("bf16", 16), ("f32", 32)):
        if marker in qt:
            bits_per_param = bits
            break
    total_params_b = (model_size_mb * MIB * 8) / bits_per_param / 1e9

    if total_params_b >= 60:
        return _ArchDefaults(80, 64, 8, 8192, 28672, 128256)
    if total_params_b >= 25:
        return _ArchDefaults(64, 40, 8, 5120, 25600, 151936)
    if total_params_b >= 11:
        return _ArchDefaults(40, 40, 8, 5120, 13824, 151936)
    if total_params_b >= 6:
        return _ArchDefaults(32, 32, 8, 4096, 14336, 128256)
    if total_params_b >= 2.5:
        return _ArchDefaults(36, 16, 2, 2048, 11008, 151936)
    if total_params_b >= 0.8:
        return _ArchDefaults(28, 12, 2, 1536, 8960, 151936)
    return _ArchDefaults(24, 14, 2, 896, 4864, 151936)


def _cache_type_bytes(cache_type: str) -> float:
    """Bytes per cached element for a ``--cache-type-k/v`` value."""
    wanted = (cache_type or "f16").strip().upper()
    for name, block_elems, block_bytes in GGML_TYPES.values():
        if name == wanted:
            return block_bytes / block_elems
    return 2.0


def _get_system_ram_mb() -> float:
    """Total system RAM in MB (16 GB if it cannot be determined)."""
    try:
        if platform.system() == "Windows":
            import ctypes
            c_ulonglong = ctypes.c_ulonglong

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong), ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong), ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong), ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            buf = _MemoryStatusEx()
            buf.dwLength = ctypes.sizeof(buf)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(buf))
            return buf.ullTotalPhys / MIB
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return int(result.stdout.strip()) / MIB
        else:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024
    except Exception:
        logger.debug("Could not read system RAM", exc_info=True)
    return 16384.0


# ---------------------------------------------------------------------------
# Layer placement, shared by weights, KV cache and recurrent state
# ---------------------------------------------------------------------------

@dataclass
class _Placement:
    """Which parts of the model the estimate assumes are on the GPU."""
    n_layer: int
    offloaded: int            # layers on GPU counting the output layer, as llama-server reports
    gpu_layers: List[int]     # indices of repeating layers on the GPU
    output_on_gpu: bool

    @property
    def offloadable(self) -> int:
        return self.n_layer + 1

    def layer_on_gpu(self, index: int) -> bool:
        return index in self._gpu_set

    def __post_init__(self):
        self._gpu_set = set(self.gpu_layers)


def _place_layers(n_layer: int, gpu_layers_cfg: int, has_gpu: bool) -> _Placement:
    """Mirror llama.cpp's offload order: output layer first, then the last layers."""
    if not has_gpu or n_layer <= 0:
        return _Placement(n_layer, 0, [], False)
    if gpu_layers_cfg < 0:
        offloaded = n_layer + 1
    else:
        offloaded = min(gpu_layers_cfg, n_layer + 1)
    if offloaded <= 0:
        return _Placement(n_layer, 0, [], False)
    n_blk = offloaded - 1
    return _Placement(n_layer, offloaded, list(range(n_layer - n_blk, n_layer)), True)


# ---------------------------------------------------------------------------
# ResourceEstimator
# ---------------------------------------------------------------------------

class ResourceEstimator:
    """Predicts VRAM and system RAM usage for a given launch configuration."""

    #: Fixed per-GPU runtime allocation (MB). CUDA measured at ~570 MiB on an
    #: RTX 5090 / CUDA 13 regardless of model or context; older driver stacks
    #: sit a little lower.
    CUDA_RUNTIME_MB = 570
    HIP_RUNTIME_MB = 450
    VULKAN_RUNTIME_MB = 300
    METAL_RUNTIME_MB = 150
    #: llama-server process overhead in RAM beyond its buffers (threads, HTTP
    #: server, tokenizer, CUDA host allocations).
    CPU_RUNTIME_MB = 400

    #: Default server slots when --parallel is auto (llama-server picks 4).
    DEFAULT_SLOTS = 4

    #: Extra VRAM for a vision projector's image-encoding buffers.
    MMPROJ_COMPUTE_MB = 256

    # Safety margin: percentage of total VRAM estimate (multi-GPU only).
    DEFAULT_SAFETY_MARGIN = 0.05
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
        system_ram_mb: float = -1.0,
        draft_model_path: str = "",
        mmproj_model_path: str = "",
    ) -> ResourceEstimate:
        """Calculate resource estimates for the given configuration.

        Args:
            system_ram_mb: RAM of the machine the model will run on. Defaults to
                this machine's, which is wrong once a remote server is selected.
            draft_model_path/mmproj_model_path: read for their own tensor
                sizes when they are local files; otherwise the ``*_size_mb``
                figures are used.
        """
        est = ResourceEstimate()
        effective_ram_mb = (system_ram_mb if system_ram_mb and system_ram_mb > 0
                            else self._system_ram_mb)
        est.system_total_ram_mb = effective_ram_mb

        if model_size_mb <= 0:
            if gpu_info:
                est.gpu_total_vram_mb = float(gpu_info.vram_mb)
                est.gpu_available_vram_mb = float(gpu_info.available_vram_mb or gpu_info.vram_mb)
            return est

        est.model_file_size_mb = model_size_mb
        est.model_family = str(model_family) if model_family else "unknown"
        est.quant_type = str(quant_type) if quant_type else "unknown"

        meta = _GgufReader.read_full_metadata(model_path) if model_path else ModelMetadata()
        est.model_metadata = meta
        geo = self._geometry(meta, model_size_mb, quant_type, total_layer_count)
        n_layer = geo.layers
        est.layer_count = n_layer
        est.hidden_size = geo.embedding_length
        est.attention_heads = geo.heads
        est.kv_heads = geo.kv_heads
        est.training_context_length = meta.context_length

        # ---- GPU topology ----
        gpu_list = self._resolve_gpu_list(gpu_info)
        num_gpus = len(gpu_list)
        has_gpu = num_gpus > 0 and any(g.vram_mb > 0 for g in gpu_list)
        backend = gpu_info.backend if gpu_info else "cpu"
        est.split_mode = config.split_mode or "layer"
        est.multi_gpu = num_gpus > 1

        placement = _place_layers(n_layer, config.gpu_layers, has_gpu)
        est.offloaded_layers = placement.offloaded
        est.offloadable_layers = placement.offloadable

        # ---- 1. Weights ----
        gpu_w, cpu_w = self._split_weights(meta, geo, model_size_mb, placement, config)
        est.model_weights_mb = (meta.total_bytes / MIB) if meta.has_tensor_sizes else model_size_mb
        est.model_weights_from_gguf_mb = est.model_weights_mb
        est.gpu_weights_mb = gpu_w
        est.cpu_weights_mb = cpu_w

        # ---- 2. KV cache + recurrent state ----
        ctx = max(int(config.context_size), 256)
        ubatch = max(int(config.ubatch_size), 1)
        slots = config.parallel if config.parallel and config.parallel > 0 else self.DEFAULT_SLOTS
        kv_gpu, kv_cpu, rs_gpu, rs_cpu = self._kv_and_state(
            meta, geo, placement, config, ctx, ubatch, slots)
        est.kv_cache_total_mb = kv_gpu + kv_cpu
        est.kv_cache_gpu_mb = kv_gpu
        est.kv_cache_cpu_mb = kv_cpu
        est.temp_allocations_mb = rs_gpu

        # ---- 3. Compute buffers ----
        flash = self._flash_attention_on(config.flash_attn, backend)
        experts_on_cpu = meta.is_moe and (config.cpu_moe or (config.n_cpu_moe or 0) > 0)
        compute_gpu, compute_cpu = self._compute_buffers(
            geo, placement, ctx, ubatch, flash, has_gpu, experts_on_cpu)
        est.compute_buffers_mb = compute_gpu + compute_cpu
        est.compute_buffers_gpu_mb = compute_gpu
        est.compute_buffers_cpu_mb = compute_cpu
        est.inference_buffers_mb = est.compute_buffers_mb

        # ---- 4. Runtime overhead ----
        runtime_per_gpu = self._runtime_mb(backend)
        gpu_runtime_mb = runtime_per_gpu * num_gpus if has_gpu else 0.0
        est.runtime_overhead_mb = self.CPU_RUNTIME_MB
        est.backend_overhead_mb = gpu_runtime_mb

        # ---- 5. Auxiliary models ----
        draft_gpu, draft_cpu = self._auxiliary_model(
            draft_model_path, draft_model_size_mb,
            on_gpu=has_gpu and config.spec_draft_ngl != 0,
            ctx=ctx, config=config, slots=slots, backend=backend)
        mmproj_gpu, mmproj_cpu = self._auxiliary_model(
            mmproj_model_path, mmproj_model_size_mb,
            on_gpu=has_gpu and config.mmproj_offload, ctx=0, config=config,
            slots=slots, backend=backend)
        if mmproj_gpu > 0:
            mmproj_gpu += self.MMPROJ_COMPUTE_MB
        est.draft_model_mb = draft_gpu + draft_cpu
        est.mmproj_model_mb = mmproj_gpu + mmproj_cpu

        # ---- 6. Totals ----
        is_cuda = backend in ("cuda", "hip")
        vram_bd = VramBreakdown(
            gpu_weights_mb=gpu_w,
            kv_cache_mb=kv_gpu,
            inference_buffers_mb=compute_gpu,
            cuda_runtime_mb=gpu_runtime_mb if is_cuda else 0.0,
            temp_allocations_mb=rs_gpu,
            backend_overhead_mb=gpu_runtime_mb if (has_gpu and not is_cuda) else 0.0,
            draft_model_mb=draft_gpu,
            mmproj_model_mb=mmproj_gpu,
        )
        ram_bd = RamBreakdown(
            cpu_weights_mb=cpu_w,
            kv_cache_mb=kv_cpu + rs_cpu,
            inference_buffers_mb=compute_cpu,
            mmap_overhead_mb=0.0,
            runtime_overhead_mb=self.CPU_RUNTIME_MB,
            draft_model_mb=draft_cpu,
            mmproj_model_mb=mmproj_cpu,
        )
        est.vram_breakdown = vram_bd
        est.ram_breakdown = ram_bd
        est.total_vram_mb = vram_bd.total_mb if has_gpu else 0.0
        est.total_ram_mb = ram_bd.total_mb

        # What a full offload would need, for hosts without a detected GPU.
        if has_gpu:
            est.estimated_vram_mb = est.total_vram_mb
        else:
            full = _place_layers(n_layer, -1, True)
            fw, _ = self._split_weights(meta, geo, model_size_mb, full, config)
            fkv, _, frs, _ = self._kv_and_state(meta, geo, full, config, ctx, ubatch, slots)
            fc, _ = self._compute_buffers(geo, full, ctx, ubatch, True, True)
            est.estimated_vram_mb = (fw + fkv + frs + fc + self.CUDA_RUNTIME_MB
                                     + draft_model_size_mb + mmproj_model_size_mb)

        # ---- 7. Multi-GPU distribution ----
        if num_gpus > 1 and placement.offloaded > 0:
            est.gpu_breakdowns = self._distribute_across_gpus(
                gpu_list, placement, est.gpu_weights_mb, kv_gpu + rs_gpu,
                compute_gpu, runtime_per_gpu, config.split_mode, config.tensor_split,
                draft_gpu, mmproj_gpu)
            est.safety_margin_mb = sum(gb.total_used_mb for gb in est.gpu_breakdowns) * self._safety_margin
            for gb in est.gpu_breakdowns:
                gb.safety_margin_mb = est.safety_margin_mb / num_gpus
            vram_bd.safety_margin_mb = est.safety_margin_mb
            est.total_vram_mb = sum(gb.total_used_mb for gb in est.gpu_breakdowns)
        elif num_gpus == 1 and placement.offloaded > 0:
            gpu = gpu_list[0]
            est.gpu_breakdowns = [GpuBreakdown(
                gpu_index=0, gpu_name=gpu.device_name or "GPU 0",
                total_vram_mb=gpu.vram_mb, available_vram_mb=gpu.available_vram_mb,
                model_weights_mb=gpu_w, kv_cache_mb=kv_gpu + rs_gpu,
                compute_buffers_mb=compute_gpu, backend_overhead_mb=gpu_runtime_mb,
                auxiliary_mb=draft_gpu + mmproj_gpu,
            )]

        # ---- 8. GPU capacity ----
        if gpu_info:
            est.gpu_total_vram_mb = float(gpu_info.vram_mb)
            est.gpu_available_vram_mb = float(gpu_info.available_vram_mb or gpu_info.vram_mb)
        if num_gpus > 1:
            est.gpu_total_vram_mb = sum(gb.total_vram_mb for gb in est.gpu_breakdowns)
            est.gpu_available_vram_mb = sum(gb.available_vram_mb for gb in est.gpu_breakdowns)
        if est.gpu_total_vram_mb > 0:
            # Measured against VRAM this configuration could actually get: the
            # caller supplies a live reading in which our own running server's
            # allocation counts as available, since relaunching releases it.
            est.gpu_remaining_vram_mb = est.gpu_available_vram_mb - est.total_vram_mb

        self._add_warnings(est, config, meta, ctx, placement, has_gpu)
        return est

    # ---- Geometry -----------------------------------------------------------

    @dataclass
    class _Geometry:
        layers: int
        heads: int
        kv_heads: int
        embedding_length: int
        key_length: int
        value_length: int
        ff_dim: int
        vocab_size: int
        n_expert_used: int
        from_gguf: bool

    def _geometry(self, meta: ModelMetadata, model_size_mb: float, quant_type: str,
                  total_layer_count: int) -> "_Geometry":
        if meta.layers and meta.embedding_length:
            ff = meta.ff_dim
            expert_ff = 0
            if meta.is_moe:
                # Per-token FFN work: the experts actually routed to (their
                # buffers are allocated per expert, so it scales ~2x a dense
                # FFN of the same width) plus any shared expert.
                per_expert = meta.expert_ff_dim or ff or meta.embedding_length * 4
                expert_ff = 2 * meta.n_expert_used * per_expert + meta.shared_expert_ff_dim
            return self._Geometry(
                layers=total_layer_count or meta.layers,
                heads=meta.heads or 1, kv_heads=meta.kv_heads or meta.heads or 1,
                embedding_length=meta.embedding_length,
                key_length=meta.key_length or meta.head_dim,
                value_length=meta.value_length or meta.key_length or meta.head_dim,
                ff_dim=expert_ff or ff or meta.embedding_length * 4,
                vocab_size=meta.vocab_size or 32000,
                n_expert_used=meta.n_expert_used, from_gguf=True)
        d = _estimate_arch_from_size(model_size_mb, quant_type)
        head_dim = d.embedding_length // d.heads
        return self._Geometry(
            layers=total_layer_count or d.layers, heads=d.heads, kv_heads=d.kv_heads,
            embedding_length=d.embedding_length, key_length=head_dim, value_length=head_dim,
            ff_dim=d.ff_dim, vocab_size=d.vocab_size, n_expert_used=0, from_gguf=False)

    # ---- Weights ------------------------------------------------------------

    def _split_weights(self, meta: ModelMetadata, geo, model_size_mb: float,
                       placement: _Placement, config: LlamaConfig):
        """(gpu_mb, cpu_mb) for the model weights.

        With per-tensor sizes: the embedding always stays on the host, the
        output layer is offloaded first (a tied embedding is copied to the GPU
        for that), then the last repeating layers. ``--cpu-moe`` /
        ``--n-cpu-moe`` keep expert tensors on the host regardless.
        """
        if meta.has_tensor_sizes and len(meta.layer_bytes) >= placement.n_layer:
            output_bytes = meta.output_bytes or meta.embd_bytes
            gpu = 0
            if placement.output_on_gpu:
                gpu += output_bytes + meta.other_bytes
            n_cpu_moe = placement.n_layer if config.cpu_moe else max(int(config.n_cpu_moe or 0), 0)
            for i in placement.gpu_layers:
                layer = meta.layer_bytes[i]
                if i < n_cpu_moe and i < len(meta.expert_bytes):
                    layer -= meta.expert_bytes[i]
                gpu += layer
            total = meta.total_bytes
            # A tied output copies the embedding to the GPU; host keeps its own.
            cpu = total - gpu + (meta.embd_bytes if placement.output_on_gpu and not meta.output_bytes else 0)
            return gpu / MIB, max(cpu, 0) / MIB

        # No tensor directory: assume uniform layers, output ~ one layer.
        units = placement.n_layer + 1
        gpu_units = placement.offloaded
        gpu = model_size_mb * gpu_units / max(units, 1)
        return gpu, model_size_mb - gpu

    # ---- KV cache + recurrent state -----------------------------------------

    def _kv_and_state(self, meta: ModelMetadata, geo, placement: _Placement,
                      config: LlamaConfig, ctx: int, ubatch: int, slots: int):
        """(kv_gpu, kv_cpu, rs_gpu, rs_cpu) in MB."""
        bytes_k = _cache_type_bytes(config.cache_type_k)
        bytes_v = _cache_type_bytes(config.cache_type_v)
        n_layer = placement.n_layer
        if n_layer <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # Sliding-window layers hold at most the window plus one micro-batch,
        # rounded up to the cache's 256-token padding.
        swa_ctx = ctx
        if meta.swa_window and meta.swa_pattern and not config.swa_full:
            swa_ctx = min(ctx, -(-(meta.swa_window + ubatch) // 256) * 256)

        rs_per_layer = 0.0
        if meta.ssm_inner_size:
            conv = max(meta.ssm_conv_kernel - 1, 0) * (
                meta.ssm_inner_size + 2 * meta.ssm_group_count * meta.ssm_state_size)
            state = meta.ssm_inner_size * meta.ssm_state_size
            rs_per_layer = (conv + state) * 4 * slots

        kv_gpu = kv_cpu = rs_gpu = rs_cpu = 0.0
        for i in range(n_layer):
            on_gpu = placement.layer_on_gpu(i) and config.kv_offload
            kv_heads = (meta.kv_heads_per_layer[i] if i < len(meta.kv_heads_per_layer)
                        else geo.kv_heads)
            is_attention = kv_heads > 0 and (meta.attn_interval <= 1
                                             or (i + 1) % meta.attn_interval == 0)
            if is_attention:
                layer_ctx = ctx
                if swa_ctx != ctx and (i + 1) % meta.swa_pattern != 0:
                    layer_ctx = swa_ctx
                nbytes = layer_ctx * kv_heads * (geo.key_length * bytes_k + geo.value_length * bytes_v)
                if on_gpu:
                    kv_gpu += nbytes
                else:
                    kv_cpu += nbytes
            elif rs_per_layer:
                if on_gpu:
                    rs_gpu += rs_per_layer
                else:
                    rs_cpu += rs_per_layer
        return kv_gpu / MIB, kv_cpu / MIB, rs_gpu / MIB, rs_cpu / MIB

    # ---- Compute buffers ----------------------------------------------------

    @staticmethod
    def _flash_attention_on(flash_attn: str, backend: str) -> bool:
        if flash_attn == "on":
            return True
        if flash_attn == "off":
            return False
        return backend in ("cuda", "hip", "metal", "vulkan")

    #: Extra GPU scratch llama.cpp reserves when MoE experts run on the CPU
    #: (measured: 76 -> 445 MiB on a 256-expert model).
    MOE_CPU_EXPERTS_COMPUTE_MB = 370

    def _compute_buffers(self, geo, placement: _Placement, ctx: int, ubatch: int,
                         flash: bool, has_gpu: bool, experts_on_cpu: bool = False):
        """(gpu_mb, cpu_mb) scratch buffers for one forward pass.

        Measured on llama-server: the GPU buffer is ~(3*n_ff + 2*n_embd) f32
        values per micro-batch token, plus the f16 attention mask over the
        context -- or, without flash attention, the full f32 KQ score matrix
        ``ubatch x ctx x heads``. The CUDA backend reserves its buffer even
        when no layer is offloaded.
        """
        per_token = (3 * geo.ff_dim + 2 * geo.embedding_length) * 4
        activations = ubatch * per_token
        if flash:
            attention = ubatch * ctx * 2
        else:
            attention = ubatch * ctx * (geo.heads * 4 + 2)
        total = activations + attention

        # Logits for the last token of each slot live on the host.
        host = ubatch * ctx * 2 + ubatch * geo.embedding_length * 4
        if has_gpu:
            gpu = total
            if experts_on_cpu:
                gpu += self.MOE_CPU_EXPERTS_COMPUTE_MB * MIB
            if placement.offloaded < placement.offloadable:
                host += activations  # CPU layers compute on the host too
            return gpu / MIB, host / MIB
        return 0.0, (total + host) / MIB

    def _runtime_mb(self, backend: str) -> float:
        return {
            "cuda": self.CUDA_RUNTIME_MB, "hip": self.HIP_RUNTIME_MB,
            "vulkan": self.VULKAN_RUNTIME_MB, "metal": self.METAL_RUNTIME_MB,
        }.get(backend, 0.0)

    # ---- Draft / mmproj -----------------------------------------------------

    def _auxiliary_model(self, path: str, size_mb: float, on_gpu: bool, ctx: int,
                         config: LlamaConfig, slots: int, backend: str):
        """(gpu_mb, cpu_mb) for a draft or projector model."""
        if not path and size_mb <= 0:
            return 0.0, 0.0
        meta = _GgufReader.read_full_metadata(path) if path else ModelMetadata()
        weights = meta.total_bytes / MIB if meta.has_tensor_sizes else size_mb
        if weights <= 0:
            return 0.0, 0.0
        kv = 0.0
        if ctx and meta.layers and meta.kv_heads:
            # A draft model keeps its own KV cache at the same context.
            geo = self._geometry(meta, weights, "", 0)
            placement = _place_layers(meta.layers, -1 if on_gpu else 0, on_gpu)
            kv_gpu, kv_cpu, rs_gpu, rs_cpu = self._kv_and_state(
                meta, geo, placement, config, ctx, max(int(config.ubatch_size), 1), slots)
            kv = kv_gpu + kv_cpu + rs_gpu + rs_cpu
        if on_gpu:
            return weights + kv, 0.0
        return 0.0, weights + kv

    # ---- Multi-GPU ----------------------------------------------------------

    def _resolve_gpu_list(self, gpu_info: Optional[GPUInfo]) -> list:
        if not gpu_info or gpu_info.backend == "cpu":
            return []
        gpu_infos = getattr(gpu_info, "gpu_infos", None)
        if gpu_infos and len(gpu_infos) > 1:
            return gpu_infos[:self.MAX_GPUS]
        gpu_count = getattr(gpu_info, "gpu_count", 1)
        if gpu_count > 1:
            return [GPUInfo(
                backend=gpu_info.backend, vram_mb=gpu_info.vram_mb,
                available_vram_mb=gpu_info.available_vram_mb,
                device_name=f"{gpu_info.device_name} :{i}" if gpu_info.device_name else f"GPU {i}",
                driver_version=gpu_info.driver_version,
            ) for i in range(min(gpu_count, self.MAX_GPUS))]
        return [gpu_info]

    def _distribute_across_gpus(self, gpu_list, placement: _Placement, gpu_weights_mb,
                                kv_cache_mb, compute_mb, runtime_per_gpu_mb, split_mode,
                                tensor_split, draft_mb, mmproj_mb) -> list:
        """Split the GPU-resident parts across cards.

        ``--tensor-split`` ratios win; otherwise layer split follows available
        VRAM and row split is even. Compute buffers and runtime are per GPU.
        """
        num_gpus = len(gpu_list)
        ratios = None
        if tensor_split:
            try:
                ratios = [max(float(x.strip()), 0.0) for x in tensor_split.split(",")][:num_gpus]
            except ValueError:
                ratios = None
        if ratios is None:
            if split_mode == "row":
                ratios = [1.0] * num_gpus
            else:
                ratios = [max(float(g.available_vram_mb or g.vram_mb), 1.0) for g in gpu_list]
        while len(ratios) < num_gpus:
            ratios.append(1.0)
        total_ratio = sum(ratios) or 1.0

        breakdowns = []
        for i, gpu in enumerate(gpu_list):
            share = ratios[i] / total_ratio
            breakdowns.append(GpuBreakdown(
                gpu_index=i, gpu_name=gpu.device_name or f"GPU {i}",
                total_vram_mb=gpu.vram_mb, available_vram_mb=gpu.available_vram_mb,
                model_weights_mb=gpu_weights_mb * share,
                kv_cache_mb=kv_cache_mb * share,
                compute_buffers_mb=compute_mb,
                backend_overhead_mb=runtime_per_gpu_mb,
                auxiliary_mb=(draft_mb + mmproj_mb) if i == 0 else 0.0,
            ))
        return breakdowns

    # ---- Warnings -----------------------------------------------------------

    def _add_warnings(self, est: ResourceEstimate, config: LlamaConfig, meta: ModelMetadata,
                      ctx: int, placement: _Placement, has_gpu: bool) -> None:
        est.warnings = []
        held_by_others_mb = max(0.0, est.gpu_total_vram_mb - est.gpu_available_vram_mb)

        if est.gpu_total_vram_mb > 0 and est.total_vram_mb > est.gpu_total_vram_mb:
            est.vram_exceeded = True
            hint = ("Reduce context size, use a q8_0 KV cache, or lower gpu_layers."
                    if placement.offloaded == placement.offloadable
                    else "Reduce gpu_layers or context size.")
            est.warnings.append(
                f"Estimated VRAM usage ({self._fmt_mb(est.total_vram_mb)}) exceeds "
                f"total VRAM ({self._fmt_mb(est.gpu_total_vram_mb)}). {hint}")
        elif (est.gpu_available_vram_mb > 0
                and est.total_vram_mb > est.gpu_available_vram_mb
                and held_by_others_mb > _VRAM_IN_USE_FLOOR_MB):
            est.vram_exceeded = True
            est.warnings.append(
                f"Estimated VRAM usage ({self._fmt_mb(est.total_vram_mb)}) exceeds the "
                f"{self._fmt_mb(est.gpu_available_vram_mb)} currently free "
                f"({self._fmt_mb(held_by_others_mb)} is held by other processes). "
                f"Stop the other model, or reduce gpu_layers or context size.")
        elif est.gpu_total_vram_mb > 0 and est.vram_percentage > 92:
            est.warnings.append(
                f"VRAM usage is at {est.vram_percentage:.0f}%: close to the limit, "
                f"long prompts may fail to allocate.")

        if est.total_ram_mb > est.system_total_ram_mb * 0.9:
            est.ram_warning = True
            est.warnings.append(
                f"Estimated RAM usage ({self._fmt_mb(est.total_ram_mb)}) is near "
                f"the system limit ({self._fmt_mb(est.system_total_ram_mb)}).")

        for gb in est.gpu_breakdowns:
            if len(est.gpu_breakdowns) > 1 and gb.usage_percentage > 100:
                est.warnings.append(
                    f"GPU {gb.gpu_index} ({gb.gpu_name}) exceeds capacity: "
                    f"{self._fmt_mb(gb.total_used_mb)} / {self._fmt_mb(gb.total_vram_mb)}.")

        if has_gpu and 0 < placement.offloaded < placement.offloadable:
            est.warnings.append(
                f"{placement.offloaded}/{placement.offloadable} layers on the GPU; the rest "
                f"run on the CPU, so generation will be much slower.")

        if config.embeddings:
            est.warnings.append(
                "Embeddings mode: the KV cache is not needed, actual VRAM usage will be lower.")

        if meta.context_length > 0 and ctx > meta.context_length:
            est.warnings.append(
                f"Selected context ({ctx} tokens) exceeds the model's trained context "
                f"({meta.context_length} tokens); RoPE scaling will be used and quality may degrade.")

    @staticmethod
    def _fmt_mb(mb: float) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"
