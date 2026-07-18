"""Resource Estimation Types."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResourceBreakdown:
    """Detailed breakdown of a single memory region."""
    label: str
    size_mb: float
    description: str


@dataclass
class ModelMetadata:
    """Architecture metadata extracted from GGUF header."""
    arch: str = "unknown"
    layers: int = 0
    heads: int = 0
    kv_heads: int = 0
    embedding_length: int = 0
    head_dim: int = 0
    ff_dim: int = 0
    kv_dim: int = 0  # Gemma uses ff_dim for V cache dimension
    context_length: int = 0
    total_params: int = 0
    quant_types: dict[str, int] = field(default_factory=dict)
    tensor_count: int = 0
    attn_interval: int = 1  # >1 for hybrid SSM/attention archs (e.g. qwen35)


@dataclass
class GpuBreakdown:
    """Per-GPU memory breakdown for multi-GPU setups."""
    gpu_index: int = 0
    gpu_name: str = ""
    total_vram_mb: float = 0.0
    available_vram_mb: float = 0.0

    # Usage
    model_weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    compute_buffers_mb: float = 0.0
    backend_overhead_mb: float = 0.0
    safety_margin_mb: float = 0.0
    auxiliary_mb: float = 0.0

    @property
    def total_used_mb(self) -> float:
        return (self.model_weights_mb + self.kv_cache_mb + self.compute_buffers_mb +
                self.backend_overhead_mb + self.safety_margin_mb + self.auxiliary_mb)

    @property
    def remaining_vram_mb(self) -> float:
        return self.available_vram_mb - self.total_used_mb

    @property
    def usage_percentage(self) -> float:
        if self.available_vram_mb <= 0:
            return 0.0
        return (self.total_used_mb / self.available_vram_mb) * 100.0


@dataclass
class VramBreakdown:
    """Detailed VRAM usage breakdown."""
    gpu_weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    inference_buffers_mb: float = 0.0
    cuda_runtime_mb: float = 0.0
    temp_allocations_mb: float = 0.0
    draft_model_mb: float = 0.0
    mmproj_model_mb: float = 0.0
    backend_overhead_mb: float = 0.0
    safety_margin_mb: float = 0.0

    @property
    def total_mb(self) -> float:
        return (self.gpu_weights_mb + self.kv_cache_mb + self.inference_buffers_mb +
                self.cuda_runtime_mb + self.temp_allocations_mb + self.draft_model_mb +
                self.mmproj_model_mb + self.backend_overhead_mb + self.safety_margin_mb)

    def as_breakdown_list(self) -> list[ResourceBreakdown]:
        items = [
            ResourceBreakdown("Model Weights (GPU)", self.gpu_weights_mb,
                "Weights for layers offloaded to GPU. Depends on gpu_layers setting and model quantization."),
            ResourceBreakdown("KV Cache (GPU)", self.kv_cache_mb,
                "Key/Value cache for attention on GPU. Depends on context size, batch size, number of sequences, and cache type."),
            ResourceBreakdown("Inference Buffers", self.inference_buffers_mb,
                "Temporary buffers for forward pass computation (layer activations, intermediate results)."),
        ]
        if self.cuda_runtime_mb > 0:
            items.append(ResourceBreakdown("CUDA Runtime", self.cuda_runtime_mb,
                "CUDA context and driver overhead. Typically ~200-400 MB per GPU."))
        if self.temp_allocations_mb > 0:
            items.append(ResourceBreakdown("Temp Allocations", self.temp_allocations_mb,
                "Temporary allocations during inference (softmax buffers, gemm scratch space)."))
        if self.backend_overhead_mb > 0:
            items.append(ResourceBreakdown("Backend Overhead", self.backend_overhead_mb,
                "Estimated backend overhead (Vulkan/Metal/CUDA workspace)."))
        if self.safety_margin_mb > 0:
            items.append(ResourceBreakdown("Safety Margin", self.safety_margin_mb,
                "Configurable safety margin to account for estimation uncertainty."))
        if self.draft_model_mb > 0:
            items.append(ResourceBreakdown("Draft Model", self.draft_model_mb,
                "Draft model weights for speculative decoding."))
        if self.mmproj_model_mb > 0:
            items.append(ResourceBreakdown("mmproj Model", self.mmproj_model_mb,
                "Multimodal projector model for vision-language models."))
        return items


@dataclass
class RamBreakdown:
    """Detailed system RAM usage breakdown."""
    cpu_weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    inference_buffers_mb: float = 0.0
    mmap_overhead_mb: float = 0.0
    runtime_overhead_mb: float = 0.0
    draft_model_mb: float = 0.0
    mmproj_model_mb: float = 0.0

    @property
    def total_mb(self) -> float:
        return (self.cpu_weights_mb + self.kv_cache_mb + self.inference_buffers_mb +
                self.mmap_overhead_mb + self.runtime_overhead_mb + self.draft_model_mb +
                self.mmproj_model_mb)

    def as_breakdown_list(self) -> list[ResourceBreakdown]:
        return [
            ResourceBreakdown("Model Weights (CPU)", self.cpu_weights_mb,
                "Weights for layers running on CPU. If mmap is enabled, this is mostly page cache, not physical RAM."),
            ResourceBreakdown("KV Cache (CPU)", self.kv_cache_mb,
                "Key/Value cache for attention on CPU. Used when kv_offload is disabled or GPU KV cache is full."),
            ResourceBreakdown("Inference Buffers", self.inference_buffers_mb,
                "CPU-side temporary buffers for computation."),
            ResourceBreakdown("mmap Overhead", self.mmap_overhead_mb,
                "Memory-mapped file overhead when mmap is enabled."),
            ResourceBreakdown("Runtime Overhead", self.runtime_overhead_mb,
                "llama-server process overhead, thread stacks, internal data structures."),
            ResourceBreakdown("Draft Model", self.draft_model_mb,
                "Draft model weights on CPU for speculative decoding."),
            ResourceBreakdown("mmproj Model", self.mmproj_model_mb,
                "Multimodal projector model on CPU."),
        ]


@dataclass
class ResourceEstimate:
    """Complete resource estimate for a launch configuration."""

    # Model weights
    model_weights_mb: float = 0.0
    model_weights_from_gguf_mb: float = 0.0
    gpu_weights_mb: float = 0.0
    cpu_weights_mb: float = 0.0

    # KV Cache
    kv_cache_total_mb: float = 0.0
    kv_cache_gpu_mb: float = 0.0
    kv_cache_cpu_mb: float = 0.0

    # Compute buffers
    compute_buffers_mb: float = 0.0
    compute_buffers_gpu_mb: float = 0.0
    compute_buffers_cpu_mb: float = 0.0

    # Buffers & overhead
    inference_buffers_mb: float = 0.0
    cuda_runtime_mb: float = 0.0
    temp_allocations_mb: float = 0.0
    runtime_overhead_mb: float = 0.0
    mmap_overhead_mb: float = 0.0
    backend_overhead_mb: float = 0.0
    safety_margin_mb: float = 0.0

    # Auxiliary models
    draft_model_mb: float = 0.0
    mmproj_model_mb: float = 0.0

    # Totals
    total_vram_mb: float = 0.0
    total_ram_mb: float = 0.0

    # Single-GPU info (backward compat)
    gpu_total_vram_mb: float = 0.0
    gpu_available_vram_mb: float = 0.0
    gpu_remaining_vram_mb: float = 0.0

    # Estimated VRAM if all layers were offloaded (for display when no GPU detected)
    estimated_vram_mb: float = 0.0

    # Multi-GPU
    gpu_breakdowns: list[GpuBreakdown] = field(default_factory=list)
    multi_gpu: bool = False
    split_mode: str = "layer"

    # System info
    system_total_ram_mb: float = 0.0

    # Warnings
    vram_exceeded: bool = False
    ram_warning: bool = False
    warnings: list[str] = field(default_factory=list)

    # Breakdowns
    vram_breakdown: VramBreakdown = field(default_factory=VramBreakdown)
    ram_breakdown: RamBreakdown = field(default_factory=RamBreakdown)

    # Model metadata from GGUF
    model_metadata: ModelMetadata = field(default_factory=ModelMetadata)

    # Legacy fields for backward compat
    model_file_size_mb: float = 0.0
    model_family: str = "unknown"
    quant_type: str = "unknown"
    layer_count: int = 0
    hidden_size: int = 0
    attention_heads: int = 0
    kv_heads: int = 0
    training_context_length: int = 0

    @property
    def vram_percentage(self) -> float:
        if self.gpu_total_vram_mb <= 0:
            return 0.0
        return (self.total_vram_mb / self.gpu_total_vram_mb) * 100.0

    @property
    def ram_percentage(self) -> float:
        if self.system_total_ram_mb <= 0:
            return 0.0
        return (self.total_ram_mb / self.system_total_ram_mb) * 100.0

    @property
    def is_remote(self) -> bool:
        return self.gpu_total_vram_mb == 0 and self.system_total_ram_mb == 0

    @property
    def total_params_display(self) -> str:
        """Human-readable parameter count."""
        p = self.model_metadata.total_params
        if not isinstance(p, int) or p == 0:
            return "unknown"
        if p >= 1_000_000_000:
            return f"{p / 1_000_000_000:.1f}B"
        if p >= 1_000_000:
            return f"{p / 1_000_000:.1f}M"
        if p >= 1_000:
            return f"{p / 1_000:.0f}K"
        return str(p)

    @property
    def arch_name(self) -> str:
        a = self.model_metadata.arch
        if isinstance(a, str) and a:
            return a
        if isinstance(a, bytes):
            return a.decode("utf-8", errors="replace")
        return self.model_family if self.model_family else "unknown"
