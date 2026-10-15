"""Resource Estimation Types."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResourceBreakdown:
    """Detailed breakdown of a single memory region."""
    label: str
    size_mb: float
    description: str


@dataclass
class ModelMetadata:
    """Architecture metadata and tensor sizes read from a GGUF header."""
    arch: str = "unknown"
    layers: int = 0
    heads: int = 0
    kv_heads: int = 0
    embedding_length: int = 0
    head_dim: int = 0
    #: K and V head dimensions; equal to head_dim unless the model says otherwise.
    key_length: int = 0
    value_length: int = 0
    ff_dim: int = 0
    kv_dim: int = 0  # kept for callers that read it; equals value_length
    context_length: int = 0
    vocab_size: int = 0
    total_params: int = 0
    quant_types: dict = field(default_factory=dict)
    #: Quantization label from general.file_type (e.g. "Q4_K_M"), "" if absent.
    file_type: str = ""
    tensor_count: int = 0
    #: >1 for hybrid SSM/attention archs where only every Nth layer has a KV cache.
    attn_interval: int = 1
    #: Per-layer KV head counts when the model varies them (else empty).
    kv_heads_per_layer: List[int] = field(default_factory=list)
    #: Sliding-window size for SWA layers (0 = none) and the pattern period
    #: (every Nth layer is a full-attention layer; 0 = unknown).
    swa_window: int = 0
    swa_pattern: int = 0
    #: Recurrent-state geometry for SSM / linear-attention layers.
    ssm_conv_kernel: int = 0
    ssm_inner_size: int = 0
    ssm_state_size: int = 0
    ssm_group_count: int = 0
    #: Mixture-of-experts geometry (0 for dense models).
    n_expert: int = 0
    n_expert_used: int = 0
    expert_ff_dim: int = 0
    shared_expert_ff_dim: int = 0
    #: The chat template embedded in the model ("" if none).
    chat_template: str = ""

    # ---- tensor sizes, in bytes ----
    #: Bytes of each repeating layer (``blk.N.*``), index = layer.
    layer_bytes: List[int] = field(default_factory=list)
    #: Bytes of the MoE expert tensors inside each layer (subset of layer_bytes).
    expert_bytes: List[int] = field(default_factory=list)
    #: Token embedding, output head (0 when tied to the embedding), and
    #: everything else that is not a repeating layer.
    embd_bytes: int = 0
    output_bytes: int = 0
    other_bytes: int = 0
    total_bytes: int = 0

    @property
    def has_tensor_sizes(self) -> bool:
        return self.total_bytes > 0 and bool(self.layer_bytes)

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1


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
        return self.total_vram_mb - self.total_used_mb

    @property
    def usage_percentage(self) -> float:
        if self.total_vram_mb <= 0:
            return 0.0
        return (self.total_used_mb / self.total_vram_mb) * 100.0


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

    def as_breakdown_list(self) -> list:
        items = [
            ResourceBreakdown("Model Weights (GPU)", self.gpu_weights_mb,
                "Weights of the layers offloaded to the GPU (per-tensor sizes from the GGUF)."),
            ResourceBreakdown("KV Cache (GPU)", self.kv_cache_mb,
                "Key/Value cache for the offloaded attention layers: context size x KV heads x head size x cache type."),
            ResourceBreakdown("Compute Buffers", self.inference_buffers_mb,
                "Scratch space for the forward pass; grows with ubatch size and, without flash attention, with context."),
        ]
        if self.cuda_runtime_mb > 0:
            items.append(ResourceBreakdown("CUDA Runtime", self.cuda_runtime_mb,
                "CUDA context and driver overhead, roughly 0.5 GB per GPU."))
        if self.temp_allocations_mb > 0:
            items.append(ResourceBreakdown("Recurrent State", self.temp_allocations_mb,
                "State buffers for SSM / linear-attention layers, one per server slot."))
        if self.backend_overhead_mb > 0:
            items.append(ResourceBreakdown("Backend Overhead", self.backend_overhead_mb,
                "Estimated backend overhead (Vulkan/Metal/HIP runtime)."))
        if self.safety_margin_mb > 0:
            items.append(ResourceBreakdown("Safety Margin", self.safety_margin_mb,
                "Margin for estimation uncertainty."))
        if self.draft_model_mb > 0:
            items.append(ResourceBreakdown("Draft Model", self.draft_model_mb,
                "Draft model weights and KV cache for speculative decoding."))
        if self.mmproj_model_mb > 0:
            items.append(ResourceBreakdown("mmproj Model", self.mmproj_model_mb,
                "Multimodal projector weights plus its image-encoding buffers."))
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

    def as_breakdown_list(self) -> list:
        return [
            ResourceBreakdown("Model Weights (CPU)", self.cpu_weights_mb,
                "Weights that stay in system RAM: the token embedding plus any layer not offloaded. "
                "With mmap this is page cache, released under memory pressure."),
            ResourceBreakdown("KV Cache (CPU)", self.kv_cache_mb,
                "Key/Value cache of the layers that run on the CPU, or all of it when KV offload is off."),
            ResourceBreakdown("Compute Buffers", self.inference_buffers_mb,
                "Host-side scratch buffers (input staging, CPU layers' activations)."),
            ResourceBreakdown("mmap Overhead", self.mmap_overhead_mb,
                "Page-table and read-ahead overhead of memory-mapping the model file."),
            ResourceBreakdown("Runtime Overhead", self.runtime_overhead_mb,
                "llama-server process overhead: thread stacks, HTTP server, tokenizer."),
            ResourceBreakdown("Draft Model", self.draft_model_mb,
                "Draft model weights on the CPU for speculative decoding."),
            ResourceBreakdown("mmproj Model", self.mmproj_model_mb,
                "Multimodal projector on the CPU."),
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
    gpu_breakdowns: list = field(default_factory=list)
    multi_gpu: bool = False
    split_mode: str = "layer"

    # System info
    system_total_ram_mb: float = 0.0

    # Warnings
    vram_exceeded: bool = False
    ram_warning: bool = False
    warnings: list = field(default_factory=list)

    # Breakdowns
    vram_breakdown: VramBreakdown = field(default_factory=VramBreakdown)
    ram_breakdown: RamBreakdown = field(default_factory=RamBreakdown)

    # Model metadata from GGUF
    model_metadata: ModelMetadata = field(default_factory=ModelMetadata)

    #: How many of the model's layers (plus the output layer) the estimate
    #: assumes are on the GPU, and the total count, as llama-server reports it.
    offloaded_layers: int = 0
    offloadable_layers: int = 0

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
    def quant_display(self) -> str:
        """Best available quantization label (file type, else dominant tensor type)."""
        meta = self.model_metadata
        if meta.file_type:
            return meta.file_type
        if meta.quant_types:
            return max(meta.quant_types, key=meta.quant_types.get)
        if self.quant_type and self.quant_type != "unknown":
            return self.quant_type
        return ""

    @property
    def arch_name(self) -> str:
        a = self.model_metadata.arch
        if isinstance(a, str) and a:
            return a
        if isinstance(a, bytes):
            return a.decode("utf-8", errors="replace")
        return self.model_family if self.model_family else "unknown"
