"""VRAM-Based Configuration Optimizer."""
from core.config import LlamaConfig
from gpu.types import GPUInfo


def optimize_for_vram(config: LlamaConfig, gpu_info: GPUInfo, model_size_mb: float) -> LlamaConfig:
    """Optimize configuration based on model size and GPU VRAM."""
    optimized = LlamaConfig(
        model_path=config.model_path,
        context_size=config.context_size,
        gpu_layers=config.gpu_layers,
        threads=config.threads,
        port=config.port,
        host=config.host,
        backend=config.backend,
        use_mmap=config.use_mmap,
        kv_cache_type=config.kv_cache_type,
        batch_size=config.batch_size,
        extra_flags=list(config.extra_flags),
        name=config.name,
        description=config.description
    )
    
    if gpu_info.backend == "cpu" or gpu_info.vram_mb == 0:
        # CPU mode: use mmap and reduce batch size
        optimized.use_mmap = True
        optimized.gpu_layers = 0
        optimized.batch_size = min(config.batch_size, 256)
        return optimized
    
    # Calculate available VRAM for model + context
    # Model size in VRAM depends on quantization (approximate)
    # Assume model is quantized to Q4_K_M (~4-5 bits per weight)
    # 7B model Q4_K_M ~ 4-5GB, 13B ~ 7-8GB, etc.
    
    estimated_model_vram_mb = model_size_mb * 0.6  # Rough estimate for Q4 quantization
    
    # Required VRAM = model + context overhead + KV cache
    # Context overhead ~ context_size * batch_size * 2 bytes per token (f16)
    kv_cache_mb = (optimized.context_size * optimized.batch_size * 2) / (1024 * 1024)
    total_required_mb = estimated_model_vram_mb + kv_cache_mb + 1024  # 1GB buffer
    
    if gpu_info.vram_mb < total_required_mb:
        # Need to optimize settings
        
        # Reduce context size first
        while (estimated_model_vram_mb + (optimized.context_size * optimized.batch_size * 2) / (1024 * 1024) + 1024) > gpu_info.vram_mb * 0.9 and optimized.context_size >= 1024:
            optimized.context_size = max(1024, optimized.context_size // 2)
        
        # Change KV cache type to quantized if still over
        if (estimated_model_vram_mb + (optimized.context_size * optimized.batch_size * 2) / (1024 * 1024) + 1024) > gpu_info.vram_mb * 0.9:
            optimized.kv_cache_type = "q8_0"
        
        # Reduce batch size
        while (estimated_model_vram_mb + (optimized.context_size * optimized.batch_size * 2) / (1024 * 1024) + 1024) > gpu_info.vram_mb * 0.9 and optimized.batch_size >= 128:
            optimized.batch_size = max(128, optimized.batch_size // 2)
        
        # Set gpu_layers based on available VRAM
        if estimated_model_vram_mb > gpu_info.vram_mb * 0.7:
            # Model is large relative to VRAM, don't put all layers on GPU
            optimized.gpu_layers = max(0, int(50 * (gpu_info.vram_mb - 1024) / model_size_mb))
        else:
            optimized.gpu_layers = -1  # All layers to GPU
    
    return optimized
