"""Configuration Validator."""
from .config import LlamaConfig


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


def validate_config(config: LlamaConfig) -> None:
    """Validate a LlamaConfig object."""
    # Validate model_path
    if not config.model_path or not isinstance(config.model_path, str):
        raise ValidationError("model_path must be a non-empty string")
    
    # Validate context_size
    if not isinstance(config.context_size, int) or not (256 <= config.context_size <= 524288):
        raise ValidationError("context_size must be an integer between 256 and 524288")
    
    # Validate gpu_layers
    if not isinstance(config.gpu_layers, int) or not (-1 <= config.gpu_layers <= 999):
        raise ValidationError("gpu_layers must be an integer between -1 and 999")
    
    # Validate threads
    if not isinstance(config.threads, int) or not (0 <= config.threads <= 256):
        raise ValidationError("threads must be an integer between 0 and 256")
    
    # Validate port
    if not isinstance(config.port, int) or not (1 <= config.port <= 65535):
        raise ValidationError("port must be an integer between 1 and 65535")
    
    # Validate host
    if not isinstance(config.host, str) or not config.host:
        raise ValidationError("host must be a non-empty string")
    
    # Validate backend
    valid_backends = {"auto", "cuda", "metal", "vulkan", "cpu"}
    if config.backend not in valid_backends:
        raise ValidationError(f"backend must be one of {valid_backends}")
    
    # Validate kv_cache_type
    valid_kv_types = {"f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"}
    if config.kv_cache_type not in valid_kv_types:
        raise ValidationError(f"kv_cache_type must be one of {valid_kv_types}")
    
    # Validate batch_size
    if not isinstance(config.batch_size, int) or not (32 <= config.batch_size <= 4096):
        raise ValidationError("batch_size must be an integer between 32 and 4096")
