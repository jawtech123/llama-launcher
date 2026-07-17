"""GPU Detection Module."""
from .types import GPUInfo
from .detector import detect_gpu

__all__ = ['GPUInfo', 'detect_gpu']
