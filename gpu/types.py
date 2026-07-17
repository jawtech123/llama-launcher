"""GPU Information Types."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    """Information about detected GPU and backend."""
    backend: str  # cuda, metal, vulkan, cpu
    vram_mb: int = 0
    available_vram_mb: int = 0
    device_name: Optional[str] = None
    driver_version: Optional[str] = None
    supported: bool = True
    gpu_count: int = 1
    gpu_infos: list["GPUInfo"] = field(default_factory=list)

    def __post_init__(self):
        if self.available_vram_mb == 0:
            self.available_vram_mb = self.vram_mb
