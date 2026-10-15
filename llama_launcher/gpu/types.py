"""GPU Information Types."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GpuProcess:
    """A process holding GPU memory."""
    pid: int = 0
    name: str = ""
    vram_mb: float = 0.0
    gpu_index: int = 0


@dataclass
class GPUInfo:
    """Information about detected GPU(s) and the llama.cpp backend to use.

    For several cards the top-level object aggregates them (summed VRAM, first
    card's name) and ``gpu_infos`` holds one entry per card. Negative values
    for the live readings mean "not available", never zero.
    """
    backend: str  # cuda, hip, vulkan, metal, sycl, cpu
    vram_mb: int = 0
    available_vram_mb: int = 0
    device_name: Optional[str] = None
    driver_version: Optional[str] = None
    supported: bool = True
    gpu_count: int = 1
    gpu_infos: list = field(default_factory=list)

    # ---- per-card details ----
    vendor: str = ""               # nvidia, amd, intel, apple, ""
    index: int = 0
    pci_bus_id: str = ""
    #: "12.0" (CUDA compute capability) or "gfx1100" (AMD target)
    compute_capability: str = ""
    integrated: bool = False
    used_vram_mb: float = -1.0
    utilization_pct: float = -1.0
    temperature_c: float = -1.0
    power_w: float = -1.0
    power_limit_w: float = -1.0
    clock_mhz: float = -1.0
    mem_clock_mhz: float = -1.0
    pcie_link: str = ""

    # ---- host-level facts (aggregate only) ----
    #: ROCm user-space tooling found on the host (rocm-smi / amd-smi / /opt/rocm).
    rocm_available: bool = False
    processes: list = field(default_factory=list)
    #: Human-readable remarks (ignored second GPU, driver mismatch, ...).
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if self.available_vram_mb == 0:
            self.available_vram_mb = self.vram_mb


def gpu_to_dict(gpu: Optional[GPUInfo]) -> Optional[Dict[str, Any]]:
    """Serialize for the host cache (nested cards and processes included)."""
    if gpu is None:
        return None
    data = {k: v for k, v in gpu.__dict__.items() if k not in ("gpu_infos", "processes")}
    data["gpu_infos"] = [gpu_to_dict(sub) for sub in (gpu.gpu_infos or [])]
    data["processes"] = [p.__dict__ for p in (gpu.processes or [])]
    return data


def gpu_from_dict(data: Optional[Dict[str, Any]]) -> Optional[GPUInfo]:
    """Inverse of :func:`gpu_to_dict`; unknown keys (older/newer files) are ignored."""
    if not data:
        return None
    try:
        known = {f for f in GPUInfo.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known and k not in ("gpu_infos", "processes")}
        kwargs.setdefault("backend", "cpu")
        gpu = GPUInfo(**kwargs)
        gpu.gpu_infos = [g for g in (gpu_from_dict(sub) for sub in data.get("gpu_infos", []) or []) if g]
        gpu.processes = [GpuProcess(**{k: v for k, v in p.items() if k in GpuProcess.__dataclass_fields__})
                         for p in data.get("processes", []) or [] if isinstance(p, dict)]
        return gpu
    except (TypeError, ValueError):
        return None
