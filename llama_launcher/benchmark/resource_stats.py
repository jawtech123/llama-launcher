"""Accumulate live GPU/VRAM/RAM usage samples over a benchmark session.

Fed by ``remote.procinfo.ResourceSample`` readings taken from the app's
existing usage poller (see ``gui/main_window.py``'s ``_on_usage_sample``) --
this module adds no polling of its own, it just summarizes samples handed
to it.
"""
from dataclasses import dataclass


@dataclass
class ResourceAccumulator:
    """Running sum/max of resource samples, summarized on demand.

    ``ResourceSample`` uses -1 to mean "not available" (no GPU, unreadable
    counters); those fields are excluded from their respective average/max
    rather than treated as zero.
    """
    gpu_sum: float = 0.0
    gpu_max: float = 0.0
    gpu_count: int = 0
    vram_sum: float = 0.0
    vram_max: float = 0.0
    vram_count: int = 0
    ram_sum: float = 0.0
    ram_max: float = 0.0
    ram_count: int = 0
    count: int = 0

    def add(self, sample) -> None:
        if sample is None:
            return
        if sample.gpu_percent >= 0:
            self.gpu_sum += sample.gpu_percent
            self.gpu_max = max(self.gpu_max, sample.gpu_percent)
            self.gpu_count += 1
        if sample.gpu_mem_used_mb >= 0:
            self.vram_sum += sample.gpu_mem_used_mb
            self.vram_max = max(self.vram_max, sample.gpu_mem_used_mb)
            self.vram_count += 1
        if sample.ram_used_mb >= 0:
            self.ram_sum += sample.ram_used_mb
            self.ram_max = max(self.ram_max, sample.ram_used_mb)
            self.ram_count += 1
        self.count += 1

    def summary(self) -> dict:
        return {
            "gpu_percent_avg": self.gpu_sum / self.gpu_count if self.gpu_count else 0.0,
            "gpu_percent_max": self.gpu_max,
            "vram_used_mb_avg": self.vram_sum / self.vram_count if self.vram_count else 0.0,
            "vram_used_mb_max": self.vram_max,
            "ram_used_mb_avg": self.ram_sum / self.ram_count if self.ram_count else 0.0,
            "ram_used_mb_max": self.ram_max,
            "sample_count": self.count,
        }
