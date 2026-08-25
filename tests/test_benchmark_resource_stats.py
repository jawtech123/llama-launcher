"""Tests for live GPU/VRAM/RAM usage accumulation over a benchmark session."""
from dataclasses import dataclass

from llama_launcher.benchmark.resource_stats import ResourceAccumulator


@dataclass
class _FakeSample:
    """Mirrors remote.procinfo.ResourceSample's field names/semantics."""
    cpu_percent: float = -1.0
    ram_used_mb: float = -1.0
    ram_total_mb: float = -1.0
    gpu_percent: float = -1.0
    gpu_mem_used_mb: float = -1.0
    gpu_mem_total_mb: float = -1.0
    own_vram_mb: float = 0.0


def test_empty_accumulator_summary_is_zeroed():
    acc = ResourceAccumulator()
    summary = acc.summary()
    assert summary["sample_count"] == 0
    assert summary["gpu_percent_avg"] == 0.0
    assert summary["vram_used_mb_max"] == 0.0


def test_add_none_sample_is_ignored():
    acc = ResourceAccumulator()
    acc.add(None)
    assert acc.count == 0


def test_avg_and_max_across_samples():
    acc = ResourceAccumulator()
    acc.add(_FakeSample(gpu_percent=10.0, gpu_mem_used_mb=1000.0, ram_used_mb=2000.0))
    acc.add(_FakeSample(gpu_percent=30.0, gpu_mem_used_mb=3000.0, ram_used_mb=4000.0))
    acc.add(_FakeSample(gpu_percent=20.0, gpu_mem_used_mb=2000.0, ram_used_mb=3000.0))

    summary = acc.summary()
    assert summary["sample_count"] == 3
    assert summary["gpu_percent_avg"] == 20.0
    assert summary["gpu_percent_max"] == 30.0
    assert summary["vram_used_mb_avg"] == 2000.0
    assert summary["vram_used_mb_max"] == 3000.0
    assert summary["ram_used_mb_avg"] == 3000.0
    assert summary["ram_used_mb_max"] == 4000.0


def test_negative_unavailable_values_excluded_from_average():
    """-1 means 'not available' and must not drag the average toward zero."""
    acc = ResourceAccumulator()
    acc.add(_FakeSample(gpu_percent=-1.0, gpu_mem_used_mb=-1.0, ram_used_mb=1000.0))
    acc.add(_FakeSample(gpu_percent=50.0, gpu_mem_used_mb=5000.0, ram_used_mb=1000.0))

    summary = acc.summary()
    assert summary["sample_count"] == 2
    # Only one valid GPU reading -> average is that reading, not (50+(-1))/2
    assert summary["gpu_percent_avg"] == 50.0
    assert summary["vram_used_mb_avg"] == 5000.0
    assert summary["ram_used_mb_avg"] == 1000.0
