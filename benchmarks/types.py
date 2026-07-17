"""Benchmark Result Types."""
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Structured benchmark results."""
    tokens_per_sec: float
    latency_ms: float
    max_context_stable: int
    model_path: str
    backend: str
