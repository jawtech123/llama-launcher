"""Benchmark: parse llama.cpp server logs for throughput statistics."""
from .fingerprint import config_fingerprint
from .labels import (
    build_session_label,
    build_session_tooltip,
    config_snapshot_fields,
    estimate_summary,
)
from .parser import TimingRecord, SessionStats, aggregate, parse_log_line, parse_log_text
from .resource_stats import ResourceAccumulator
from .runner import (
    BenchmarkRunConfig,
    DEFAULT_PROMPT,
    diff_new_log_lines,
    run_benchmark,
    run_completion,
    wait_for_server_ready,
)
from .storage import (
    load_sessions,
    append_session,
    replace_sessions_for_source,
    clear_sessions,
    latest_session_for_preset,
)

__all__ = [
    "TimingRecord",
    "SessionStats",
    "aggregate",
    "config_fingerprint",
    "parse_log_line",
    "parse_log_text",
    "load_sessions",
    "append_session",
    "replace_sessions_for_source",
    "clear_sessions",
    "latest_session_for_preset",
    "build_session_label",
    "build_session_tooltip",
    "config_snapshot_fields",
    "estimate_summary",
    "ResourceAccumulator",
    "BenchmarkRunConfig",
    "DEFAULT_PROMPT",
    "diff_new_log_lines",
    "run_benchmark",
    "run_completion",
    "wait_for_server_ready",
]
