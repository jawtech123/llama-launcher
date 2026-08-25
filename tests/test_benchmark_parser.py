"""Tests for benchmark log parsing edge cases."""
from llama_launcher.benchmark.parser import (
    SessionStats, TimingRecord, parse_log_line, parse_log_text, aggregate,
)


def test_multiple_slots_aggregation():
    """Multiple slots in log → aggregated correctly."""
    log = """\
slot print_timings: id  0 | task 0 | prompt eval time =    100.00 ms /   10 tokens (10.00 ms per token, 100.00 tokens per second)
slot print_timings: id  0 | task 0 |        eval time =    500.00 ms /   50 tokens (10.00 ms per token, 100.00 tokens per second)
slot print_timings: id  0 | task 1 | prompt eval time =    200.00 ms /   20 tokens (10.00 ms per token, 100.00 tokens per second)
slot print_timings: id  0 | task 1 |        eval time =   1000.00 ms /   100 tokens (10.00 ms per token, 100.00 tokens per second)
"""
    stats = aggregate(parse_log_text(log))
    assert stats.requests == 2
    assert stats.prompt_tokens == 30
    assert stats.gen_tokens == 150


def test_empty_log():
    """Empty log → zero stats."""
    stats = aggregate(parse_log_text(""))
    assert stats.requests == 0
    assert stats.prompt_tokens == 0
    assert stats.gen_tokens == 0


def test_malformed_timing_lines():
    """Malformed timing lines are skipped gracefully."""
    log = """\
this is not a timing line
also not a timing line
slot print_timings: id  0 | task 0 | prompt eval time =    100.00 ms /   10 tokens (10.00 ms per token, 100.00 tokens per second)
random garbage here
"""
    records = parse_log_text(log)
    assert len(records) == 1
    assert records[0].tokens == 10


def test_zero_tokens():
    """Timing line with zero tokens produces zero count."""
    log = "slot print_timings: id  0 | task 0 | prompt eval time =    0.00 ms /   0 tokens (0.00 ms per token, 0.00 tokens per second)\n"
    records = parse_log_text(log)
    assert len(records) == 1
    assert records[0].tokens == 0


def test_negative_times():
    """Negative time values should be handled without crash."""
    # The regex doesn't match negative numbers, so this returns None
    log = "slot print_timings: id  0 | task 0 | prompt eval time =   -100.00 ms /   10 tokens\n"
    records = parse_log_text(log)
    # Negative time won't match the regex pattern, so it returns None
    assert len(records) == 0


def test_extremely_high_tps():
    """Extremely high TPS values don't cause overflow."""
    log = "slot print_timings: id  0 | task 0 | prompt eval time =      1.00 ms /    1 tokens ( 1000.00 ms per token, 1000000.00 tokens per second)\n"
    records = parse_log_text(log)
    assert len(records) == 1


def test_partial_timing_line():
    """Partial timing line (missing some fields) returns None."""
    assert parse_log_line("slot print_timings: incomplete") is None
    assert parse_log_line("slot print_timings:") is None
    assert parse_log_line("slot print_timings: id  0") is None


def test_parse_log_line_returns_none_for_non_timing():
    """Non-timing lines return None."""
    assert parse_log_line("update_slots: launching") is None
    assert parse_log_line("") is None
    assert parse_log_line("random log line") is None


def test_parse_log_line_handles_various_spaces():
    """Timing lines with various spacing are parsed."""
    line = "slot print_timings: id  0 | task 1 | prompt eval time =     853.13 ms /   285 tokens (2.99 ms per token, 334.06 tokens per second)"
    record = parse_log_line(line)
    assert record is not None
    assert record.tokens == 285


def test_parse_old_cli_sample_time_ignored():
    """Old CLI format sample time lines are not treated as generation."""
    log = """\
llama_print_timings:        load time =    1234.56 ms
llama_print_timings:      sample time =      12.34 ms /    64 runs
"""
    records = parse_log_text(log)
    # sample time should not be parsed as prompt or eval
    assert all(r.kind != "prompt" for r in records)
    assert all(r.kind != "eval" for r in records)


def test_parse_old_cli_eval_format():
    """Old CLI format eval lines parsed correctly."""
    log = """\
llama_print_timings: prompt eval time =    2000.00 ms /   512 tokens (3.91 ms per token, 256.00 tokens per second)
llama_print_timings:        eval time =    4000.00 ms /   128 runs (31.25 ms per token, 32.00 tokens per second)
"""
    records = parse_log_text(log)
    assert len(records) == 2
    prompt = [r for r in records if r.kind == "prompt"]
    eval_r = [r for r in records if r.kind == "eval"]
    assert len(prompt) == 1
    assert len(eval_r) == 1
    assert prompt[0].tokens == 512
    assert eval_r[0].tokens == 128


def test_percentiles_empty_session_returns_zeros():
    stats = SessionStats()
    p = stats.percentiles()
    assert p == {
        "ttft_ms_avg": 0.0, "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0,
        "gen_tps_p50": 0.0, "gen_tps_p95": 0.0,
    }


def test_percentiles_known_distribution():
    stats = SessionStats()
    for ms in (100.0, 200.0, 300.0, 400.0, 500.0):
        stats.add(TimingRecord(kind="prompt", tokens=10, time_ms=ms, tokens_per_second=100.0))
    for tps in (10.0, 20.0, 30.0, 40.0, 50.0):
        stats.add(TimingRecord(kind="eval", tokens=10, time_ms=100.0, tokens_per_second=tps))

    p = stats.percentiles()
    assert p["ttft_ms_avg"] == 300.0
    assert p["ttft_ms_p50"] == 300.0
    assert p["ttft_ms_p95"] == 500.0
    assert p["gen_tps_p50"] == 30.0
    assert p["gen_tps_p95"] == 50.0


def test_slice_from_extracts_only_new_requests():
    stats = SessionStats()
    stats.add(TimingRecord(kind="prompt", tokens=10, time_ms=100.0, tokens_per_second=100.0))
    stats.add(TimingRecord(kind="eval", tokens=20, time_ms=200.0, tokens_per_second=100.0))
    prompt_snapshot = len(stats.prompt_ms_samples)
    gen_snapshot = len(stats.gen_tps_samples)

    stats.add(TimingRecord(kind="prompt", tokens=30, time_ms=300.0, tokens_per_second=100.0))
    stats.add(TimingRecord(kind="eval", tokens=40, time_ms=400.0, tokens_per_second=100.0))

    sliced = stats.slice_from(prompt_snapshot, gen_snapshot)
    assert sliced.requests == 1
    assert sliced.prompt_tokens == 30
    assert sliced.prompt_ms == 300.0
    assert sliced.gen_tokens == 40
    assert sliced.gen_ms == 400.0
    # Original stats are untouched.
    assert stats.requests == 2
    assert stats.prompt_tokens == 40


def test_slice_from_empty_when_nothing_new():
    stats = SessionStats()
    stats.add(TimingRecord(kind="prompt", tokens=10, time_ms=100.0, tokens_per_second=100.0))
    stats.add(TimingRecord(kind="eval", tokens=20, time_ms=200.0, tokens_per_second=100.0))
    sliced = stats.slice_from(len(stats.prompt_ms_samples), len(stats.gen_tps_samples))
    assert sliced.requests == 0
    assert sliced.prompt_tokens == 0
    assert sliced.gen_tokens == 0


def test_percentiles_do_not_get_persisted_as_raw_samples():
    """Sample lists are transient in-memory only; percentiles() is a summary."""
    stats = SessionStats()
    stats.add(TimingRecord(kind="prompt", tokens=10, time_ms=100.0, tokens_per_second=100.0))
    assert len(stats.prompt_ms_samples) == 1
    summary = stats.percentiles()
    assert isinstance(summary["ttft_ms_avg"], float)
    assert "prompt_ms_samples" not in summary
