"""Tests for llama.cpp log timing parsing and benchmark storage."""
from llama_launcher.benchmark.fingerprint import config_fingerprint
from llama_launcher.benchmark.parser import (
    aggregate,
    parse_log_line,
    parse_log_text,
)
from llama_launcher.benchmark.storage import (
    append_session,
    clear_sessions,
    latest_session_for_preset,
    load_sessions,
    replace_sessions_for_source,
)
from llama_launcher.core.config import LlamaConfig

SERVER_LOG = """\
srv  update_slots: id  0 | task 1 | launching slot
slot print_timings: id  0 | task 1 | prompt eval time =     853.13 ms /   285 tokens (    2.99 ms per token,   334.06 tokens per second)
slot print_timings: id  0 | task 1 |        eval time =   11039.20 ms /   256 tokens (   43.12 ms per token,    23.19 tokens per second)
slot print_timings: id  0 | task 1 |       total time =   11892.33 ms /   541 tokens
srv  update_slots: id  0 | task 2 | launching slot
slot print_timings: id  0 | task 2 | prompt eval time =     100.00 ms /   100 tokens (    1.00 ms per token,  1000.00 tokens per second)
slot print_timings: id  0 | task 2 |        eval time =    1000.00 ms /    50 tokens (   20.00 ms per token,    50.00 tokens per second)
"""

OLD_CLI_LOG = """\
llama_print_timings:        load time =    1234.56 ms
llama_print_timings:      sample time =      12.34 ms /    64 runs   (    0.19 ms per token,  5188.33 tokens per second)
llama_print_timings: prompt eval time =    2000.00 ms /   512 tokens (    3.91 ms per token,   256.00 tokens per second)
llama_print_timings:        eval time =    4000.00 ms /   128 runs   (   31.25 ms per token,    32.00 tokens per second)
llama_print_timings:       total time =    6000.00 ms /   640 tokens
"""


def test_parse_server_prompt_and_eval_lines():
    records = parse_log_text(SERVER_LOG)
    assert len(records) == 4
    prompts = [r for r in records if r.kind == "prompt"]
    evals = [r for r in records if r.kind == "eval"]
    assert len(prompts) == 2 and len(evals) == 2
    assert prompts[0].tokens == 285
    assert abs(prompts[0].tokens_per_second - 334.06) < 0.01
    assert evals[0].tokens == 256
    assert abs(evals[0].time_ms - 11039.20) < 0.01


def test_parse_old_cli_format_with_runs():
    records = parse_log_text(OLD_CLI_LOG)
    prompts = [r for r in records if r.kind == "prompt"]
    evals = [r for r in records if r.kind == "eval"]
    # sample time lines must not be picked up as generation stats
    assert len(prompts) == 1 and len(evals) == 1
    assert prompts[0].tokens == 512
    assert evals[0].tokens == 128
    assert abs(evals[0].tokens_per_second - 32.0) < 0.01


def test_aggregate_computes_time_weighted_speeds():
    stats = aggregate(parse_log_text(SERVER_LOG))
    assert stats.requests == 2
    assert stats.prompt_tokens == 385
    assert stats.gen_tokens == 306
    expected_in = 385 / ((853.13 + 100.0) / 1000.0)
    expected_out = 306 / ((11039.20 + 1000.0) / 1000.0)
    assert abs(stats.prompt_tps - expected_in) < 0.01
    assert abs(stats.gen_tps - expected_out) < 0.01


def test_parse_log_line_ignores_unrelated_lines():
    assert parse_log_line("srv  update_slots: id  0 | task 1 | launching slot") is None
    assert parse_log_line("") is None
    assert parse_log_line("random text") is None


def test_storage_roundtrip_and_replace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llama_launcher.benchmark.storage.config_file",
        lambda name: tmp_path / name,
    )
    clear_sessions()
    assert load_sessions() == []

    append_session({"source": "local", "prompt_tokens": 100})
    append_session({"source": "phoenix", "prompt_tokens": 200})
    assert len(load_sessions()) == 2

    # Re-importing the same source replaces instead of duplicating
    replace_sessions_for_source("phoenix",
                                {"source": "phoenix", "prompt_tokens": 999})
    sessions = load_sessions()
    assert len(sessions) == 2
    phoenix = [s for s in sessions if s["source"] == "phoenix"]
    assert len(phoenix) == 1 and phoenix[0]["prompt_tokens"] == 999

    clear_sessions()
    assert load_sessions() == []


def test_fingerprint_stable_and_parameter_sensitive():
    a = config_fingerprint(LlamaConfig(model_path="/m.gguf", context_size=4096))
    b = config_fingerprint(LlamaConfig(model_path="/m.gguf", context_size=4096))
    assert a == b  # identical configs -> identical fingerprint
    c = config_fingerprint(LlamaConfig(model_path="/m.gguf", context_size=8192))
    assert a != c  # any parameter change -> new fingerprint (benchmark resets)


def test_latest_session_for_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llama_launcher.benchmark.storage.config_file",
        lambda name: tmp_path / name,
    )
    clear_sessions()
    assert latest_session_for_preset("Large 32GB") is None

    append_session({"preset": "Large 32GB", "fingerprint": "aaa",
                    "started": 100.0, "gen_tokens": 10})
    append_session({"preset": "Small 8GB", "fingerprint": "bbb",
                    "started": 200.0, "gen_tokens": 20})
    append_session({"preset": "Large 32GB", "fingerprint": "ccc",
                    "started": 300.0, "gen_tokens": 30})

    latest = latest_session_for_preset("Large 32GB")
    assert latest["fingerprint"] == "ccc"
    assert latest["gen_tokens"] == 30
    assert latest_session_for_preset("Small 8GB")["fingerprint"] == "bbb"
