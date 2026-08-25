"""Tests for the built-in benchmark runner (HTTP calls mocked, no real server)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from llama_launcher.benchmark.runner import (
    BenchmarkRunConfig,
    diff_new_log_lines,
    run_benchmark,
    run_completion,
    wait_for_server_ready,
)


def _fake_response(status=200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = b"{}"
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_wait_for_server_ready_succeeds_immediately():
    with patch("llama_launcher.benchmark.runner.urllib.request.urlopen",
              return_value=_fake_response(200)) as mock_open:
        assert wait_for_server_ready("http://127.0.0.1:8080", timeout_s=5.0) is True
    mock_open.assert_called_once()
    assert mock_open.call_args[0][0] == "http://127.0.0.1:8080/health"


def test_wait_for_server_ready_retries_then_succeeds():
    calls = {"n": 0}

    def side_effect(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection refused")
        return _fake_response(200)

    with patch("llama_launcher.benchmark.runner.urllib.request.urlopen",
              side_effect=side_effect):
        with patch("llama_launcher.benchmark.runner.time.sleep"):
            ok = wait_for_server_ready("http://127.0.0.1:8080", timeout_s=5.0,
                                       poll_interval_s=0.01)
    assert ok is True
    assert calls["n"] == 3


def test_wait_for_server_ready_times_out():
    # Force the deadline to already be in the past after the first check.
    times = iter([0.0, 0.0, 10.0])
    with patch("llama_launcher.benchmark.runner.urllib.request.urlopen",
              side_effect=OSError("refused")):
        with patch("llama_launcher.benchmark.runner.time.sleep"):
            with patch("llama_launcher.benchmark.runner.time.monotonic",
                      side_effect=lambda: next(times)):
                ok = wait_for_server_ready("http://127.0.0.1:8080", timeout_s=1.0)
    assert ok is False


def test_run_completion_posts_expected_json_body():
    with patch("llama_launcher.benchmark.runner.urllib.request.urlopen",
              return_value=_fake_response(200)) as mock_open:
        run_completion("http://127.0.0.1:8080", "hello world", n_predict=64)

    req = mock_open.call_args[0][0]
    assert req.full_url == "http://127.0.0.1:8080/completion"
    body = json.loads(req.data.decode("utf-8"))
    assert body["prompt"] == "hello world"
    assert body["n_predict"] == 64
    assert body["stream"] is False
    assert body["cache_prompt"] is False


def test_run_completion_raises_on_http_error():
    with patch("llama_launcher.benchmark.runner.urllib.request.urlopen",
              side_effect=urllib_error_stub()):
        with pytest.raises(Exception):
            run_completion("http://127.0.0.1:8080", "hi", n_predict=8)


def urllib_error_stub():
    import urllib.error
    return urllib.error.URLError("boom")


def test_run_benchmark_fires_iterations_plus_warmup():
    with patch("llama_launcher.benchmark.runner.run_completion") as mock_run:
        cfg = BenchmarkRunConfig(prompt="test prompt", n_predict=32,
                                 iterations=3, warmup=True)
        run_benchmark("http://127.0.0.1:8080", cfg)
    assert mock_run.call_count == 4  # 3 iterations + 1 warmup


def test_run_benchmark_no_warmup():
    with patch("llama_launcher.benchmark.runner.run_completion") as mock_run:
        cfg = BenchmarkRunConfig(iterations=2, warmup=False)
        run_benchmark("http://127.0.0.1:8080", cfg)
    assert mock_run.call_count == 2


def test_run_benchmark_prompts_are_unique_per_iteration():
    seen_prompts = []

    def _capture(base_url, prompt, n_predict, timeout_s=60.0):
        seen_prompts.append(prompt)

    with patch("llama_launcher.benchmark.runner.run_completion", side_effect=_capture):
        cfg = BenchmarkRunConfig(iterations=3, warmup=False)
        run_benchmark("http://127.0.0.1:8080", cfg)
    assert len(set(seen_prompts)) == 3  # each iteration marker makes the prompt unique


def test_diff_new_log_lines_simple():
    baseline = "a\nb\nc\n"
    post = "a\nb\nc\nd\ne\n"
    assert diff_new_log_lines(baseline, post) == "d\ne"


def test_diff_new_log_lines_handles_duplicates():
    baseline = "x\nx\ny\n"
    post = "x\nx\nx\ny\nz\n"
    # Two x's already accounted for; the third x and z are new.
    assert diff_new_log_lines(baseline, post) == "x\nz"


def test_diff_new_log_lines_no_change():
    log = "a\nb\nc\n"
    assert diff_new_log_lines(log, log) == ""
