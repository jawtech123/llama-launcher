"""Controlled benchmark run against a live llama-server, over HTTP.

Stdlib-only (``urllib``) -- no new dependency. Timing numbers still come
from llama-server's own log lines via the existing parser pipeline; the
HTTP responses here are fired-and-discarded, so the exact response body
shape doesn't need to be relied on.
"""
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass

#: ~50 words, long enough to produce a measurable prompt-processing time.
DEFAULT_PROMPT = (
    "Describe, in a few paragraphs, how a modern operating system schedules "
    "processes across multiple CPU cores, covering preemption, priority "
    "levels, and how context switches are minimized for performance."
)


@dataclass
class BenchmarkRunConfig:
    prompt: str = DEFAULT_PROMPT
    n_predict: int = 128
    iterations: int = 3
    warmup: bool = True
    timeout_s: float = 60.0


def wait_for_server_ready(base_url: str, timeout_s: float = 60.0,
                          poll_interval_s: float = 0.5) -> bool:
    """Poll ``GET /health`` until it returns HTTP 200, or give up.

    llama-server returns 503 while the model is still loading and 200 once
    it can serve requests -- steadier than matching a log line across
    llama.cpp versions.
    """
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval_s)
    return False


def run_completion(base_url: str, prompt: str, n_predict: int,
                   timeout_s: float = 60.0) -> None:
    """Fire one ``POST /completion`` request and discard the body."""
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
        "cache_prompt": False,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp.read()


def run_benchmark(base_url: str, cfg: BenchmarkRunConfig) -> None:
    """Fire ``cfg.iterations`` sequential completions (+1 warmup if enabled).

    Each prompt is suffixed with a unique iteration marker so llama-server's
    prompt-prefix cache can never short-circuit re-processing, which would
    silently skew the throughput numbers.
    """
    total = cfg.iterations + (1 if cfg.warmup else 0)
    for i in range(total):
        prompt = f"{cfg.prompt}\n\n[benchmark iteration {i}]"
        run_completion(base_url, prompt, cfg.n_predict, cfg.timeout_s)


def diff_new_log_lines(baseline: str, post: str) -> str:
    """Lines present in ``post`` but not accounted for in ``baseline``.

    Both are fixed-size tail reads of the same growing log file. A multiset
    diff correctly handles duplicate lines, unlike a naive membership check.
    Written for a future tail-log-based (Server Management) benchmark path.
    """
    seen = Counter(baseline.splitlines())
    out = []
    for line in post.splitlines():
        if seen.get(line, 0) > 0:
            seen[line] -= 1
        else:
            out.append(line)
    return "\n".join(out)
