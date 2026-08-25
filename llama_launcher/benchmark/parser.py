"""Parse llama.cpp timing lines from server logs.

llama.cpp emits one pair of timing lines per request, e.g.::

    slot print_timings: id  0 | task 34 | prompt eval time = 853.13 ms / 285 tokens (2.99 ms per token, 334.06 tokens per second)
    slot print_timings: id  0 | task 34 |        eval time = 11039.20 ms / 256 tokens (43.12 ms per token, 23.19 tokens per second)

Older builds prefix with ``llama_print_timings:`` and count "runs" instead
of "tokens" on the eval line. Both variants are handled.

- "prompt eval" = prompt processing (tokens *in*)
- "eval"        = token generation  (tokens *out*)
"""
import re
import statistics
from dataclasses import dataclass, field

#: Guard against unbounded memory growth on pathologically long-running sessions.
_MAX_SAMPLES = 20000

#: 1234.56 ms /  512 tokens (    2.41 ms per token,   212.50 tokens per second)
_TIMING_RE = re.compile(
    r"=\s*([0-9.,]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)"
    r"\s*\(\s*[0-9.,]+\s*ms per token,\s*([0-9.,]+)\s*tokens per second"
)


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


@dataclass
class TimingRecord:
    """One parsed timing line."""
    kind: str               # "prompt" | "eval"
    tokens: int
    time_ms: float
    tokens_per_second: float


def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


@dataclass
class SessionStats:
    """Aggregated statistics for a set of timing records.

    ``prompt_ms_samples``/``gen_tps_samples`` hold one entry per request
    (transient, in-memory only) so ``percentiles()`` can report spread
    without ever persisting per-request rows to disk.
    """
    requests: int = 0
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    gen_tokens: int = 0
    gen_ms: float = 0.0
    prompt_ms_samples: list = field(default_factory=list)
    gen_tps_samples: list = field(default_factory=list)
    #: Parallel to the two lists above -- lets a caller slice out a sub-range
    #: (e.g. "just the requests fired by the benchmark runner") and recompute
    #: exact token/time sums for that slice, not just spread percentiles.
    prompt_tokens_samples: list = field(default_factory=list)
    gen_tokens_samples: list = field(default_factory=list)
    gen_ms_samples: list = field(default_factory=list)

    def add(self, record: TimingRecord) -> None:
        if record.kind == "prompt":
            self.requests += 1
            self.prompt_tokens += record.tokens
            self.prompt_ms += record.time_ms
            if len(self.prompt_ms_samples) < _MAX_SAMPLES:
                self.prompt_ms_samples.append(record.time_ms)
                self.prompt_tokens_samples.append(record.tokens)
        else:
            self.gen_tokens += record.tokens
            self.gen_ms += record.time_ms
            if len(self.gen_tps_samples) < _MAX_SAMPLES:
                self.gen_tps_samples.append(record.tokens_per_second)
                self.gen_tokens_samples.append(record.tokens)
                self.gen_ms_samples.append(record.time_ms)

    @property
    def prompt_tps(self) -> float:
        """Time-weighted average prompt-processing speed (tokens/second)."""
        return self.prompt_tokens / (self.prompt_ms / 1000.0) if self.prompt_ms > 0 else 0.0

    @property
    def gen_tps(self) -> float:
        """Time-weighted average generation speed (tokens/second)."""
        return self.gen_tokens / (self.gen_ms / 1000.0) if self.gen_ms > 0 else 0.0

    @property
    def busy_seconds(self) -> float:
        return (self.prompt_ms + self.gen_ms) / 1000.0

    def slice_from(self, prompt_index: int, gen_index: int) -> "SessionStats":
        """Build a fresh SessionStats over only the requests recorded after
        the given sample indices.

        Used to extract "just what happened during the benchmark run" from a
        live session's running stats, without disturbing the live totals
        (which keep counting everything -- benchmark traffic included -- as
        real server activity too).
        """
        prompt_ms = self.prompt_ms_samples[prompt_index:]
        prompt_tokens = self.prompt_tokens_samples[prompt_index:]
        gen_tps = self.gen_tps_samples[gen_index:]
        gen_tokens = self.gen_tokens_samples[gen_index:]
        gen_ms = self.gen_ms_samples[gen_index:]
        return SessionStats(
            requests=len(prompt_ms),
            prompt_tokens=sum(prompt_tokens),
            prompt_ms=sum(prompt_ms),
            gen_tokens=sum(gen_tokens),
            gen_ms=sum(gen_ms),
            prompt_ms_samples=list(prompt_ms),
            gen_tps_samples=list(gen_tps),
            prompt_tokens_samples=list(prompt_tokens),
            gen_tokens_samples=list(gen_tokens),
            gen_ms_samples=list(gen_ms),
        )

    def percentiles(self) -> dict:
        """Per-request time-to-first-token and generation-speed spread.

        Prompt processing time *is* time-to-first-token (the first
        generated token can't appear until the prompt finishes), so this
        needs no new log parsing -- just percentiles over the per-request
        samples already collected by ``add()``.
        """
        return {
            "ttft_ms_avg": statistics.fmean(self.prompt_ms_samples) if self.prompt_ms_samples else 0.0,
            "ttft_ms_p50": _percentile(self.prompt_ms_samples, 50),
            "ttft_ms_p95": _percentile(self.prompt_ms_samples, 95),
            "gen_tps_p50": _percentile(self.gen_tps_samples, 50),
            "gen_tps_p95": _percentile(self.gen_tps_samples, 95),
        }


def parse_log_line(line: str):
    """Parse a single log line; return a TimingRecord or None."""
    if "eval time" not in line:
        return None
    if "prompt eval time" in line:
        kind = "prompt"
    elif "eval time" in line:
        kind = "eval"
    else:
        return None
    m = _TIMING_RE.search(line)
    if not m:
        return None
    return TimingRecord(
        kind=kind,
        tokens=int(m.group(2)),
        time_ms=_to_float(m.group(1)),
        tokens_per_second=_to_float(m.group(3)),
    )


def parse_log_text(text: str) -> list:
    """Parse all timing records from a block of log text."""
    records = []
    for line in text.splitlines():
        rec = parse_log_line(line)
        if rec is not None:
            records.append(rec)
    return records


def aggregate(records) -> SessionStats:
    """Aggregate a list of TimingRecords into SessionStats."""
    stats = SessionStats()
    for rec in records:
        stats.add(rec)
    return stats
