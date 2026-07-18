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
from dataclasses import dataclass

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


@dataclass
class SessionStats:
    """Aggregated statistics for a set of timing records."""
    requests: int = 0
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    gen_tokens: int = 0
    gen_ms: float = 0.0

    def add(self, record: TimingRecord) -> None:
        if record.kind == "prompt":
            self.requests += 1
            self.prompt_tokens += record.tokens
            self.prompt_ms += record.time_ms
        else:
            self.gen_tokens += record.tokens
            self.gen_ms += record.time_ms

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
