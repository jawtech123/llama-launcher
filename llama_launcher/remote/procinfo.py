"""Parsers for process/port probe output, shared by both transports.

The local and SSH connections run the same shell probes -- ``ps``, ``ss``,
``nvidia-smi`` -- and differ only in how the command is executed. Keeping the
parsing here means one implementation and one set of tests for both.
"""
import logging
import re
import shlex
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from llama_launcher.gpu.probe import build_live_probe, parse_gpu_probe, split_sections
from llama_launcher.gpu.types import GPUInfo

logger = logging.getLogger(__name__)

_PID_RE = re.compile(r"pid=(\d+)")

#: Marker separating the sections of the combined status probe.
_SECTION = "@@LL@@"

#: How long the CPU sampler waits between /proc/stat reads.
CPU_SAMPLE_SECONDS = 0.3


@dataclass
class ResourceSample:
    """A live utilization reading from one machine.

    Negative values mean "not available here" (no GPU, unreadable counters), so
    the UI can distinguish that from a genuine zero.
    """
    cpu_percent: float = -1.0
    ram_used_mb: float = -1.0
    ram_total_mb: float = -1.0
    gpu_percent: float = -1.0
    gpu_mem_used_mb: float = -1.0
    gpu_mem_total_mb: float = -1.0
    #: VRAM held by this server's own llama-server, if one is running.
    own_vram_mb: float = 0.0
    #: Full live reading (per card, temperature, power, processes), if any.
    gpu: Optional[GPUInfo] = None


def parse_listener_pid(ss_output: str) -> Optional[int]:
    """Extract the listening process PID from ``ss -tlnp`` output.

    ``ss`` only reveals the PID for processes the caller owns, so a foreign
    listener yields None rather than a misattributed PID.
    """
    match = _PID_RE.search(ss_output or "")
    return int(match.group(1)) if match else None


def parse_ps_stats(ps_output: str) -> Tuple[float, float, float]:
    """Parse ``ps -o %cpu,rss,etimes`` into (cpu_percent, ram_mb, uptime_s)."""
    parts = (ps_output or "").strip().split()
    if len(parts) < 3:
        return (0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]) / 1024.0, float(parts[2]))
    except ValueError:
        return (0.0, 0.0, 0.0)


def parse_gpu_stats(gpu_output: str) -> Tuple[float, float]:
    """Parse the status probe's GPU section into (util_pct, mem_used_mb).

    Accepts either the sectioned live probe (NVIDIA or AMD) or the legacy
    ``utilization, memory`` nvidia-smi CSV. Multi-GPU: utilization is the max
    across devices, memory the sum, so a model split over several cards
    reports its true total footprint.
    """
    if "@@GPU:" in (gpu_output or ""):
        gpu = parse_gpu_probe(gpu_output)
        if gpu is None or gpu.vram_mb <= 0:
            return (0.0, 0.0)
        return (max(gpu.utilization_pct, 0.0), max(gpu.used_vram_mb, 0.0))
    utils, mems = [], []
    for line in (gpu_output or "").strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) < 2:
            continue
        try:
            utils.append(float(values[0]))
            mems.append(float(values[1]))
        except ValueError:
            continue
    if not utils:
        return (0.0, 0.0)
    return (max(utils), sum(mems))


def parse_cpu_times(proc_stat_line: str) -> Optional[Tuple[float, float]]:
    """Return ``(total, idle)`` jiffies from a ``cpu `` line of /proc/stat."""
    parts = (proc_stat_line or "").split()
    if len(parts) < 5 or not parts[0].startswith("cpu"):
        return None
    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    return (sum(values), idle)


def cpu_percent_from_samples(first_line: str, second_line: str) -> float:
    """CPU utilization between two /proc/stat readings. -1 when unavailable."""
    first = parse_cpu_times(first_line)
    second = parse_cpu_times(second_line)
    if first is None or second is None:
        return -1.0
    total_delta = second[0] - first[0]
    idle_delta = second[1] - first[1]
    if total_delta <= 0:
        return -1.0
    return max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))


def parse_meminfo(text: str) -> Tuple[float, float]:
    """Parse /proc/meminfo into ``(used_mb, total_mb)``.

    Uses MemAvailable rather than MemFree: cache and buffers are reclaimable, so
    MemFree wildly overstates pressure on a machine that has been running a while.
    """
    total_kb = available_kb = 0.0
    for line in (text or "").splitlines():
        if line.startswith("MemTotal:"):
            total_kb = _first_number(line)
        elif line.startswith("MemAvailable:"):
            available_kb = _first_number(line)
        if total_kb and available_kb:
            break
    if total_kb <= 0:
        return (-1.0, -1.0)
    total_mb = total_kb / 1024.0
    used_mb = (total_kb - available_kb) / 1024.0
    return (max(0.0, used_mb), total_mb)


def _first_number(line: str) -> float:
    for token in line.split():
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def parse_compute_apps(text: str, pids) -> float:
    """Sum the VRAM held by ``pids`` from nvidia-smi --query-compute-apps output."""
    wanted = {int(p) for p in pids if p}
    if not wanted:
        return 0.0
    total = 0.0
    for line in (text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) in wanted:
                total += float(parts[1])
        except ValueError:
            continue
    return total


def build_status_probe(pid: Optional[int], port: int) -> str:
    """Build one shell command gathering every status field at once.

    Status is polled on a timer; issuing four separate commands meant four SSH
    round trips per refresh. Emitting delimited sections keeps that to one.
    """
    pid_part = (
        f"ps -p {int(pid)} -o %cpu,rss,etimes --no-headers 2>/dev/null"
        if pid else "true"
    )
    return (
        f"echo {shlex.quote(_SECTION)}; {pid_part}; "
        f"echo {shlex.quote(_SECTION)}; "
        f"ss -tlnpH 'sport = :{int(port)}' 2>/dev/null; "
        f"echo {shlex.quote(_SECTION)}; "
        + build_live_probe()
    )


def split_status_probe(output: str) -> Tuple[str, str, str]:
    """Split combined probe output into (ps, ss, nvidia-smi) sections."""
    ps_out, ss_out, gpu_out = _split_sections(output, 3)
    return (ps_out, ss_out, gpu_out)


def build_usage_probe(own_pids: Iterable[int] = ()) -> str:
    """One shell command sampling CPU, RAM and GPU utilization.

    The CPU figure needs two /proc/stat reads separated in time; doing that
    inside a single command keeps it to one SSH round trip instead of two, and
    keeps the interval accurate regardless of network latency. The GPU part is
    the shared live probe (NVIDIA or AMD); ``own_pids`` lets an AMD host report
    the VRAM our own llama-server holds.
    """
    return (
        f"echo {shlex.quote(_SECTION)}; grep '^cpu ' /proc/stat; "
        f"sleep {CPU_SAMPLE_SECONDS}; "
        f"echo {shlex.quote(_SECTION)}; grep '^cpu ' /proc/stat; "
        f"echo {shlex.quote(_SECTION)}; cat /proc/meminfo 2>/dev/null; "
        f"echo {shlex.quote(_SECTION)}; "
        + build_live_probe(own_pids)
    )


def parse_usage_probe(output: str, own_pids=()) -> ResourceSample:
    """Turn ``build_usage_probe`` output into a ResourceSample."""
    cpu_first, cpu_second, meminfo, gpu_text, apps_text = _split_sections(output, 5)

    sample = ResourceSample()
    sample.cpu_percent = cpu_percent_from_samples(cpu_first, cpu_second)
    sample.ram_used_mb, sample.ram_total_mb = parse_meminfo(meminfo)

    if "@@GPU:" in gpu_text:
        gpu = parse_gpu_probe(gpu_text)
        if gpu is not None and gpu.vram_mb > 0:
            sample.gpu = gpu
            sample.gpu_percent = gpu.utilization_pct
            sample.gpu_mem_used_mb = gpu.used_vram_mb
            sample.gpu_mem_total_mb = float(gpu.vram_mb)
            wanted = {int(p) for p in own_pids if p}
            sample.own_vram_mb = sum(p.vram_mb for p in gpu.processes if p.pid in wanted)
        return sample

    # Legacy layout: "util, used, total" rows then "pid, used" rows.
    utils, useds, totals = [], [], []
    for line in (gpu_text or "").strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) < 3:
            continue
        try:
            utils.append(float(values[0]))
            useds.append(float(values[1]))
            totals.append(float(values[2]))
        except ValueError:
            continue
    if utils:
        # Multi-GPU: highest utilization, summed memory -- matches the estimator.
        sample.gpu_percent = max(utils)
        sample.gpu_mem_used_mb = sum(useds)
        sample.gpu_mem_total_mb = sum(totals)
    sample.own_vram_mb = parse_compute_apps(apps_text, own_pids)
    return sample


def _split_sections(output: str, count: int) -> list:
    """Split marker-delimited probe output into exactly ``count`` sections.

    The last section absorbs any surplus, so a trailing multi-section GPU
    probe stays intact.
    """
    sections = (output or "").split(_SECTION)
    sections = sections[1:] if len(sections) > 1 else sections
    if len(sections) > count:
        sections = sections[:count - 1] + [_SECTION.join(sections[count - 1:])]
    while len(sections) < count:
        sections.append("")
    return [s.strip() for s in sections[:count]]
