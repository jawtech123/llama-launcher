"""One shell probe that describes the GPUs of any Linux host, NVIDIA or AMD.

The same script runs on this machine (``sh -c``) and on a server reached over
SSH, so local and remote hosts are described identically. Every tool is
optional: sections for missing tools come back empty and the parser uses
whatever is there, in order of preference:

- NVIDIA: ``nvidia-smi`` (always present with the driver).
- AMD: ``amd-smi`` (ROCm 6+), else ``rocm-smi``, else the ``amdgpu`` kernel
  driver's sysfs files, which exist on every AMD box even without ROCm. Names
  come from ``lspci`` when no SMI tool is installed.
- Anything else: ``vulkaninfo``.

The *live* probe is the cheap subset polled every few seconds (utilization,
memory, temperature, power, per-process VRAM); the *full* probe adds the
static facts gathered once per connect.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .types import GPUInfo, GpuProcess

logger = logging.getLogger(__name__)

_SECTION = "@@GPU:"

#: nvidia-smi columns, in the order the probe requests them.
_NV_FIELDS = ("index", "name", "memory.total", "memory.used", "memory.free",
              "driver_version", "temperature.gpu", "power.draw", "power.limit",
              "utilization.gpu", "clocks.gr", "clocks.mem", "pci.bus_id")
_NV_QUERY = ("nvidia-smi --query-gpu=" + ",".join(_NV_FIELDS)
             + " --format=csv,noheader,nounits 2>&1")
#: compute_cap is a newer query field; asked separately so an old driver
#: that rejects it does not take the whole row with it.
_NV_CC_QUERY = "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits 2>/dev/null"
_NV_PROC_QUERY = ("nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory"
                  " --format=csv,noheader,nounits 2>/dev/null")

#: sysfs attributes copied per card; hwmon values follow.
_SYSFS_ATTRS = ("vendor", "device", "product_name", "mem_info_vram_total",
                "mem_info_vram_used", "gpu_busy_percent", "current_link_speed",
                "current_link_width")
_HWMON_ATTRS = ("temp1_input", "power1_average", "power1_input", "power1_cap",
                "freq1_input", "freq2_input")

_SYSFS_SCRIPT = (
    "for d in /sys/class/drm/card[0-9]*/device; do "
    "[ -f \"$d/vendor\" ] || continue; "
    "echo \"card=$(basename $(dirname \"$d\"))\"; "
    "echo \"slot=$(basename $(readlink -f \"$d\"))\"; "
    + " ".join(f"[ -f \"$d/{a}\" ] && echo \"{a}=$(cat \"$d/{a}\" 2>/dev/null)\";" for a in _SYSFS_ATTRS)
    + " for h in \"$d\"/hwmon/hwmon*; do [ -d \"$h\" ] || continue; "
    + " ".join(f"[ -f \"$h/{a}\" ] && echo \"{a}=$(cat \"$h/{a}\" 2>/dev/null)\";" for a in _HWMON_ATTRS)
    + " done; done; "
    "echo \"amdgpu_version=$(cat /sys/module/amdgpu/version 2>/dev/null || uname -r)\""
)

_FDINFO_SCRIPT = (
    # Per-process VRAM on amdgpu without ROCm: the DRM fdinfo of each fd.
    "for p in {pids}; do for f in /proc/$p/fdinfo/*; do "
    "grep -qs '^drm-driver:.*amdgpu' \"$f\" || continue; "
    "echo \"pid=$p\"; grep -E '^drm-(client-id|memory-vram)' \"$f\" 2>/dev/null; done; done"
)


def _section(name: str, cmd: str) -> str:
    return f"echo '{_SECTION}{name}'; {cmd}; "


def build_live_probe(pids: Iterable[int] = ()) -> str:
    """Shell script reporting utilization/memory/temperature/power per GPU."""
    pid_list = " ".join(str(int(p)) for p in pids if p)
    script = (
        _section("nv", _NV_QUERY)
        + _section("nvproc", _NV_PROC_QUERY)
        + _section("sysfs", _SYSFS_SCRIPT)
    )
    if pid_list:
        script += _section("fdinfo", _FDINFO_SCRIPT.format(pids=pid_list))
    return script + "true"


def build_full_probe() -> str:
    """Live probe plus the static facts (names, driver, ROCm, Vulkan)."""
    return (
        build_live_probe().rstrip("true")
        + _section("nvcc", _NV_CC_QUERY)
        + _section("amdsmi_static", "amd-smi static --json 2>/dev/null")
        + _section("amdsmi_metric", "amd-smi metric --json 2>/dev/null")
        + _section("rocm", "rocm-smi --showproductname --showmeminfo vram --showuse "
                           "--showtemp --showpower --showclocks --showdriverversion --json 2>/dev/null")
        + _section("rocm_present", "(command -v rocm-smi || command -v amd-smi || "
                                   "command -v rocminfo || [ -d /opt/rocm ]) >/dev/null 2>&1 && echo yes")
        + _section("lspci", "lspci -D -nn 2>/dev/null | grep -Ei 'VGA|3D controller|Display controller'")
        + _section("vulkan", "vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|deviceType|driverVersion' | head -20")
        + "true"
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def split_sections(output: str) -> Dict[str, str]:
    """Map section name -> text. Text without any marker is returned as ``nv``."""
    if _SECTION not in (output or ""):
        return {"nv": output or ""}
    sections: Dict[str, str] = {}
    current = None
    buf: List[str] = []
    for line in output.splitlines():
        if line.startswith(_SECTION):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[len(_SECTION):].strip()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _num(text, default: float = -1.0) -> float:
    """Parse ``"1234"``, ``"12.5 W"``, ``"(2482Mhz)"`` or ``{"value": 12}``."""
    if isinstance(text, dict):
        text = text.get("value", default)
    if isinstance(text, bool):
        return default
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return default
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else default


def _nv_rows(csv_text: str) -> List[List[str]]:
    rows = []
    for line in (csv_text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append(parts)
    return rows


def parse_nvidia_rows(csv_text: str) -> List[GPUInfo]:
    """Parse the live-probe nvidia-smi rows (or the legacy 4-column query)."""
    gpus: List[GPUInfo] = []
    for i, parts in enumerate(_nv_rows(csv_text)):
        if len(parts) >= len(_NV_FIELDS) - 1 and parts[0].isdigit():
            fields = dict(zip(_NV_FIELDS, parts))
            total = _num(fields.get("memory.total"), 0)
            used = _num(fields.get("memory.used"), -1)
            free = _num(fields.get("memory.free"), -1)
            if total <= 0:
                continue
            gpu = GPUInfo(
                backend="cuda", vendor="nvidia", index=int(fields["index"]),
                vram_mb=int(total), available_vram_mb=int(free if free >= 0 else max(total - used, 0)),
                device_name=fields.get("name") or f"NVIDIA GPU {fields['index']}",
                driver_version=fields.get("driver_version") or None,
                used_vram_mb=used, temperature_c=_num(fields.get("temperature.gpu")),
                power_w=_num(fields.get("power.draw")), power_limit_w=_num(fields.get("power.limit")),
                utilization_pct=_num(fields.get("utilization.gpu")),
                clock_mhz=_num(fields.get("clocks.gr")), mem_clock_mhz=_num(fields.get("clocks.mem")),
                pci_bus_id=fields.get("pci.bus_id", ""),
            )
            gpus.append(gpu)
            continue
        # Legacy "name, total MiB, free MiB, driver" rows.
        total = _num(parts[1], 0)
        if total <= 0:
            continue
        free = _num(parts[2], -1) if len(parts) > 2 else -1
        gpus.append(GPUInfo(
            backend="cuda", vendor="nvidia", index=i, vram_mb=int(total),
            available_vram_mb=int(free if free >= 0 else total),
            device_name=parts[0] or f"NVIDIA GPU {i}",
            driver_version=parts[3] if len(parts) > 3 and parts[3] else None,
            used_vram_mb=(total - free) if free >= 0 else -1,
        ))
    return gpus


def _parse_nv_processes(text: str, gpus: List[GPUInfo]) -> List[GpuProcess]:
    bus_to_index = {g.pci_bus_id.upper(): g.index for g in gpus if g.pci_bus_id}
    procs = []
    for parts in _nv_rows(text):
        if len(parts) == 3:              # old query without bus id: pid, name, mem
            bus, pid, name, mem = "", parts[0], parts[1], parts[2]
        elif len(parts) >= 4:
            bus, pid, name, mem = parts[0], parts[1], ",".join(parts[2:-1]), parts[-1]
        else:
            continue
        if not pid.isdigit():
            continue
        # process_name is the full command line; keep the executable's basename.
        exe = name.split()[0] if name.split() else name
        procs.append(GpuProcess(pid=int(pid), name=exe.split("/")[-1], vram_mb=_num(mem, 0),
                                gpu_index=bus_to_index.get(bus.upper(), 0)))
    return procs


def parse_sysfs(text: str) -> Tuple[List[GPUInfo], str]:
    """Parse the sysfs section into AMD cards (NVIDIA cards are skipped)."""
    cards: List[dict] = []
    driver = ""
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "card":
            cards.append({"card": value})
        elif key == "amdgpu_version":
            driver = value
        elif cards:
            cards[-1][key] = value
    gpus = []
    for idx, c in enumerate(cards):
        if c.get("vendor", "").lower() != "0x1002":
            continue
        total = _num(c.get("mem_info_vram_total"), 0) / (1024 * 1024)
        used = _num(c.get("mem_info_vram_used"), -1)
        used_mb = used / (1024 * 1024) if used >= 0 else -1.0
        power = _num(c.get("power1_average"), -1)
        if power < 0:
            power = _num(c.get("power1_input"), -1)
        gpu = GPUInfo(
            backend="vulkan", vendor="amd", index=len(gpus),
            vram_mb=int(total), available_vram_mb=int(max(total - used_mb, 0)) if used_mb >= 0 else int(total),
            device_name=c.get("product_name") or "",
            used_vram_mb=used_mb, utilization_pct=_num(c.get("gpu_busy_percent")),
            temperature_c=_num(c.get("temp1_input"), -1000) / 1000 if "temp1_input" in c else -1.0,
            power_w=power / 1e6 if power >= 0 else -1.0,
            power_limit_w=_num(c.get("power1_cap"), -1) / 1e6 if "power1_cap" in c else -1.0,
            clock_mhz=_num(c.get("freq1_input"), -1) / 1e6 if "freq1_input" in c else -1.0,
            mem_clock_mhz=_num(c.get("freq2_input"), -1) / 1e6 if "freq2_input" in c else -1.0,
            pci_bus_id=c.get("slot", ""),
            pcie_link=(f"{c['current_link_speed']} x{c['current_link_width']}"
                       if c.get("current_link_speed") and c.get("current_link_width") else ""),
            integrated=total > 0 and total <= 1024,
        )
        gpu.notes.append(c.get("device", ""))   # PCI device id, for the lspci name lookup
        gpus.append(gpu)
    return gpus, driver


def parse_lspci(text: str) -> Dict[str, str]:
    """Map PCI slot ("0000:0c:00.0") and device id ("13c0") -> product name."""
    names: Dict[str, str] = {}
    for line in (text or "").splitlines():
        m = re.match(r"^(\S+)\s+[^:]+:\s*(.*)\s+\[([0-9a-f]{4}):([0-9a-f]{4})\]", line.strip(), re.I)
        if not m:
            continue
        slot, name, _vendor, device = m.groups()
        name = re.sub(r"^(Advanced Micro Devices, Inc\. \[AMD/ATI\]|NVIDIA Corporation|Intel Corporation)\s*", "", name)
        name = re.sub(r"\s*\(rev [^)]*\)\s*$", "", name).strip()
        names[slot] = name
        names[device.lower()] = name
    return names


def _apply_lspci_names(gpus: List[GPUInfo], names: Dict[str, str]) -> None:
    for g in gpus:
        if g.device_name:
            continue
        device_id = (g.notes[0] if g.notes else "").lower().replace("0x", "")
        g.device_name = names.get(g.pci_bus_id) or names.get(device_id) or f"AMD GPU {device_id or g.index}"
    for g in gpus:
        g.notes = [n for n in g.notes if not re.fullmatch(r"0x[0-9a-f]+", n or "", re.I)]


def parse_amd_smi(static_text: str, metric_text: str) -> List[GPUInfo]:
    """Parse ``amd-smi static --json`` and ``amd-smi metric --json``."""
    def load(text):
        try:
            data = json.loads(text) if text and text.strip() else []
        except ValueError:
            return {}
        if isinstance(data, dict):
            data = [data]
        return {int(_num(d.get("gpu"), i)): d for i, d in enumerate(data) if isinstance(d, dict)}

    statics, metrics = load(static_text), load(metric_text)
    gpus = []
    for idx in sorted(set(statics) | set(metrics)):
        st, me = statics.get(idx, {}), metrics.get(idx, {})
        asic, vram, driver = st.get("asic", {}) or {}, st.get("vram", {}) or {}, st.get("driver", {}) or {}
        mem = me.get("mem_usage", {}) or {}
        total = _num((vram.get("size") if isinstance(vram, dict) else None), 0)
        if total <= 0:
            total = _num(mem.get("total_vram"), 0)
        used = _num(mem.get("used_vram"), -1)
        temp = me.get("temperature", {}) or {}
        power = me.get("power", {}) or {}
        clocks = me.get("clock", {}) or {}
        gfx_clock = clocks.get("gfx_0", {}) if isinstance(clocks, dict) else {}
        mem_clock = clocks.get("mem_0", {}) if isinstance(clocks, dict) else {}
        limit = (st.get("limit", {}) or {})
        gpu = GPUInfo(
            backend="hip", vendor="amd", index=idx,
            vram_mb=int(total), available_vram_mb=int(max(total - used, 0)) if used >= 0 else int(total),
            device_name=asic.get("market_name") or asic.get("product_name") or f"AMD GPU {idx}",
            driver_version=str(driver.get("version") or "") or None,
            compute_capability=str(asic.get("target_graphics_version") or ""),
            used_vram_mb=used, utilization_pct=_num((me.get("usage", {}) or {}).get("gfx_activity")),
            temperature_c=_num(temp.get("edge", temp.get("hotspot"))),
            power_w=_num(power.get("socket_power", power.get("average_socket_power"))),
            power_limit_w=_num(limit.get("max_power", limit.get("current_power"))),
            clock_mhz=_num(gfx_clock.get("clk") if isinstance(gfx_clock, dict) else None),
            mem_clock_mhz=_num(mem_clock.get("clk") if isinstance(mem_clock, dict) else None),
            pci_bus_id=str((st.get("bus", {}) or {}).get("bdf", "") or ""),
        )
        gpus.append(gpu)
    return gpus


def parse_rocm_smi(text: str) -> Tuple[List[GPUInfo], str]:
    """Parse ``rocm-smi ... --json``; keys vary between ROCm releases, so match loosely."""
    try:
        data = json.loads(text) if text and text.strip() else {}
    except ValueError:
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    driver = ""
    system = data.get("system", {}) or {}
    for k, v in system.items():
        if "driver" in k.lower():
            driver = str(v)
    gpus = []
    for key in sorted(k for k in data if k.lower().startswith("card")):
        card = data[key]
        if not isinstance(card, dict):
            continue
        fields = {k.lower(): v for k, v in card.items()}

        def find(*needles, exclude=()):
            for k, v in fields.items():
                if all(n in k for n in needles) and not any(x in k for x in exclude):
                    return v
            return None

        total = _num(find("vram total memory"), 0) / (1024 * 1024)
        used_raw = find("vram total used")
        used = _num(used_raw, -1) / (1024 * 1024) if used_raw is not None else -1.0
        temp = find("temperature", "edge")
        if temp is None:
            temp = find("temperature", "junction")
        gpu = GPUInfo(
            backend="hip", vendor="amd", index=int(_num(key, len(gpus))),
            vram_mb=int(total), available_vram_mb=int(max(total - used, 0)) if used >= 0 else int(total),
            device_name=str(find("card series") or find("device name") or find("card model") or f"AMD GPU {len(gpus)}"),
            driver_version=driver or None,
            compute_capability=str(find("gfx version") or ""),
            used_vram_mb=used, utilization_pct=_num(find("gpu use")),
            temperature_c=_num(temp),
            power_w=_num(find("power", "(w)")),
            clock_mhz=_num(find("sclk")), mem_clock_mhz=_num(find("mclk")),
            pci_bus_id=str(find("pci bus") or ""),
        )
        gpus.append(gpu)
    return gpus, driver


def parse_fdinfo(text: str) -> Dict[int, float]:
    """pid -> VRAM MB from amdgpu DRM fdinfo (one entry per DRM client, not per fd)."""
    per_pid: Dict[int, Dict[str, float]] = {}
    pid = None
    client = None
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("pid="):
            pid = int(_num(line, 0)); client = None
            per_pid.setdefault(pid, {})
        elif line.startswith("drm-client-id:"):
            client = line.split(":", 1)[1].strip()
        elif line.startswith("drm-memory-vram:") and pid is not None:
            value = line.split(":", 1)[1].strip().lower()
            amount = _num(value, 0)
            if "kib" in value or value.endswith("k"):
                mb = amount / 1024
            elif "mib" in value:
                mb = amount
            elif "gib" in value:
                mb = amount * 1024
            else:
                mb = amount / (1024 * 1024)
            per_pid[pid][client or "?"] = mb
    return {p: sum(v.values()) for p, v in per_pid.items()}


def parse_vulkan(text: str) -> List[str]:
    names = []
    for line in (text or "").splitlines():
        if "deviceName" in line and "=" in line:
            name = line.split("=", 1)[1].strip()
            if name and name not in names:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _aggregate(cards: List[GPUInfo], backend: str) -> GPUInfo:
    first = cards[0]
    agg = GPUInfo(
        backend=backend, vendor=first.vendor,
        vram_mb=sum(c.vram_mb for c in cards),
        available_vram_mb=sum(c.available_vram_mb for c in cards),
        device_name=first.device_name, driver_version=first.driver_version,
        supported=True, gpu_count=len(cards), gpu_infos=cards,
        compute_capability=first.compute_capability,
        used_vram_mb=sum(c.used_vram_mb for c in cards if c.used_vram_mb >= 0)
        if any(c.used_vram_mb >= 0 for c in cards) else -1.0,
        utilization_pct=max((c.utilization_pct for c in cards), default=-1.0),
        temperature_c=max((c.temperature_c for c in cards), default=-1.0),
        power_w=sum(c.power_w for c in cards if c.power_w >= 0)
        if any(c.power_w >= 0 for c in cards) else -1.0,
        power_limit_w=sum(c.power_limit_w for c in cards if c.power_limit_w >= 0)
        if any(c.power_limit_w >= 0 for c in cards) else -1.0,
    )
    for c in cards:
        c.backend = backend
    return agg


def parse_gpu_probe(output: str, stderr: str = "") -> Optional[GPUInfo]:
    """Turn probe output (live or full) into one GPUInfo, or None if no GPU.

    NVIDIA cards win when present (llama.cpp's CUDA build only sees those);
    other GPUs are then listed in ``notes`` so the user knows they are ignored.
    """
    sec = split_sections(output)
    nv_text = sec.get("nv", "")
    nvidia = parse_nvidia_rows(nv_text)
    if not nvidia and nv_text.strip():
        # nvidia-smi printed something that is not a table: surface it.
        detail = nv_text.strip().splitlines()[0]
        if "not found" not in detail.lower() and "nvidia-smi" not in detail.lower()[:12]:
            info = GPUInfo(backend="cpu", vram_mb=0, available_vram_mb=0,
                           device_name=detail[:140], supported=False)
            info.notes.append(detail)
            return info

    for line in sec.get("nvcc", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            for g in nvidia:
                if str(g.index) == parts[0] and re.fullmatch(r"\d+(\.\d+)?", parts[1]):
                    g.compute_capability = parts[1]

    amd = parse_amd_smi(sec.get("amdsmi_static", ""), sec.get("amdsmi_metric", ""))
    rocm_driver = ""
    if not amd:
        amd, rocm_driver = parse_rocm_smi(sec.get("rocm", ""))
    sysfs_cards, amdgpu_version = parse_sysfs(sec.get("sysfs", ""))
    names = parse_lspci(sec.get("lspci", ""))
    _apply_lspci_names(sysfs_cards, names)
    if amd:
        # SMI tools know names/gfx; sysfs knows integrated-ness and link speed.
        for g, s in zip(amd, sysfs_cards):
            g.integrated = s.integrated
            g.pcie_link = g.pcie_link or s.pcie_link
            if g.used_vram_mb < 0:
                g.used_vram_mb, g.available_vram_mb = s.used_vram_mb, s.available_vram_mb
        for g in amd:
            g.driver_version = g.driver_version or rocm_driver or amdgpu_version or None
    else:
        amd = sysfs_cards
        for g in amd:
            g.driver_version = amdgpu_version or None
    rocm_present = sec.get("rocm_present", "").strip() == "yes" or bool(
        parse_amd_smi(sec.get("amdsmi_static", ""), "") or parse_rocm_smi(sec.get("rocm", ""))[0])

    if nvidia:
        info = _aggregate(nvidia, "cuda")
        info.processes = _parse_nv_processes(sec.get("nvproc", ""), nvidia)
        for g in amd:
            info.notes.append(
                f"Also present, not used by the CUDA backend: {g.device_name} "
                f"({g.vram_mb} MB{', integrated' if g.integrated else ''}).")
        info.rocm_available = rocm_present
        return info

    if amd:
        backend = "hip" if rocm_present else "vulkan"
        info = _aggregate(amd, backend)
        info.rocm_available = rocm_present
        own = parse_fdinfo(sec.get("fdinfo", ""))
        info.processes = [GpuProcess(pid=p, name="", vram_mb=mb) for p, mb in own.items()]
        if not rocm_present:
            info.notes.append("No ROCm tooling found: use llama.cpp's Vulkan build, or install "
                              "ROCm for the HIP build.")
        return info

    vulkan_names = parse_vulkan(sec.get("vulkan", ""))
    vulkan_names = [n for n in vulkan_names if "llvmpipe" not in n.lower()]
    if vulkan_names:
        return GPUInfo(backend="vulkan", vram_mb=0, available_vram_mb=0,
                       device_name=vulkan_names[0], gpu_count=len(vulkan_names),
                       notes=["VRAM size is not reported by vulkaninfo."])
    return None


def merge_live(profile: GPUInfo, live: Optional[GPUInfo]) -> GPUInfo:
    """Overlay a live reading on a cached profile (names/driver from the profile)."""
    if live is None:
        return profile
    by_index = {g.index: g for g in profile.gpu_infos} if profile.gpu_infos else {}
    for g in live.gpu_infos:
        base = by_index.get(g.index)
        if base is not None:
            g.device_name = g.device_name or base.device_name
            g.driver_version = g.driver_version or base.driver_version
            g.compute_capability = g.compute_capability or base.compute_capability
            g.integrated = g.integrated or base.integrated
            g.pcie_link = g.pcie_link or base.pcie_link
    live.device_name = live.device_name or profile.device_name
    live.driver_version = live.driver_version or profile.driver_version
    live.compute_capability = live.compute_capability or profile.compute_capability
    live.rocm_available = live.rocm_available or profile.rocm_available
    live.notes = live.notes or profile.notes
    return live
