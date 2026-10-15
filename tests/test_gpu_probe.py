"""Tests for the vendor-neutral GPU probe parser and the backend advice.

No AMD card is needed: the parser is fed the documented output of amd-smi,
rocm-smi and the amdgpu sysfs files (the latter captured from a real Ryzen
iGPU), plus nvidia-smi rows.
"""
import json

import pytest

from llama_launcher.gpu.advice import cuda_arch_flag, suggest_backend
from llama_launcher.gpu.detector import vram_sample_from_gpu
from llama_launcher.gpu.probe import (
    build_full_probe, build_live_probe, merge_live, parse_fdinfo, parse_gpu_probe,
    parse_lspci, parse_rocm_smi, parse_sysfs, split_sections,
)
from llama_launcher.gpu.types import GPUInfo, GpuProcess, gpu_from_dict, gpu_to_dict
from llama_launcher.remote import procinfo
from llama_launcher.remote.types import ServerConfig, environment_prefix, parse_environment


def sections(**kw) -> str:
    return "".join(f"@@GPU:{k}\n{v}\n" for k, v in kw.items())


NV_ROW = ("0, NVIDIA GeForce RTX 5090, 32607, 2824, 29783, 595.91.07, 56, 80.78, 600.00, 5, 2610, 405, "
          "00000000:01:00.0")
NV_PROCS = "00000000:01:00.0, 8713, /usr/libexec/gnome-remote-desktop-daemon, 504\n00000000:01:00.0, 4242, llama-server, 20000"

SYSFS_IGPU = """card=card2
slot=0000:0c:00.0
vendor=0x1002
device=0x13c0
mem_info_vram_total=536870912
mem_info_vram_used=16883712
gpu_busy_percent=7
current_link_speed=16.0 GT/s PCIe
current_link_width=16
temp1_input=44000
power1_input=22000
freq1_input=600000000
amdgpu_version=7.0.0-34-generic"""

SYSFS_DGPU = """card=card1
slot=0000:03:00.0
vendor=0x1002
device=0x744c
mem_info_vram_total=25753026560
mem_info_vram_used=1073741824
gpu_busy_percent=42
temp1_input=61000
power1_average=180000000
power1_cap=355000000
freq1_input=2400000000
freq2_input=1250000000
amdgpu_version=6.8.5"""

LSPCI = ("0000:0c:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
         "Granite Ridge [Radeon Graphics] [1002:13c0] (rev ca)\n"
         "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
         "Navi 31 [Radeon RX 7900 XTX] [1002:744c] (rev c8)")

AMD_SMI_STATIC = json.dumps([{
    "gpu": 0,
    "asic": {"market_name": "Radeon RX 7900 XTX", "vendor_id": "0x1002", "device_id": "0x744c",
             "target_graphics_version": "gfx1100"},
    "bus": {"bdf": "0000:03:00.0"},
    "vram": {"type": "GDDR6", "size": {"value": 24560, "unit": "MB"}},
    "driver": {"name": "amdgpu", "version": "6.8.5"},
    "limit": {"max_power": {"value": 355, "unit": "W"}},
}])
AMD_SMI_METRIC = json.dumps([{
    "gpu": 0,
    "usage": {"gfx_activity": {"value": 42, "unit": "%"}},
    "power": {"socket_power": {"value": 180, "unit": "W"}},
    "clock": {"gfx_0": {"clk": {"value": 2400, "unit": "MHz"}}, "mem_0": {"clk": {"value": 1250, "unit": "MHz"}}},
    "temperature": {"edge": {"value": 61, "unit": "C"}, "hotspot": {"value": 75, "unit": "C"}},
    "mem_usage": {"total_vram": {"value": 24560, "unit": "MB"}, "used_vram": {"value": 1024, "unit": "MB"}},
}])
ROCM_SMI = json.dumps({
    "card0": {
        "Temperature (Sensor edge) (C)": "61.0", "Temperature (Sensor junction) (C)": "75.0",
        "Average Graphics Package Power (W)": "180.0", "GPU use (%)": "42",
        "VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "1073741824",
        "Card Series": "Navi 31 [Radeon RX 7900 XTX]", "Card Model": "0x744c",
        "GFX Version": "gfx1100", "sclk clock speed:": "(2400Mhz)", "mclk clock speed:": "(1250Mhz)",
    },
    "system": {"Driver version": "6.8.5"},
})


class TestNvidia:
    def test_live_row_has_every_field(self):
        gpu = parse_gpu_probe(sections(nv=NV_ROW, nvproc=NV_PROCS, nvcc="0, 12.0"))
        assert gpu.backend == "cuda" and gpu.vendor == "nvidia"
        assert gpu.device_name == "NVIDIA GeForce RTX 5090"
        assert gpu.vram_mb == 32607 and gpu.available_vram_mb == 29783 and gpu.used_vram_mb == 2824
        card = gpu.gpu_infos[0]
        assert card.temperature_c == 56 and card.power_w == pytest.approx(80.78)
        assert card.power_limit_w == 600 and card.utilization_pct == 5
        assert card.clock_mhz == 2610 and card.mem_clock_mhz == 405
        assert card.compute_capability == "12.0" and gpu.compute_capability == "12.0"
        assert card.pci_bus_id == "00000000:01:00.0"
        assert [(p.pid, p.name, p.vram_mb) for p in gpu.processes] == [
            (8713, "gnome-remote-desktop-daemon", 504.0), (4242, "llama-server", 20000.0)]

    def test_legacy_four_column_rows_still_parse(self):
        gpu = parse_gpu_probe("NVIDIA A100, 40000 MiB, 30000 MiB, 550.1\nNVIDIA A100, 40000 MiB, 10000 MiB, 550.1")
        assert gpu.gpu_count == 2 and gpu.vram_mb == 80000 and gpu.available_vram_mb == 40000

    def test_nvml_error_is_surfaced(self):
        gpu = parse_gpu_probe("Failed to initialize NVML: Driver/library version mismatch")
        assert gpu.supported is False and "NVML" in gpu.device_name

    def test_nothing_found(self):
        assert parse_gpu_probe("") is None
        assert parse_gpu_probe(sections(nv="", sysfs="", lspci="")) is None

    def test_cuda_wins_over_an_integrated_amd_gpu(self):
        gpu = parse_gpu_probe(sections(nv=NV_ROW, sysfs=SYSFS_IGPU, lspci=LSPCI))
        assert gpu.backend == "cuda" and gpu.gpu_count == 1
        assert any("Granite Ridge" in n and "integrated" in n for n in gpu.notes)


class TestAmd:
    def test_sysfs_only_without_rocm_means_vulkan(self):
        gpu = parse_gpu_probe(sections(nv="", sysfs=SYSFS_IGPU, lspci=LSPCI, rocm_present=""))
        assert gpu.backend == "vulkan" and gpu.vendor == "amd" and not gpu.rocm_available
        card = gpu.gpu_infos[0]
        assert card.device_name == "Granite Ridge [Radeon Graphics]"   # from lspci
        assert card.vram_mb == 512 and card.used_vram_mb == pytest.approx(16.1, abs=0.1)
        assert card.integrated is True
        assert card.temperature_c == 44.0 and card.utilization_pct == 7
        assert card.power_w == pytest.approx(0.022)
        assert card.clock_mhz == 600 and card.pcie_link == "16.0 GT/s PCIe x16"
        assert card.driver_version == "7.0.0-34-generic"
        assert any("Vulkan" in n for n in gpu.notes)

    def test_sysfs_with_rocm_present_means_hip(self):
        gpu = parse_gpu_probe(sections(nv="", sysfs=SYSFS_DGPU, lspci=LSPCI, rocm_present="yes"))
        assert gpu.backend == "hip" and gpu.rocm_available
        card = gpu.gpu_infos[0]
        assert card.device_name == "Navi 31 [Radeon RX 7900 XTX]"
        assert card.vram_mb == 24560 and card.integrated is False
        assert card.power_w == 180 and card.power_limit_w == 355
        assert card.clock_mhz == 2400 and card.mem_clock_mhz == 1250

    def test_amd_smi_json(self):
        gpu = parse_gpu_probe(sections(nv="", amdsmi_static=AMD_SMI_STATIC, amdsmi_metric=AMD_SMI_METRIC,
                                       sysfs=SYSFS_DGPU, lspci=LSPCI, rocm_present="yes"))
        assert gpu.backend == "hip"
        card = gpu.gpu_infos[0]
        assert card.device_name == "Radeon RX 7900 XTX"
        assert card.compute_capability == "gfx1100" and gpu.compute_capability == "gfx1100"
        assert card.vram_mb == 24560 and card.used_vram_mb == 1024 and card.available_vram_mb == 23536
        assert card.temperature_c == 61 and card.power_w == 180 and card.power_limit_w == 355
        assert card.utilization_pct == 42 and card.clock_mhz == 2400 and card.mem_clock_mhz == 1250
        assert card.driver_version == "6.8.5" and card.pci_bus_id == "0000:03:00.0"
        assert card.integrated is False

    def test_rocm_smi_json(self):
        cards, driver = parse_rocm_smi(ROCM_SMI)
        assert driver == "6.8.5"
        card = cards[0]
        assert card.device_name == "Navi 31 [Radeon RX 7900 XTX]"
        assert card.compute_capability == "gfx1100"
        assert card.vram_mb == 24560 and card.used_vram_mb == 1024
        assert card.temperature_c == 61.0 and card.power_w == 180.0 and card.utilization_pct == 42
        assert card.clock_mhz == 2400 and card.mem_clock_mhz == 1250
        gpu = parse_gpu_probe(sections(nv="", rocm=ROCM_SMI, sysfs=SYSFS_DGPU, lspci=LSPCI))
        assert gpu.backend == "hip" and gpu.rocm_available   # rocm-smi answering proves ROCm

    def test_garbage_json_is_ignored(self):
        assert parse_rocm_smi("not json") == ([], "")
        gpu = parse_gpu_probe(sections(nv="", amdsmi_static="{bad", amdsmi_metric="", sysfs=SYSFS_DGPU, lspci=LSPCI))
        assert gpu.vendor == "amd" and gpu.gpu_infos[0].vram_mb == 24560

    def test_sysfs_skips_nvidia_cards(self):
        cards, _ = parse_sysfs("card=card1\nslot=0000:01:00.0\nvendor=0x10de\ndevice=0x2b85\n" + SYSFS_IGPU)
        assert len(cards) == 1 and cards[0].vram_mb == 512

    def test_fdinfo_sums_one_entry_per_drm_client(self):
        text = ("pid=4242\ndrm-client-id:\t7\ndrm-memory-vram:\t1048576 KiB\n"
                "drm-client-id:\t7\ndrm-memory-vram:\t1048576 KiB\n"       # same client, second fd
                "drm-client-id:\t9\ndrm-memory-vram:\t512 MiB\n"
                "pid=1\ndrm-client-id:\t3\ndrm-memory-vram:\t0 KiB\n")
        assert parse_fdinfo(text) == {4242: 1024 + 512, 1: 0}
        gpu = parse_gpu_probe(sections(nv="", sysfs=SYSFS_DGPU, lspci=LSPCI, fdinfo=text))
        assert {p.pid: p.vram_mb for p in gpu.processes} == {4242: 1536.0, 1: 0.0}
        sample = vram_sample_from_gpu(gpu, own_pids=[4242])
        assert sample.valid and sample.own_used_mb == 1536 and sample.total_mb == 24560

    def test_lspci_names(self):
        names = parse_lspci(LSPCI)
        assert names["0000:0c:00.0"] == "Granite Ridge [Radeon Graphics]"
        assert names["744c"] == "Navi 31 [Radeon RX 7900 XTX]"


class TestProbeScripts:
    def test_live_probe_only_runs_cheap_tools(self):
        script = build_live_probe([12, 34])
        assert "nvidia-smi" in script and "/sys/class/drm" in script
        assert "for p in 12 34;" in script and "fdinfo" in script
        assert "amd-smi" not in script and "vulkaninfo" not in script

    def test_full_probe_covers_every_tool(self):
        script = build_full_probe()
        for tool in ("nvidia-smi", "amd-smi", "rocm-smi", "lspci", "vulkaninfo", "/opt/rocm"):
            assert tool in script

    def test_split_sections_without_markers_is_nvidia_text(self):
        assert split_sections("a, b, c") == {"nv": "a, b, c"}

    def test_usage_probe_carries_full_gpu_reading(self):
        probe = ("@@LL@@\ncpu  100 0 100 800 0 0 0 0\n@@LL@@\ncpu  120 0 120 860 0 0 0 0\n"
                 "@@LL@@\nMemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n@@LL@@\n"
                 + sections(nv=NV_ROW, nvproc=NV_PROCS, sysfs=""))
        sample = procinfo.parse_usage_probe(probe, own_pids=[4242])
        assert sample.cpu_percent == pytest.approx(40.0, abs=0.1)
        assert sample.gpu_percent == 5 and sample.gpu_mem_used_mb == 2824 and sample.gpu_mem_total_mb == 32607
        assert sample.own_vram_mb == 20000
        assert sample.gpu.gpu_infos[0].temperature_c == 56

    def test_status_probe_gpu_section(self):
        ps, ss, gpu = procinfo.split_status_probe(
            "@@LL@@\n 1.0 100 5\n@@LL@@\nLISTEN pid=7\n@@LL@@\n" + sections(nv=NV_ROW, nvproc="", sysfs=""))
        assert procinfo.parse_gpu_stats(gpu) == (5.0, 2824.0)


class TestMergeAndCache:
    def test_live_reading_keeps_profile_names(self):
        profile = parse_gpu_probe(sections(nv=NV_ROW, nvcc="0, 12.0", sysfs=SYSFS_IGPU, lspci=LSPCI))
        live = parse_gpu_probe(sections(nv=NV_ROW.replace("2824", "20000"), nvproc=""))
        merged = merge_live(profile, live)
        assert merged.used_vram_mb == 20000
        assert merged.gpu_infos[0].compute_capability == "12.0"
        assert merged.notes == profile.notes

    def test_round_trip_through_dict(self):
        gpu = parse_gpu_probe(sections(nv=NV_ROW, nvproc=NV_PROCS, nvcc="0, 12.0"))
        back = gpu_from_dict(gpu_to_dict(gpu))
        assert back.gpu_infos[0].temperature_c == 56 and back.compute_capability == "12.0"
        assert [p.pid for p in back.processes] == [8713, 4242]
        # Older cache files without the new keys still load.
        assert gpu_from_dict({"backend": "cuda", "vram_mb": 1, "gpu_infos": [{"backend": "cuda"}]}).vram_mb == 1


class TestAdvice:
    def test_nvidia(self):
        title, lines = suggest_backend(GPUInfo(backend="cuda", vendor="nvidia", compute_capability="12.0"))
        assert title == "CUDA" and any("CMAKE_CUDA_ARCHITECTURES=120" in l for l in lines)
        assert cuda_arch_flag("8.6") == "86"

    def test_amd_with_rocm_and_unsupported_target(self):
        gpu = GPUInfo(backend="hip", vendor="amd", rocm_available=True,
                      gpu_infos=[GPUInfo(backend="hip", vendor="amd", compute_capability="gfx1103")])
        title, lines = suggest_backend(gpu)
        assert title == "HIP (ROCm)"
        assert any("GGML_HIP=ON" in l and "gfx1103" in l for l in lines)
        assert any("HSA_OVERRIDE_GFX_VERSION=11.0.0" in l for l in lines)

    def test_amd_without_rocm(self):
        title, lines = suggest_backend(GPUInfo(backend="vulkan", vendor="amd"))
        assert title == "Vulkan" and any("GGML_VULKAN=ON" in l for l in lines)

    def test_cpu(self):
        assert suggest_backend(GPUInfo(backend="cpu"))[0] == "CPU"


class TestServerEnvironment:
    def test_prefix_quotes_values(self):
        cfg = ServerConfig(environment="HSA_OVERRIDE_GFX_VERSION=11.0.0\n# comment\nexport X=a b\n")
        assert environment_prefix(cfg) == "env HSA_OVERRIDE_GFX_VERSION=11.0.0 'X=a b' "
        assert environment_prefix(ServerConfig()) == ""

    def test_malformed_lines_rejected_by_parser_but_skipped_by_prefix(self):
        with pytest.raises(ValueError):
            parse_environment("1BAD=x")
        assert environment_prefix(ServerConfig(environment="nope\nOK=1")) == "env OK=1 "

    def test_environment_is_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        from llama_launcher.remote.storage import load_servers, save_server
        save_server(ServerConfig(friendly_name="amd", environment="HIP_VISIBLE_DEVICES=0"))
        assert load_servers()[0].environment == "HIP_VISIBLE_DEVICES=0"
