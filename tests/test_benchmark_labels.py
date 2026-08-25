"""Tests for benchmark session labels and detail tooltips."""
from llama_launcher.benchmark.labels import (
    build_session_label,
    build_session_tooltip,
    config_snapshot_fields,
    estimate_summary,
)
from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.presets import PresetManager


def test_label_live_with_preset():
    label = build_session_label(
        "/models/qwen2.5-7b-instruct-q4_k_m.gguf", "Medium 12GB",
        kind="live", source="local")
    assert label == "qwen Q4_K_M · Medium 12GB"


def test_label_live_without_preset():
    label = build_session_label(
        "/models/llama-3-8b-q5_k_m.gguf", "", kind="live", source="local")
    assert label == "llama Q5_K_M · Local"


def test_label_remote_import():
    label = build_session_label(
        "/models/mistral-7b-q4_0.gguf", "", kind="remote_import", source="phoenix")
    assert label == "mistral Q4_0 · remote:phoenix"


def test_label_benchmark_kind():
    label = build_session_label(
        "/models/gemma-2-9b-q8_0.gguf", "MyPreset", kind="benchmark", source="local")
    assert label == "gemma Q8_0 · Benchmark"


def test_label_unknown_model_falls_back_to_source():
    label = build_session_label("", "", kind="remote_import", source="phoenix")
    assert label == "remote:phoenix"


def test_label_completely_unknown_falls_back_to_session():
    label = build_session_label("", "", kind="live", source="")
    assert label == "Local"


def test_label_includes_context_size_when_provided():
    label = build_session_label(
        "/models/qwen2.5-7b-instruct-q4_k_m.gguf", "Medium 12GB",
        kind="live", source="local", context_size=8192)
    assert label == "qwen Q4_K_M · 8K ctx · Medium 12GB"


def test_label_non_power_of_1024_context_shown_verbatim():
    label = build_session_label(
        "/models/llama-3-8b-q5_k_m.gguf", "", kind="live", source="local",
        context_size=6000)
    assert label == "llama Q5_K_M · 6000 ctx · Local"


def test_label_omits_context_when_absent():
    label = build_session_label(
        "/models/llama-3-8b-q5_k_m.gguf", "", kind="live", source="local",
        context_size=None)
    assert label == "llama Q5_K_M · Local"


def test_config_snapshot_fields_whitelisted_subset():
    cfg = LlamaConfig(model_path="/m.gguf", context_size=8192, gpu_layers=33,
                      threads=8, port=8080, host="127.0.0.1")
    snap = config_snapshot_fields(cfg)
    assert snap["model_path"] == "/m.gguf"
    assert snap["context_size"] == 8192
    assert snap["gpu_layers"] == 33
    assert "temp" not in snap  # not in the whitelist


def test_estimate_summary_none_returns_empty_dict():
    assert estimate_summary(None) == {}


def test_tooltip_minimal_session_does_not_crash():
    tooltip = build_session_tooltip({})
    assert "Model: unknown" in tooltip


def test_tooltip_includes_config_and_estimate():
    session = {
        "model_path": "/m.gguf",
        "model_family": "llama",
        "model_quant": "Q4_K_M",
        "config_snapshot": {
            "context_size": 8192, "gpu_layers": 33, "device": "",
            "n_cpu_moe": 0, "cache_type_k": "f16", "cache_type_v": "f16",
            "batch_size": 2048, "ubatch_size": 512, "flash_attn": "auto",
            "threads": 8, "port": 8080, "host": "127.0.0.1",
        },
        "estimate": {
            "total_vram_mb": 6000.0, "total_ram_mb": 2000.0,
            "arch_name": "llama", "total_params_display": "7.6B",
        },
    }
    tooltip = build_session_tooltip(session)
    assert "Context: 8192" in tooltip
    assert "Estimated VRAM/RAM: 6000 MB / 2000 MB" in tooltip


def test_tooltip_reports_missing_preset_gracefully():
    session = {"preset": "definitely-not-a-real-preset-xyz"}
    tooltip = build_session_tooltip(session)
    assert "no longer exists" in tooltip


def test_tooltip_includes_valid_preset_description():
    session = {"preset": "Small 8GB"}
    tooltip = build_session_tooltip(session)
    expected_desc = PresetManager.get_preset("Small 8GB").description
    assert expected_desc in tooltip
