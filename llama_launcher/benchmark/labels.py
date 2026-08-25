"""Descriptive session labels and detail text for the Benchmark tab.

Session dicts persisted to disk stay small (flat scalars + a couple of
whitelisted sub-dicts); the richer detail text is built on demand from
that stored data for tooltips/detail popups, never persisted itself.
"""
from llama_launcher.models.scanner import infer_model_hints

_CONFIG_SNAPSHOT_FIELDS = (
    "model_path", "context_size", "gpu_layers", "device", "n_cpu_moe",
    "cache_type_k", "cache_type_v", "batch_size", "ubatch_size",
    "flash_attn", "threads", "port", "host",
)


def config_snapshot_fields(config) -> dict:
    """Whitelisted subset of a LlamaConfig worth persisting per session."""
    return {f: getattr(config, f, None) for f in _CONFIG_SNAPSHOT_FIELDS}


def estimate_summary(estimate) -> dict:
    """Compact summary of a ResourceEstimate, or {} if unavailable."""
    if estimate is None:
        return {}
    return {
        "total_vram_mb": estimate.total_vram_mb,
        "total_ram_mb": estimate.total_ram_mb,
        "arch_name": estimate.arch_name,
        "total_params_display": estimate.total_params_display,
    }


def _format_context(context_size) -> str:
    """Compact context-size label, e.g. 8192 -> "8K ctx"."""
    if not context_size:
        return ""
    if context_size % 1024 == 0:
        return f"{context_size // 1024}K ctx"
    return f"{context_size} ctx"


def build_session_label(model_path: str, preset: str, kind: str, source: str,
                        context_size=None) -> str:
    """Compact, table-friendly label: "<model family/quant> · <context> · <tail>".

    Context size is surfaced directly in the label (not just the tooltip)
    because it materially changes throughput numbers -- comparing two
    sessions at a glance without it is misleading.
    """
    family, quant = infer_model_hints(model_path) if model_path else ("unknown", None)
    model_part = f"{family} {quant}".strip() if family != "unknown" else ""
    ctx_part = _format_context(context_size)

    if kind == "benchmark":
        tail = "Benchmark"
    elif kind == "remote_import":
        tail = f"remote:{source}" if source else "remote"
    elif preset:
        tail = preset
    else:
        tail = "Local"

    label = " · ".join(p for p in (model_part, ctx_part, tail) if p)
    return label or (source or "Session")


def build_session_tooltip(session: dict) -> str:
    """Multi-line detail text for a hover-tooltip / double-click popup.

    Every field is read defensively since older sessions on disk may
    predate any given key.
    """
    lines = [f"Model: {session.get('model_path') or 'unknown'}"]

    family = session.get("model_family")
    quant = session.get("model_quant")
    if family:
        lines.append(f"Family/Quant: {family} {quant or ''}".rstrip())

    preset = session.get("preset")
    if preset:
        try:
            from llama_launcher.core.presets import PresetManager
            desc = PresetManager.get_preset(preset).description
            lines.append(f"Preset: {preset}" + (f" — {desc}" if desc else ""))
        except ValueError:
            lines.append(f"Preset: {preset} (no longer exists)")

    cfg = session.get("config_snapshot") or {}
    if cfg:
        lines.append(
            f"Context: {cfg.get('context_size')}  GPU layers: {cfg.get('gpu_layers')}  "
            f"Device: {cfg.get('device') or 'auto'}  n_cpu_moe: {cfg.get('n_cpu_moe')}")
        lines.append(
            f"Cache K/V: {cfg.get('cache_type_k')}/{cfg.get('cache_type_v')}  "
            f"Batch: {cfg.get('batch_size')}/{cfg.get('ubatch_size')}  "
            f"Flash-attn: {cfg.get('flash_attn')}  Threads: {cfg.get('threads')}")
        lines.append(f"Endpoint: {cfg.get('host')}:{cfg.get('port')}")

    est = session.get("estimate") or {}
    if est:
        lines.append(
            f"Estimated VRAM/RAM: {est.get('total_vram_mb', 0):.0f} MB / "
            f"{est.get('total_ram_mb', 0):.0f} MB   Arch: {est.get('arch_name')}  "
            f"Params: {est.get('total_params_display')}")

    if session.get("ttft_ms_p50"):
        lines.append(
            f"TTFT: avg {session.get('ttft_ms_avg', 0):.0f} ms, "
            f"p50 {session.get('ttft_ms_p50', 0):.0f} ms, "
            f"p95 {session.get('ttft_ms_p95', 0):.0f} ms")
        lines.append(
            f"Gen speed: p50 {session.get('gen_tps_p50', 0):.1f} t/s, "
            f"p95 {session.get('gen_tps_p95', 0):.1f} t/s")

    res = session.get("resource") or {}
    if res:
        lines.append(
            f"GPU util: avg {res.get('gpu_percent_avg', 0):.0f}%, "
            f"max {res.get('gpu_percent_max', 0):.0f}%   "
            f"VRAM: avg {res.get('vram_used_mb_avg', 0):.0f} MB, "
            f"max {res.get('vram_used_mb_max', 0):.0f} MB")

    bp = session.get("benchmark_params")
    if bp:
        lines.append(
            f"Benchmark: n_predict={bp.get('n_predict')}, "
            f"iterations={bp.get('iterations')}, "
            f"warmup={'yes' if bp.get('warmup') else 'no'}")

    return "\n".join(lines)
