"""Parse llama-server command lines into LlamaConfig field values.

This module is intentionally Qt-free so it can be shared by the GUI importer,
the CLI, and unit tests. Value coercion is driven by the ``LlamaConfig``
dataclass field types, so new fields only need a flag mapping here.
"""
import dataclasses
import logging
import shlex
from typing import Any, Dict, List, Tuple

from llama_launcher.core.config import LlamaConfig

logger = logging.getLogger(__name__)

_FIELD_TYPES: Dict[str, Any] = {f.name: f.type for f in dataclasses.fields(LlamaConfig)}

#: llama-server flag -> LlamaConfig field name.
FLAG_MAP: Dict[str, str] = {
    # Model paths
    "--model": "model_path",
    "-m": "model_path",
    "--mmproj": "mmproj_model_path",
    "--mmproj-model": "mmproj_model_path",
    "--draft-model": "draft_model_path",
    "--draft": "draft_model_path",
    # Threads and batching
    "--threads": "threads",
    "-t": "threads",
    "--threads-batch": "threads_batch",
    "-tb": "threads_batch",
    "--cpu-mask": "cpu_mask",
    "--cpu-range": "cpu_range",
    "--cpu-strict": "cpu_strict",
    "--prio": "prio",
    "--poll": "poll",
    "--cpu-mask-batch": "cpu_mask_batch",
    "--cpu-range-batch": "cpu_range_batch",
    "--cpu-strict-batch": "cpu_strict_batch",
    "--prio-batch": "prio_batch",
    "--poll-batch": "poll_batch",
    # Context and predict
    "--ctx-size": "context_size",
    "-c": "context_size",
    "--predict": "predict",
    "-n": "predict",
    "--n-predict": "predict",
    "--batch-size": "batch_size",
    "-b": "batch_size",
    "--ubatch-size": "ubatch_size",
    "-ub": "ubatch_size",
    "--keep": "keep",
    "--swa-full": "swa_full",
    "--no-swa-full": "swa_full",
    "--flash-attn": "flash_attn",
    "-fa": "flash_attn",
    "--perf": "perf",
    "--no-perf": "perf",
    "--escape": "escape",
    "--no-escape": "escape",
    # RoPE and YaRN
    "--rope-scaling": "rope_scaling",
    "--rope-scale": "rope_scale",
    "--rope-freq-base": "rope_freq_base",
    "--rope-freq-scale": "rope_freq_scale",
    "--yarn-orig-ctx": "yarn_orig_ctx",
    "--yarn-ext-factor": "yarn_ext_factor",
    "--yarn-attn-factor": "yarn_attn_factor",
    "--yarn-beta-slow": "yarn_beta_slow",
    "--yarn-beta-fast": "yarn_beta_fast",
    # Cache and memory
    "--kv-offload": "kv_offload",
    "--no-kv-offload": "kv_offload",
    "--repack": "repack",
    "--no-repack": "repack",
    "--no-host": "no_host",
    "--cache-type-k": "cache_type_k",
    "-ctk": "cache_type_k",
    "--cache-type-v": "cache_type_v",
    "-ctv": "cache_type_v",
    "--defrag-thold": "defrag_thold",
    "--mlock": "mlock",
    "--no-mlock": "mlock",
    "--mmap": "use_mmap",
    "--no-mmap": "use_mmap",
    "--direct-io": "direct_io",
    "--no-direct-io": "direct_io",
    # GPU and NUMA
    "--numa": "numa",
    "--device": "device",
    "--override-tensor": "override_tensor",
    "--cpu-moe": "cpu_moe",
    "--no-cpu-moe": "cpu_moe",
    "--n-cpu-moe": "n_cpu_moe",
    "--gpu-layers": "gpu_layers",
    "-ngl": "gpu_layers",
    "--n-gpu-layers": "gpu_layers",
    "--split-mode": "split_mode",
    "-sm": "split_mode",
    "--tensor-split": "tensor_split",
    "-ts": "tensor_split",
    "--main-gpu": "main_gpu",
    "-mg": "main_gpu",
    "--fit": "fit",
    "--fit-target": "fit_target",
    "--fit-ctx": "fit_ctx",
    "--check-tensors": "check_tensors",
    "--no-check-tensors": "check_tensors",
    "--override-kv": "override_kv",
    "--op-offload": "op_offload",
    "--no-op-offload": "op_offload",
    # LoRA and control vectors
    "--lora": "lora",
    "--lora-scaled": "lora_scaled",
    "--control-vector": "control_vector",
    "--control-vector-scaled": "control_vector_scaled",
    "--control-vector-layer-range": "control_vector_layer_range",
    # Model sources
    "--model-url": "model_url",
    "--docker-repo": "docker_repo",
    "--hf-repo": "hf_repo",
    "--hf-file": "hf_file",
    "--hf-repo-v": "hf_repo_v",
    "--hf-file-v": "hf_file_v",
    "--hf-token": "hf_token",
    # Logging
    "--log-disable": "log_disable",
    "--log-file": "log_file",
    "--log-colors": "log_colors",
    "--verbose": "verbose",
    "--offline": "offline",
    "--verbosity": "verbosity",
    "--log-prefix": "log_prefix",
    "--no-log-prefix": "log_prefix",
    "--log-timestamps": "log_timestamps",
    "--no-log-timestamps": "log_timestamps",
    # Spec draft cache types
    "--spec-draft-type-k": "spec_draft_type_k",
    "--spec-draft-type-v": "spec_draft_type_v",
    # Sampling
    "--samplers": "samplers",
    "--seed": "seed",
    "--sampler-seq": "sampler_seq",
    "--ignore-eos": "ignore_eos",
    "--no-ignore-eos": "ignore_eos",
    "--temp": "temp",
    "--temperature": "temp",
    "--top-k": "top_k",
    "--top-p": "top_p",
    "--min-p": "min_p",
    "--top-n-sigma": "top_n_sigma",
    "--xtc-probability": "xtc_probability",
    "--xtc-threshold": "xtc_threshold",
    "--typical-p": "typical_p",
    "--repeat-last-n": "repeat_last_n",
    "--repeat-penalty": "repeat_penalty",
    "--presence-penalty": "presence_penalty",
    "--frequency-penalty": "frequency_penalty",
    "--dry-multiplier": "dry_multiplier",
    "--dry-mul": "dry_multiplier",
    "--dry-base": "dry_base",
    "--dry-allowed-length": "dry_allowed_length",
    "--dry-penalty-last-n": "dry_penalty_last_n",
    "--dry-sequence-breaker": "dry_sequence_breaker",
    "--adaptive-target": "adaptive_target",
    "--adaptive-decay": "adaptive_decay",
    "--dynatemp-range": "dynatemp_range",
    "--dynatemp-exp": "dynatemp_exp",
    "--mirostat": "mirostat",
    "--mirostat-lr": "mirostat_lr",
    "--mirostat-ent": "mirostat_ent",
    "--logit-bias": "logit_bias",
    "--grammar": "grammar",
    "--grammar-file": "grammar_file",
    "--json-schema": "json_schema",
    "--json-schema-file": "json_schema_file",
    "--backend-sampling": "backend_sampling",
    "--no-backend-sampling": "backend_sampling",
    # Speculative decoding
    "--spec-draft-hf": "spec_draft_hf",
    "--spec-draft-threads": "spec_draft_threads",
    "--spec-draft-threads-batch": "spec_draft_threads_batch",
    "--spec-draft-cpu-mask": "spec_draft_cpu_mask",
    "--spec-draft-cpu-range": "spec_draft_cpu_range",
    "--spec-draft-cpu-strict": "spec_draft_cpu_strict",
    "--spec-draft-prio": "spec_draft_prio",
    "--spec-draft-poll": "spec_draft_poll",
    "--spec-draft-cpu-mask-batch": "spec_draft_cpu_mask_batch",
    "--spec-draft-cpu-strict-batch": "spec_draft_cpu_strict_batch",
    "--spec-draft-prio-batch": "spec_draft_prio_batch",
    "--spec-draft-poll-batch": "spec_draft_poll_batch",
    "--spec-draft-override-tensor": "spec_draft_override_tensor",
    "--spec-draft-cpu-moe": "spec_draft_cpu_moe",
    "--no-spec-draft-cpu-moe": "spec_draft_cpu_moe",
    "--spec-draft-n-cpu-moe": "spec_draft_n_cpu_moe",
    "--spec-draft-n-max": "spec_draft_n_max",
    "--spec-draft-n-min": "spec_draft_n_min",
    "--draft-n-max": "spec_draft_n_max",
    "--draft-n-min": "spec_draft_n_min",
    "--spec-draft-p-split": "spec_draft_p_split",
    "--spec-draft-p-min": "spec_draft_p_min",
    "--spec-draft-backend-sampling": "spec_draft_backend_sampling",
    "--no-spec-draft-backend-sampling": "spec_draft_backend_sampling",
    "--spec-draft-device": "spec_draft_device",
    "--spec-draft-ngl": "spec_draft_ngl",
    "--spec-draft-model": "spec_draft_model",
    "--spec-type": "spec_type",
    "--spec-ngram-mod-n-min": "spec_ngram_mod_n_min",
    "--spec-ngram-mod-n-max": "spec_ngram_mod_n_max",
    "--spec-ngram-mod-n-match": "spec_ngram_mod_n_match",
    "--spec-ngram-simple-size-n": "spec_ngram_simple_size_n",
    "--spec-ngram-simple-size-m": "spec_ngram_simple_size_m",
    "--spec-ngram-simple-min-hits": "spec_ngram_simple_min_hits",
    "--spec-ngram-map-k-size-n": "spec_ngram_map_k_size_n",
    "--spec-ngram-map-k-size-m": "spec_ngram_map_k_size_m",
    "--spec-ngram-map-k-min-hits": "spec_ngram_map_k_min_hits",
    "--spec-ngram-map-k4v-size-n": "spec_ngram_map_k4v_size_n",
    "--spec-ngram-map-k4v-size-m": "spec_ngram_map_k4v_size_m",
    "--spec-ngram-map-k4v-min-hits": "spec_ngram_map_k4v_min_hits",
    # Cache and context extras
    "--lookup-cache-static": "lookup_cache_static",
    "--lookup-cache-dynamic": "lookup_cache_dynamic",
    "--ctx-checkpoints": "ctx_checkpoints",
    "--checkpoint-min-step": "checkpoint_min_step",
    "--cache-ram": "cache_ram",
    "--kv-unified": "kv_unified",
    "--no-kv-unified": "kv_unified",
    "--cache-idle-slots": "cache_idle_slots",
    "--no-cache-idle-slots": "cache_idle_slots",
    "--context-shift": "context_shift",
    "--no-context-shift": "context_shift",
    # Generation and server
    "--reverse-prompt": "reverse_prompt",
    "--special": "special",
    "--no-special": "special",
    "--warmup": "warmup",
    "--no-warmup": "warmup",
    "--spm-infill": "spm_infill",
    "--no-spm-infill": "spm_infill",
    "--pooling": "pooling",
    "--parallel": "parallel",
    "--cont-batching": "cont_batching",
    "--no-cont-batching": "cont_batching",
    # Multimodal
    "--mmproj-url": "mmproj_url",
    "--mmproj-auto": "mmproj_auto",
    "--no-mmproj-auto": "mmproj_auto",
    "--mmproj-offload": "mmproj_offload",
    "--no-mmproj-offload": "mmproj_offload",
    "--image-min-tokens": "image_min_tokens",
    "--image-max-tokens": "image_max_tokens",
    "--mtmd-batch-max-tokens": "mtmd_batch_max_tokens",
    # Server and UI
    "--alias": "alias",
    "--tags": "tags",
    "--embd-normalize": "embd_normalize",
    "--port": "port",
    "-p": "port",
    "--host": "host",
    "--reuse-port": "reuse_port",
    "--no-reuse-port": "reuse_port",
    "--path": "path",
    "--api-prefix": "api_prefix",
    "--ui-config": "ui_config",
    "--ui-config-file": "ui_config_file",
    "--ui-mcp-proxy": "ui_mcp_proxy",
    "--no-ui-mcp-proxy": "ui_mcp_proxy",
    "--tools": "tools",
    "--agent": "agent",
    "--no-agent": "agent",
    "--ui": "ui",
    "--no-ui": "ui",
    "--embeddings": "embeddings",
    "--no-embeddings": "embeddings",
    "--reranking": "reranking",
    "--no-reranking": "reranking",
    # Backend and jinja
    "--backend": "backend",
    "--jinja": "jinja",
    "--no-jinja": "jinja",
}

_FALSE_STRINGS = {"false", "0", "off", "no"}


def _coerce(config_key: str, value: str) -> Any:
    """Coerce a string value to the LlamaConfig field's declared type."""
    field_type = _FIELD_TYPES.get(config_key)
    try:
        if field_type is bool:
            return value.strip().lower() not in _FALSE_STRINGS
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
    except (ValueError, TypeError):
        logger.warning("Could not coerce %r for field %s; keeping raw value", value, config_key)
    return value


def _is_bool_field(config_key: str) -> bool:
    return _FIELD_TYPES.get(config_key) is bool


def _looks_like_value(token: str) -> bool:
    """A token is a value unless it looks like a flag.

    Negative numbers (e.g. ``-1`` for ``--seed``) count as values.
    """
    if not token.startswith("-"):
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_launch_command(command_str: str) -> Tuple[Dict[str, Any], List[str]]:
    """Parse a llama-server command line.

    Returns:
        (params, unrecognized_flags): a dict of LlamaConfig field values and a
        list of tokens that could not be mapped to a known field.
    """
    try:
        args = shlex.split(command_str)
    except ValueError as e:
        logger.warning("shlex failed (%s); falling back to naive split", e)
        args = command_str.split()

    # Drop the executable name if present (e.g. "llama-server", "./llama-server").
    if args and not args[0].startswith("-") and not args[0].endswith(
        (".gguf", ".bin", ".safetensors")
    ):
        args = args[1:]

    params: Dict[str, Any] = {}
    unrecognized_flags: List[str] = []
    i = 0

    while i < len(args):
        arg = args[i]

        # --flag=value form
        if arg.startswith("-") and "=" in arg:
            key, value = arg.split("=", 1)
            config_key = FLAG_MAP.get(key)
            if config_key:
                params[config_key] = _coerce(config_key, value)
            else:
                unrecognized_flags.append(arg)
            i += 1
            continue

        if arg.startswith("-"):
            config_key = FLAG_MAP.get(arg)
            if config_key is None:
                unrecognized_flags.append(arg)
                i += 1
                continue

            if _is_bool_field(config_key):
                params[config_key] = not arg.startswith("--no-")
                i += 1
                continue

            # Value-taking flag: consume the next token if it is a value.
            if i + 1 < len(args) and _looks_like_value(args[i + 1]):
                params[config_key] = _coerce(config_key, args[i + 1])
                i += 2
            else:
                logger.warning("Flag %s missing a value; skipped", arg)
                i += 1
            continue

        # Bare token: treat as model path if we don't have one yet.
        if "model_path" not in params:
            if arg.endswith((".gguf", ".bin", ".safetensors")) or "/" in arg or "\\" in arg:
                params["model_path"] = arg
        i += 1

    return params, unrecognized_flags
