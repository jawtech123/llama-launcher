"""CLI Command Builder.

Builds a llama-server command string from a :class:`LlamaConfig`. Values are
shell-quoted where necessary, so the result round-trips through ``shlex.split``
(used for QProcess launch) and can be pasted directly into a terminal.

Only options whose value differs from the llama-server default are emitted.
"""
import shlex
from typing import List

from .config import LlamaConfig


def build_command(config: LlamaConfig) -> str:
    """Build the llama-server command for the given configuration."""
    args: List[str] = ["llama-server"]

    def flag(name: str) -> None:
        """Append a bare boolean flag (e.g. --mlock)."""
        args.append(name)

    def opt(name: str, value) -> None:
        """Append a flag/value pair, quoting the value when required."""
        args.append(name)
        args.append(shlex.quote(str(value)))

    g = lambda name, default=None: getattr(config, name, default)

    # --- Model paths ---
    if config.model_path:
        opt("--model", config.model_path)

    # mmproj_model_path (basic tab) and mmproj (advanced tab) are the same flag;
    # emit it exactly once.
    mmproj_path = config.mmproj_model_path or g("mmproj", "")
    if mmproj_path:
        opt("--mmproj", mmproj_path)

    if config.draft_model_path:
        opt("--draft-model", config.draft_model_path)

    # --- Server / network ---
    if config.port != 8080:
        opt("--port", config.port)
    if config.host != "127.0.0.1":
        opt("--host", config.host)

    # --- Context & predict ---
    if config.context_size != 4096:
        opt("--ctx-size", config.context_size)

    if g("predict", -1) != -1:
        opt("--n-predict", g("predict"))

    if g("keep", 0) != 0:
        opt("--keep", g("keep"))

    # --- Batching ---
    if config.batch_size != 2048:
        opt("--batch-size", config.batch_size)

    if g("ubatch_size", 512) != 512:
        opt("--ubatch-size", g("ubatch_size"))

    # --- GPU layers ---
    if config.gpu_layers != -1:
        opt("-ngl", config.gpu_layers)

    # --- Threading ---
    if config.threads > 0:
        opt("--threads", config.threads)

    if config.threads_batch != -1:
        opt("--threads-batch", config.threads_batch)

    if g("cpu_mask"):
        opt("--cpu-mask", g("cpu_mask"))
    if g("cpu_range"):
        opt("--cpu-range", g("cpu_range"))
    if g("cpu_strict", 0):
        opt("--cpu-strict", g("cpu_strict"))
    if g("prio", 0) != 0:
        opt("--prio", g("prio"))
    if g("poll", 50) != 50:
        opt("--poll", g("poll"))
    if g("cpu_mask_batch"):
        opt("--cpu-mask-batch", g("cpu_mask_batch"))
    if g("cpu_range_batch"):
        opt("--cpu-range-batch", g("cpu_range_batch"))
    if g("cpu_strict_batch", 0):
        opt("--cpu-strict-batch", g("cpu_strict_batch"))
    if g("prio_batch", 0) != 0:
        opt("--prio-batch", g("prio_batch"))
    if g("poll_batch", 0) != 0:
        opt("--poll-batch", g("poll_batch"))

    # --- KV cache ---
    cache_type_k = g("cache_type_k", "f16")
    if cache_type_k and cache_type_k != "f16":
        opt("--cache-type-k", cache_type_k)

    cache_type_v = g("cache_type_v", "f16")
    if cache_type_v and cache_type_v != "f16":
        opt("--cache-type-v", cache_type_v)

    if not g("kv_unified", True):
        flag("--no-kv-unified")

    # --- RoPE scaling ---
    rope_scaling = g("rope_scaling", "none")
    if rope_scaling and rope_scaling != "none":
        opt("--rope-scaling", rope_scaling)

    if g("rope_scale", 1) != 1:
        opt("--rope-scale", g("rope_scale"))
    if g("rope_freq_base", 0) != 0:
        opt("--rope-freq-base", g("rope_freq_base"))
    if g("rope_freq_scale", 0) != 0:
        opt("--rope-freq-scale", g("rope_freq_scale"))

    # --- YaRN ---
    if g("yarn_orig_ctx", 0) != 0:
        opt("--yarn-orig-ctx", g("yarn_orig_ctx"))
    if g("yarn_ext_factor", -1.0) != -1.0:
        opt("--yarn-ext-factor", g("yarn_ext_factor"))
    if g("yarn_attn_factor", -1.0) != -1.0:
        opt("--yarn-attn-factor", g("yarn_attn_factor"))
    if g("yarn_beta_slow", -1.0) != -1.0:
        opt("--yarn-beta-slow", g("yarn_beta_slow"))
    if g("yarn_beta_fast", -1.0) != -1.0:
        opt("--yarn-beta-fast", g("yarn_beta_fast"))

    # --- Flash attention & SWA ---
    flash_attn = g("flash_attn", "auto")
    if flash_attn and flash_attn != "auto":
        if flash_attn in ("on", "true", "enable"):
            opt("--flash-attn", "on")
        elif flash_attn == "disable":
            opt("--flash-attn", "off")
        else:
            opt("--flash-attn", flash_attn)

    if g("swa_full", False):
        flag("--swa-full")
    if g("perf", False):
        flag("--perf")
    if not g("escape", True):
        flag("--no-escape")

    # --- Cache & memory ---
    if not g("kv_offload", True):
        flag("--no-kv-offload")
    if not g("repack", True):
        flag("--no-repack")
    if g("no_host", False):
        flag("--no-host")
    if g("defrag_thold", 0) != 0:
        opt("--defrag-thold", g("defrag_thold"))
    if g("direct_io", False):
        flag("--direct-io")
    if g("mlock", False):
        flag("--mlock")
    if not g("use_mmap", True):
        flag("--no-mmap")

    # --- GPU & NUMA ---
    if g("numa"):
        opt("--numa", g("numa"))
    if g("device"):
        opt("--device", g("device"))
    if g("override_tensor"):
        opt("--override-tensor", g("override_tensor"))
    if g("cpu_moe", False):
        flag("--cpu-moe")
    if g("n_cpu_moe", 0) != 0:
        opt("--n-cpu-moe", g("n_cpu_moe"))

    split_mode = g("split_mode", "layer")
    if split_mode and split_mode != "layer":
        opt("--split-mode", split_mode)

    if g("tensor_split"):
        opt("--tensor-split", g("tensor_split"))
    if g("main_gpu", 0) != 0:
        opt("--main-gpu", g("main_gpu"))

    fit = g("fit", "on")
    if fit and fit != "on":
        opt("--fit", fit)
    if g("fit_target"):
        opt("--fit-target", g("fit_target"))
    if g("fit_ctx", 4096) != 4096:
        opt("--fit-ctx", g("fit_ctx"))
    if g("check_tensors", False):
        flag("--check-tensors")
    if g("override_kv"):
        opt("--override-kv", g("override_kv"))
    if not g("op_offload", True):
        flag("--no-op-offload")

    # --- LoRA & control vectors ---
    if g("lora"):
        opt("--lora", g("lora"))
    if g("lora_scaled"):
        opt("--lora-scaled", g("lora_scaled"))
    if g("control_vector"):
        opt("--control-vector", g("control_vector"))
    if g("control_vector_scaled"):
        opt("--control-vector-scaled", g("control_vector_scaled"))
    if g("control_vector_layer_range"):
        opt("--control-vector-layer-range", g("control_vector_layer_range"))

    # --- Model sources ---
    if g("model_url"):
        opt("--model-url", g("model_url"))
    if g("docker_repo"):
        opt("--docker-repo", g("docker_repo"))
    if g("hf_repo"):
        opt("--hf-repo", g("hf_repo"))
    if g("hf_file"):
        opt("--hf-file", g("hf_file"))
    if g("hf_repo_v"):
        opt("--hf-repo-v", g("hf_repo_v"))
    if g("hf_file_v"):
        opt("--hf-file-v", g("hf_file_v"))
    if g("hf_token"):
        opt("--hf-token", g("hf_token"))

    # --- Logging ---
    if g("log_disable", False):
        flag("--log-disable")
    if g("log_file"):
        opt("--log-file", g("log_file"))

    log_colors = g("log_colors", "auto")
    if log_colors and log_colors != "auto":
        opt("--log-colors", log_colors)

    if g("verbose", False):
        flag("--verbose")
    if g("offline", False):
        flag("--offline")
    if g("verbosity", 3) != 3:
        opt("--verbosity", g("verbosity"))
    if not g("log_prefix", True):
        flag("--no-log-prefix")
    if not g("log_timestamps", True):
        flag("--no-log-timestamps")

    # --- Speculative decoding ---
    spec_type = g("spec_type", "none")
    if spec_type and spec_type != "none":
        opt("--spec-type", spec_type)

    # draft_n_* and spec_draft_n_* map to the same flags; emit each exactly
    # once, preferring the spec_draft_* value.
    draft_n_max = g("spec_draft_n_max", 3)
    if draft_n_max == 3:
        draft_n_max = g("draft_n_max", 3)
    if draft_n_max != 3:
        opt("--spec-draft-n-max", draft_n_max)

    draft_n_min = g("spec_draft_n_min", 0)
    if draft_n_min == 0:
        draft_n_min = g("draft_n_min", 0)
    if draft_n_min != 0:
        opt("--spec-draft-n-min", draft_n_min)

    if g("spec_draft_hf"):
        opt("--spec-draft-hf", g("spec_draft_hf"))
    if g("spec_draft_threads", 0) != 0:
        opt("--spec-draft-threads", g("spec_draft_threads"))
    if g("spec_draft_threads_batch", -1) != -1:
        opt("--spec-draft-threads-batch", g("spec_draft_threads_batch"))
    if g("spec_draft_override_tensor"):
        opt("--spec-draft-override-tensor", g("spec_draft_override_tensor"))
    if g("spec_draft_device"):
        opt("--spec-draft-device", g("spec_draft_device"))

    spec_draft_type_k = g("spec_draft_type_k", "f16")
    if spec_draft_type_k and spec_draft_type_k != "f16":
        opt("--spec-draft-type-k", spec_draft_type_k)

    spec_draft_type_v = g("spec_draft_type_v", "f16")
    if spec_draft_type_v and spec_draft_type_v != "f16":
        opt("--spec-draft-type-v", spec_draft_type_v)

    if g("spec_draft_model"):
        opt("--spec-draft-model", g("spec_draft_model"))

    # --- Sampling ---
    if g("temp", 0.8) != 0.8:
        opt("--temp", g("temp"))
    if g("top_p", 0.95) != 0.95:
        opt("--top-p", g("top_p"))
    if g("top_k", 40) != 40:
        opt("--top-k", g("top_k"))
    if g("min_p", 0.05) != 0.05:
        opt("--min-p", g("min_p"))
    if g("repeat_last_n", 64) != 64:
        opt("--repeat-last-n", g("repeat_last_n"))
    if g("repeat_penalty", 1.0) != 1.0:
        opt("--repeat-penalty", g("repeat_penalty"))
    if g("presence_penalty", 0.0) != 0.0:
        opt("--presence-penalty", g("presence_penalty"))
    if g("frequency_penalty", 0.0) != 0.0:
        opt("--frequency-penalty", g("frequency_penalty"))
    if g("dry_multiplier", 0.0) != 0.0:
        opt("--dry-mul", g("dry_multiplier"))
    if g("dry_base", 1.75) != 1.75:
        opt("--dry-base", g("dry_base"))
    if g("dry_allowed_length", 2) != 2:
        opt("--dry-allowed-length", g("dry_allowed_length"))
    if g("dry_penalty_last_n", -1) != -1:
        opt("--dry-penalty-last-n", g("dry_penalty_last_n"))
    if g("dry_sequence_breaker"):
        opt("--dry-sequence-breaker", g("dry_sequence_breaker"))
    if g("mirostat", 0) != 0:
        opt("--mirostat", g("mirostat"))
    if g("mirostat_lr", 0.1) != 0.1:
        opt("--mirostat-lr", g("mirostat_lr"))
    if g("mirostat_ent", 5.0) != 5.0:
        opt("--mirostat-ent", g("mirostat_ent"))
    if g("grammar"):
        opt("--grammar", g("grammar"))
    if g("grammar_file"):
        opt("--grammar-file", g("grammar_file"))
    if g("json_schema"):
        opt("--json-schema", g("json_schema"))
    if g("json_schema_file"):
        opt("--json-schema-file", g("json_schema_file"))
    if g("backend_sampling", False):
        flag("--backend-sampling")
    if g("ignore_eos", False):
        flag("--ignore-eos")

    # --- Server & UI ---
    parallel = g("parallel", -1)
    if parallel != -1 and parallel > 0:
        opt("--parallel", parallel)
    if g("jinja", False):
        flag("--jinja")
    if not g("cont_batching", True):
        flag("--no-cont-batching")
    if g("embeddings", False):
        flag("--embeddings")
    if g("reranking", False):
        flag("--reranking")
    if not g("ui", True):
        flag("--no-ui")
    if g("api_prefix"):
        opt("--api-prefix", g("api_prefix"))
    if g("ui_config"):
        opt("--ui-config", g("ui_config"))
    if g("ui_config_file"):
        opt("--ui-config-file", g("ui_config_file"))
    if g("tools"):
        opt("--tools", g("tools"))
    if g("agent", False):
        flag("--agent")

    backend = g("backend", "auto")
    if backend and backend != "auto":
        opt("--backend", backend)

    # --- Example-specific ---
    if g("reverse_prompt"):
        opt("--reverse-prompt", g("reverse_prompt"))
    if not g("warmup", True):
        flag("--no-warmup")

    pooling = g("pooling", "none")
    if pooling and pooling != "none":
        opt("--pooling", pooling)

    if g("lookup_cache_static"):
        opt("--lookup-cache-static", g("lookup_cache_static"))
    if g("lookup_cache_dynamic"):
        opt("--lookup-cache-dynamic", g("lookup_cache_dynamic"))

    # --- Extra sampling ---
    samplers = g("samplers", "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature")
    if samplers and samplers != "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature":
        opt("--samplers", samplers)

    sampler_seq = g("sampler_seq", "edskypmxt")
    if sampler_seq and sampler_seq != "edskypmxt":
        opt("--sampler-seq", sampler_seq)

    if g("seed", -1) != -1:
        opt("--seed", g("seed"))
    if g("top_n_sigma", -1.0) != -1.0:
        opt("--top-n-sigma", g("top_n_sigma"))
    if g("typical_p", 1.0) != 1.0:
        opt("--typical-p", g("typical_p"))
    if g("xtc_probability", 0.0) != 0.0:
        opt("--xtc-probability", g("xtc_probability"))
    if g("xtc_threshold", 0.1) != 0.1:
        opt("--xtc-threshold", g("xtc_threshold"))
    if g("logit_bias"):
        opt("--logit-bias", g("logit_bias"))

    # --- Adaptive & DynaTemp ---
    if g("adaptive_target", -1.0) != -1.0:
        opt("--adaptive-target", g("adaptive_target"))
    if g("adaptive_decay", 0.9) != 0.9:
        opt("--adaptive-decay", g("adaptive_decay"))
    if g("dynatemp_range", 0.0) != 0.0:
        opt("--dynatemp-range", g("dynatemp_range"))
    if g("dynatemp_exp", 1.0) != 1.0:
        opt("--dynatemp-exp", g("dynatemp_exp"))

    # --- Spec draft extras ---
    if g("spec_draft_p_split", 0.1) != 0.1:
        opt("--spec-draft-p-split", g("spec_draft_p_split"))
    if g("spec_draft_p_min", 0.0) != 0.0:
        opt("--spec-draft-p-min", g("spec_draft_p_min"))
    if not g("spec_draft_backend_sampling", True):
        flag("--no-spec-draft-backend-sampling")
    if g("spec_draft_ngl", -1) != -1:
        opt("--spec-draft-ngl", g("spec_draft_ngl"))

    # --- Spec draft CPU affinity ---
    if g("spec_draft_cpu_mask", ""):
        opt("--spec-draft-cpu-mask", g("spec_draft_cpu_mask"))
    if g("spec_draft_cpu_range", ""):
        opt("--spec-draft-cpu-range", g("spec_draft_cpu_range"))
    if g("spec_draft_cpu_strict", 0) != 0:
        opt("--spec-draft-cpu-strict", g("spec_draft_cpu_strict"))
    if g("spec_draft_prio", 0) != 0:
        opt("--spec-draft-prio", g("spec_draft_prio"))
    if g("spec_draft_poll", 0) != 0:
        opt("--spec-draft-poll", g("spec_draft_poll"))
    if g("spec_draft_cpu_mask_batch", ""):
        opt("--spec-draft-cpu-mask-batch", g("spec_draft_cpu_mask_batch"))
    if g("spec_draft_cpu_strict_batch", 0) != 0:
        opt("--spec-draft-cpu-strict-batch", g("spec_draft_cpu_strict_batch"))
    if g("spec_draft_prio_batch", 0) != 0:
        opt("--spec-draft-prio-batch", g("spec_draft_prio_batch"))
    if g("spec_draft_poll_batch", 0) != 0:
        opt("--spec-draft-poll-batch", g("spec_draft_poll_batch"))
    if g("spec_draft_cpu_moe", False):
        flag("--spec-draft-cpu-moe")
    if g("spec_draft_n_cpu_moe", 0) != 0:
        opt("--spec-draft-n-cpu-moe", g("spec_draft_n_cpu_moe"))

    # --- Spec N-gram / mod params ---
    if g("spec_ngram_mod_n_min", 48) != 48:
        opt("--spec-ngram-mod-n-min", g("spec_ngram_mod_n_min"))
    if g("spec_ngram_mod_n_max", 64) != 64:
        opt("--spec-ngram-mod-n-max", g("spec_ngram_mod_n_max"))
    if g("spec_ngram_mod_n_match", 24) != 24:
        opt("--spec-ngram-mod-n-match", g("spec_ngram_mod_n_match"))
    if g("spec_ngram_simple_size_n", 12) != 12:
        opt("--spec-ngram-simple-size-n", g("spec_ngram_simple_size_n"))
    if g("spec_ngram_simple_size_m", 48) != 48:
        opt("--spec-ngram-simple-size-m", g("spec_ngram_simple_size_m"))
    if g("spec_ngram_simple_min_hits", 1) != 1:
        opt("--spec-ngram-simple-min-hits", g("spec_ngram_simple_min_hits"))
    if g("spec_ngram_map_k_size_n", 12) != 12:
        opt("--spec-ngram-map-k-size-n", g("spec_ngram_map_k_size_n"))
    if g("spec_ngram_map_k_size_m", 48) != 48:
        opt("--spec-ngram-map-k-size-m", g("spec_ngram_map_k_size_m"))
    if g("spec_ngram_map_k_min_hits", 1) != 1:
        opt("--spec-ngram-map-k-min-hits", g("spec_ngram_map_k_min_hits"))
    if g("spec_ngram_map_k4v_size_n", 12) != 12:
        opt("--spec-ngram-map-k4v-size-n", g("spec_ngram_map_k4v_size_n"))
    if g("spec_ngram_map_k4v_size_m", 48) != 48:
        opt("--spec-ngram-map-k4v-size-m", g("spec_ngram_map_k4v_size_m"))
    if g("spec_ngram_map_k4v_min_hits", 1) != 1:
        opt("--spec-ngram-map-k4v-min-hits", g("spec_ngram_map_k4v_min_hits"))

    # --- Extra cache ---
    if g("ctx_checkpoints", 32) != 32:
        opt("--ctx-checkpoints", g("ctx_checkpoints"))
    if g("checkpoint_min_step", 256) != 256:
        opt("--checkpoint-min-step", g("checkpoint_min_step"))
    if g("cache_ram", 8192) != 8192:
        opt("--cache-ram", g("cache_ram"))
    if not g("cache_idle_slots", True):
        flag("--no-cache-idle-slots")
    if g("context_shift", False):
        flag("--context-shift")

    # --- Generation extras ---
    if g("special", False):
        flag("--special")
    if g("spm_infill", False):
        flag("--spm-infill")

    # --- Multimodal ---
    if g("mmproj_url", ""):
        opt("--mmproj-url", g("mmproj_url"))
    if not g("mmproj_auto", True):
        flag("--no-mmproj-auto")
    if not g("mmproj_offload", True):
        flag("--no-mmproj-offload")
    if g("image_min_tokens", 0) != 0:
        opt("--image-min-tokens", g("image_min_tokens"))
    if g("image_max_tokens", 0) != 0:
        opt("--image-max-tokens", g("image_max_tokens"))
    if g("mtmd_batch_max_tokens", 1024) != 1024:
        opt("--mtmd-batch-max-tokens", g("mtmd_batch_max_tokens"))

    # --- Server extras ---
    if g("alias", ""):
        opt("--alias", g("alias"))
    if g("tags", ""):
        opt("--tags", g("tags"))
    if g("embd_normalize", 2) != 2:
        opt("--embd-normalize", g("embd_normalize"))
    if g("path", ""):
        opt("--path", g("path"))
    if g("reuse_port", False):
        flag("--reuse-port")
    if g("ui_mcp_proxy", False):
        flag("--ui-mcp-proxy")

    # --- Extra raw flags (user-provided fragments, appended verbatim) ---
    for extra in config.extra_flags:
        args.append(str(extra))

    return " ".join(args)
