"""Llama Configuration Schema."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LlamaConfig:
    """Shared configuration object for llama-server launch configurations."""
    # Model paths
    model_path: str = ""
    mmproj_model_path: Optional[str] = None
    draft_model_path: Optional[str] = None
    
    # Common params - threads and batching
    threads: int = 0
    threads_batch: int = -1
    cpu_mask: str = ""
    cpu_range: str = ""
    cpu_strict: int = 0
    prio: int = 0
    poll: int = 50
    cpu_mask_batch: str = ""
    cpu_range_batch: str = ""
    cpu_strict_batch: int = 0
    prio_batch: int = 0
    poll_batch: int = 0
    
    # Common params - context and predict
    context_size: int = 4096
    predict: int = -1
    batch_size: int = 2048
    logical_batch_size: int = 2048
    ubatch_size: int = 512
    keep: int = 0
    swa_full: bool = False
    flash_attn: str = "auto"
    perf: bool = False
    escape: bool = True
    
    # Common params - RoPE and YaRN
    rope_scaling: str = "none"
    rope_scale: int = 1
    rope_freq_base: int = 0
    rope_freq_scale: int = 0
    yarn_orig_ctx: int = 0
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = -1.0
    yarn_beta_slow: float = -1.0
    yarn_beta_fast: float = -1.0
    
    # Common params - cache and memory
    kv_offload: bool = True
    repack: bool = True
    no_host: bool = False
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    kv_cache_type: str = "f16"
    defrag_thold: int = 0
    mlock: bool = False
    use_mmap: bool = True
    direct_io: bool = False
    
    # Common params - GPU and NUMA
    numa: str = ""
    device: str = ""
    override_tensor: str = ""
    cpu_moe: bool = False
    n_cpu_moe: int = 0
    gpu_layers: int = -1
    split_mode: str = "layer"
    tensor_split: Optional[str] = None
    main_gpu: int = 0
    fit: str = "on"
    fit_target: str = ""
    fit_ctx: int = 4096
    check_tensors: bool = False
    override_kv: str = ""
    op_offload: bool = True
    
    # Common params - LoRA and control vectors
    lora: str = ""
    lora_scaled: str = ""
    control_vector: str = ""
    control_vector_scaled: str = ""
    control_vector_layer_range: str = ""
    
    # Common params - model sources
    model_url: str = ""
    docker_repo: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    hf_repo_v: str = ""
    hf_file_v: str = ""
    hf_token: str = ""
    
    # Common params - logging
    log_disable: bool = False
    log_file: str = ""
    log_colors: str = "auto"
    verbose: bool = False
    offline: bool = False
    verbosity: int = 3
    log_prefix: bool = True
    log_timestamps: bool = True
    
    # Spec draft cache types
    spec_draft_type_k: str = "f16"
    spec_draft_type_v: str = "f16"
    
    # Sampling params
    samplers: str = "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
    seed: int = -1
    sampler_seq: str = "edskypmxt"
    ignore_eos: bool = False
    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    top_n_sigma: float = -1.0
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    typical_p: float = 1.0
    repeat_last_n: int = 64
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1
    dry_sequence_breaker: str = ""
    adaptive_target: float = -1.0
    adaptive_decay: float = 0.9
    dynatemp_range: float = 0.0
    dynatemp_exp: float = 1.0
    mirostat: int = 0
    mirostat_lr: float = 0.1
    mirostat_ent: float = 5.0
    logit_bias: str = ""
    grammar: str = ""
    grammar_file: str = ""
    json_schema: str = ""
    json_schema_file: str = ""
    backend_sampling: bool = False
    
    # Speculative params - draft model
    spec_draft_hf: str = ""
    spec_draft_threads: int = 0
    spec_draft_threads_batch: int = -1
    spec_draft_cpu_mask: str = ""
    spec_draft_cpu_range: str = ""
    spec_draft_cpu_strict: int = 0
    spec_draft_prio: int = 0
    spec_draft_poll: int = 0
    spec_draft_cpu_mask_batch: str = ""
    spec_draft_cpu_strict_batch: int = 0
    spec_draft_prio_batch: int = 0
    spec_draft_poll_batch: int = 0
    spec_draft_override_tensor: str = ""
    spec_draft_cpu_moe: bool = False
    spec_draft_n_cpu_moe: int = 0
    spec_draft_n_max: int = 3
    spec_draft_n_min: int = 0
    draft_n_max: int = 3
    draft_n_min: int = 0
    spec_draft_p_split: float = 0.1
    spec_draft_p_min: float = 0.0
    spec_draft_backend_sampling: bool = True
    spec_draft_device: str = ""
    spec_draft_ngl: int = -1
    spec_draft_model: str = ""
    spec_type: str = "none"
    spec_ngram_mod_n_min: int = 48
    spec_ngram_mod_n_max: int = 64
    spec_ngram_mod_n_match: int = 24
    spec_ngram_simple_size_n: int = 12
    spec_ngram_simple_size_m: int = 48
    spec_ngram_simple_min_hits: int = 1
    spec_ngram_map_k_size_n: int = 12
    spec_ngram_map_k_size_m: int = 48
    spec_ngram_map_k_min_hits: int = 1
    spec_ngram_map_k4v_size_n: int = 12
    spec_ngram_map_k4v_size_m: int = 48
    spec_ngram_map_k4v_min_hits: int = 1
    
    # Example-specific params - cache and context
    lookup_cache_static: str = ""
    lookup_cache_dynamic: str = ""
    ctx_checkpoints: int = 32
    checkpoint_min_step: int = 256
    cache_ram: int = 8192
    kv_unified: bool = True
    cache_idle_slots: bool = True
    context_shift: bool = False
    
    # Example-specific params - generation and server
    reverse_prompt: str = ""
    special: bool = False
    warmup: bool = True
    spm_infill: bool = False
    pooling: str = "none"
    parallel: int = -1
    cont_batching: bool = True
    
    # Example-specific params - multimodal
    mmproj: str = ""
    mmproj_url: str = ""
    mmproj_auto: bool = True
    mmproj_offload: bool = True
    image_min_tokens: int = 0
    image_max_tokens: int = 0
    mtmd_batch_max_tokens: int = 1024
    
    # Example-specific params - server and UI
    alias: str = ""
    tags: str = ""
    embd_normalize: int = 2
    port: int = 8080
    host: str = "127.0.0.1"
    reuse_port: bool = False
    path: str = ""
    api_prefix: str = ""
    ui_config: str = ""
    ui_config_file: str = ""
    ui_mcp_proxy: bool = False
    tools: str = ""
    agent: bool = False
    ui: bool = True
    embeddings: bool = False
    reranking: bool = False
    
    # Backend and jinja
    backend: str = "auto"
    jinja: bool = False
    
    # Target server for remote execution
    target_server: Optional[str] = None

    # Metadata for presets/profiles
    name: Optional[str] = None
    description: Optional[str] = None
    extra_flags: List[str] = field(default_factory=list)
