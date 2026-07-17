"""CLI Command Builder."""
from .config import LlamaConfig


def _add_if(value, flag, fmt=None):
    """Helper: return formatted flag string if value is set, else empty."""
    if value is None or value == "":
        return ""
    if fmt:
        return fmt.format(value)
    if isinstance(value, bool):
        return f" {flag}" if value else ""
    return f" {flag} {value}"


def build_command(config: LlamaConfig) -> str:
    """Build llama-server command from LlamaConfig."""
    cmd = "llama-server"

    # --- Model paths ---
    cmd += f" --model {config.model_path}"

    if config.mmproj_model_path:
        cmd += f" --mmproj {config.mmproj_model_path}"

    if config.draft_model_path:
        cmd += f" --draft-model {config.draft_model_path}"

    # --- Server / network ---
    if config.port != 8080:
        cmd += f" --port {config.port}"
    if config.host != "127.0.0.1":
        cmd += f" --host {config.host}"

    # --- Context & predict ---
    if config.context_size != 4096:
        cmd += f" --ctx-size {config.context_size}"

    predict = getattr(config, 'predict', -1)
    if predict != -1:
        cmd += f" --n-predict {predict}"

    keep = getattr(config, 'keep', 0)
    if keep != 0:
        cmd += f" --keep {keep}"

    # --- Batching ---
    if config.batch_size != 2048:
        cmd += f" --batch-size {config.batch_size}"

    logical_batch_size = getattr(config, 'logical_batch_size', 2048)
    if logical_batch_size != 2048:
        cmd += f" --logical-batch-size {logical_batch_size}"

    ubatch_size = getattr(config, 'ubatch_size', 512)
    if ubatch_size != 512:
        cmd += f" --ubatch-size {ubatch_size}"

    # --- GPU layers ---
    if config.gpu_layers != -1:
        cmd += f" -ngl {config.gpu_layers}"

    # --- Threading ---
    if config.threads > 0:
        cmd += f" --threads {config.threads}"

    if config.threads_batch != -1:
        cmd += f" --threads-batch {config.threads_batch}"

    # CPU affinity
    cpu_mask = getattr(config, 'cpu_mask', None)
    if cpu_mask:
        cmd += f" --cpu-mask {cpu_mask}"

    cpu_range = getattr(config, 'cpu_range', None)
    if cpu_range:
        cmd += f" --cpu-range {cpu_range}"

    cpu_strict = getattr(config, 'cpu_strict', 0)
    if cpu_strict:
        cmd += f" --cpu-strict {cpu_strict}"

    prio = getattr(config, 'prio', 0)
    if prio != 0:
        cmd += f" --prio {prio}"

    poll = getattr(config, 'poll', 50)
    if poll != 50:
        cmd += f" --poll {poll}"

    cpu_mask_batch = getattr(config, 'cpu_mask_batch', None)
    if cpu_mask_batch:
        cmd += f" --cpu-mask-batch {cpu_mask_batch}"

    cpu_range_batch = getattr(config, 'cpu_range_batch', None)
    if cpu_range_batch:
        cmd += f" --cpu-range-batch {cpu_range_batch}"

    cpu_strict_batch = getattr(config, 'cpu_strict_batch', 0)
    if cpu_strict_batch:
        cmd += f" --cpu-strict-batch {cpu_strict_batch}"

    prio_batch = getattr(config, 'prio_batch', 0)
    if prio_batch != 0:
        cmd += f" --prio-batch {prio_batch}"

    poll_batch = getattr(config, 'poll_batch', 0)
    if poll_batch != 0:
        cmd += f" --poll-batch {poll_batch}"

    # --- KV cache ---
    cache_type_k = getattr(config, 'cache_type_k', 'f16')
    if cache_type_k and cache_type_k != 'f16':
        cmd += f" --cache-type-k {cache_type_k}"

    cache_type_v = getattr(config, 'cache_type_v', 'f16')
    if cache_type_v and cache_type_v != 'f16':
        cmd += f" --cache-type-v {cache_type_v}"

    kv_cache_type = getattr(config, 'kv_cache_type', 'f16')
    if kv_cache_type and kv_cache_type != 'f16':
        cmd += f" --kv-cache-type {kv_cache_type}"

    kv_unified = getattr(config, 'kv_unified', True)
    if not kv_unified:
        cmd += " --no-kv-unified"

    # --- RoPE scaling ---
    rope_scaling = getattr(config, 'rope_scaling', 'none')
    if rope_scaling and rope_scaling != 'none':
        cmd += f" --rope-scaling {rope_scaling}"

    rope_scale = getattr(config, 'rope_scale', 1)
    if rope_scale != 1:
        cmd += f" --rope-scale {rope_scale}"

    rope_freq_base = getattr(config, 'rope_freq_base', 0)
    if rope_freq_base != 0:
        cmd += f" --rope-freq-base {rope_freq_base}"

    rope_freq_scale = getattr(config, 'rope_freq_scale', 0)
    if rope_freq_scale != 0:
        cmd += f" --rope-freq-scale {rope_freq_scale}"

    # --- YaRN ---
    yarn_orig_ctx = getattr(config, 'yarn_orig_ctx', 0)
    if yarn_orig_ctx != 0:
        cmd += f" --yarn-orig-ctx {yarn_orig_ctx}"

    yarn_ext_factor = getattr(config, 'yarn_ext_factor', -1.0)
    if yarn_ext_factor != -1.0:
        cmd += f" --yarn-ext-factor {yarn_ext_factor}"

    yarn_attn_factor = getattr(config, 'yarn_attn_factor', -1.0)
    if yarn_attn_factor != -1.0:
        cmd += f" --yarn-attn-factor {yarn_attn_factor}"

    yarn_beta_slow = getattr(config, 'yarn_beta_slow', -1.0)
    if yarn_beta_slow != -1.0:
        cmd += f" --yarn-beta-slow {yarn_beta_slow}"

    yarn_beta_fast = getattr(config, 'yarn_beta_fast', -1.0)
    if yarn_beta_fast != -1.0:
        cmd += f" --yarn-beta-fast {yarn_beta_fast}"

    # --- Flash attention & SWA ---
    flash_attn = getattr(config, 'flash_attn', 'auto')
    if flash_attn and flash_attn != 'auto':
        if flash_attn in ('on', 'true', 'enable'):
            cmd += " --flash-attn on"
        elif flash_attn == 'disable':
            cmd += " --flash-attn off"
        else:
            cmd += f" --flash-attn {flash_attn}"

    swa_full = getattr(config, 'swa_full', False)
    if swa_full:
        cmd += " --swa-full"

    perf = getattr(config, 'perf', False)
    if perf:
        cmd += " --perf"

    escape = getattr(config, 'escape', True)
    if not escape:
        cmd += " --no-escape"

    # --- Cache & memory ---
    kv_offload = getattr(config, 'kv_offload', True)
    if not kv_offload:
        cmd += " --no-kv-offload"

    repack = getattr(config, 'repack', True)
    if not repack:
        cmd += " --no-repack"

    no_host = getattr(config, 'no_host', False)
    if no_host:
        cmd += " --no-host"

    defrag_thold = getattr(config, 'defrag_thold', 0)
    if defrag_thold != 0:
        cmd += f" --defrag-thold {defrag_thold}"

    direct_io = getattr(config, 'direct_io', False)
    if direct_io:
        cmd += " --direct-io"

    mlock = getattr(config, 'mlock', False)
    if mlock:
        cmd += " --mlock"

    use_mmap = getattr(config, 'use_mmap', True)
    if not use_mmap:
        cmd += " --no-mmap"

    # --- GPU & NUMA ---
    numa = getattr(config, 'numa', None)
    if numa:
        cmd += f" --numa {numa}"

    device = getattr(config, 'device', None)
    if device:
        cmd += f" --device {device}"

    override_tensor = getattr(config, 'override_tensor', None)
    if override_tensor:
        cmd += f" --override-tensor {override_tensor}"

    cpu_moe = getattr(config, 'cpu_moe', False)
    if cpu_moe:
        cmd += " --cpu-moe"

    n_cpu_moe = getattr(config, 'n_cpu_moe', 0)
    if n_cpu_moe != 0:
        cmd += f" --n-cpu-moe {n_cpu_moe}"

    split_mode = getattr(config, 'split_mode', 'layer')
    if split_mode and split_mode != 'layer':
        cmd += f" --split-mode {split_mode}"

    tensor_split = getattr(config, 'tensor_split', None)
    if tensor_split:
        cmd += f" --tensor-split {tensor_split}"

    main_gpu = getattr(config, 'main_gpu', 0)
    if main_gpu != 0:
        cmd += f" --main-gpu {main_gpu}"

    fit = getattr(config, 'fit', 'on')
    if fit and fit != 'on':
        cmd += f" --fit {fit}"

    fit_target = getattr(config, 'fit_target', None)
    if fit_target:
        cmd += f" --fit-target {fit_target}"

    fit_ctx = getattr(config, 'fit_ctx', 4096)
    if fit_ctx != 4096:
        cmd += f" --fit-ctx {fit_ctx}"

    check_tensors = getattr(config, 'check_tensors', False)
    if check_tensors:
        cmd += " --check-tensors"

    override_kv = getattr(config, 'override_kv', None)
    if override_kv:
        cmd += f" --override-kv {override_kv}"

    op_offload = getattr(config, 'op_offload', True)
    if not op_offload:
        cmd += " --no-op-offload"

    # --- LoRA & control vectors ---
    lora = getattr(config, 'lora', None)
    if lora:
        cmd += f" --lora {lora}"

    lora_scaled = getattr(config, 'lora_scaled', None)
    if lora_scaled:
        cmd += f" --lora-scaled {lora_scaled}"

    control_vector = getattr(config, 'control_vector', None)
    if control_vector:
        cmd += f" --control-vector {control_vector}"

    control_vector_scaled = getattr(config, 'control_vector_scaled', None)
    if control_vector_scaled:
        cmd += f" --control-vector-scaled {control_vector_scaled}"

    control_vector_layer_range = getattr(config, 'control_vector_layer_range', None)
    if control_vector_layer_range:
        cmd += f" --control-vector-layer-range {control_vector_layer_range}"

    # --- Model sources ---
    model_url = getattr(config, 'model_url', None)
    if model_url:
        cmd += f" --model-url {model_url}"

    docker_repo = getattr(config, 'docker_repo', None)
    if docker_repo:
        cmd += f" --docker-repo {docker_repo}"

    hf_repo = getattr(config, 'hf_repo', None)
    if hf_repo:
        cmd += f" --hf-repo {hf_repo}"

    hf_file = getattr(config, 'hf_file', None)
    if hf_file:
        cmd += f" --hf-file {hf_file}"

    hf_repo_v = getattr(config, 'hf_repo_v', None)
    if hf_repo_v:
        cmd += f" --hf-repo-v {hf_repo_v}"

    hf_file_v = getattr(config, 'hf_file_v', None)
    if hf_file_v:
        cmd += f" --hf-file-v {hf_file_v}"

    hf_token = getattr(config, 'hf_token', None)
    if hf_token:
        cmd += f" --hf-token {hf_token}"

    # --- Logging ---
    log_disable = getattr(config, 'log_disable', False)
    if log_disable:
        cmd += " --log-disable"

    log_file = getattr(config, 'log_file', None)
    if log_file:
        cmd += f" --log-file {log_file}"

    log_colors = getattr(config, 'log_colors', 'auto')
    if log_colors and log_colors != 'auto':
        cmd += f" --log-colors {log_colors}"

    verbose = getattr(config, 'verbose', False)
    if verbose:
        cmd += " --verbose"

    offline = getattr(config, 'offline', False)
    if offline:
        cmd += " --offline"

    verbosity = getattr(config, 'verbosity', 3)
    if verbosity != 3:
        cmd += f" --verbosity {verbosity}"

    log_prefix = getattr(config, 'log_prefix', True)
    if not log_prefix:
        cmd += " --no-log-prefix"

    log_timestamps = getattr(config, 'log_timestamps', True)
    if not log_timestamps:
        cmd += " --no-log-timestamps"

    # --- Speculative decoding ---
    spec_type = getattr(config, 'spec_type', 'none')
    if spec_type and spec_type != 'none':
        cmd += f" --spec-type {spec_type}"

    draft_n_max = getattr(config, 'draft_n_max', 3)
    if draft_n_max != 3:
        cmd += f" --spec-draft-n-max {draft_n_max}"

    draft_n_min = getattr(config, 'draft_n_min', 0)
    if draft_n_min != 0:
        cmd += f" --spec-draft-n-min {draft_n_min}"

    spec_draft_hf = getattr(config, 'spec_draft_hf', None)
    if spec_draft_hf:
        cmd += f" --spec-draft-hf {spec_draft_hf}"

    spec_draft_threads = getattr(config, 'spec_draft_threads', 0)
    if spec_draft_threads != 0:
        cmd += f" --spec-draft-threads {spec_draft_threads}"

    spec_draft_threads_batch = getattr(config, 'spec_draft_threads_batch', -1)
    if spec_draft_threads_batch != -1:
        cmd += f" --spec-draft-threads-batch {spec_draft_threads_batch}"

    spec_draft_override_tensor = getattr(config, 'spec_draft_override_tensor', None)
    if spec_draft_override_tensor:
        cmd += f" --spec-draft-override-tensor {spec_draft_override_tensor}"

    spec_draft_device = getattr(config, 'spec_draft_device', None)
    if spec_draft_device:
        cmd += f" --spec-draft-device {spec_draft_device}"

    spec_draft_type_k = getattr(config, 'spec_draft_type_k', 'f16')
    if spec_draft_type_k and spec_draft_type_k != 'f16':
        cmd += f" --spec-draft-type-k {spec_draft_type_k}"

    spec_draft_type_v = getattr(config, 'spec_draft_type_v', 'f16')
    if spec_draft_type_v and spec_draft_type_v != 'f16':
        cmd += f" --spec-draft-type-v {spec_draft_type_v}"

    spec_draft_model = getattr(config, 'spec_draft_model', None)
    if spec_draft_model:
        cmd += f" --spec-draft-model {spec_draft_model}"

    # --- Sampling ---
    temp = getattr(config, 'temp', 0.8)
    if temp != 0.8:
        cmd += f" --temp {temp}"

    top_p = getattr(config, 'top_p', 0.95)
    if top_p != 0.95:
        cmd += f" --top-p {top_p}"

    top_k = getattr(config, 'top_k', 40)
    if top_k != 40:
        cmd += f" --top-k {top_k}"

    min_p = getattr(config, 'min_p', 0.05)
    if min_p != 0.05:
        cmd += f" --min-p {min_p}"

    repeat_last_n = getattr(config, 'repeat_last_n', 64)
    if repeat_last_n != 64:
        cmd += f" --repeat-last-n {repeat_last_n}"

    repeat_penalty = getattr(config, 'repeat_penalty', 1.0)
    if repeat_penalty != 1.0:
        cmd += f" --repeat-penalty {repeat_penalty}"

    presence_penalty = getattr(config, 'presence_penalty', 0.0)
    if presence_penalty != 0.0:
        cmd += f" --presence-penalty {presence_penalty}"

    frequency_penalty = getattr(config, 'frequency_penalty', 0.0)
    if frequency_penalty != 0.0:
        cmd += f" --frequency-penalty {frequency_penalty}"

    dry_multiplier = getattr(config, 'dry_multiplier', 0.0)
    if dry_multiplier != 0.0:
        cmd += f" --dry-mul {dry_multiplier}"

    dry_base = getattr(config, 'dry_base', 1.75)
    if dry_base != 1.75:
        cmd += f" --dry-base {dry_base}"

    dry_allowed_length = getattr(config, 'dry_allowed_length', 2)
    if dry_allowed_length != 2:
        cmd += f" --dry-allowed-length {dry_allowed_length}"

    dry_penalty_last_n = getattr(config, 'dry_penalty_last_n', -1)
    if dry_penalty_last_n != -1:
        cmd += f" --dry-penalty-last-n {dry_penalty_last_n}"

    dry_sequence_breaker = getattr(config, 'dry_sequence_breaker', None)
    if dry_sequence_breaker:
        cmd += f" --dry-sequence-breaker \"{dry_sequence_breaker}\""

    mirostat = getattr(config, 'mirostat', 0)
    if mirostat != 0:
        cmd += f" --mirostat {mirostat}"

    mirostat_lr = getattr(config, 'mirostat_lr', 0.1)
    if mirostat_lr != 0.1:
        cmd += f" --mirostat-lr {mirostat_lr}"

    mirostat_ent = getattr(config, 'mirostat_ent', 5.0)
    if mirostat_ent != 5.0:
        cmd += f" --mirostat-ent {mirostat_ent}"

    grammar = getattr(config, 'grammar', None)
    if grammar:
        cmd += f" --grammar \"{grammar}\""

    grammar_file = getattr(config, 'grammar_file', None)
    if grammar_file:
        cmd += f" --grammar-file {grammar_file}"

    json_schema = getattr(config, 'json_schema', None)
    if json_schema:
        cmd += f" --json-schema \"{json_schema}\""

    json_schema_file = getattr(config, 'json_schema_file', None)
    if json_schema_file:
        cmd += f" --json-schema-file {json_schema_file}"

    backend_sampling = getattr(config, 'backend_sampling', False)
    if backend_sampling:
        cmd += " --backend-sampling"

    ignore_eos = getattr(config, 'ignore_eos', False)
    if ignore_eos:
        cmd += " --ignore-eos"

    # --- Server & UI ---
    parallel = getattr(config, 'parallel', -1)
    if parallel != -1 and parallel > 0:
        cmd += f" --parallel {parallel}"

    jinja = getattr(config, 'jinja', False)
    if jinja:
        cmd += " --jinja"

    cont_batching = getattr(config, 'cont_batching', True)
    if not cont_batching:
        cmd += " --no-cont-batching"

    embeddings = getattr(config, 'embeddings', False)
    if embeddings:
        cmd += " --embeddings"

    reranking = getattr(config, 'reranking', False)
    if reranking:
        cmd += " --reranking"

    ui = getattr(config, 'ui', True)
    if not ui:
        cmd += " --no-ui"

    api_prefix = getattr(config, 'api_prefix', None)
    if api_prefix:
        cmd += f" --api-prefix {api_prefix}"

    ui_config = getattr(config, 'ui_config', None)
    if ui_config:
        cmd += f" --ui-config \"{ui_config}\""

    ui_config_file = getattr(config, 'ui_config_file', None)
    if ui_config_file:
        cmd += f" --ui-config-file {ui_config_file}"

    tools = getattr(config, 'tools', None)
    if tools:
        cmd += f" --tools \"{tools}\""

    agent = getattr(config, 'agent', False)
    if agent:
        cmd += " --agent"

    # --- Backend ---
    backend = getattr(config, 'backend', 'auto')
    if backend and backend != 'auto':
        cmd += f" --backend {backend}"

    # --- Example-specific ---
    reverse_prompt = getattr(config, 'reverse_prompt', None)
    if reverse_prompt:
        cmd += f" --reverse-prompt \"{reverse_prompt}\""

    warmup = getattr(config, 'warmup', True)
    if not warmup:
        cmd += " --no-warmup"

    pooling = getattr(config, 'pooling', 'none')
    if pooling and pooling != 'none':
        cmd += f" --pooling {pooling}"

    lookup_cache_static = getattr(config, 'lookup_cache_static', None)
    if lookup_cache_static:
        cmd += f" --lookup-cache-static {lookup_cache_static}"

    lookup_cache_dynamic = getattr(config, 'lookup_cache_dynamic', None)
    if lookup_cache_dynamic:
        cmd += f" --lookup-cache-dynamic {lookup_cache_dynamic}"

    # --- Extra sampling ---
    samplers = getattr(config, 'samplers', "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature")
    if samplers and samplers != "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature":
        cmd += f" --samplers \"{samplers}\""

    sampler_seq = getattr(config, 'sampler_seq', "edskypmxt")
    if sampler_seq and sampler_seq != "edskypmxt":
        cmd += f" --sampler-seq \"{sampler_seq}\""

    seed = getattr(config, 'seed', -1)
    if seed != -1:
        cmd += f" --seed {seed}"

    top_n_sigma = getattr(config, 'top_n_sigma', -1.0)
    if top_n_sigma != -1.0:
        cmd += f" --top-n-sigma {top_n_sigma}"

    typical_p = getattr(config, 'typical_p', 1.0)
    if typical_p != 1.0:
        cmd += f" --typical-p {typical_p}"

    xtc_probability = getattr(config, 'xtc_probability', 0.0)
    if xtc_probability != 0.0:
        cmd += f" --xtc-probability {xtc_probability}"

    xtc_threshold = getattr(config, 'xtc_threshold', 0.1)
    if xtc_threshold != 0.1:
        cmd += f" --xtc-threshold {xtc_threshold}"

    logit_bias = getattr(config, 'logit_bias', "")
    if logit_bias:
        cmd += f" --logit-bias \"{logit_bias}\""

    # --- Adaptive & DynaTemp ---
    adaptive_target = getattr(config, 'adaptive_target', -1.0)
    if adaptive_target != -1.0:
        cmd += f" --adaptive-target {adaptive_target}"

    adaptive_decay = getattr(config, 'adaptive_decay', 0.9)
    if adaptive_decay != 0.9:
        cmd += f" --adaptive-decay {adaptive_decay}"

    dynatemp_range = getattr(config, 'dynatemp_range', 0.0)
    if dynatemp_range != 0.0:
        cmd += f" --dynatemp-range {dynatemp_range}"

    dynatemp_exp = getattr(config, 'dynatemp_exp', 1.0)
    if dynatemp_exp != 1.0:
        cmd += f" --dynatemp-exp {dynatemp_exp}"

    # --- Spec draft extras ---
    spec_draft_n_max = getattr(config, 'spec_draft_n_max', 3)
    if spec_draft_n_max != 3:
        cmd += f" --spec-draft-n-max {spec_draft_n_max}"

    spec_draft_n_min = getattr(config, 'spec_draft_n_min', 0)
    if spec_draft_n_min != 0:
        cmd += f" --spec-draft-n-min {spec_draft_n_min}"

    spec_draft_p_split = getattr(config, 'spec_draft_p_split', 0.1)
    if spec_draft_p_split != 0.1:
        cmd += f" --spec-draft-p-split {spec_draft_p_split}"

    spec_draft_p_min = getattr(config, 'spec_draft_p_min', 0.0)
    if spec_draft_p_min != 0.0:
        cmd += f" --spec-draft-p-min {spec_draft_p_min}"

    spec_draft_backend_sampling = getattr(config, 'spec_draft_backend_sampling', True)
    if not spec_draft_backend_sampling:
        cmd += " --no-spec-draft-backend-sampling"

    spec_draft_ngl = getattr(config, 'spec_draft_ngl', -1)
    if spec_draft_ngl != -1:
        cmd += f" --spec-draft-ngl {spec_draft_ngl}"

    # --- Spec draft CPU affinity ---
    spec_draft_cpu_mask = getattr(config, 'spec_draft_cpu_mask', "")
    if spec_draft_cpu_mask:
        cmd += f" --spec-draft-cpu-mask \"{spec_draft_cpu_mask}\""

    spec_draft_cpu_range = getattr(config, 'spec_draft_cpu_range', "")
    if spec_draft_cpu_range:
        cmd += f" --spec-draft-cpu-range \"{spec_draft_cpu_range}\""

    spec_draft_cpu_strict = getattr(config, 'spec_draft_cpu_strict', 0)
    if spec_draft_cpu_strict != 0:
        cmd += f" --spec-draft-cpu-strict {spec_draft_cpu_strict}"

    spec_draft_prio = getattr(config, 'spec_draft_prio', 0)
    if spec_draft_prio != 0:
        cmd += f" --spec-draft-prio {spec_draft_prio}"

    spec_draft_poll = getattr(config, 'spec_draft_poll', 0)
    if spec_draft_poll != 0:
        cmd += f" --spec-draft-poll {spec_draft_poll}"

    spec_draft_cpu_mask_batch = getattr(config, 'spec_draft_cpu_mask_batch', "")
    if spec_draft_cpu_mask_batch:
        cmd += f" --spec-draft-cpu-mask-batch \"{spec_draft_cpu_mask_batch}\""

    spec_draft_cpu_strict_batch = getattr(config, 'spec_draft_cpu_strict_batch', 0)
    if spec_draft_cpu_strict_batch != 0:
        cmd += f" --spec-draft-cpu-strict-batch {spec_draft_cpu_strict_batch}"

    spec_draft_prio_batch = getattr(config, 'spec_draft_prio_batch', 0)
    if spec_draft_prio_batch != 0:
        cmd += f" --spec-draft-prio-batch {spec_draft_prio_batch}"

    spec_draft_poll_batch = getattr(config, 'spec_draft_poll_batch', 0)
    if spec_draft_poll_batch != 0:
        cmd += f" --spec-draft-poll-batch {spec_draft_poll_batch}"

    spec_draft_cpu_moe = getattr(config, 'spec_draft_cpu_moe', False)
    if spec_draft_cpu_moe:
        cmd += " --spec-draft-cpu-moe"

    spec_draft_n_cpu_moe = getattr(config, 'spec_draft_n_cpu_moe', 0)
    if spec_draft_n_cpu_moe != 0:
        cmd += f" --spec-draft-n-cpu-moe {spec_draft_n_cpu_moe}"

    # --- Spec N-gram / mod params ---
    spec_ngram_mod_n_min = getattr(config, 'spec_ngram_mod_n_min', 48)
    if spec_ngram_mod_n_min != 48:
        cmd += f" --spec-ngram-mod-n-min {spec_ngram_mod_n_min}"

    spec_ngram_mod_n_max = getattr(config, 'spec_ngram_mod_n_max', 64)
    if spec_ngram_mod_n_max != 64:
        cmd += f" --spec-ngram-mod-n-max {spec_ngram_mod_n_max}"

    spec_ngram_mod_n_match = getattr(config, 'spec_ngram_mod_n_match', 24)
    if spec_ngram_mod_n_match != 24:
        cmd += f" --spec-ngram-mod-n-match {spec_ngram_mod_n_match}"

    spec_ngram_simple_size_n = getattr(config, 'spec_ngram_simple_size_n', 12)
    if spec_ngram_simple_size_n != 12:
        cmd += f" --spec-ngram-simple-size-n {spec_ngram_simple_size_n}"

    spec_ngram_simple_size_m = getattr(config, 'spec_ngram_simple_size_m', 48)
    if spec_ngram_simple_size_m != 48:
        cmd += f" --spec-ngram-simple-size-m {spec_ngram_simple_size_m}"

    spec_ngram_simple_min_hits = getattr(config, 'spec_ngram_simple_min_hits', 1)
    if spec_ngram_simple_min_hits != 1:
        cmd += f" --spec-ngram-simple-min-hits {spec_ngram_simple_min_hits}"

    spec_ngram_map_k_size_n = getattr(config, 'spec_ngram_map_k_size_n', 12)
    if spec_ngram_map_k_size_n != 12:
        cmd += f" --spec-ngram-map-k-size-n {spec_ngram_map_k_size_n}"

    spec_ngram_map_k_size_m = getattr(config, 'spec_ngram_map_k_size_m', 48)
    if spec_ngram_map_k_size_m != 48:
        cmd += f" --spec-ngram-map-k-size-m {spec_ngram_map_k_size_m}"

    spec_ngram_map_k_min_hits = getattr(config, 'spec_ngram_map_k_min_hits', 1)
    if spec_ngram_map_k_min_hits != 1:
        cmd += f" --spec-ngram-map-k-min-hits {spec_ngram_map_k_min_hits}"

    spec_ngram_map_k4v_size_n = getattr(config, 'spec_ngram_map_k4v_size_n', 12)
    if spec_ngram_map_k4v_size_n != 12:
        cmd += f" --spec-ngram-map-k4v-size-n {spec_ngram_map_k4v_size_n}"

    spec_ngram_map_k4v_size_m = getattr(config, 'spec_ngram_map_k4v_size_m', 48)
    if spec_ngram_map_k4v_size_m != 48:
        cmd += f" --spec-ngram-map-k4v-size-m {spec_ngram_map_k4v_size_m}"

    spec_ngram_map_k4v_min_hits = getattr(config, 'spec_ngram_map_k4v_min_hits', 1)
    if spec_ngram_map_k4v_min_hits != 1:
        cmd += f" --spec-ngram-map-k4v-min-hits {spec_ngram_map_k4v_min_hits}"

    # --- Extra cache ---
    ctx_checkpoints = getattr(config, 'ctx_checkpoints', 32)
    if ctx_checkpoints != 32:
        cmd += f" --ctx-checkpoints {ctx_checkpoints}"

    checkpoint_min_step = getattr(config, 'checkpoint_min_step', 256)
    if checkpoint_min_step != 256:
        cmd += f" --checkpoint-min-step {checkpoint_min_step}"

    cache_ram = getattr(config, 'cache_ram', 8192)
    if cache_ram != 8192:
        cmd += f" --cache-ram {cache_ram}"

    cache_idle_slots = getattr(config, 'cache_idle_slots', True)
    if not cache_idle_slots:
        cmd += " --no-cache-idle-slots"

    context_shift = getattr(config, 'context_shift', False)
    if context_shift:
        cmd += " --context-shift"

    # --- Generation extras ---
    special = getattr(config, 'special', False)
    if special:
        cmd += " --special"

    spm_infill = getattr(config, 'spm_infill', False)
    if spm_infill:
        cmd += " --spm-infill"

    # --- Multimodal ---
    mmproj = getattr(config, 'mmproj', "")
    if mmproj:
        cmd += f" --mmproj {mmproj}"

    mmproj_url = getattr(config, 'mmproj_url', "")
    if mmproj_url:
        cmd += f" --mmproj-url {mmproj_url}"

    mmproj_auto = getattr(config, 'mmproj_auto', True)
    if not mmproj_auto:
        cmd += " --no-mmproj-auto"

    mmproj_offload = getattr(config, 'mmproj_offload', True)
    if not mmproj_offload:
        cmd += " --no-mmproj-offload"

    image_min_tokens = getattr(config, 'image_min_tokens', 0)
    if image_min_tokens != 0:
        cmd += f" --image-min-tokens {image_min_tokens}"

    image_max_tokens = getattr(config, 'image_max_tokens', 0)
    if image_max_tokens != 0:
        cmd += f" --image-max-tokens {image_max_tokens}"

    mtmd_batch_max_tokens = getattr(config, 'mtmd_batch_max_tokens', 1024)
    if mtmd_batch_max_tokens != 1024:
        cmd += f" --mtmd-batch-max-tokens {mtmd_batch_max_tokens}"

    # --- Server extras ---
    alias = getattr(config, 'alias', "")
    if alias:
        cmd += f" --alias {alias}"

    tags = getattr(config, 'tags', "")
    if tags:
        cmd += f" --tags \"{tags}\""

    embd_normalize = getattr(config, 'embd_normalize', 2)
    if embd_normalize != 2:
        cmd += f" --embd-normalize {embd_normalize}"

    path = getattr(config, 'path', "")
    if path:
        cmd += f" --path {path}"

    reuse_port = getattr(config, 'reuse_port', False)
    if reuse_port:
        cmd += " --reuse-port"

    ui_mcp_proxy = getattr(config, 'ui_mcp_proxy', False)
    if ui_mcp_proxy:
        cmd += " --ui-mcp-proxy"

    # --- Extra flags ---
    for flag in config.extra_flags:
        cmd += f" {flag}"

    return cmd
