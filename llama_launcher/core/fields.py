"""Declarative field registry for llama-server parameters.

This is the single source of truth that drives:

- :mod:`llama_launcher.core.builder` — command emission
- :mod:`llama_launcher.core.parser` — command import
- :mod:`llama_launcher.core.validator` — range/choice validation
- the generated Advanced Config tab (widget creation)
- the GUI config controller (widget <-> config binding)

Adding or changing a parameter means editing exactly one :class:`Field` here.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Field:
    """Specification for one llama-server parameter."""

    name: str            # LlamaConfig attribute name
    flag: str            # llama-server long flag; "" for metadata-only fields
    kind: str            # "int" | "float" | "bool" | "str" | "choice"
    default: Any         # llama-server default; emission is suppressed at default
    group: str = ""      # advanced-tab group box; "" = not shown in advanced tab
    label: str = ""      # human-readable label
    tooltip: str = ""
    choices: Tuple[str, ...] = ()      # kind="choice"
    minimum: float = 0
    maximum: float = 0
    step: float = 1
    aliases: Tuple[str, ...] = ()      # extra parse aliases (short flags, legacy names)
    negates: str = ""                  # "--no-x" form for bool fields
    placeholder: str = ""
    special_text: str = ""             # spinbox text for the minimum value
    secret: bool = False               # password echo for text fields
    decimals: int = 2


CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
BACKENDS = ("auto", "cuda", "metal", "vulkan", "cpu", "sycl")

GRP_COMMON = "Common Parameters"
GRP_GPU = "GPU Parameters"
GRP_SAMPLING = "Sampling Parameters"
GRP_EXTRA_SAMPLING = "Extra Sampling Parameters"
GRP_ADAPTIVE = "Adaptive & DynaTemp"
GRP_MIROSTAT = "Mirostat Sampling"
GRP_DRY = "DRY (Anti-Repetition) Controls"
GRP_SPEC = "Speculative Decoding Parameters"
GRP_SPEC_CACHE = "Spec Draft Cache Types"
GRP_SPEC_CPU = "Spec Draft CPU Affinity"
GRP_NGRAM = "Speculative N-Gram / Mod Parameters"
GRP_GRAMMAR = "Grammar & Schema Constraints"
GRP_THREADS = "Threads & Batching Parameters"
GRP_CONTEXT = "Context & Predict Parameters"
GRP_YARN = "YaRN RoPE Extension Parameters"
GRP_CACHE = "Cache & Memory Parameters"
GRP_EXTRA_CACHE = "Extra Cache & Context"
GRP_NUMA = "GPU & NUMA Parameters"
GRP_LORA = "LoRA & Control Vectors Parameters"
GRP_SOURCES = "Model Sources Parameters"
GRP_LOGGING = "Logging Parameters"
GRP_SERVER = "Server Options"
GRP_GEN = "Generation & Cache Options"
GRP_MM = "Multimodal Parameters"
GRP_SERVER_EXTRA = "Server & Embeddings Extras"

#: Advanced-tab group display order.
GROUP_ORDER: Tuple[str, ...] = (
    GRP_COMMON, GRP_GPU, GRP_SAMPLING, GRP_EXTRA_SAMPLING, GRP_ADAPTIVE,
    GRP_MIROSTAT, GRP_DRY, GRP_SPEC, GRP_SPEC_CACHE, GRP_SPEC_CPU, GRP_NGRAM,
    GRP_GRAMMAR, GRP_THREADS, GRP_CONTEXT, GRP_YARN, GRP_CACHE,
    GRP_EXTRA_CACHE, GRP_NUMA, GRP_LORA, GRP_SOURCES, GRP_LOGGING,
    GRP_SERVER, GRP_GEN, GRP_MM, GRP_SERVER_EXTRA,
)


FIELDS: Tuple[Field, ...] = (
    # ------------------------------------------------------------------
    # Basic-tab only fields (never shown in the advanced tab)
    # ------------------------------------------------------------------
    Field("model_path", "--model", "str", "", aliases=("-m",),
          label="Main Model", tooltip="Path to the main model file (.gguf, .bin, .safetensors)."),
    Field("mmproj_model_path", "--mmproj", "str", "", aliases=("--mmproj-model",),
          label="mmproj Model", tooltip="Path to the mmproj model for vision-language models."),
    Field("draft_model_path", "--draft-model", "str", "", aliases=("--draft",),
          label="Draft Model", tooltip="Path to a draft model for speculative decoding."),
    Field("port", "--port", "int", 8080, minimum=1, maximum=65535, aliases=("-p",),
          label="Port", tooltip="Port for llama-server to listen on. Default: 8080."),
    Field("host", "--host", "str", "127.0.0.1",
          label="Host", tooltip="Host address for llama-server to bind to. Default: 127.0.0.1."),

    # ------------------------------------------------------------------
    # Common Parameters
    # ------------------------------------------------------------------
    Field("threads", "--threads", "int", 0, group=GRP_COMMON, minimum=0, maximum=256,
          aliases=("-t",), special_text="auto", label="Threads",
          tooltip="CPU threads for generation. 0 = auto-detect (physical cores)."),
    Field("threads_batch", "--threads-batch", "int", -1, group=GRP_COMMON, minimum=-1, maximum=256,
          aliases=("-tb",), special_text="same as threads", label="Threads Batch",
          tooltip="Threads for batch/prompt processing. -1 = same as --threads."),
    Field("context_size", "--ctx-size", "int", 4096, group=GRP_COMMON, minimum=256, maximum=1048576,
          aliases=("-c",), label="Context Size",
          tooltip="Prompt context size in tokens. Default: 4096."),
    Field("predict", "--n-predict", "int", -1, group=GRP_COMMON, minimum=-1, maximum=65536,
          aliases=("-n", "--predict"), special_text="infinite", label="Predict",
          tooltip="Tokens to predict. -1 = generate until stop token or context limit."),
    Field("batch_size", "--batch-size", "int", 2048, group=GRP_COMMON, minimum=1, maximum=65536,
          aliases=("-b",), label="Batch Size",
          tooltip="Logical maximum batch size. Default: 2048."),
    Field("ubatch_size", "--ubatch-size", "int", 512, group=GRP_COMMON, minimum=1, maximum=4096,
          aliases=("-ub",), label="UBatch Size",
          tooltip="Physical maximum batch size for internal batching. Default: 512."),
    Field("keep", "--keep", "int", 0, group=GRP_COMMON, minimum=-1, maximum=131072,
          special_text="keep all", label="Keep",
          tooltip="Tokens to keep from the initial prompt. 0 = none, -1 = all."),
    Field("rope_scaling", "--rope-scaling", "choice", "none", group=GRP_COMMON,
          choices=("none", "linear", "yarn"), label="RoPE Scaling",
          tooltip="RoPE frequency scaling method for extending context."),
    Field("rope_scale", "--rope-scale", "int", 1, group=GRP_COMMON, minimum=1, maximum=100,
          label="RoPE Scale", tooltip="RoPE context scaling factor N. Default: 1."),
    Field("rope_freq_base", "--rope-freq-base", "int", 0, group=GRP_COMMON, minimum=0, maximum=10000000,
          special_text="from model", label="RoPE Freq Base",
          tooltip="RoPE base frequency (NTK-aware scaling). 0 = load from model."),
    Field("rope_freq_scale", "--rope-freq-scale", "int", 0, group=GRP_COMMON, minimum=0, maximum=100,
          special_text="from model", label="RoPE Freq Scale",
          tooltip="RoPE frequency scaling factor. 0 = load from model."),
    Field("cache_type_k", "--cache-type-k", "choice", "f16", group=GRP_COMMON,
          choices=CACHE_TYPES, aliases=("-ctk",), label="Cache Type K",
          tooltip="KV cache data type for K. Default: f16."),
    Field("cache_type_v", "--cache-type-v", "choice", "f16", group=GRP_COMMON,
          choices=CACHE_TYPES, aliases=("-ctv",), label="Cache Type V",
          tooltip="KV cache data type for V. Default: f16."),
    Field("use_mmap", "--mmap", "bool", True, group=GRP_COMMON, negates="--no-mmap",
          label="Memory-map model (--mmap)",
          tooltip="Memory-map the model file. Disable for slower load but fewer pageouts."),
    Field("mlock", "--mlock", "bool", False, group=GRP_COMMON, negates="--no-mlock",
          label="Lock model in RAM (--mlock)",
          tooltip="Keep the model in RAM rather than swapping. Default: disabled."),

    # ------------------------------------------------------------------
    # GPU Parameters
    # ------------------------------------------------------------------
    Field("gpu_layers", "--gpu-layers", "int", -1, group=GRP_GPU, minimum=-1, maximum=999,
          aliases=("-ngl", "--n-gpu-layers"), special_text="all", label="GPU Layers",
          tooltip="Max layers to store in VRAM. -1 = all layers to GPU."),
    Field("split_mode", "--split-mode", "choice", "layer", group=GRP_GPU,
          choices=("none", "layer", "row", "tensor"), aliases=("-sm",), label="Split Mode",
          tooltip="How to split the model across multiple GPUs. Default: layer."),
    Field("tensor_split", "--tensor-split", "str", "", group=GRP_GPU, aliases=("-ts",),
          label="Tensor Split", placeholder="e.g. 3,1",
          tooltip="Fraction of model on each GPU, comma-separated. e.g. '3,1' = 75%/25%."),
    Field("main_gpu", "--main-gpu", "int", 0, group=GRP_GPU, minimum=0, maximum=256,
          aliases=("-mg",), label="Main GPU",
          tooltip="GPU used for the model / intermediate results. Default: 0."),

    # ------------------------------------------------------------------
    # Sampling Parameters
    # ------------------------------------------------------------------
    Field("temp", "--temp", "float", 0.8, group=GRP_SAMPLING, minimum=0.0, maximum=2.0, step=0.05,
          aliases=("--temperature",), label="Temperature",
          tooltip="Randomness: lower = deterministic, higher = creative. Default: 0.8."),
    Field("top_k", "--top-k", "int", 40, group=GRP_SAMPLING, minimum=0, maximum=128,
          special_text="disabled", label="Top K",
          tooltip="Top-K sampling. 0 = disabled. Default: 40."),
    Field("top_p", "--top-p", "float", 0.95, group=GRP_SAMPLING, minimum=0.0, maximum=1.0, step=0.01,
          label="Top P", tooltip="Nucleus sampling. 1.0 = disabled. Default: 0.95."),
    Field("min_p", "--min-p", "float", 0.05, group=GRP_SAMPLING, minimum=0.0, maximum=1.0, step=0.01,
          label="Min P", tooltip="Min-P sampling. 0.0 = disabled. Default: 0.05."),
    Field("repeat_last_n", "--repeat-last-n", "int", 64, group=GRP_SAMPLING, minimum=-1, maximum=131072,
          special_text="ctx size", label="Repeat Last N",
          tooltip="Tokens considered for repeat penalty. 0 = off, -1 = context size. Default: 64."),
    Field("repeat_penalty", "--repeat-penalty", "float", 1.0, group=GRP_SAMPLING,
          minimum=0.0, maximum=2.0, step=0.05, label="Repeat Penalty",
          tooltip="Penalize repeated token sequences. 1.0 = disabled."),
    Field("presence_penalty", "--presence-penalty", "float", 0.0, group=GRP_SAMPLING,
          minimum=-2.0, maximum=2.0, step=0.05, label="Presence Penalty",
          tooltip="Penalize tokens already present in the text. Default: 0.0."),
    Field("frequency_penalty", "--frequency-penalty", "float", 0.0, group=GRP_SAMPLING,
          minimum=-2.0, maximum=2.0, step=0.05, label="Frequency Penalty",
          tooltip="Penalize tokens by their frequency in the text. Default: 0.0."),
    Field("ignore_eos", "--ignore-eos", "bool", False, group=GRP_SAMPLING, negates="--no-ignore-eos",
          label="Ignore EOS token", tooltip="Allow generation beyond the end-of-sequence token."),
    Field("backend_sampling", "--backend-sampling", "bool", False, group=GRP_SAMPLING,
          negates="--no-backend-sampling", label="Backend Sampling",
          tooltip="Use backend (GPU) sampling for faster generation."),

    # ------------------------------------------------------------------
    # Extra Sampling Parameters
    # ------------------------------------------------------------------
    Field("samplers", "--samplers", "str", "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature",
          group=GRP_EXTRA_SAMPLING, label="Samplers",
          placeholder="penalties;dry;top_k;top_p;temperature",
          tooltip="Semi-colon separated sampler chain."),
    Field("sampler_seq", "--sampler-seq", "str", "edskypmxt", group=GRP_EXTRA_SAMPLING,
          label="Sampler Seq", placeholder="edskypmxt", tooltip="Sampler sequence shorthand."),
    Field("seed", "--seed", "int", -1, group=GRP_EXTRA_SAMPLING, minimum=-1, maximum=2147483647,
          special_text="random", label="Seed", tooltip="RNG seed. -1 = random."),
    Field("top_n_sigma", "--top-n-sigma", "float", -1.0, group=GRP_EXTRA_SAMPLING,
          minimum=-1.0, maximum=10.0, step=0.1, label="Top N Sigma",
          tooltip="Top-N-Sigma sampling. -1 = disabled."),
    Field("typical_p", "--typical-p", "float", 1.0, group=GRP_EXTRA_SAMPLING,
          minimum=0.0, maximum=1.0, step=0.01, label="Typical P",
          tooltip="Locally typical sampling. 1.0 = disabled."),
    Field("xtc_probability", "--xtc-probability", "float", 0.0, group=GRP_EXTRA_SAMPLING,
          minimum=0.0, maximum=1.0, step=0.01, label="XTC Probability",
          tooltip="XTC turn probability. 0 = disabled."),
    Field("xtc_threshold", "--xtc-threshold", "float", 0.1, group=GRP_EXTRA_SAMPLING,
          minimum=0.0, maximum=1.0, step=0.1, label="XTC Threshold",
          tooltip="XTC token probability threshold. Default: 0.1."),
    Field("logit_bias", "--logit-bias", "str", "", group=GRP_EXTRA_SAMPLING,
          label="Logit Bias", placeholder="15032:1.0,15033:-1.0",
          tooltip="Logit bias for token control."),

    # ------------------------------------------------------------------
    # Adaptive & DynaTemp
    # ------------------------------------------------------------------
    Field("adaptive_target", "--adaptive-target", "float", -1.0, group=GRP_ADAPTIVE,
          minimum=-1.0, maximum=10.0, step=0.1, label="Adaptive Target",
          tooltip="Adaptive temperature target. -1 = disabled."),
    Field("adaptive_decay", "--adaptive-decay", "float", 0.9, group=GRP_ADAPTIVE,
          minimum=0.0, maximum=1.0, step=0.05, label="Adaptive Decay",
          tooltip="Adaptive temperature decay rate. Default: 0.9."),
    Field("dynatemp_range", "--dynatemp-range", "float", 0.0, group=GRP_ADAPTIVE,
          minimum=0.0, maximum=5.0, step=0.1, label="DynaTemp Range",
          tooltip="DynaTemp range. 0 = disabled."),
    Field("dynatemp_exp", "--dynatemp-exp", "float", 1.0, group=GRP_ADAPTIVE,
          minimum=0.0, maximum=10.0, step=0.1, label="DynaTemp Exp",
          tooltip="DynaTemp exponent. Default: 1.0."),

    # ------------------------------------------------------------------
    # Mirostat
    # ------------------------------------------------------------------
    Field("mirostat", "--mirostat", "int", 0, group=GRP_MIROSTAT, minimum=0, maximum=2,
          label="Mirostat", tooltip="0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0."),
    Field("mirostat_lr", "--mirostat-lr", "float", 0.1, group=GRP_MIROSTAT,
          minimum=0.0, maximum=1.0, step=0.01, label="Mirostat LR",
          tooltip="Mirostat learning rate. Default: 0.1."),
    Field("mirostat_ent", "--mirostat-ent", "float", 5.0, group=GRP_MIROSTAT,
          minimum=0.0, maximum=20.0, step=0.1, label="Mirostat Entropy",
          tooltip="Target Mirostat entropy. Default: 5.0."),

    # ------------------------------------------------------------------
    # DRY
    # ------------------------------------------------------------------
    Field("dry_multiplier", "--dry-mul", "float", 0.0, group=GRP_DRY,
          minimum=0.0, maximum=10.0, step=0.05, aliases=("--dry-multiplier",),
          label="DRY Multiplier", tooltip="DRY penalty multiplier. 0 = disabled."),
    Field("dry_base", "--dry-base", "float", 1.75, group=GRP_DRY,
          minimum=0.0, maximum=10.0, step=0.05, label="DRY Base",
          tooltip="DRY penalty base value. Default: 1.75."),
    Field("dry_allowed_length", "--dry-allowed-length", "int", 2, group=GRP_DRY,
          minimum=0, maximum=32, label="DRY Allowed Length",
          tooltip="DRY allowed repetition length. Default: 2."),
    Field("dry_penalty_last_n", "--dry-penalty-last-n", "int", -1, group=GRP_DRY,
          minimum=-1, maximum=131072, special_text="ctx size", label="DRY Penalty Last N",
          tooltip="Tokens to look back for DRY penalty. -1 = context size."),
    Field("dry_sequence_breaker", "--dry-sequence-breaker", "str", "", group=GRP_DRY,
          label="DRY Sequence Breaker", placeholder="\\n",
          tooltip="Sequence breaker token for DRY penalty."),

    # ------------------------------------------------------------------
    # Speculative Decoding
    # ------------------------------------------------------------------
    Field("spec_type", "--spec-type", "choice", "none", group=GRP_SPEC,
          choices=("none", "ngram", "mod", "simple", "map_k", "map_k4v"),
          label="Spec Type", tooltip="Speculative decoding type. Default: none."),
    Field("spec_draft_model", "--spec-draft-model", "str", "", group=GRP_SPEC,
          label="Spec Draft Model", placeholder="/path/to/draft-model.gguf",
          tooltip="Path to the speculative draft model."),
    Field("spec_draft_n_max", "--spec-draft-n-max", "int", 3, group=GRP_SPEC,
          minimum=1, maximum=32, label="Draft N Max",
          tooltip="Max tokens to draft per iteration. Default: 3."),
    Field("spec_draft_n_min", "--spec-draft-n-min", "int", 0, group=GRP_SPEC,
          minimum=0, maximum=32, label="Draft N Min",
          tooltip="Min draft tokens to use. 0 = no minimum."),
    Field("spec_draft_threads", "--spec-draft-threads", "int", 0, group=GRP_SPEC,
          minimum=0, maximum=256, special_text="auto", label="Spec Draft Threads",
          tooltip="Threads for the draft model. 0 = auto."),
    Field("spec_draft_threads_batch", "--spec-draft-threads-batch", "int", -1, group=GRP_SPEC,
          minimum=-1, maximum=256, special_text="same as draft", label="Spec Draft Threads Batch",
          tooltip="Threads for draft batch processing. -1 = same as draft threads."),
    Field("spec_draft_hf", "--spec-draft-hf", "str", "", group=GRP_SPEC,
          label="Spec Draft HF Repo", placeholder="org/model-name",
          tooltip="Hugging Face repo for the draft model."),
    Field("spec_draft_override_tensor", "--spec-draft-override-tensor", "str", "", group=GRP_SPEC,
          label="Spec Draft Override Tensor", placeholder="KQV=1,O=1",
          tooltip="Override tensor split for the draft model."),
    Field("spec_draft_device", "--spec-draft-device", "str", "", group=GRP_SPEC,
          label="Spec Draft Device", placeholder="cuda:0,cpu",
          tooltip="Device for the draft model."),
    Field("spec_draft_p_split", "--spec-draft-p-split", "float", 0.1, group=GRP_SPEC,
          minimum=0.0, maximum=1.0, step=0.05, label="Spec Draft P Split",
          tooltip="Spec draft p-split threshold. Default: 0.1."),
    Field("spec_draft_p_min", "--spec-draft-p-min", "float", 0.0, group=GRP_SPEC,
          minimum=0.0, maximum=1.0, step=0.05, label="Spec Draft P Min",
          tooltip="Spec draft p-min threshold. Default: 0.0."),
    Field("spec_draft_backend_sampling", "--spec-draft-backend-sampling", "bool", True,
          group=GRP_SPEC, negates="--no-spec-draft-backend-sampling",
          label="Spec Draft Backend Sampling",
          tooltip="Use backend sampling for the spec draft. Default: enabled."),
    Field("spec_draft_ngl", "--spec-draft-ngl", "int", -1, group=GRP_SPEC,
          minimum=-1, maximum=999, special_text="all", label="Spec Draft NGL",
          tooltip="GPU layers for the draft model. -1 = all."),

    # ------------------------------------------------------------------
    # Spec Draft Cache Types
    # ------------------------------------------------------------------
    Field("spec_draft_type_k", "--spec-draft-type-k", "choice", "f16", group=GRP_SPEC_CACHE,
          choices=CACHE_TYPES, label="Spec Draft Type K",
          tooltip="KV cache type for K in spec draft. Default: f16."),
    Field("spec_draft_type_v", "--spec-draft-type-v", "choice", "f16", group=GRP_SPEC_CACHE,
          choices=CACHE_TYPES, label="Spec Draft Type V",
          tooltip="KV cache type for V in spec draft. Default: f16."),

    # ------------------------------------------------------------------
    # Spec Draft CPU Affinity
    # ------------------------------------------------------------------
    Field("spec_draft_cpu_mask", "--spec-draft-cpu-mask", "str", "", group=GRP_SPEC_CPU,
          label="CPU Mask", placeholder="0x1,0x2", tooltip="CPU affinity mask for spec draft."),
    Field("spec_draft_cpu_range", "--spec-draft-cpu-range", "str", "", group=GRP_SPEC_CPU,
          label="CPU Range", placeholder="0-3,4-7", tooltip="CPU range for spec draft."),
    Field("spec_draft_cpu_strict", "--spec-draft-cpu-strict", "int", 0, group=GRP_SPEC_CPU,
          minimum=0, maximum=1, label="CPU Strict", tooltip="Strict CPU affinity for spec draft."),
    Field("spec_draft_prio", "--spec-draft-prio", "int", 0, group=GRP_SPEC_CPU,
          minimum=-2, maximum=2, label="Prio", tooltip="Thread priority for spec draft."),
    Field("spec_draft_poll", "--spec-draft-poll", "int", 0, group=GRP_SPEC_CPU,
          minimum=0, maximum=1000, label="Poll", tooltip="Polling interval (ms) for spec draft."),
    Field("spec_draft_cpu_mask_batch", "--spec-draft-cpu-mask-batch", "str", "", group=GRP_SPEC_CPU,
          label="CPU Mask Batch", placeholder="0x1,0x2",
          tooltip="CPU mask for spec draft batch."),
    Field("spec_draft_cpu_strict_batch", "--spec-draft-cpu-strict-batch", "int", 0,
          group=GRP_SPEC_CPU, minimum=0, maximum=1, label="CPU Strict Batch",
          tooltip="Strict CPU affinity for spec draft batch."),
    Field("spec_draft_prio_batch", "--spec-draft-prio-batch", "int", 0, group=GRP_SPEC_CPU,
          minimum=-2, maximum=2, label="Prio Batch",
          tooltip="Thread priority for spec draft batch."),
    Field("spec_draft_poll_batch", "--spec-draft-poll-batch", "int", 0, group=GRP_SPEC_CPU,
          minimum=0, maximum=1000, label="Poll Batch",
          tooltip="Polling interval (ms) for spec draft batch."),
    Field("spec_draft_cpu_moe", "--spec-draft-cpu-moe", "bool", False, group=GRP_SPEC_CPU,
          negates="--no-spec-draft-cpu-moe", label="CPU MoE",
          tooltip="Offload spec draft MoE layers to CPU."),
    Field("spec_draft_n_cpu_moe", "--spec-draft-n-cpu-moe", "int", 0, group=GRP_SPEC_CPU,
          minimum=0, maximum=256, special_text="auto", label="N CPU MoE",
          tooltip="CPU threads for spec draft MoE. 0 = auto."),

    # ------------------------------------------------------------------
    # Speculative N-Gram / Mod
    # ------------------------------------------------------------------
    Field("spec_ngram_mod_n_min", "--spec-ngram-mod-n-min", "int", 48, group=GRP_NGRAM,
          minimum=1, maximum=128, label="N-Gram Mod N Min", tooltip="Min n-gram mod sequence length."),
    Field("spec_ngram_mod_n_max", "--spec-ngram-mod-n-max", "int", 64, group=GRP_NGRAM,
          minimum=1, maximum=128, label="N-Gram Mod N Max", tooltip="Max n-gram mod sequence length."),
    Field("spec_ngram_mod_n_match", "--spec-ngram-mod-n-match", "int", 24, group=GRP_NGRAM,
          minimum=1, maximum=128, label="N-Gram Mod N Match", tooltip="N-gram mod match length."),
    Field("spec_ngram_simple_size_n", "--spec-ngram-simple-size-n", "int", 12, group=GRP_NGRAM,
          minimum=1, maximum=128, label="Simple Size N", tooltip="Simple n-gram size N."),
    Field("spec_ngram_simple_size_m", "--spec-ngram-simple-size-m", "int", 48, group=GRP_NGRAM,
          minimum=1, maximum=128, label="Simple Size M", tooltip="Simple n-gram size M."),
    Field("spec_ngram_simple_min_hits", "--spec-ngram-simple-min-hits", "int", 1, group=GRP_NGRAM,
          minimum=0, maximum=32, label="Simple Min Hits", tooltip="Simple n-gram min hits."),
    Field("spec_ngram_map_k_size_n", "--spec-ngram-map-k-size-n", "int", 12, group=GRP_NGRAM,
          minimum=1, maximum=128, label="Map K Size N", tooltip="Map K size N."),
    Field("spec_ngram_map_k_size_m", "--spec-ngram-map-k-size-m", "int", 48, group=GRP_NGRAM,
          minimum=1, maximum=128, label="Map K Size M", tooltip="Map K size M."),
    Field("spec_ngram_map_k_min_hits", "--spec-ngram-map-k-min-hits", "int", 1, group=GRP_NGRAM,
          minimum=0, maximum=32, label="Map K Min Hits", tooltip="Map K min hits."),
    Field("spec_ngram_map_k4v_size_n", "--spec-ngram-map-k4v-size-n", "int", 12, group=GRP_NGRAM,
          minimum=1, maximum=128, label="Map K4V Size N", tooltip="Map K4V size N."),
    Field("spec_ngram_map_k4v_size_m", "--spec-ngram-map-k4v-size-m", "int", 48, group=GRP_NGRAM,
          minimum=1, maximum=128, label="Map K4V Size M", tooltip="Map K4V size M."),
    Field("spec_ngram_map_k4v_min_hits", "--spec-ngram-map-k4v-min-hits", "int", 1, group=GRP_NGRAM,
          minimum=0, maximum=32, label="Map K4V Min Hits", tooltip="Map K4V min hits."),

    # ------------------------------------------------------------------
    # Grammar & Schema
    # ------------------------------------------------------------------
    Field("grammar", "--grammar", "str", "", group=GRP_GRAMMAR, label="Grammar",
          placeholder="root ::= [a-z]+", tooltip="GBNF grammar string for constrained generation."),
    Field("grammar_file", "--grammar-file", "str", "", group=GRP_GRAMMAR, label="Grammar File",
          placeholder="/path/to/grammar.gbnf", tooltip="Path to a GBNF grammar file."),
    Field("json_schema", "--json-schema", "str", "", group=GRP_GRAMMAR, label="JSON Schema",
          placeholder='{"type": "object"}', tooltip="JSON schema string for constrained generation."),
    Field("json_schema_file", "--json-schema-file", "str", "", group=GRP_GRAMMAR,
          label="JSON Schema File", placeholder="/path/to/schema.json",
          tooltip="Path to a JSON schema file."),

    # ------------------------------------------------------------------
    # Threads & Batching (CPU affinity)
    # ------------------------------------------------------------------
    Field("cpu_mask", "--cpu-mask", "str", "", group=GRP_THREADS, label="CPU Mask",
          placeholder="0x1,0x2", tooltip="CPU affinity mask for the main thread."),
    Field("cpu_range", "--cpu-range", "str", "", group=GRP_THREADS, label="CPU Range",
          placeholder="0-3,4-7", tooltip="CPU range for the main thread."),
    Field("cpu_strict", "--cpu-strict", "int", 0, group=GRP_THREADS, minimum=0, maximum=1,
          label="CPU Strict", tooltip="Strict CPU affinity for the main thread."),
    Field("prio", "--prio", "int", 0, group=GRP_THREADS, minimum=-2, maximum=2,
          label="Prio", tooltip="Thread priority: -2 lowest .. 2 highest. Default: 0."),
    Field("poll", "--poll", "int", 50, group=GRP_THREADS, minimum=0, maximum=1000,
          label="Poll", tooltip="Polling interval (ms) for the main thread. Default: 50."),
    Field("cpu_mask_batch", "--cpu-mask-batch", "str", "", group=GRP_THREADS,
          label="CPU Mask Batch", placeholder="0x1,0x2", tooltip="CPU mask for the batch thread."),
    Field("cpu_range_batch", "--cpu-range-batch", "str", "", group=GRP_THREADS,
          label="CPU Range Batch", placeholder="0-3,4-7", tooltip="CPU range for the batch thread."),
    Field("cpu_strict_batch", "--cpu-strict-batch", "int", 0, group=GRP_THREADS,
          minimum=0, maximum=1, label="CPU Strict Batch",
          tooltip="Strict CPU affinity for the batch thread."),
    Field("prio_batch", "--prio-batch", "int", 0, group=GRP_THREADS, minimum=-2, maximum=2,
          label="Prio Batch", tooltip="Thread priority for the batch thread. Default: 0."),
    Field("poll_batch", "--poll-batch", "int", 0, group=GRP_THREADS, minimum=0, maximum=1000,
          label="Poll Batch", tooltip="Polling interval (ms) for the batch thread. 0 = inherit."),

    # ------------------------------------------------------------------
    # Context & Predict extras
    # ------------------------------------------------------------------
    Field("swa_full", "--swa-full", "bool", False, group=GRP_CONTEXT, negates="--no-swa-full",
          label="Full Sliding Window Attention", tooltip="Enable full SWA. Default: disabled."),
    Field("flash_attn", "--flash-attn", "choice", "auto", group=GRP_CONTEXT,
          choices=("auto", "on", "off"), aliases=("-fa",), label="Flash Attention",
          tooltip="Flash attention mode. Default: auto."),
    Field("perf", "--perf", "bool", False, group=GRP_CONTEXT, negates="--no-perf",
          label="Print Performance Info", tooltip="Print performance information on exit."),
    Field("escape", "--escape", "bool", True, group=GRP_CONTEXT, negates="--no-escape",
          label="Process Escape Sequences", tooltip="Process escape sequences in input. Default: enabled."),

    # ------------------------------------------------------------------
    # YaRN
    # ------------------------------------------------------------------
    Field("yarn_orig_ctx", "--yarn-orig-ctx", "int", 0, group=GRP_YARN, minimum=0, maximum=131072,
          special_text="from model", label="YaRN Orig Context",
          tooltip="Original context size for YaRN scaling. 0 = load from model."),
    Field("yarn_ext_factor", "--yarn-ext-factor", "float", -1.0, group=GRP_YARN,
          minimum=-1.0, maximum=100.0, step=0.1, label="YaRN Ext Factor",
          tooltip="YaRN extrapolation mix factor. -1 = auto."),
    Field("yarn_attn_factor", "--yarn-attn-factor", "float", -1.0, group=GRP_YARN,
          minimum=-1.0, maximum=100.0, step=0.1, label="YaRN Attn Factor",
          tooltip="YaRN attention scaling factor. -1 = auto."),
    Field("yarn_beta_slow", "--yarn-beta-slow", "float", -1.0, group=GRP_YARN,
          minimum=-1.0, maximum=100.0, step=0.1, label="YaRN Beta Slow",
          tooltip="YaRN beta slow factor. -1 = auto."),
    Field("yarn_beta_fast", "--yarn-beta-fast", "float", -1.0, group=GRP_YARN,
          minimum=-1.0, maximum=100.0, step=0.1, label="YaRN Beta Fast",
          tooltip="YaRN beta fast factor. -1 = auto."),

    # ------------------------------------------------------------------
    # Cache & Memory
    # ------------------------------------------------------------------
    Field("kv_offload", "--kv-offload", "bool", True, group=GRP_CACHE, negates="--no-kv-offload",
          label="KV Offload", tooltip="Offload KV cache to system RAM if VRAM is full. Default: enabled."),
    Field("repack", "--repack", "bool", True, group=GRP_CACHE, negates="--no-repack",
          label="Repack", tooltip="Repack KV cache for memory efficiency. Default: enabled."),
    Field("no_host", "--no-host", "bool", False, group=GRP_CACHE,
          label="No Host RAM", tooltip="Do not keep the model in host RAM. Default: disabled."),
    Field("defrag_thold", "--defrag-thold", "int", 0, group=GRP_CACHE, minimum=0, maximum=100,
          special_text="disabled", label="Defrag Threshold",
          tooltip="KV cache defragmentation threshold. 0 = disabled."),
    Field("direct_io", "--direct-io", "bool", False, group=GRP_CACHE, negates="--no-direct-io",
          label="Direct I/O", tooltip="Use direct I/O for model loading. Default: disabled."),

    # ------------------------------------------------------------------
    # Extra Cache & Context
    # ------------------------------------------------------------------
    Field("ctx_checkpoints", "--ctx-checkpoints", "int", 32, group=GRP_EXTRA_CACHE,
          minimum=0, maximum=1024, label="Context Checkpoints",
          tooltip="Number of context checkpoints. Default: 32."),
    Field("checkpoint_min_step", "--checkpoint-min-step", "int", 256, group=GRP_EXTRA_CACHE,
          minimum=0, maximum=4096, label="Checkpoint Min Step",
          tooltip="Minimum checkpoint step size. Default: 256."),
    Field("cache_ram", "--cache-ram", "int", 8192, group=GRP_EXTRA_CACHE,
          minimum=0, maximum=65536, label="Cache RAM (MB)",
          tooltip="Cache RAM size in MB. Default: 8192."),
    Field("cache_idle_slots", "--cache-idle-slots", "bool", True, group=GRP_EXTRA_CACHE,
          negates="--no-cache-idle-slots", label="Cache Idle Slots",
          tooltip="Enable idle slot caching. Default: enabled."),
    Field("context_shift", "--context-shift", "bool", False, group=GRP_EXTRA_CACHE,
          negates="--no-context-shift", label="Context Shift",
          tooltip="Enable context shifting. Default: disabled."),

    # ------------------------------------------------------------------
    # GPU & NUMA
    # ------------------------------------------------------------------
    Field("numa", "--numa", "str", "", group=GRP_NUMA, label="NUMA",
          placeholder="0,1 or prefer-numa-node", tooltip="NUMA allocation policy."),
    Field("device", "--device", "str", "", group=GRP_NUMA, label="Device",
          placeholder="cuda:0,cpu", tooltip="Device to use. Empty = auto-detect."),
    Field("override_tensor", "--override-tensor", "str", "", group=GRP_NUMA,
          label="Override Tensor", placeholder="KQV=1,O=1",
          tooltip="Force specific tensors to GPU/CPU."),
    Field("cpu_moe", "--cpu-moe", "bool", False, group=GRP_NUMA, negates="--no-cpu-moe",
          label="CPU MoE", tooltip="Offload MoE layers to CPU."),
    Field("n_cpu_moe", "--n-cpu-moe", "int", 0, group=GRP_NUMA, minimum=0, maximum=256,
          special_text="auto", label="N CPU MoE", tooltip="CPU threads for MoE layers. 0 = auto."),
    Field("fit", "--fit", "choice", "on", group=GRP_NUMA, choices=("on", "off"),
          label="Fit", tooltip="Fit model to VRAM. Default: on."),
    Field("fit_target", "--fit-target", "str", "", group=GRP_NUMA, label="Fit Target",
          placeholder="80%", tooltip="Target VRAM usage for fit mode."),
    Field("fit_ctx", "--fit-ctx", "int", 4096, group=GRP_NUMA, minimum=512, maximum=131072,
          label="Fit Context", tooltip="Context size for fit mode calculation. Default: 4096."),
    Field("check_tensors", "--check-tensors", "bool", False, group=GRP_NUMA,
          negates="--no-check-tensors", label="Check Tensors",
          tooltip="Check tensor sizes and types. Default: disabled."),
    Field("override_kv", "--override-kv", "str", "", group=GRP_NUMA, label="Override KV",
          placeholder="k_scale=0.5,v_scale=0.5", tooltip="Override KV scale factors."),
    Field("op_offload", "--op-offload", "bool", True, group=GRP_NUMA, negates="--no-op-offload",
          label="Op Offload", tooltip="Offload operations to GPU. Default: enabled."),

    # ------------------------------------------------------------------
    # LoRA & Control Vectors
    # ------------------------------------------------------------------
    Field("lora", "--lora", "str", "", group=GRP_LORA, label="LoRA",
          placeholder="/path/to/lora.gguf", tooltip="LoRA adapter path."),
    Field("lora_scaled", "--lora-scaled", "str", "", group=GRP_LORA, label="LoRA Scaled",
          placeholder="/path/to/lora.gguf=1.5", tooltip="LoRA adapter with scale factor."),
    Field("control_vector", "--control-vector", "str", "", group=GRP_LORA, label="Control Vector",
          placeholder="/path/to/cv.gguf", tooltip="Control vector path."),
    Field("control_vector_scaled", "--control-vector-scaled", "str", "", group=GRP_LORA,
          label="Control Vector Scaled", placeholder="/path/to/cv.gguf=1.5",
          tooltip="Control vector with scale factor."),
    Field("control_vector_layer_range", "--control-vector-layer-range", "str", "", group=GRP_LORA,
          label="Control Vector Layer Range", placeholder="0-20",
          tooltip="Layer range the control vector applies to."),

    # ------------------------------------------------------------------
    # Model Sources
    # ------------------------------------------------------------------
    Field("model_url", "--model-url", "str", "", group=GRP_SOURCES, label="Model URL",
          placeholder="https://example.com/model.gguf", tooltip="Model download URL."),
    Field("docker_repo", "--docker-repo", "str", "", group=GRP_SOURCES, label="Docker Repo",
          placeholder="ghcr.io/ggml-org/llama.cpp", tooltip="Docker repository for the model."),
    Field("hf_repo", "--hf-repo", "str", "", group=GRP_SOURCES, label="HF Repo",
          placeholder="org/model-name", tooltip="Hugging Face repository."),
    Field("hf_file", "--hf-file", "str", "", group=GRP_SOURCES, label="HF File",
          placeholder="model.gguf", tooltip="Hugging Face file name."),
    Field("hf_repo_v", "--hf-repo-v", "str", "", group=GRP_SOURCES, label="HF Repo V",
          placeholder="org/validator-model", tooltip="Hugging Face validator repository."),
    Field("hf_file_v", "--hf-file-v", "str", "", group=GRP_SOURCES, label="HF File V",
          placeholder="validator.gguf", tooltip="Hugging Face validator file name."),
    Field("hf_token", "--hf-token", "str", "", group=GRP_SOURCES, label="HF Token",
          placeholder="hf_...", secret=True, tooltip="Hugging Face access token."),

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    Field("log_disable", "--log-disable", "bool", False, group=GRP_LOGGING,
          label="Disable Logging", tooltip="Disable logging."),
    Field("log_file", "--log-file", "str", "", group=GRP_LOGGING, label="Log File",
          placeholder="/path/to/log.txt", tooltip="Log file path. Empty = stdout."),
    Field("log_colors", "--log-colors", "choice", "auto", group=GRP_LOGGING,
          choices=("auto", "enable", "disable"), label="Log Colors",
          tooltip="Colored log output. Default: auto."),
    Field("verbose", "--verbose", "bool", False, group=GRP_LOGGING, label="Verbose",
          tooltip="Enable verbose logging."),
    Field("offline", "--offline", "bool", False, group=GRP_LOGGING, label="Offline Mode",
          tooltip="Disable network access."),
    Field("verbosity", "--verbosity", "int", 3, group=GRP_LOGGING, minimum=0, maximum=5,
          label="Verbosity", tooltip="0=off 1=error 2=warn 3=info 4=debug 5=trace. Default: 3."),
    Field("log_prefix", "--log-prefix", "bool", True, group=GRP_LOGGING, negates="--no-log-prefix",
          label="Log Prefix", tooltip="Include timestamp/level prefix in logs. Default: enabled."),
    Field("log_timestamps", "--log-timestamps", "bool", True, group=GRP_LOGGING,
          negates="--no-log-timestamps", label="Log Timestamps",
          tooltip="Include timestamps in logs. Default: enabled."),

    # ------------------------------------------------------------------
    # Server Options
    # ------------------------------------------------------------------
    Field("parallel", "--parallel", "int", -1, group=GRP_SERVER, minimum=-1, maximum=128,
          special_text="auto", label="Parallel Slots",
          tooltip="Number of parallel request streams. -1 = auto."),
    Field("backend", "--backend", "choice", "auto", group=GRP_SERVER, choices=BACKENDS,
          label="Backend", tooltip="Compute backend. Default: auto."),
    Field("jinja", "--jinja", "bool", False, group=GRP_SERVER, negates="--no-jinja",
          label="Jinja Templates", tooltip="Use Jinja template for prompt formatting."),
    Field("warmup", "--warmup", "bool", True, group=GRP_SERVER, negates="--no-warmup",
          label="Warmup", tooltip="Warm up the model on startup. Default: enabled."),
    Field("embeddings", "--embeddings", "bool", False, group=GRP_SERVER, negates="--no-embeddings",
          label="Embeddings", tooltip="Return embeddings for prompts."),
    Field("reranking", "--reranking", "bool", False, group=GRP_SERVER, negates="--no-reranking",
          label="Reranking", tooltip="Enable reranking mode."),
    Field("ui", "--ui", "bool", True, group=GRP_SERVER, negates="--no-ui",
          label="Web UI", tooltip="Enable the built-in web UI. Default: enabled."),
    Field("agent", "--agent", "bool", False, group=GRP_SERVER, negates="--no-agent",
          label="Agent Mode", tooltip="Enable agent mode."),
    Field("cont_batching", "--cont-batching", "bool", True, group=GRP_SERVER,
          negates="--no-cont-batching", label="Continuous Batching",
          tooltip="Enable continuous batching. Default: enabled."),
    Field("api_prefix", "--api-prefix", "str", "", group=GRP_SERVER, label="API Prefix",
          placeholder="/v1", tooltip="API endpoint prefix."),
    Field("ui_config", "--ui-config", "str", "", group=GRP_SERVER, label="UI Config",
          placeholder='{"theme": "dark"}', tooltip="UI configuration JSON string."),
    Field("ui_config_file", "--ui-config-file", "str", "", group=GRP_SERVER,
          label="UI Config File", placeholder="/path/to/ui-config.json",
          tooltip="Path to a UI configuration file."),
    Field("tools", "--tools", "str", "", group=GRP_SERVER, label="Tools",
          placeholder="tool1,tool2", tooltip="Comma-separated list of available tools."),

    # ------------------------------------------------------------------
    # Generation & Cache Options
    # ------------------------------------------------------------------
    Field("reverse_prompt", "--reverse-prompt", "str", "", group=GRP_GEN, label="Reverse Prompt",
          placeholder="User:", tooltip="Reverse prompt to stop generation."),
    Field("pooling", "--pooling", "choice", "none", group=GRP_GEN,
          choices=("none", "mean", "cls", "last"), label="Pooling",
          tooltip="Pooling method for embeddings. Default: none."),
    Field("kv_unified", "--kv-unified", "bool", True, group=GRP_GEN, negates="--no-kv-unified",
          label="Unified KV Cache", tooltip="Use a unified KV cache. Default: enabled."),
    Field("lookup_cache_static", "--lookup-cache-static", "str", "", group=GRP_GEN,
          label="Lookup Cache Static", placeholder="/path/to/static-cache",
          tooltip="Static lookup cache path."),
    Field("lookup_cache_dynamic", "--lookup-cache-dynamic", "str", "", group=GRP_GEN,
          label="Lookup Cache Dynamic", placeholder="/path/to/dynamic-cache",
          tooltip="Dynamic lookup cache path."),
    Field("special", "--special", "bool", False, group=GRP_GEN, negates="--no-special",
          label="Special Tokens", tooltip="Render special tokens in output."),
    Field("spm_infill", "--spm-infill", "bool", False, group=GRP_GEN, negates="--no-spm-infill",
          label="SPM Infill", tooltip="Enable SPM fill-in-the-middle mode."),

    # ------------------------------------------------------------------
    # Multimodal
    # ------------------------------------------------------------------
    Field("mmproj_url", "--mmproj-url", "str", "", group=GRP_MM, label="MMProj URL",
          placeholder="https://example.com/mmproj.gguf", tooltip="URL to download the mmproj."),
    Field("mmproj_auto", "--mmproj-auto", "bool", True, group=GRP_MM, negates="--no-mmproj-auto",
          label="MMProj Auto", tooltip="Auto-detect the multimodal projector. Default: enabled."),
    Field("mmproj_offload", "--mmproj-offload", "bool", True, group=GRP_MM,
          negates="--no-mmproj-offload", label="MMProj Offload",
          tooltip="Offload the multimodal projector to GPU. Default: enabled."),
    Field("image_min_tokens", "--image-min-tokens", "int", 0, group=GRP_MM,
          minimum=0, maximum=65536, label="Image Min Tokens",
          tooltip="Minimum tokens for image processing. Default: 0."),
    Field("image_max_tokens", "--image-max-tokens", "int", 0, group=GRP_MM,
          minimum=0, maximum=65536, label="Image Max Tokens",
          tooltip="Maximum tokens for image processing. 0 = unlimited."),
    Field("mtmd_batch_max_tokens", "--mtmd-batch-max-tokens", "int", 1024, group=GRP_MM,
          minimum=0, maximum=65536, label="MTMD Batch Max Tokens",
          tooltip="Maximum tokens per batch for mtmd. Default: 1024."),

    # ------------------------------------------------------------------
    # Server & Embeddings Extras
    # ------------------------------------------------------------------
    Field("alias", "--alias", "str", "", group=GRP_SERVER_EXTRA, label="Alias",
          placeholder="my-model", tooltip="Model alias for the API."),
    Field("tags", "--tags", "str", "", group=GRP_SERVER_EXTRA, label="Tags",
          placeholder="coding,chat", tooltip="Comma-separated tags for the model."),
    Field("embd_normalize", "--embd-normalize", "int", 2, group=GRP_SERVER_EXTRA,
          minimum=0, maximum=2, label="Embedding Normalize",
          tooltip="0 = none, 1 = max absolute, 2 = L2. Default: 2."),
    Field("path", "--path", "str", "", group=GRP_SERVER_EXTRA, label="Path",
          placeholder="/api", tooltip="Custom API path."),
    Field("reuse_port", "--reuse-port", "bool", False, group=GRP_SERVER_EXTRA,
          negates="--no-reuse-port", label="Reuse Port", tooltip="Allow port reuse."),
    Field("ui_mcp_proxy", "--ui-mcp-proxy", "bool", False, group=GRP_SERVER_EXTRA,
          negates="--no-ui-mcp-proxy", label="UI MCP Proxy", tooltip="Enable UI MCP proxy."),
)

FIELDS_BY_NAME: Dict[str, Field] = {f.name: f for f in FIELDS}

#: Fields that produce CLI flags, in registry order.
CLI_FIELDS: Tuple[Field, ...] = tuple(f for f in FIELDS if f.flag)

#: flag/alias/negation -> Field, for the parser.
FLAG_TO_FIELD: Dict[str, Field] = {}
for _f in CLI_FIELDS:
    FLAG_TO_FIELD[_f.flag] = _f
    for _a in _f.aliases:
        FLAG_TO_FIELD[_a] = _f
    if _f.negates:
        FLAG_TO_FIELD[_f.negates] = _f


def fields_in_group(group: str) -> List[Field]:
    """Return fields belonging to an advanced-tab group, in registry order."""
    return [f for f in FIELDS if f.group == group]
