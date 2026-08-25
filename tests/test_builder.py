"""Tests for core.builder.build_command."""
import shlex
import pytest
from llama_launcher.core.builder import build_command
from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.parser import parse_launch_command


# ---- Basic command building ----

def test_default_config_emits_only_model():
    cfg = LlamaConfig(model_path="/models/m.gguf")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert tokens == ["llama-server", "--model", "/models/m.gguf"]


def test_non_default_values_are_emitted():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        context_size=8192,
        gpu_layers=33,
        threads=8,
        port=9090,
        host="0.0.0.0",
        batch_size=1024,
        ubatch_size=256,
        cache_type_k="q8_0",
        temp=0.7,
    )
    tokens = shlex.split(build_command(cfg))
    assert "--ctx-size" in tokens and tokens[tokens.index("--ctx-size") + 1] == "8192"
    assert "--gpu-layers" in tokens and tokens[tokens.index("--gpu-layers") + 1] == "33"
    assert "--threads" in tokens and tokens[tokens.index("--threads") + 1] == "8"
    assert "--port" in tokens and tokens[tokens.index("--port") + 1] == "9090"
    assert "--host" in tokens and tokens[tokens.index("--host") + 1] == "0.0.0.0"
    assert "--batch-size" in tokens and tokens[tokens.index("--batch-size") + 1] == "1024"
    assert "--ubatch-size" in tokens and tokens[tokens.index("--ubatch-size") + 1] == "256"
    assert "--cache-type-k" in tokens and tokens[tokens.index("--cache-type-k") + 1] == "q8_0"
    assert "--temp" in tokens and tokens[tokens.index("--temp") + 1] == "0.7"


def test_paths_with_spaces_are_quoted_and_roundtrip():
    cfg = LlamaConfig(model_path="/my models/some model.gguf")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert tokens == ["llama-server", "--model", "/my models/some model.gguf"]


def test_invalid_legacy_flags_are_never_emitted():
    cfg = LlamaConfig(model_path="/m.gguf", cache_type_k="q8_0", cache_type_v="q8_0")
    cmd = build_command(cfg)
    assert "--kv-cache-type" not in cmd
    assert "--logical-batch-size" not in cmd
    assert cmd.count("--cache-type-k") == 1
    assert cmd.count("--cache-type-v") == 1


def test_mmproj_emitted_exactly_once():
    cfg = LlamaConfig(model_path="/m.gguf", mmproj_model_path="/a.gguf")
    tokens = shlex.split(build_command(cfg))
    assert tokens.count("--mmproj") == 1
    assert tokens[tokens.index("--mmproj") + 1] == "/a.gguf"


def test_spec_draft_n_emitted_exactly_once():
    cfg = LlamaConfig(model_path="/m.gguf", spec_draft_n_max=7, spec_draft_n_min=2)
    tokens = shlex.split(build_command(cfg))
    assert tokens.count("--spec-draft-n-max") == 1
    assert tokens[tokens.index("--spec-draft-n-max") + 1] == "7"
    assert tokens.count("--spec-draft-n-min") == 1
    assert tokens[tokens.index("--spec-draft-n-min") + 1] == "2"


def test_boolean_flags():
    cfg = LlamaConfig(model_path="/m.gguf", use_mmap=False, mlock=True, kv_offload=False)
    tokens = shlex.split(build_command(cfg))
    assert "--no-mmap" in tokens
    assert "--mlock" in tokens
    assert "--no-kv-offload" in tokens


def test_extra_flags_appended_verbatim():
    cfg = LlamaConfig(model_path="/m.gguf", extra_flags=["--custom-flag", "42"])
    tokens = shlex.split(build_command(cfg))
    assert tokens[-2:] == ["--custom-flag", "42"]


def test_build_parse_roundtrip():
    cfg = LlamaConfig(
        model_path="/models/model.gguf",
        context_size=16384,
        gpu_layers=40,
        threads=12,
        port=8081,
        temp=0.65,
        top_p=0.9,
        seed=42,
        use_mmap=False,
        cache_type_k="q4_0",
        spec_draft_n_max=6,
    )
    cmd = build_command(cfg)
    params, unrecognized, dups = parse_launch_command(cmd)
    assert unrecognized == []
    assert dups == []
    assert params["model_path"] == cfg.model_path
    assert params["context_size"] == cfg.context_size
    assert params["gpu_layers"] == cfg.gpu_layers
    assert params["threads"] == cfg.threads
    assert params["port"] == cfg.port
    assert params["temp"] == cfg.temp
    assert params["top_p"] == cfg.top_p
    assert params["seed"] == cfg.seed
    assert params["use_mmap"] is False
    assert params["cache_type_k"] == cfg.cache_type_k
    assert params["spec_draft_n_max"] == cfg.spec_draft_n_max


# ---- Boolean field emission ----

def test_default_true_bool_not_emitted_when_true():
    """Default-True bool (use_mmap=True) should NOT appear in command."""
    cfg = LlamaConfig(model_path="/m.gguf", use_mmap=True)
    cmd = build_command(cfg)
    assert "--mmap" not in cmd
    assert "--no-mmap" not in cmd


def test_default_true_bool_emitted_as_negate_when_false():
    cfg = LlamaConfig(model_path="/m.gguf", use_mmap=False)
    cmd = build_command(cfg)
    assert "--no-mmap" in cmd


def test_default_false_bool_emitted_when_true():
    cfg = LlamaConfig(model_path="/m.gguf", mlock=True)
    cmd = build_command(cfg)
    assert "--mlock" in cmd
    assert "--no-mlock" not in cmd


def test_default_false_bool_not_emitted_when_false():
    cfg = LlamaConfig(model_path="/m.gguf", mlock=False)
    cmd = build_command(cfg)
    assert "--mlock" not in cmd
    assert "--no-mlock" not in cmd


def test_all_default_true_bools_not_in_default_command():
    cfg = LlamaConfig(model_path="/m.gguf")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    # These are default-True bools that should NOT appear at defaults
    no_form_flags = ["--no-mmap", "--no-kv-offload", "--no-repack", "--no-escape"]
    for flag in no_form_flags:
        assert flag not in tokens, f"{flag} should not appear at default values"


def test_all_default_false_bools_not_in_default_command():
    cfg = LlamaConfig(model_path="/m.gguf")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    yes_form_flags = ["--mlock", "--verbose", "--ignore-eos"]
    for flag in yes_form_flags:
        assert flag not in tokens, f"{flag} should not appear at default values"


# ---- Int/float emission ----

def test_int_not_emitted_when_at_default():
    cfg = LlamaConfig(model_path="/m.gguf", context_size=4096)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--ctx-size" not in tokens


def test_int_emitted_when_different_from_default():
    cfg = LlamaConfig(model_path="/m.gguf", context_size=8192)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--ctx-size" in tokens
    idx = tokens.index("--ctx-size")
    assert tokens[idx + 1] == "8192"


def test_float_not_emitted_when_at_default():
    cfg = LlamaConfig(model_path="/m.gguf", temp=0.8)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--temp" not in tokens


def test_float_emitted_when_different_from_default():
    cfg = LlamaConfig(model_path="/m.gguf", temp=0.5)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--temp" in tokens
    idx = tokens.index("--temp")
    assert tokens[idx + 1] == "0.5"


def test_zero_int_is_emitted():
    """Zero is not the default for threads (default=0), so should not appear."""
    cfg = LlamaConfig(model_path="/m.gguf", threads=0)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--threads" not in tokens


def test_zero_int_different_from_default_is_emitted():
    # threads default is 0, so 0 is not emitted
    # But poll default is 50, so 0 should be emitted
    cfg = LlamaConfig(model_path="/m.gguf", poll=0)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--poll" in tokens


# ---- Choice field emission ----

def test_choice_not_emitted_when_at_default():
    cfg = LlamaConfig(model_path="/m.gguf", rope_scaling="none")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--rope-scaling" not in tokens


def test_choice_emitted_when_different_from_default():
    cfg = LlamaConfig(model_path="/m.gguf", rope_scaling="yarn")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--rope-scaling" in tokens
    idx = tokens.index("--rope-scaling")
    assert tokens[idx + 1] == "yarn"


def test_choice_backend_emitted():
    cfg = LlamaConfig(model_path="/m.gguf", backend="cuda")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--backend" in tokens
    idx = tokens.index("--backend")
    assert tokens[idx + 1] == "cuda"


# ---- Str field emission ----

def test_str_field_not_emitted_when_empty():
    cfg = LlamaConfig(model_path="/m.gguf", alias="")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--alias" not in tokens


def test_str_field_emitted_when_non_empty():
    cfg = LlamaConfig(model_path="/m.gguf", alias="my-model")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--alias" in tokens
    idx = tokens.index("--alias")
    assert tokens[idx + 1] == "my-model"


def test_str_host_emitted_when_non_default():
    cfg = LlamaConfig(model_path="/m.gguf", host="0.0.0.0")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--host" in tokens
    idx = tokens.index("--host")
    assert tokens[idx + 1] == "0.0.0.0"


def test_str_host_not_emitted_at_default():
    cfg = LlamaConfig(model_path="/m.gguf", host="127.0.0.1")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--host" not in tokens


# ---- Special emit overrides ----

def test_parallel_emitted_when_positive():
    cfg = LlamaConfig(model_path="/m.gguf", parallel=4)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--parallel" in tokens
    idx = tokens.index("--parallel")
    assert tokens[idx + 1] == "4"


def test_parallel_not_emitted_when_zero():
    cfg = LlamaConfig(model_path="/m.gguf", parallel=0)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--parallel" not in tokens


def test_parallel_not_emitted_when_negative():
    cfg = LlamaConfig(model_path="/m.gguf", parallel=-1)
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--parallel" not in tokens


# ---- Extra flags ----

def test_extra_flags_appended_at_end():
    cfg = LlamaConfig(model_path="/m.gguf", extra_flags=["--log-file", "/tmp/log.txt"])
    cmd = build_command(cfg)
    assert "--log-file" in cmd
    assert "/tmp/log.txt" in cmd
    # Should be at end
    tokens = shlex.split(cmd)
    assert tokens[-2:] == ["--log-file", "/tmp/log.txt"]


def test_extra_flags_with_single_value():
    cfg = LlamaConfig(model_path="/m.gguf", extra_flags=["--unknown-flag"])
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--unknown-flag" in tokens


def test_extra_flags_empty_list():
    cfg = LlamaConfig(model_path="/m.gguf", extra_flags=[])
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--unknown-flag" not in cmd


def test_extra_flags_with_special_characters():
    cfg = LlamaConfig(model_path="/m.gguf", extra_flags=["--logit-bias", "15032:1.0,15033:-1.0"])
    cmd = build_command(cfg)
    assert "15032:1.0,15033:-1.0" in cmd


# ---- Command structure ----

def test_command_starts_with_llama_server():
    cfg = LlamaConfig(model_path="/m.gguf")
    cmd = build_command(cfg)
    assert cmd.startswith("llama-server")


def test_command_contains_model_flag():
    cfg = LlamaConfig(model_path="/m.gguf")
    cmd = build_command(cfg)
    assert "--model" in cmd


def test_command_all_model_types():
    """GGUF, BIN, and SAFETENSORS model types should all work."""
    for ext in [".gguf", ".bin", ".safetensors"]:
        cfg = LlamaConfig(model_path=f"/model{ext}")
        cmd = build_command(cfg)
        tokens = shlex.split(cmd)
        assert tokens == ["llama-server", "--model", f"/model{ext}"]


# ---- Comprehensive field coverage ----

def test_comprehensive_config():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        context_size=8192,
        gpu_layers=33,
        threads=8,
        threads_batch=4,
        predict=256,
        batch_size=2048,
        ubatch_size=512,
        keep=100,
        rope_scaling="yarn",
        rope_scale=2,
        rope_freq_base=500000,
        rope_freq_scale=0.5,
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        use_mmap=False,
        mlock=True,
        temp=0.7,
        top_k=30,
        top_p=0.9,
        min_p=0.1,
        repeat_penalty=1.1,
        presence_penalty=0.5,
        frequency_penalty=0.3,
        seed=42,
        flash_attn="on",
        backend="cuda",
    )
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--ctx-size" in tokens
    assert "--gpu-layers" in tokens
    assert "--threads" in tokens
    assert "--threads-batch" in tokens
    assert "--n-predict" in tokens
    assert "--rope-scaling" in tokens
    assert "--rope-scale" in tokens
    assert "--rope-freq-base" in tokens
    assert "--rope-freq-scale" in tokens
    assert "--cache-type-k" in tokens
    assert "--cache-type-v" in tokens
    assert "--no-mmap" in tokens
    assert "--mlock" in tokens
    assert "--temp" in tokens
    assert "--top-k" in tokens
    assert "--top-p" in tokens
    assert "--min-p" in tokens
    assert "--repeat-penalty" in tokens
    assert "--presence-penalty" in tokens
    assert "--frequency-penalty" in tokens
    assert "--seed" in tokens
    assert "--flash-attn" in tokens
    assert "--backend" in tokens


def test_speculative_decoding_fields():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        spec_type="ngram",
        spec_draft_n_max=5,
        spec_draft_n_min=2,
        spec_draft_model="/draft.gguf",
        spec_draft_backend_sampling=False,
    )
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--spec-type" in tokens
    assert "--spec-draft-n-max" in tokens
    assert "--spec-draft-n-min" in tokens
    assert "--spec-draft-model" in tokens
    assert "--no-spec-draft-backend-sampling" in tokens


def test_numa_fields():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        numa="prefer",
        split_mode="row",
        main_gpu=1,
        op_offload=False,
        cpu_moe=True,
    )
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--numa" in tokens
    assert "--split-mode" in tokens
    assert "--main-gpu" in tokens
    assert "--no-op-offload" in tokens
    assert "--cpu-moe" in tokens


def test_lora_fields():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        lora="/lora.gguf",
        lora_scaled="/lora-scaled.gguf=1.5",
        control_vector="/cv.gguf",
    )
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--lora" in tokens
    assert "--lora-scaled" in tokens
    assert "--control-vector" in tokens


def test_logging_fields():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        log_file="/tmp/log.txt",
        verbose=True,
        log_colors="enable",
        offline=True,
        verbosity=5,
    )
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--log-file" in tokens
    assert "--verbose" in tokens
    assert "--log-colors" in tokens
    assert "--offline" in tokens
    assert "--verbosity" in tokens


def test_server_fields():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        port=9090,
        host="0.0.0.0",
        parallel=2,
        embeddings=True,
        reranking=True,
        jinja=True,
        agent=True,
    )
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--port" in tokens
    assert "--host" in tokens
    assert "--parallel" in tokens
    assert "--embeddings" in tokens
    assert "--reranking" in tokens
    assert "--jinja" in tokens
    assert "--agent" in tokens


# ---- Edge cases ----

def test_empty_model_path():
    cfg = LlamaConfig(model_path="")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert "--model" not in tokens


def test_command_is_deterministic():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        context_size=8192,
        gpu_layers=33,
        threads=8,
    )
    cmd1 = build_command(cfg)
    cmd2 = build_command(cfg)
    assert cmd1 == cmd2


def test_special_chars_in_model_path_escaped():
    cfg = LlamaConfig(model_path="/path/with spaces/model.gguf")
    cmd = build_command(cfg)
    tokens = shlex.split(cmd)
    assert tokens == ["llama-server", "--model", "/path/with spaces/model.gguf"]
