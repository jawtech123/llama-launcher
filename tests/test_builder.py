"""Tests for core.builder.build_command."""
import shlex

from llama_launcher.core.builder import build_command
from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.parser import parse_launch_command


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
    params, unrecognized = parse_launch_command(cmd)
    assert unrecognized == []
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
