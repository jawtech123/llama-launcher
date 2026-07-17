"""Tests for core.parser.parse_launch_command."""
from llama_launcher.core.parser import parse_launch_command


def test_basic_command_with_executable_prefix():
    params, unrec = parse_launch_command(
        "llama-server --model /models/m.gguf --ctx-size 4096 -ngl 33 --port 8080"
    )
    assert unrec == []
    assert params["model_path"] == "/models/m.gguf"
    assert params["context_size"] == 4096
    assert params["gpu_layers"] == 33
    assert params["port"] == 8080


def test_flag_equals_value_form():
    params, unrec = parse_launch_command(
        "llama-server --model=/m.gguf --ctx-size=8192 --temp=0.5 --no-mmap"
    )
    assert unrec == []
    assert params["model_path"] == "/m.gguf"
    assert params["context_size"] == 8192
    assert params["temp"] == 0.5
    assert params["use_mmap"] is False


def test_negative_values_are_consumed_not_treated_as_flags():
    params, unrec = parse_launch_command("llama-server -m /m.gguf --seed -1 --yarn-ext-factor -1.0")
    assert unrec == []
    assert params["seed"] == -1
    assert params["yarn_ext_factor"] == -1.0


def test_bare_boolean_flags_and_no_variants():
    params, unrec = parse_launch_command(
        "llama-server -m /m.gguf --reuse-port --no-mmap --mlock --verbose --no-cont-batching"
    )
    assert unrec == []
    assert params["reuse_port"] is True
    assert params["use_mmap"] is False
    assert params["mlock"] is True
    assert params["verbose"] is True
    assert params["cont_batching"] is False


def test_unrecognized_flags_are_collected():
    params, unrec = parse_launch_command(
        "llama-server -m /m.gguf --brand-new-flag 5 --another-unknown"
    )
    assert params["model_path"] == "/m.gguf"
    assert "--brand-new-flag" in unrec
    assert "--another-unknown" in unrec


def test_bare_model_path_token():
    params, unrec = parse_launch_command("llama-server /models/m.gguf --ctx-size 2048")
    assert params["model_path"] == "/models/m.gguf"
    assert params["context_size"] == 2048


def test_quoted_path_with_spaces():
    params, unrec = parse_launch_command(
        'llama-server --model "/my models/some model.gguf" --threads 4'
    )
    assert unrec == []
    assert params["model_path"] == "/my models/some model.gguf"
    assert params["threads"] == 4


def test_short_flags():
    params, unrec = parse_launch_command("llama-server -m /m.gguf -c 8192 -t 8 -b 1024 -ub 256 -ctk q8_0")
    assert unrec == []
    assert params["context_size"] == 8192
    assert params["threads"] == 8
    assert params["batch_size"] == 1024
    assert params["ubatch_size"] == 256
    assert params["cache_type_k"] == "q8_0"


def test_type_coercion_stays_string_for_text_fields():
    params, _ = parse_launch_command("llama-server -m /m.gguf --tensor-split 3,1 --rope-scaling yarn")
    assert params["tensor_split"] == "3,1"
    assert params["rope_scaling"] == "yarn"
