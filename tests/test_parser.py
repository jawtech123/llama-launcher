"""Tests for core.parser.parse_launch_command."""
import logging
import pytest
from llama_launcher.core.parser import parse_launch_command


# ---- Existing tests (updated for 3-tuple) ----

def test_basic_command_with_executable_prefix():
    params, unrec, dups = parse_launch_command(
        "llama-server --model /models/m.gguf --ctx-size 4096 -ngl 33 --port 8080"
    )
    assert unrec == []
    assert dups == []
    assert params["model_path"] == "/models/m.gguf"
    assert params["context_size"] == 4096
    assert params["gpu_layers"] == 33
    assert params["port"] == 8080


def test_flag_equals_value_form():
    params, unrec, dups = parse_launch_command(
        "llama-server --model=/m.gguf --ctx-size=8192 --temp=0.5 --no-mmap"
    )
    assert unrec == []
    assert dups == []
    assert params["model_path"] == "/m.gguf"
    assert params["context_size"] == 8192
    assert params["temp"] == 0.5
    assert params["use_mmap"] is False


def test_negative_values_are_consumed_not_treated_as_flags():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --seed -1 --yarn-ext-factor -1.0")
    assert unrec == []
    assert dups == []
    assert params["seed"] == -1
    assert params["yarn_ext_factor"] == -1.0


def test_bare_boolean_flags_and_no_variants():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --reuse-port --no-mmap --mlock --verbose --no-cont-batching"
    )
    assert unrec == []
    assert dups == []
    assert params["reuse_port"] is True
    assert params["use_mmap"] is False
    assert params["mlock"] is True
    assert params["verbose"] is True
    assert params["cont_batching"] is False


def test_unrecognized_flags_are_collected():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --brand-new-flag 5 --another-unknown"
    )
    assert params["model_path"] == "/m.gguf"
    assert "--brand-new-flag" in unrec
    assert "--another-unknown" in unrec
    assert dups == []


def test_bare_model_path_token():
    params, unrec, dups = parse_launch_command("llama-server /models/m.gguf --ctx-size 2048")
    assert params["model_path"] == "/models/m.gguf"
    assert params["context_size"] == 2048
    assert dups == []


def test_quoted_path_with_spaces():
    params, unrec, dups = parse_launch_command(
        'llama-server --model "/my models/some model.gguf" --threads 4'
    )
    assert unrec == []
    assert dups == []
    assert params["model_path"] == "/my models/some model.gguf"
    assert params["threads"] == 4


def test_short_flags():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf -c 8192 -t 8 -b 1024 -ub 256 -ctk q8_0")
    assert unrec == []
    assert dups == []
    assert params["context_size"] == 8192
    assert params["threads"] == 8
    assert params["batch_size"] == 1024
    assert params["ubatch_size"] == 256
    assert params["cache_type_k"] == "q8_0"


def test_type_coercion_stays_string_for_text_fields():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --tensor-split 3,1 --rope-scaling yarn")
    assert params["tensor_split"] == "3,1"
    assert params["rope_scaling"] == "yarn"


# ---- Duplicate detection tests ----

def test_duplicate_long_flag_same_position():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --ctx-size 4096 --ctx-size 8192"
    )
    assert params["context_size"] == 8192  # last value wins
    assert len(dups) == 1
    assert "--ctx-size" in dups[0]


def test_duplicate_short_flag():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf -t 4 -t 8"
    )
    assert params["threads"] == 8
    assert len(dups) == 1
    assert "-t" in dups[0]


def test_duplicate_flag_equals_form():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --temp=0.8 --temp=0.5"
    )
    assert params["temp"] == 0.5
    assert len(dups) == 1


def test_duplicate_bool_flags():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --mlock --mlock"
    )
    assert params["mlock"] is True
    assert len(dups) == 1


def test_duplicate_no_bool_flags():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --no-mmap --no-mmap"
    )
    assert params["use_mmap"] is False
    assert len(dups) == 1


def test_duplicate_across_alias_forms():
    """Using -t and --threads for the same field should detect duplicate."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf -t 4 --threads 8"
    )
    assert params["threads"] == 8
    assert len(dups) >= 1


def test_no_false_duplicates_different_fields():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --ctx-size 4096 --threads 8 --port 8080"
    )
    assert dups == []


def test_multiple_different_duplicates():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --ctx-size 4096 --ctx-size 8192 --threads 4 --threads 8 --threads 16"
    )
    assert params["context_size"] == 8192
    assert params["threads"] == 16
    assert len(dups) >= 2  # 1 ctx-size dup + 2 threads dups = 3 total


# ---- Edge case: empty / malformed input ----

def test_empty_string():
    params, unrec, dups = parse_launch_command("")
    assert params == {}
    assert unrec == []
    assert dups == []


def test_only_whitespace():
    params, unrec, dups = parse_launch_command("   ")
    assert params == {}
    assert unrec == []
    assert dups == []


def test_only_executable_name():
    params, unrec, dups = parse_launch_command("llama-server")
    assert params == {}
    assert unrec == []
    assert dups == []


def test_no_model_path_only_flags():
    params, unrec, dups = parse_launch_command(
        "llama-server --ctx-size 4096 --threads 8"
    )
    assert "model_path" not in params
    assert params["context_size"] == 4096


def test_multiple_bare_model_paths_uses_first():
    params, unrec, dups = parse_launch_command(
        "llama-server /m1.gguf /m2.gguf --ctx-size 4096"
    )
    assert params["model_path"] == "/m1.gguf"


def test_executable_with_dot_slash():
    params, unrec, dups = parse_launch_command(
        "./llama-server --model /m.gguf --ctx-size 4096"
    )
    assert params["model_path"] == "/m.gguf"
    assert params["context_size"] == 4096


def test_executable_with_absolute_path():
    params, unrec, dups = parse_launch_command(
        "/usr/local/bin/llama-server --model /m.gguf --threads 4"
    )
    assert params["model_path"] == "/m.gguf"
    assert params["threads"] == 4


# ---- shlex fallback tests ----

def test_shlex_fallback_on_bad_quotes():
    """Malformed quotes cause shlex to fail; parser falls back to split."""
    params, unrec, dups = parse_launch_command('llama-server --model /m.gguf --ctx-size 4096 broken')
    assert params["model_path"] == "/m.gguf"
    assert params["context_size"] == 4096


# ---- Boolean flag behavior ----

def test_bool_flag_defaults_false_when_not_present():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf")
    assert "use_mmap" not in params


def test_bool_flag_true_no_prefix():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --mlock")
    assert params["mlock"] is True


def test_bool_flag_false_with_no_prefix():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --no-mmap")
    assert params["use_mmap"] is False


def test_no_flag_takes_no_value_when_missing():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --threads"
    )
    assert "threads" not in params


# ---- Value coercion tests ----

def test_int_coercion():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --ctx-size 4096")
    assert params["context_size"] == 4096
    assert isinstance(params["context_size"], int)


def test_float_coercion():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --temp 0.85")
    assert params["temp"] == 0.85
    assert isinstance(params["temp"], float)


def test_bool_coerce_from_string_true():
    """--verbose is a bool field, but if it takes a value it should coerce."""
    # Note: --verbose is a bool flag that doesn't take a value in typical usage
    # Testing coercion of boolean-like string values in = form
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=true")
    assert params["use_mmap"] is True


def test_bool_coerce_from_string_false():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=false")
    assert params["use_mmap"] is False


def test_bool_coerce_from_zero():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=0")
    assert params["use_mmap"] is False


def test_bool_coerce_from_off():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=off")
    assert params["use_mmap"] is False


def test_bool_coerce_from_no():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=no")
    assert params["use_mmap"] is False


def test_bool_coerce_from_on():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=on")
    assert params["use_mmap"] is True


def test_bool_coerce_from_one():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --mmap=1")
    assert params["use_mmap"] is True


def test_non_coercible_float_keeps_raw():
    """If coercion fails, the raw string is kept."""
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --temp abc")
    assert params["temp"] == "abc"


def test_non_coercible_int_keeps_raw():
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --ctx-size abc")
    assert params["context_size"] == "abc"


# ---- Negative number handling ----

def test_negative_int_as_value():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --gpu-layers -1")
    assert params["gpu_layers"] == -1


def test_negative_float_as_value():
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --repeat-penalty -1.5")
    assert params["repeat_penalty"] == -1.5


def test_flag_after_negative_number():
    """A flag after a negative number should not be consumed as its value."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --seed -1 --port 8080"
    )
    assert params["seed"] == -1
    assert params["port"] == 8080


# ---- Path handling ----

def test_gguf_extension_accepted_as_model():
    params, _, dups = parse_launch_command("llama-server model.gguf")
    assert params["model_path"] == "model.gguf"


def test_bin_extension_accepted_as_model():
    params, _, dups = parse_launch_command("llama-server model.bin")
    assert params["model_path"] == "model.bin"


def test_safetensors_extension_accepted_as_model():
    params, _, dups = parse_launch_command("llama-server model.safetensors")
    assert params["model_path"] == "model.safetensors"


def test_path_with_forward_slash():
    params, _, dups = parse_launch_command("llama-server /path/to/model")
    assert params["model_path"] == "/path/to/model"


def test_path_with_backslash():
    params, _, dups = parse_launch_command(r"llama-server C:\models\model.gguf")
    # shlex strips backslashes, but the path has a / so it's still detected
    assert params["model_path"] == "C:modelsmodel.gguf"


def test_model_path_in_middle_of_command():
    """Model path appearing after other flags."""
    params, _, dups = parse_launch_command("llama-server --ctx-size 4096 /models/m.gguf --threads 4")
    assert params["model_path"] == "/models/m.gguf"
    assert params["context_size"] == 4096
    assert params["threads"] == 4


# ---- Unrecognized flag handling ----

def test_unrecognized_flag_with_value_not_consumed():
    """Unrecognized flags should NOT consume the next token as their value."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --unknown-flag value1 --ctx-size 4096"
    )
    assert "--unknown-flag" in unrec
    assert params["context_size"] == 4096


def test_unrecognized_flag_at_end():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --ctx-size 4096 --unknown"
    )
    assert "--unknown" in unrec


def test_unrecognized_flag_with_equals():
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --unknown=value"
    )
    assert "--unknown=value" in unrec


# ---- Return type verification ----

def test_returns_three_tuple():
    result = parse_launch_command("llama-server -m /m.gguf")
    assert isinstance(result, tuple)
    assert len(result) == 3
    params, unrec, dups = result
    assert isinstance(params, dict)
    assert isinstance(unrec, list)
    assert isinstance(dups, list)


# ---- Complex real-world commands ----

def test_complex_command_roundtrip():
    cmd = (
        "llama-server --model /models/llama-2-7b.gguf --ctx-size 8192 "
        "--gpu-layers 35 --threads 8 --port 8080 --host 0.0.0.0 "
        "--temp 0.7 --top-p 0.95 --mlock --no-mmap --batch-size 2048"
    )
    params, unrec, dups = parse_launch_command(cmd)
    assert unrec == []
    assert dups == []
    assert params["model_path"] == "/models/llama-2-7b.gguf"
    assert params["context_size"] == 8192
    assert params["gpu_layers"] == 35
    assert params["threads"] == 8
    assert params["port"] == 8080
    assert params["host"] == "0.0.0.0"
    assert params["temp"] == 0.7
    assert params["top_p"] == 0.95
    assert params["mlock"] is True
    assert params["use_mmap"] is False


def test_mixed_alias_and_long_flags():
    params, _, dups = parse_launch_command(
        "llama-server -m /m.gguf --ctx-size 4096 -t 8 --batch-size 1024"
    )
    assert params["context_size"] == 4096
    assert params["threads"] == 8
    assert params["batch_size"] == 1024


def test_flag_with_no_value_at_end():
    """Boolean flag at end should work fine."""
    params, _, dups = parse_launch_command("llama-server -m /m.gguf --verbose")
    assert params["verbose"] is True


def test_empty_command_returns_empty():
    params, unrec, dups = parse_launch_command("")
    assert params == {}
    assert unrec == []
    assert dups == []


def test_logging_on_duplicate(caplog):
    with caplog.at_level(logging.WARNING):
        parse_launch_command("llama-server -m /m.gguf --ctx-size 4096 --ctx-size 8192")
    assert any("Duplicate flag" in r.message for r in caplog.records)


def test_logging_on_unrecognized(caplog):
    with caplog.at_level(logging.WARNING):
        parse_launch_command("llama-server -m /m.gguf --totally-new-flag")
    assert any("Unrecognized flag" in r.message for r in caplog.records)


def test_logging_on_missing_value(caplog):
    with caplog.at_level(logging.WARNING):
        parse_launch_command("llama-server -m /m.gguf --threads")
    assert any("missing a value" in r.message for r in caplog.records)


# ---- Additional edge case tests ----

def test_all_short_aliases():
    """All short alias flags (-m, -t, -c, -b, -ngl, -ub) are parsed."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf -t 8 -c 4096 -b 1024 -ub 256 -ngl 33"
    )
    assert params["model_path"] == "/m.gguf"
    assert params["threads"] == 8
    assert params["context_size"] == 4096
    assert params["batch_size"] == 1024
    assert params["ubatch_size"] == 256
    assert params["gpu_layers"] == 33
    assert unrec == []
    assert dups == []


def test_boolean_equals_value_form():
    """Boolean flags with =value form (true/false/0/1)."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --mmap=false --mlock=true --verbose=1 --no-mmap=0"
    )
    assert params["use_mmap"] is False
    assert params["mlock"] is True
    assert params["verbose"] is True
    assert params["use_mmap"] is False


def test_float_coercion_edge_cases():
    """Float values with various decimal representations."""
    params, _, _ = parse_launch_command("llama-server -m /m.gguf --temp 0.05")
    assert params["temp"] == 0.05

    params2, _, _ = parse_launch_command("llama-server -m /m.gguf --temp 1.75")
    assert params2["temp"] == 1.75

    params3, _, _ = parse_launch_command("llama-server -m /m.gguf --top-p 2.0")
    assert params3["top_p"] == 2.0


def test_integer_overflow_stays_int():
    """Large integers stay as int (validator catches later)."""
    params, _, _ = parse_launch_command("llama-server -m /m.gguf --ctx-size 999999999")
    assert params["context_size"] == 999999999
    assert isinstance(params["context_size"], int)


def test_flag_with_equals_and_value():
    """--flag=value form parsed correctly."""
    params, unrec, dups = parse_launch_command(
        "llama-server --model=/path/to/model.gguf --ctx-size=8192 --temp=0.7"
    )
    assert params["model_path"] == "/path/to/model.gguf"
    assert params["context_size"] == 8192
    assert params["temp"] == 0.7
    assert unrec == []
    assert dups == []


def test_negative_float_as_value():
    """Negative float values for float fields."""
    params, _, _ = parse_launch_command("llama-server -m /m.gguf --top-n-sigma -1.0")
    assert params["top_n_sigma"] == -1.0


def test_missing_value_after_flag():
    """Flag with no value following → warning logged, flag skipped."""
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --threads --ctx-size 4096")
    assert "threads" not in params
    assert params["context_size"] == 4096


def test_all_negation_forms():
    """All --no-* negation forms are parsed correctly."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --no-mmap --no-ignore-eos --no-backend-sampling "
        "--no-cont-batching --no-mlock --no-perf --no-escape --no-kv-offload "
        "--no-repack --no-direct-io --no-cache-idle-slots --no-context-shift "
        "--no-jinja --no-embeddings --no-reranking --no-agent --no-ui "
        "--no-log-prefix --no-log-timestamps --no-kv-unified --no-special "
        "--no-spm-infill --no-mmproj-auto --no-mmproj-offload --no-warmup "
        "--no-spec-draft-backend-sampling --no-cpu-moe --no-op-offload "
        "--no-reuse-port --no-ui-mcp-proxy --no-swa-full --no-check-tensors"
    )
    assert params["use_mmap"] is False
    assert params["ignore_eos"] is False
    assert params["backend_sampling"] is False
    assert params["cont_batching"] is False
    assert params["mlock"] is False
    assert params["kv_offload"] is False
    assert params["repack"] is False
    assert params["use_mmap"] is False
    assert params["cache_idle_slots"] is False
    assert params["kv_unified"] is False
    assert params["embeddings"] is False
    assert params["reranking"] is False
    assert params["agent"] is False


def test_bare_executable_prefix_variants():
    """Various executable prefixes are dropped."""
    for exe in ("./llama-server", "llama-server", "/usr/bin/llama-server"):
        params, unrec, dups = parse_launch_command(f"{exe} --model /m.gguf --threads 4")
        assert params["model_path"] == "/m.gguf"
        assert params["threads"] == 4
        assert unrec == []


def test_command_only_unrecognized_flags():
    """Command with only unrecognized flags → params empty, unrecognized populated."""
    params, unrec, dups = parse_launch_command("llama-server --unknown1 --unknown2")
    assert params == {}
    assert "--unknown1" in unrec
    assert "--unknown2" in unrec
    assert dups == []


def test_command_only_duplicate_flags():
    """Command with only duplicate flags → duplicates populated, unrecognized empty."""
    params, unrec, dups = parse_launch_command("llama-server --threads 4 --threads 8 --threads 16")
    assert params["threads"] == 16
    assert len(dups) >= 2
    assert unrec == []


def test_complex_mixed_command():
    """Command with short + long + negation + =value + duplicates + unrecognized."""
    params, unrec, dups = parse_launch_command(
        "llama-server -m /m.gguf --ctx-size 4096 --ctx-size 8192 "
        "-t 8 --threads 16 --no-mmap --unknown-flag hello --port=8080"
    )
    assert params["model_path"] == "/m.gguf"
    assert params["context_size"] == 8192
    assert params["threads"] == 16
    assert params["use_mmap"] is False
    assert params["port"] == 8080
    assert "--unknown-flag" in unrec
    assert len(dups) >= 2


def test_duplicate_different_fields_ok():
    """Different fields can have same value without being flagged as duplicate."""
    params, unrec, dups = parse_launch_command("llama-server -m /m.gguf --ctx-size 4096 --threads 8")
    assert params["context_size"] == 4096
    assert params["threads"] == 8
    assert dups == []
    assert unrec == []


def test_negation_boolean_with_value():
    """--no-mmap=false should set mmap=True."""
    params, _, _ = parse_launch_command("llama-server -m /m.gguf --mmap=false")
    assert params["use_mmap"] is False
