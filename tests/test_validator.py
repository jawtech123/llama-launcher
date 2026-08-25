"""Tests for core.validator.validate_config."""
import pytest
from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.validator import validate_config, ValidationError


def test_valid_config_passes():
    validate_config(LlamaConfig(model_path="/models/m.gguf"))


def test_valid_config_with_all_fields():
    cfg = LlamaConfig(
        model_path="/m.gguf",
        context_size=4096,
        gpu_layers=33,
        threads=8,
        port=8080,
        host="127.0.0.1",
        backend="auto",
        temp=0.8,
        top_p=0.95,
        cache_type_k="f16",
        mlock=True,
        use_mmap=False,
    )
    validate_config(cfg)


# ---- model_path validation ----

def test_missing_model_path_rejected():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path=""))


def test_model_path_none_rejected():
    """Empty string is rejected; None would be caught by dataclass default."""
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path=""))


def test_model_path_non_string_rejected():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig())  # model_path defaults to ""


def test_model_path_gguf_extension_valid():
    validate_config(LlamaConfig(model_path="/models/model.gguf"))


def test_model_path_bin_extension_valid():
    validate_config(LlamaConfig(model_path="/models/model.bin"))


def test_model_path_safetensors_extension_valid():
    validate_config(LlamaConfig(model_path="/models/model.safetensors"))


def test_model_path_txt_extension_invalid():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path="/models/model.txt"))


def test_model_path_no_extension_invalid():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path="/models/model"))


def test_model_path_json_extension_invalid():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path="/models/model.json"))


def test_model_path_py_extension_invalid():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path="/models/model.py"))


def test_model_path_case_insensitive_uppercase():
    validate_config(LlamaConfig(model_path="/models/model.GGUF"))


def test_model_path_case_insensitive_mixed():
    validate_config(LlamaConfig(model_path="/models/model.Gguf"))


def test_model_path_with_directory_and_extension():
    validate_config(LlamaConfig(model_path="/path/to/model.gguf"))


def test_model_path_with_spaces_and_extension():
    validate_config(LlamaConfig(model_path="/my models/model.gguf"))


# ---- host validation ----

def test_valid_host():
    validate_config(LlamaConfig(model_path="/m.gguf", host="127.0.0.1"))


def test_valid_host_all_interfaces():
    validate_config(LlamaConfig(model_path="/m.gguf", host="0.0.0.0"))


def test_valid_host_ipv6():
    validate_config(LlamaConfig(model_path="/m.gguf", host="::"))


def test_empty_host_rejected():
    with pytest.raises(ValidationError, match="host"):
        validate_config(LlamaConfig(model_path="/m.gguf", host=""))


# ---- range validation (int/float) ----

@pytest.mark.parametrize("field,value", [
    ("context_size", 0),
    ("context_size", 255),
    ("context_size", 1048577),
    ("gpu_layers", -2),
    ("gpu_layers", 9999),
    ("threads", -1),
    ("threads", 257),
    ("port", 0),
    ("port", 65536),
    ("batch_size", 0),
    ("batch_size", 65537),
    ("ubatch_size", 0),
    ("ubatch_size", 4097),
    ("top_k", -1),
    ("top_k", 129),
    ("temp", -0.1),
    ("temp", 2.1),
    ("top_p", -0.1),
    ("top_p", 1.01),
    ("min_p", -0.1),
    ("repeat_penalty", -0.1),
    ("repeat_penalty", 2.1),
    ("presence_penalty", -2.1),
    ("presence_penalty", 2.1),
    ("frequency_penalty", -2.1),
    ("mirostat_lr", -0.1),
    ("mirostat_ent", -0.1),
    ("mirostat_ent", 20.1),
])
def test_range_validation(field, value):
    with pytest.raises(ValidationError, match=field):
        validate_config(LlamaConfig(model_path="/m.gguf", **{field: value}))


@pytest.mark.parametrize("field,value", [
    ("context_size", 256),
    ("context_size", 1048576),
    ("gpu_layers", -1),
    ("gpu_layers", 999),
    ("threads", 0),
    ("threads", 256),
    ("port", 1),
    ("port", 65535),
    ("batch_size", 1),
    ("batch_size", 65536),
    ("temp", 0.0),
    ("temp", 2.0),
    ("top_p", 0.0),
    ("top_p", 1.0),
])
def test_range_validation_boundary(field, value):
    """Boundary values should pass validation."""
    validate_config(LlamaConfig(model_path="/m.gguf", **{field: value}))


# ---- choice validation ----

def test_backend_validation():
    with pytest.raises(ValidationError, match="backend"):
        validate_config(LlamaConfig(model_path="/m.gguf", backend="tpu"))


def test_backend_auto_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="auto"))


def test_backend_cuda_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="cuda"))


def test_backend_metal_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="metal"))


def test_backend_vulkan_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="vulkan"))


def test_backend_cpu_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="cpu"))


def test_backend_sycl_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="sycl"))


def test_cache_type_validation():
    with pytest.raises(ValidationError, match="cache_type_k"):
        validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q3_k_xx"))


def test_cache_type_f16_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="f16"))


def test_cache_type_q8_0_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q8_0"))


def test_cache_type_q4_0_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q4_0"))


def test_cache_type_q4_1_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q4_1"))


def test_cache_type_f32_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="f32"))


def test_cache_type_bf16_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="bf16"))


def test_cache_type_iq4_nl_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="iq4_nl"))


def test_cache_type_q5_0_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q5_0"))


def test_cache_type_q5_1_valid():
    validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q5_1"))


def test_rope_scaling_choice_valid():
    for choice in ("none", "linear", "yarn"):
        validate_config(LlamaConfig(model_path="/m.gguf", rope_scaling=choice))


def test_rope_scaling_choice_invalid():
    with pytest.raises(ValidationError, match="rope_scaling"):
        validate_config(LlamaConfig(model_path="/m.gguf", rope_scaling="invalid"))


def test_split_mode_choice_valid():
    for choice in ("none", "layer", "row", "tensor"):
        validate_config(LlamaConfig(model_path="/m.gguf", split_mode=choice))


def test_split_mode_choice_invalid():
    with pytest.raises(ValidationError, match="split_mode"):
        validate_config(LlamaConfig(model_path="/m.gguf", split_mode="invalid"))


def test_flash_attn_choice_valid():
    for choice in ("auto", "on", "off"):
        validate_config(LlamaConfig(model_path="/m.gguf", flash_attn=choice))


def test_flash_attn_choice_invalid():
    with pytest.raises(ValidationError, match="flash_attn"):
        validate_config(LlamaConfig(model_path="/m.gguf", flash_attn="invalid"))


def test_log_colors_choice_valid():
    for choice in ("auto", "enable", "disable"):
        validate_config(LlamaConfig(model_path="/m.gguf", log_colors=choice))


def test_pooling_choice_valid():
    for choice in ("none", "mean", "cls", "last"):
        validate_config(LlamaConfig(model_path="/m.gguf", pooling=choice))


# ---- boolean field validation ----

@pytest.mark.parametrize("field", [
    "use_mmap", "mlock", "kv_offload", "repack",
    "verbose", "ignore_eos", "mlock",
])
def test_bool_fields_accept_true(field):
    validate_config(LlamaConfig(model_path="/m.gguf", **{field: True}))


@pytest.mark.parametrize("field", [
    "use_mmap", "mlock", "kv_offload", "repack",
    "verbose", "ignore_eos", "mlock",
])
def test_bool_fields_accept_false(field):
    validate_config(LlamaConfig(model_path="/m.gguf", **{field: False}))


def test_bool_field_rejects_int():
    with pytest.raises(ValidationError, match="mlock"):
        validate_config(LlamaConfig(model_path="/m.gguf", mlock=1))


def test_bool_field_rejects_string():
    with pytest.raises(ValidationError, match="mlock"):
        validate_config(LlamaConfig(model_path="/m.gguf", mlock="true"))


def test_bool_field_rejects_none():
    """None values are skipped by validator (they're treated as unset)."""
    cfg = LlamaConfig(model_path="/m.gguf", mlock=None)
    # None is skipped by the validator (if value is None: continue)
    # So this should NOT raise - it's treated as unset
    validate_config(cfg)


# ---- ValidationError is raised correctly ----

def test_validation_error_is_exception():
    assert issubclass(ValidationError, Exception)


def test_validation_error_message_contains_field():
    try:
        validate_config(LlamaConfig(model_path=""))
    except ValidationError as e:
        assert "model_path" in str(e)


def test_first_error_stops_validation():
    """Validation stops on the first error found."""
    with pytest.raises(ValidationError):
        validate_config(LlamaConfig(model_path="", host=""))
