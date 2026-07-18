"""Tests for core.validator.validate_config."""
import pytest

from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.validator import validate_config, ValidationError


def test_valid_config_passes():
    validate_config(LlamaConfig(model_path="/models/m.gguf"))


def test_missing_model_path_rejected():
    with pytest.raises(ValidationError, match="model_path"):
        validate_config(LlamaConfig(model_path=""))


@pytest.mark.parametrize("field,value", [
    ("context_size", 0),
    ("context_size", 999999999),
    ("gpu_layers", -2),
    ("gpu_layers", 1000),
    ("threads", -1),
    ("threads", 9999),
    ("port", 0),
    ("port", 70000),
    ("batch_size", 0),
    ("batch_size", 99999),
])
def test_range_validation(field, value):
    with pytest.raises(ValidationError):
        validate_config(LlamaConfig(model_path="/m.gguf", **{field: value}))


def test_backend_validation():
    with pytest.raises(ValidationError, match="backend"):
        validate_config(LlamaConfig(model_path="/m.gguf", backend="tpu"))


def test_cache_type_validation():
    with pytest.raises(ValidationError, match="cache_type_k"):
        validate_config(LlamaConfig(model_path="/m.gguf", cache_type_k="q3_k_xx"))


def test_sycl_backend_is_accepted():
    validate_config(LlamaConfig(model_path="/m.gguf", backend="sycl"))
