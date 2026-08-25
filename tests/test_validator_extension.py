"""Tests for model_path extension validation in validator."""
import pytest
from llama_launcher.core.validator import validate_config, ValidationError
from llama_launcher.core.config import LlamaConfig


def _make_config(model_path: str, **overrides) -> LlamaConfig:
    kwargs = {"model_path": model_path, "host": "127.0.0.1"}
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def test_gguf_extension_valid():
    """Models with .gguf extension pass validation."""
    config = _make_config("/path/to/model.gguf")
    validate_config(config)  # should not raise


def test_bin_extension_valid():
    """Models with .bin extension pass validation."""
    config = _make_config("/path/to/model.bin")
    validate_config(config)  # should not raise


def test_safetensors_extension_valid():
    """Models with .safetensors extension pass validation."""
    config = _make_config("/path/to/model.safetensors")
    validate_config(config)  # should not raise


def test_txt_extension_invalid():
    """Models with .txt extension raise ValidationError."""
    config = _make_config("/path/to/model.txt")
    with pytest.raises(ValidationError, match="must end with one of"):
        validate_config(config)


def test_no_extension_invalid():
    """Model without extension raises ValidationError."""
    config = _make_config("/path/to/model")
    with pytest.raises(ValidationError, match="must end with one of"):
        validate_config(config)


def test_uppercase_extension_accepted():
    """Uppercase extensions (.GGUF, .BIN, .SAFETENSORS) are accepted."""
    for ext in (".GGUF", ".BIN", ".SAFETENSORS"):
        config = _make_config(f"/path/to/model{ext}")
        validate_config(config)


def test_mixed_case_extension_accepted():
    """Mixed case extensions (.Gguf, .BiN) are accepted (case-insensitive)."""
    config = _make_config("/path/to/model.Gguf")
    validate_config(config)


def test_invalid_extension_txt_error_message():
    """Error message mentions the valid extensions for .txt."""
    config = _make_config("/path/to/model.txt")
    with pytest.raises(ValidationError) as exc_info:
        validate_config(config)
    assert ".gguf" in str(exc_info.value)
    assert ".bin" in str(exc_info.value)
    assert ".safetensors" in str(exc_info.value)


def test_empty_model_path_fails_before_extension_check():
    """Empty model_path fails validation (caught by earlier check)."""
    config = _make_config("")
    with pytest.raises(ValidationError):
        validate_config(config)


def test_nested_path_with_valid_extension():
    """Nested paths with valid extensions pass."""
    config = _make_config("/deep/nested/path/to/model.gguf")
    validate_config(config)


def test_url_like_path_fails():
    """A URL-like path without valid extension fails."""
    config = _make_config("https://example.com/model")
    with pytest.raises(ValidationError):
        validate_config(config)
