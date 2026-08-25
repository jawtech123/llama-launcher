"""Tests for model file scanner."""
import os
import tempfile
import pytest

from llama_launcher.models.scanner import scan_models, _parse_model_file, is_mmproj_file
from llama_launcher.models.types import ModelInfo
from pathlib import Path


class TestScanModels:
    """Test model directory scanning."""

    def test_scan_gguf_files(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"fake gguf content " * 100)
        models = scan_models(str(tmp_path))
        assert len(models) == 1
        assert models[0].format == "gguf"

    def test_scan_bin_files(self, tmp_path):
        (tmp_path / "model.bin").write_bytes(b"fake bin content " * 100)
        models = scan_models(str(tmp_path))
        assert len(models) == 1
        assert models[0].format == "bin"

    def test_scan_safetensors_files(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"fake safetensors " * 100)
        models = scan_models(str(tmp_path))
        assert len(models) == 1
        assert models[0].format == "safetensors"

    def test_scan_multiple_files(self, tmp_path):
        (tmp_path / "llama.gguf").write_bytes(b"fake")
        (tmp_path / "mistral.bin").write_bytes(b"fake")
        (tmp_path / "qwen.safetensors").write_bytes(b"fake")
        models = scan_models(str(tmp_path))
        assert len(models) == 3

    def test_scan_subdirectories(self, tmp_path):
        subdir = tmp_path / "sub" / "deep"
        subdir.mkdir(parents=True)
        (subdir / "model.gguf").write_bytes(b"fake")
        models = scan_models(str(tmp_path))
        assert len(models) == 1
        assert "sub/deep/model.gguf" in models[0].file_path

    def test_scan_nonexistent_directory(self, tmp_path):
        models = scan_models(str(tmp_path / "nope"))
        assert models == []

    def test_scan_empty_directory(self, tmp_path):
        models = scan_models(str(tmp_path))
        assert models == []

    def test_scan_ignores_other_extensions(self, tmp_path):
        (tmp_path / "model.txt").write_text("ignore")
        (tmp_path / "model.json").write_text("{}")
        (tmp_path / "model.gguf").write_bytes(b"fake")
        models = scan_models(str(tmp_path))
        assert len(models) == 1

    def test_scan_returns_model_info(self, tmp_path):
        (tmp_path / "llama-q4_0.gguf").write_bytes(b"fake" * 100000)
        models = scan_models(str(tmp_path))
        info = models[0]
        assert isinstance(info, ModelInfo)
        assert info.file_name == "llama-q4_0.gguf"
        assert info.format == "gguf"
        assert info.family == "llama"
        assert info.quant_type == "Q4_0"
        assert info.file_size_mb > 0


class TestParseModelFile:
    """Test individual model file parsing."""

    def test_parse_gguf(self, tmp_path):
        f = tmp_path / "test.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info is not None
        assert info.format == "gguf"

    def test_parse_bin(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info is not None
        assert info.format == "bin"

    def test_parse_safetensors(self, tmp_path):
        f = tmp_path / "test.safetensors"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info is not None
        assert info.format == "safetensors"

    def test_parse_family_llama(self, tmp_path):
        f = tmp_path / "llama-2-7b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "llama"

    def test_parse_family_llm(self, tmp_path):
        f = tmp_path / "llm-model.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "llama"

    def test_parse_family_mistral(self, tmp_path):
        f = tmp_path / "mistral-7b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "mistral"

    def test_parse_family_qwen(self, tmp_path):
        f = tmp_path / "qwen-7b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "qwen"

    def test_parse_family_qwen2(self, tmp_path):
        f = tmp_path / "qwen2-7b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "qwen"

    def test_parse_family_qwq(self, tmp_path):
        f = tmp_path / "qwq-32b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "qwen"

    def test_parse_family_codellama(self, tmp_path):
        f = tmp_path / "codellama-7b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        # "codellama" contains "llama", so the specific hint must be tested
        # first or this collapses to the generic family.
        assert info.family == "codellama"

    def test_parse_family_tinyllama_is_llama(self, tmp_path):
        f = tmp_path / "tinyllama-1.1b.gguf"
        f.write_bytes(b"fake")
        assert _parse_model_file(f).family == "llama"

    def test_parse_family_phi(self, tmp_path):
        f = tmp_path / "phi-3.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "phi"

    def test_parse_family_gemma(self, tmp_path):
        f = tmp_path / "gemma-7b.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "gemma"

    def test_parse_family_unknown(self, tmp_path):
        f = tmp_path / "random-model.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.family == "unknown"

    def test_parse_quant_q8_0(self, tmp_path):
        f = tmp_path / "model-q8_0.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type == "Q8_0"

    def test_parse_quant_q6_k(self, tmp_path):
        f = tmp_path / "model-q6_k.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type == "Q6_K"

    def test_parse_quant_q5_k_m(self, tmp_path):
        f = tmp_path / "model-q5_k_m.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type == "Q5_K_M"

    def test_parse_quant_q4_k_m(self, tmp_path):
        f = tmp_path / "model-q4_k_m.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type == "Q4_K_M"

    def test_parse_quant_q4_0(self, tmp_path):
        f = tmp_path / "model-q4_0.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type == "Q4_0"

    def test_parse_quant_q2_k(self, tmp_path):
        f = tmp_path / "model-q2_k.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type == "Q2_K"

    def test_parse_quant_none(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"fake")
        info = _parse_model_file(f)
        assert info.quant_type is None

    def test_parse_returns_none_on_error(self):
        """Should return None for inaccessible files."""
        info = _parse_model_file(Path("/nonexistent/model.gguf"))
        assert info is None


class TestSharedHeuristics:
    """One implementation classifies models for local scan, SSH scan, estimator."""

    def test_single_traversal_finds_every_extension(self, tmp_path):
        # Globbing per extension walked the whole tree once per extension.
        (tmp_path / "nested" / "deep").mkdir(parents=True)
        for name in ("a.gguf", "b.bin", "c.safetensors", "ignore.txt"):
            (tmp_path / "nested" / "deep" / name).write_bytes(b"x")
        found = {m.file_name for m in scan_models(str(tmp_path))}
        assert found == {"a.gguf", "b.bin", "c.safetensors"}

    def test_scan_skips_directories_named_like_models(self, tmp_path):
        (tmp_path / "notamodel.gguf").mkdir()
        (tmp_path / "real.gguf").write_bytes(b"x")
        assert {m.file_name for m in scan_models(str(tmp_path))} == {"real.gguf"}

    def test_missing_directory_returns_empty(self):
        assert scan_models("/definitely/not/here") == []

    def test_file_path_instead_of_directory_returns_empty(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x")
        assert scan_models(str(f)) == []

    def test_build_model_info_matches_local_parse(self, tmp_path):
        from llama_launcher.models.scanner import build_model_info
        f = tmp_path / "qwen3-30b-Q6_K.gguf"
        f.write_bytes(b"x" * 2048)
        local = _parse_model_file(f)
        # The remote scanner has size and path but cannot stat the file.
        remote = build_model_info(str(f), 2048)
        assert (remote.family, remote.quant_type, remote.format) == \
               (local.family, local.quant_type, local.format)
        assert remote.file_size_mb == local.file_size_mb

    def test_ssh_scanner_uses_the_shared_classifier(self):
        from llama_launcher.remote.ssh import SSHServerConnection
        from llama_launcher.remote.types import ServerConfig
        conn = SSHServerConnection(ServerConfig(
            friendly_name="r", hostname="h", username="u"))
        conn._run_remote = lambda cmd, capture=False, timeout=30: (
            True, "5368709120\t/srv/models/codellama-13b-Q4_K_M.gguf\n")
        models = conn.scan_model_directory("/srv/models")
        assert len(models) == 1
        assert models[0].family == "codellama"      # not the generic "llama"
        assert models[0].quant_type == "Q4_K_M"
        assert models[0].file_size_mb == 5120.0


class TestIsMmprojFile:
    """Test the mmproj filename heuristic."""

    def test_detects_mmproj_in_name(self):
        assert is_mmproj_file("mmproj-model-f16.gguf") is True

    def test_detects_mmproj_mid_filename(self):
        assert is_mmproj_file("llava-v1.6-mmproj-f16.gguf") is True

    def test_case_insensitive(self):
        assert is_mmproj_file("MMPROJ-Model-F16.GGUF") is True

    def test_regular_model_is_not_mmproj(self):
        assert is_mmproj_file("llama-3-8b-instruct-q4_k_m.gguf") is False

    def test_empty_or_none_is_not_mmproj(self):
        assert is_mmproj_file("") is False
        assert is_mmproj_file(None) is False
