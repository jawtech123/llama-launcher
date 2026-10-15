"""Tests for CLI main module commands."""
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from llama_launcher.cli.main import main, cmd_launch, cmd_scan, cmd_detect
import argparse


def _make_args(**overrides):
    defaults = {
        "model": None,
        "preset": None,
        "context": 4096,
        "gpu_layers": -1,
        "threads": 0,
        "port": 8080,
        "host": "127.0.0.1",
        "kv_cache": "f16",
        "batch": 2048,
        "dry_run": True,
        "extra_flags": [],
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdDetect:
    """GPU detection CLI command tests."""

    def test_detect_prints_gpu_info(self, capsys):
        with patch("llama_launcher.cli.main.detect_gpu") as mock_detect:
            mock_detect.return_value = MagicMock(backend="cuda", device_name="RTX 4090", vram_mb=24576, supported=True)
            result = cmd_detect(argparse.Namespace())
            assert result == 0
            output = capsys.readouterr().out
            assert "cuda" in output
            assert "RTX 4090" in output


class TestCmdScan:
    """Model scanning CLI command tests."""

    def test_scan_subcommand_lists_models(self, capsys, tmp_path):
        # Create a dummy model file
        model_file = tmp_path / "test.gguf"
        model_file.touch()

        with patch("llama_launcher.cli.main.scan_models") as mock_scan:
            mock_scan.return_value = []
            result = cmd_scan(argparse.Namespace(directory=str(tmp_path)))
            assert result == 0


class TestCmdLaunch:
    """Launch command tests."""

    def test_launch_dry_run_with_model(self, capsys):
        args = _make_args(model="/path/to/model.gguf", dry_run=True)
        result = cmd_launch(args)
        assert result == 0
        output = capsys.readouterr().out
        assert "Launch Command:" in output

    def test_launch_dry_run_with_preset(self, capsys):
        # Builtin presets carry no model_path, so --model supplies it; the
        # preset's own context_size proves the preset was actually applied.
        args = _make_args(preset="Small 8GB", model="/path/to/model.gguf", dry_run=True)
        result = cmd_launch(args)
        assert result == 0
        output = capsys.readouterr().out
        assert "Launch Command:" in output
        assert "8192" in output

    def test_launch_preset_without_model_fails_validation(self, capsys):
        args = _make_args(preset="Small 8GB", dry_run=True)
        assert cmd_launch(args) == 1

    def test_launch_invalid_preset_error_exit_1(self, capsys):
        args = _make_args(preset="NonExistentPreset", dry_run=True)
        result = cmd_launch(args)
        assert result == 1

    def test_launch_missing_model_validation_error(self, capsys):
        args = _make_args(model="", dry_run=True)
        result = cmd_launch(args)
        # Should fail validation
        assert result == 1

    def test_launch_extra_flags_shown(self, capsys):
        args = _make_args(model="/path/to/model.gguf", extra_flags=["--foo", "bar"], dry_run=True)
        result = cmd_launch(args)
        assert result == 0
        output = capsys.readouterr().out
        assert "Extra flags:" in output


class TestMain:
    """Main CLI entry point tests."""

    def test_no_subcommand_help_and_exit_1(self):
        with patch("sys.argv", ["llama-launcher"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_unknown_subcommand(self):
        with patch("sys.argv", ["llama-launcher", "nonexistent"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_scan_help(self):
        with patch("sys.argv", ["llama-launcher", "scan", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_detect_help(self):
        with patch("sys.argv", ["llama-launcher", "detect", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_launch_help(self):
        with patch("sys.argv", ["llama-launcher", "launch", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
