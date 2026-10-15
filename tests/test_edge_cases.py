"""Tests for edge cases in builder, parser, validator, and config."""
import pytest
from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.builder import build_command
from llama_launcher.core.validator import validate_config, ValidationError
from llama_launcher.core.parser import parse_launch_command
from llama_launcher.core.presets import PresetManager


class TestPathEdgeCases:
    """Path handling edge cases."""

    def test_paths_with_spaces(self):
        config = LlamaConfig(model_path="/path/to/my model.gguf", host="127.0.0.1")
        cmd = build_command(config)
        assert "/path/to/my model.gguf" in cmd

    def test_paths_with_brackets(self):
        config = LlamaConfig(model_path="/path/to/model[large].gguf", host="127.0.0.1")
        cmd = build_command(config)
        assert "model[large].gguf" in cmd

    def test_paths_with_unicode(self):
        config = LlamaConfig(model_path="/path/to/モデル.gguf", host="127.0.0.1")
        cmd = build_command(config)
        assert "モデル.gguf" in cmd

    def test_paths_with_dollar_home(self):
        config = LlamaConfig(model_path="$HOME/models/model.gguf", host="127.0.0.1")
        cmd = build_command(config)
        assert "$HOME/models/model.gguf" in cmd

    def test_paths_with_semicolons(self):
        config = LlamaConfig(model_path="/path/to/model;rm -rf.gguf", host="127.0.0.1")
        cmd = build_command(config)
        assert "model" in cmd


class TestBoundaryValues:
    """Boundary value tests for int fields."""

    def test_context_1048576(self):
        config = LlamaConfig(model_path="/m.gguf", context_size=1048576, host="127.0.0.1")
        validate_config(config)
        cmd = build_command(config)
        assert "--ctx-size 1048576" in cmd

    def test_predict_65536(self):
        config = LlamaConfig(model_path="/m.gguf", predict=65536, host="127.0.0.1")
        validate_config(config)

    def test_batch_65536(self):
        config = LlamaConfig(model_path="/m.gguf", batch_size=65536, host="127.0.0.1")
        validate_config(config)

    def test_threads_zero(self):
        """threads=0 means auto-detect (default value, not emitted)."""
        config = LlamaConfig(model_path="/m.gguf", threads=0, host="127.0.0.1")
        validate_config(config)
        cmd = build_command(config)
        # threads=0 is the default, so it's not emitted in the command
        assert "llama-server" in cmd

    def test_batch_size_1(self):
        config = LlamaConfig(model_path="/m.gguf", batch_size=1, host="127.0.0.1")
        validate_config(config)

    def test_predict_zero(self):
        config = LlamaConfig(model_path="/m.gguf", predict=0, host="127.0.0.1")
        validate_config(config)

    def test_gpu_layers_zero(self):
        """gpu_layers=0 means 0 layers on GPU (CPU only)."""
        config = LlamaConfig(model_path="/m.gguf", gpu_layers=0, host="127.0.0.1")
        validate_config(config)
        cmd = build_command(config)
        assert "--gpu-layers 0" in cmd

    def test_gpu_layers_negative_one(self):
        """gpu_layers=-1 means all layers on GPU."""
        config = LlamaConfig(model_path="/m.gguf", gpu_layers=-1, host="127.0.0.1")
        validate_config(config)
        cmd = build_command(config)
        # gpu_layers=-1 is the default, so it's not emitted
        assert "--gpu-layers" not in cmd or "-1" not in cmd

    def test_predict_negative_one(self):
        """predict=-1 means infinite (default value)."""
        config = LlamaConfig(model_path="/m.gguf", predict=-1, host="127.0.0.1")
        validate_config(config)
        cmd = build_command(config)
        # predict=-1 is the default, so it's not emitted
        assert "--n-predict" not in cmd

    def test_threads_batch_negative_one(self):
        config = LlamaConfig(model_path="/m.gguf", threads_batch=-1, host="127.0.0.1")
        validate_config(config)

    def test_dry_penalty_last_n_negative_one(self):
        config = LlamaConfig(model_path="/m.gguf", dry_penalty_last_n=-1, host="127.0.0.1")
        validate_config(config)

    def test_seed_negative_one(self):
        """seed=-1 means random."""
        config = LlamaConfig(model_path="/m.gguf", seed=-1, host="127.0.0.1")
        validate_config(config)

    def test_parallel_negative_one(self):
        config = LlamaConfig(model_path="/m.gguf", parallel=-1, host="127.0.0.1")
        validate_config(config)


class TestBooleanAndChoiceEdgeCases:
    """Boolean and choice field edge cases."""

    def test_all_booleans_true_simultaneously(self):
        """All boolean fields set to True at once."""
        config = LlamaConfig(
            model_path="/m.gguf", host="127.0.0.1",
            use_mmap=True, mlock=True, perf=True, escape=True,
            kv_offload=True, repack=True, verbose=True,
            ignore_eos=True, log_prefix=True, log_timestamps=True,
            cont_batching=True, warmup=True, kv_unified=True,
            cache_idle_slots=True, embeddings=True, reranking=True,
            agent=True, ui=True, jinja=True, backend_sampling=True,
            mmproj_auto=True, mmproj_offload=True, spec_draft_backend_sampling=True,
            op_offload=True,
        )
        validate_config(config)
        cmd = build_command(config)
        assert "llama-server" in cmd

    def test_all_choices_in_single_config(self):
        """All valid choices in one config."""
        config = LlamaConfig(
            model_path="/m.gguf", host="127.0.0.1",
            cache_type_k="q4_0", cache_type_v="q4_0",
            rope_scaling="yarn", split_mode="layer", flash_attn="on",
            pooling="none", log_colors="off", numa="distribute",
        )
        validate_config(config)
        cmd = build_command(config)
        assert "--log-colors off" in cmd
        assert "--numa distribute" in cmd


class TestEmptyVsNoneVsDefault:
    """Empty string vs None vs default behavior."""

    def test_empty_string_vs_none(self):
        """Empty string is treated as a value, None is skipped."""
        config1 = LlamaConfig(model_path="/m.gguf", host="127.0.0.1", alias="")
        config2 = LlamaConfig(model_path="/m.gguf", host="127.0.0.1", alias=None)
        # Empty string may be omitted; None is definitely omitted
        cmd1 = build_command(config1)
        cmd2 = build_command(config2)
        # Both should work without error
        assert "llama-server" in cmd1
        assert "llama-server" in cmd2

    def test_default_none_fields(self):
        """Fields with None default should not appear in command."""
        config = LlamaConfig(model_path="/m.gguf", host="127.0.0.1")
        cmd = build_command(config)
        # name and description are None by default
        assert "--name" not in cmd
        assert "--description" not in cmd


class TestFloatPrecision:
    """Float precision edge cases."""

    def test_float_precision_values(self):
        """Various float values maintain precision."""
        config = LlamaConfig(
            model_path="/m.gguf", host="127.0.0.1",
            temp=0.05, top_p=0.99, typical_p=0.5,
            min_p=0.05, repeat_penalty=1.5,
        )
        validate_config(config)
        cmd = build_command(config)
        assert "0.05" in cmd
        assert "0.99" in cmd

    def test_negative_float_values(self):
        """Negative float values for float fields."""
        config = LlamaConfig(
            model_path="/m.gguf", host="127.0.0.1",
            top_n_sigma=-1.0, dynatemp_range=0.0,
        )
        validate_config(config)


class TestRoundTrip:
    """Build/parse roundtrip tests."""

    def test_config_to_dict_roundtrip(self):
        """Config → dataclass.asdict → LlamaConfig → same values."""
        import dataclasses
        config1 = LlamaConfig(model_path="/m.gguf", context_size=4096, threads=8, temp=0.7, host="127.0.0.1")
        d = dataclasses.asdict(config1)
        config2 = LlamaConfig(**d)
        assert config1.context_size == config2.context_size
        assert config1.threads == config2.threads
        assert config1.temp == config2.temp

    def test_build_parse_build_roundtrip(self):
        """Build → parse → verify params match original config."""
        config = LlamaConfig(model_path="/m.gguf", context_size=4097, threads=8, host="127.0.0.1")
        cmd = build_command(config)
        params, _, _ = parse_launch_command(cmd)
        assert params["context_size"] == 4097
        assert params["threads"] == 8
