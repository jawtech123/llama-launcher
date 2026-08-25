"""Tests for parser 3-tuple return (params, unrecognized_flags, duplicate_flags)."""
from llama_launcher.core.parser import parse_launch_command


def test_three_tuple_return():
    """Verify parse_launch_command returns exactly 3 values."""
    result = parse_launch_command("--threads 8")
    assert isinstance(result, tuple)
    assert len(result) == 3
    params, unrecognized, duplicates = result


def test_duplicate_same_field_in_duplicates_list():
    """Same flag repeated twice → appears in duplicate_flags."""
    _, _, duplicates = parse_launch_command("--threads 8 --threads 16")
    assert "--threads" in duplicates


def test_unrecognized_flag_in_unrecognized_list():
    """Unknown flag appears in unrecognized_flags."""
    _, unrecognized, _ = parse_launch_command("--threads 8 --unknown-flag")
    assert "--unknown-flag" in unrecognized


def test_both_lists_populated_simultaneously():
    """Duplicates and unrecognized can coexist."""
    params, unrecognized, duplicates = parse_launch_command(
        "--threads 8 --threads 16 --bogus-flag --ctx-size 4096 --ctx-size 8192"
    )
    assert "--threads" in duplicates
    assert "--bogus-flag" in unrecognized
    assert params["threads"] == 16
    assert params["context_size"] == 8192


def test_logger_warning_called_for_duplicate(capsys, caplog):
    """Duplicate flag triggers a warning log."""
    with caplog.at_level("WARNING"):
        parse_launch_command("--threads 8 --threads 16")
    assert any("Duplicate flag" in record.message for record in caplog.records)


def test_logger_warning_called_for_unrecognized(capsys, caplog):
    """Unrecognized flag triggers a warning log."""
    with caplog.at_level("WARNING"):
        parse_launch_command("--threads 8 --totally-bogus")
    assert any("Unrecognized flag" in record.message for record in caplog.records)


def test_no_crash_on_empty_lists():
    """Simple command with no duplicates or unrecognized should not crash."""
    params, unrecognized, duplicates = parse_launch_command("--threads 8")
    assert "threads" in params
    assert "--threads" not in unrecognized
    assert len(duplicates) == 0


def test_duplicate_keeps_last_value():
    """When a flag is duplicated, the last value is used."""
    params, _, _ = parse_launch_command("--threads 4 --threads 8 --threads 16")
    assert params["threads"] == 16

    params2, _, _ = parse_launch_command("--ctx-size 4096 --ctx-size 8192")
    assert params2["context_size"] == 8192


def test_unrecognized_does_not_affect_params():
    """Unrecognized flags should not populate params."""
    params, unrecognized, _ = parse_launch_command("--threads 8 --weird-flag hello --ctx-size 4096")
    assert "threads" in params
    assert "context_size" in params
    assert "weird_flag" not in params
    assert "--weird-flag" in unrecognized


def test_duplicate_same_field_twice_counts_two():
    """Same field repeated 3 times → 2 entries in duplicates."""
    _, _, duplicates = parse_launch_command("--threads 4 --threads 8 --threads 16")
    assert duplicates.count("--threads") == 2


def test_equals_form_duplicate():
    """Duplicate with =value form also detected."""
    params, _, duplicates = parse_launch_command("--threads=4 --threads=8")
    assert params["threads"] == 8
    assert len(duplicates) == 1


def test_mixed_equals_and_space_duplicates():
    """--threads 4 --threads=8 counts as duplicate."""
    params, _, duplicates = parse_launch_command("--threads 4 --threads=8")
    assert params["threads"] == 8
    assert len(duplicates) == 1


def test_bare_token_not_consumed_by_duplicate():
    """Bare model path tokens are not treated as flag values for dedup."""
    params, _, duplicates = parse_launch_command(
        "--threads 8 --model /path/to/model.gguf --threads 16"
    )
    assert params["threads"] == 16
    assert params["model_path"] == "/path/to/model.gguf"
    assert len(duplicates) == 1


def test_empty_command():
    """Empty string → empty params, no duplicates, no unrecognized."""
    params, unrecognized, duplicates = parse_launch_command("")
    assert params == {}
    assert unrecognized == []
    assert duplicates == []


def test_only_whitespace():
    """Whitespace-only string → empty params, no duplicates, no unrecognized."""
    params, unrecognized, duplicates = parse_launch_command("   \t  ")
    assert params == {}
    assert unrecognized == []
    assert duplicates == []


def test_executable_dropped_not_a_flag():
    """Executable prefix is dropped and not treated as unrecognized."""
    params, unrecognized, _ = parse_launch_command("llama-server --threads 8")
    assert "threads" in params
    assert "llama-server" not in unrecognized


def test_unrecognized_equals_form():
    """Unrecognized flags with =value form collected as-is."""
    _, unrecognized, _ = parse_launch_command("--unknown-arg=value")
    assert "--unknown-arg=value" in unrecognized


def test_duplicate_boolean_flag():
    """Boolean flag repeated → duplicate detected, last wins."""
    params, _, duplicates = parse_launch_command("--mmap --no-mmap")
    assert params["use_mmap"] is False
    assert len(duplicates) == 1


def test_negative_value_not_confused_for_flag():
    """Negative numbers are consumed as values, not flags."""
    params, _, _ = parse_launch_command("--threads -1 --seed -1")
    assert params["threads"] == -1
    assert params["seed"] == -1
