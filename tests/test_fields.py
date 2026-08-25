"""Integrity tests for the field registry and config migration."""
import dataclasses
import pytest
from llama_launcher.core.config import LlamaConfig, migrate_config_dict
from llama_launcher.core.fields import (
    Field,
    CLI_FIELDS, FIELDS, FIELDS_BY_NAME, FLAG_TO_FIELD, GROUP_ORDER,
    CACHE_TYPES, BACKENDS, fields_in_group,
    GRP_COMMON, GRP_GPU, GRP_SAMPLING, GRP_THREADS, GRP_CONTEXT,
    GRP_YARN, GRP_CACHE, GRP_NUMA, GRP_LORA, GRP_SOURCES, GRP_LOGGING,
    GRP_SERVER, GRP_GEN, GRP_MM, GRP_SERVER_EXTRA,
    GRP_EXTRA_SAMPLING, GRP_ADAPTIVE, GRP_MIROSTAT, GRP_DRY,
    GRP_SPEC, GRP_SPEC_CACHE, GRP_SPEC_CPU, GRP_NGRAM, GRP_GRAMMAR,
    GRP_EXTRA_CACHE,
)

#: LlamaConfig fields intentionally absent from the registry.
METADATA_FIELDS = {"name", "description", "extra_flags"}


def test_every_registry_field_exists_on_config_with_matching_default():
    cfg = LlamaConfig()
    for f in FIELDS:
        assert hasattr(cfg, f.name), f"missing config field: {f.name}"
        assert getattr(cfg, f.name) == f.default, (
            f"default mismatch for {f.name}: "
            f"config={getattr(cfg, f.name)!r} registry={f.default!r}"
        )


def test_all_config_fields_covered_by_registry():
    cfg_fields = {f.name for f in dataclasses.fields(LlamaConfig)} - METADATA_FIELDS
    missing = cfg_fields - set(FIELDS_BY_NAME)
    assert not missing, f"config fields missing from registry: {sorted(missing)}"


def test_flags_are_unique():
    flags = [f.flag for f in CLI_FIELDS]
    assert len(flags) == len(set(flags)), "duplicate flags in registry"
    assert len(FLAG_TO_FIELD) == len(set(FLAG_TO_FIELD))


def test_groups_are_valid():
    groups = {f.group for f in FIELDS if f.group}
    assert groups <= set(GROUP_ORDER)


def test_bool_fields_with_default_true_have_negation():
    for f in FIELDS:
        if f.kind == "bool" and f.default is True:
            assert f.negates, f"{f.name}: default-True bool needs a --no- form"


def test_parser_map_covers_flags_aliases_negations():
    for f in CLI_FIELDS:
        assert FLAG_TO_FIELD[f.flag] is f
        for a in f.aliases:
            assert FLAG_TO_FIELD[a] is f
        if f.negates:
            assert FLAG_TO_FIELD[f.negates] is f


def test_migrate_config_dict_legacy_keys():
    m = migrate_config_dict({
        "kv_cache_type": "q8_0",
        "mmproj": "/legacy.mmproj.gguf",
        "draft_n_max": 5,
        "draft_n_min": 1,
        "logical_batch_size": 4096,
        "flash_attn": "enable",
        "tensor_split": None,
    })
    assert m["cache_type_k"] == "q8_0"
    assert m["cache_type_v"] == "q8_0"
    assert m["mmproj_model_path"] == "/legacy.mmproj.gguf"
    assert m["spec_draft_n_max"] == 5
    assert m["spec_draft_n_min"] == 1
    assert "logical_batch_size" not in m
    assert "kv_cache_type" not in m
    assert m["flash_attn"] == "on"
    assert m["tensor_split"] == ""


def test_migrate_prefers_explicit_new_values():
    m = migrate_config_dict({
        "mmproj": "/old.gguf",
        "mmproj_model_path": "/new.gguf",
        "draft_n_max": 5,
        "spec_draft_n_max": 9,
    })
    assert m["mmproj_model_path"] == "/new.gguf"
    assert m["spec_draft_n_max"] == 9


def test_migrate_is_noop_for_current_schema():
    data = {"model_path": "/m.gguf", "context_size": 8192, "temp": 0.5}
    m = migrate_config_dict(data)
    assert m["model_path"] == "/m.gguf"
    assert m["context_size"] == 8192
    assert m["temp"] == 0.5


# ---- Additional fields tests ----

def test_fields_has_name_and_flag():
    for f in FIELDS:
        assert f.name, f"FIELD {f} has empty name"
        # flag can be empty for metadata-only fields


def test_cli_fields_have_non_empty_flag():
    for f in CLI_FIELDS:
        assert f.flag, f"CLI field {f.name} has empty flag"


def test_cli_fields_are_subset_of_fields():
    cli_names = {f.name for f in CLI_FIELDS}
    all_names = {f.name for f in FIELDS}
    assert cli_names <= all_names


def test_flag_to_field_coverage():
    """Every CLI field flag, alias, and negation should be in FLAG_TO_FIELD."""
    for f in CLI_FIELDS:
        assert f.flag in FLAG_TO_FIELD
        for alias in f.aliases:
            assert alias in FLAG_TO_FIELD
        if f.negates:
            assert f.negates in FLAG_TO_FIELD


def test_negation_flags_map_to_same_field():
    for f in CLI_FIELDS:
        if f.negates:
            assert FLAG_TO_FIELD[f.negates] is f


def test_aliases_map_to_same_field():
    for f in CLI_FIELDS:
        for alias in f.aliases:
            assert FLAG_TO_FIELD[alias] is f


def test_fields_by_name_keys_match_fields():
    assert set(FIELDS_BY_NAME.keys()) == {f.name for f in FIELDS}


def test_group_order_contains_all_groups():
    groups_in_fields = {f.group for f in FIELDS if f.group}
    missing = groups_in_fields - set(GROUP_ORDER)
    assert not missing, f"groups not in GROUP_ORDER: {sorted(missing)}"


def test_fields_have_valid_kinds():
    valid_kinds = {"int", "float", "bool", "str", "choice"}
    for f in FIELDS:
        assert f.kind in valid_kinds, f"{f.name}: kind={f.kind!r} not valid"


def test_choice_fields_have_choices():
    for f in FIELDS:
        if f.kind == "choice":
            assert f.choices, f"{f.name}: choice field has no choices"
            assert isinstance(f.choices, tuple)


def test_non_choice_fields_have_empty_choices():
    for f in FIELDS:
        if f.kind != "choice":
            assert f.choices == (), f"{f.name}: non-choice field has choices"


def test_cache_types_is_tuple():
    assert isinstance(CACHE_TYPES, tuple)
    assert "f16" in CACHE_TYPES
    assert "q8_0" in CACHE_TYPES
    assert "q4_0" in CACHE_TYPES


def test_backends_is_tuple():
    assert isinstance(BACKENDS, tuple)
    assert "auto" in BACKENDS
    assert "cuda" in BACKENDS
    assert "metal" in BACKENDS
    assert "cpu" in BACKENDS


def test_all_groups_have_name_constants():
    for group in GROUP_ORDER:
        assert group, f"Empty group name in GROUP_ORDER"


def test_fields_in_group_function():
    common_fields = fields_in_group(GRP_COMMON)
    assert len(common_fields) > 0
    for f in common_fields:
        assert f.group == GRP_COMMON


def test_fields_in_group_returns_empty_for_unknown():
    result = fields_in_group("NonExistentGroup")
    assert result == []


def test_fields_in_group_respects_exact_match():
    common = fields_in_group(GRP_COMMON)
    gpu = fields_in_group(GRP_GPU)
    common_names = {f.name for f in common}
    gpu_names = {f.name for f in gpu}
    assert common_names & gpu_names == set()


def test_metadata_fields_not_in_cli_fields():
    cli_names = {f.name for f in CLI_FIELDS}
    for mf in METADATA_FIELDS:
        assert mf not in cli_names, f"{mf} should not be in CLI_FIELDS"


def test_metadata_fields_on_config():
    cfg = LlamaConfig(
        name="test",
        description="desc",
        extra_flags=["--extra"],
    )
    assert cfg.name == "test"
    assert cfg.description == "desc"
    assert cfg.extra_flags == ["--extra"]


def test_migrate_dropped_keys():
    m = migrate_config_dict({
        "logical_batch_size": 4096,
        "kv_cache_type": "f16",
    })
    assert "logical_batch_size" not in m
    assert "kv_cache_type" not in m


def test_migrate_flash_attn_legacy_true():
    m = migrate_config_dict({"flash_attn": "true"})
    assert m["flash_attn"] == "on"


def test_migrate_flash_attn_legacy_false():
    m = migrate_config_dict({"flash_attn": "false"})
    assert m["flash_attn"] == "off"


def test_migrate_flash_attn_legacy_enable():
    m = migrate_config_dict({"flash_attn": "enable"})
    assert m["flash_attn"] == "on"


def test_migrate_flash_attn_legacy_disable():
    m = migrate_config_dict({"flash_attn": "disable"})
    assert m["flash_attn"] == "off"


def test_migrate_preserves_unknown_keys():
    m = migrate_config_dict({"model_path": "/m.gguf", "custom_key": "custom_value"})
    assert m["custom_key"] == "custom_value"


def test_field_has_tooltip():
    for f in FIELDS:
        assert f.tooltip, f"{f.name}: missing tooltip"


def test_field_has_label():
    for f in FIELDS:
        assert f.label, f"{f.name}: missing label"


def test_field_dataclass_is_frozen():
    """Field should be a frozen dataclass (immutable)."""
    import dataclasses
    assert dataclasses.is_dataclass(Field)
    assert dataclasses.fields(Field)  # verify it has fields
    assert Field.__dataclass_params__.frozen is True
