"""Integrity tests for the field registry and config migration."""
import dataclasses

from llama_launcher.core.config import LlamaConfig, migrate_config_dict
from llama_launcher.core.fields import (
    CLI_FIELDS, FIELDS, FIELDS_BY_NAME, FLAG_TO_FIELD, GROUP_ORDER,
)

#: LlamaConfig fields intentionally absent from the registry.
METADATA_FIELDS = {"target_server", "name", "description", "extra_flags"}


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
