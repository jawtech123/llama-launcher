# llama-launcher Comprehensive Test Plan

## Overview

This document defines the exhaustive test suite for llama-launcher, covering all viable parameter combinations, connection modes (local + SSH remote), edge cases, and error handling paths.

**Target:** ~456 new tests + 8 code changes
**Remote host:** 192.168.10.41, user: jaw, password: [redacted]

---

## Pre-Implementation Code Changes (Phase 0)

### 0a. `core/parser.py` — Unrecognized + duplicate detection with alerting
- Return signature: `(params, unrecognized_flags, duplicates)` — 3-tuple (was 2-tuple)
- Track seen fields in a set; on same-field repeat → add to `duplicates` list + `logger.warning("Duplicate flag: %s (keeping last value)", arg)`
- Unrecognized flags already collected; add `logger.warning("Unrecognized flag: %s", flag)` per flag
- Return type: `Tuple[Dict[str, Any], List[str], List[str]]`

### 0b. `gui/tabs/import_tab.py` — GUI alert for unrecognized/duplicate
- Update call: `params, unrecognized, duplicates = parse_launch_command(command_str)`
- Add duplicates section to result display
- Update QMessageBox to mention both counts

### 0c. `gui/tabs/preview_tab.py` — Make command editable
- Change `setReadOnly(True)` → `setReadOnly(False)`
- Add "Regenerate" button next to "Copy Command"
- When user edits command manually → show note: "✎ Edited command — click Regenerate to restore"
- When user clicks "Regenerate" → overwrite with generated command from config
- Launch uses text content as-is (whether generated or manually edited)
- Editing the command does NOT trigger config regeneration (stays until regenerated)

### 0d. `remote/ssh.py` — Retry transient failures
- Wrap connect/exec calls with retry loop: max 3 retries, fixed 1-second delay
- Retries only on: ConnectionResetError, TimeoutError, paramiko transport errors
- Not on: auth failures (3 attempts then ERROR state)
- Fixed 1-second delay between retries (no exponential backoff)

### 0e. `cli/main.py` — Add `--extra-flags` argument
- Add `--extra-flags` argument (N=N) to accept arbitrary flags
- Pass through to `config.extra_flags`
- Shown in command preview output

### 0f. `core/validator.py` — model_path extension check
- Validate `model_path` ends with `.gguf`, `.bin`, or `.safetensors`
- Raises `ValidationError` if not

---

## Test Modules

### Module 1: Core Parser & Builder Tests

#### 1a. `test_parser.py` — Expand existing 8 → ~40 tests

**Existing tests (keep as-is):**
- `test_basic_command_with_executable_prefix`
- `test_flag_equals_value_form`
- `test_negative_values_are_consumed_not_treated_as_flags`
- `test_bare_boolean_flags_and_no_variants`
- `test_unrecognized_flags_are_collected`
- `test_bare_model_path_token`
- `test_quoted_path_with_spaces`
- `test_type_coercion_stays_string_for_text_fields`

**New tests:**
- `test_duplicate_same_field_keeps_last_value` — `--ctx-size 4096 --ctx-size 8192` → 8192, 1 duplicate
- `test_duplicate_same_field_three_times` — `--threads 4 --threads 8 --threads 16` → 16, 2 duplicates
- `test_mixed_duplicates_and_unrecognized` — both lists populated simultaneously
- `test_all_short_aliases` — -m, -t, -c, -b, -ngl, -ts, -mg, -ctk, -ctv, -fa, -tb, -ub
- `test_all_negation_forms` — --no-mmap, --no-ignore-eos, --no-backend-sampling, --no-cont-batching, --no-mlock, --no-perf, --no-escape, --no-kv-offload, --no-repack, --no-direct-io, --no-cache-idle-slots, --no-context-shift, --no-jinja, --no-embeddings, --no-reranking, --no-ui, --no-agent, --no-log-prefix, --no-log-timestamps, --no-kv-unified, --no-special, --no-spm-infill, --no-mmproj-auto, --no-mmproj-offload, --no-warmup, --no-spec-draft-backend-sampling, --no-cpu-moe, --no-op-offload, --no-reuse-port, --no-ui-mcp-proxy, --no-swa-full, --no-check-tensors
- `test_boolean_equals_value_form` — --mmap=false, --mmap=true, --verbose=1, --verbose=0
- `test_float_coercion_edge_cases` — 0.05, 1.75, 2.0, -1.0, scientific notation
- `test_integer_overflow_stays_int` — --context-size=999999999 (stays int, validator catches later)
- `test_missing_value_after_flag` — --model with no path following → warning logged, skipped
- `test_bare_executable_prefix_variants` — ./llama-server, nohup llama-server, /usr/bin/llama-server
- `test_empty_command_string` → empty params, empty unrecognized, empty duplicates
- `test_command_only_unrecognized_flags` → no params, unrecognized populated, no duplicates
- `test_command_only_duplicate_flags` → duplicates populated, unrecognized empty
- `test_complex_mixed_command` — all short + long + negation + =value + duplicates + unrecognized
- `test_three_tuple_return_type` — verify `params, unrec, dups = parse_launch_command(...)` works
- `test_duplicate_different_fields_ok` — --ctx-size 4096 --threads 8 (different fields, no duplicates)
- `test_flag_with_equals_and_value` — --model=/path/to/model.gguf --ctx-size=8192
- `test_negation_boolean_with_value` — --no-mmap=false (should set mmap=True)

#### 1b. `test_builder.py` — Expand existing 9 → ~50 tests

**Existing tests (keep as-is):**
- `test_default_command_structure`
- `test_non_default_values_emitted`
- `test_path_quoting`
- `test_legacy_flag_exclusion`
- `test_mmproj_draft_uniqueness`
- `test_boolean_flags`
- `test_extra_flags_appended`
- `test_build_parse_roundtrip`
- `test_bare_model_path`

**New tests:**
- `test_all_cache_types` — f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
- `test_all_backends` — auto, cuda, metal, vulkan, cpu, sycl
- `test_all_split_modes` — none, layer, row, tensor
- `test_all_flash_attn` — auto, on, off
- `test_all_pooling` — none, mean, cls, last
- `test_all_log_colors` — auto, enable, disable
- `test_all_fit_modes` — on, off
- `test_all_spec_types` — none, ngram, mod, simple, map_k, map_k4v
- `test_all_negation_boolean_forms` — every default-True boolean produces --no-* when False
- `test_path_quoting_spaces` — paths with spaces properly quoted
- `test_path_quoting_brackets` — paths with [brackets] properly quoted
- `test_path_quoting_special_chars` — paths with $HOME, semicolons, pipes, backticks
- `test_sentinel_values` — threads=0 (auto), predict=-1 (infinite), gpu_layers=-1 (all)
- `test_empty_str_fields_not_emitted` — --tensor-split "" should NOT appear
- `test_parallel_emit_override` — value > 0 emits, 0/-1 suppressed
- `test_extra_flags_verbatim` — extra_flags = ["--foo", "bar"] → appended as-is
- `test_partial_roundtrip` — config → build → parse → modify → build (predictable diff)
- `test_all_25_groups_represented` — every advanced-tab group field appears in output
- `test_dry_sequence_breaker` — --dry-sequence-breaker with \\n value
- `test_logit_bias` — --logit-bias 15032:1.0,15033:-1.0
- `test_samplers_chain` — --samplers with semi-colon chain
- `test_control_vector_layer_range` — --control-vector-layer-range 0-20
- `test_multi_gpu_combo` — tensor_split + split_mode + main_gpu together
- `test_full_speculative_decoding_ngram` — all spec fields with ngram type
- `test_full_speculative_decoding_mod` — all spec fields with mod type
- `test_yarn_all_fields` — all 5 YaRN fields with non-default values
- `test_cpu_affinity_main` — all main thread CPU fields
- `test_cpu_affinity_batch` — all batch thread CPU fields
- `test_spec_draft_cpu_affinity` — all 10 spec-draft CPU fields
- `test_server_options_combo` — jinja + embeddings + reranking + agent + cont_batching + warmup + ui
- `test_generation_fields` — reverse_prompt + pooling + lookup cache + spm_infill + special
- `test_multimodal_fields` — all mmproj + image token fields
- `test_server_extras_fields` — alias + tags + embd_normalize + reuse_port + ui_mcp_proxy
- `test_model_sources_fields` — model_url + docker_repo + hf_repo + hf_file + hf_token
- `test_logging_fields` — all 7 logging fields
- `test_numa_fields` — numa + fit + fit_target + override_tensor + check_tensors + override_kv + op_offload
- `test_lora_fields` — lora + lora_scaled + control_vector + control_vector_scaled + control_vector_layer_range
- `test_each_field_individual` — every non-default field individually set → correct flag emission

#### 1c. `test_validator.py` — Expand existing 10 → ~40 tests

**Existing tests (keep as-is):**
- `test_valid_config_passes`
- `test_missing_model_path_raises`
- `test_range_validation`
- `test_backend_validation`
- `test_cache_type_validation`
- `test_sycl_support`
- `test_negative_sentinel_values_valid`
- `test_zero_values_valid`
- `test_none_values_skipped`
- `test_bool_validation`

**New tests:**
- `test_model_path_gguf_extension_valid`
- `test_model_path_bin_extension_valid`
- `test_model_path_safetensors_extension_valid`
- `test_model_path_txt_extension_invalid`
- `test_model_path_no_extension_invalid`
- `test_all_int_field_range_boundaries` — parametrized: at min, at max, below min, above max
- `test_all_float_field_range_boundaries` — parametrized: at min, at max, below, above
- `test_all_choice_fields_valid` — each valid choice passes
- `test_all_choice_fields_invalid` — each invalid choice fails
- `test_bool_true_valid`
- `test_bool_false_valid`
- `test_bool_int_rejected` — value=1 for bool field raises
- `test_bool_string_rejected` — value="true" for bool field raises
- `test_host_empty_raises`
- `test_host_whitespace_only_raises`
- `test_model_path_none_raises`
- `test_model_path_empty_raises`
- `test_model_path_non_string_raises`
- `test_150_field_sweep_all_valid` — every field set to valid non-default value → passes
- `test_150_field_sweep_all_invalid` — every field set to invalid boundary → fails on first
- `test_extra_fields_not_in_registry_no_crash`
- `test_validator_does_not_crash_on_none_values`

#### 1d. `test_fields.py` — Expand existing 8 → ~15 tests

**Existing tests (keep as-is):**
- `test_fields_count_matches_config`
- `test_flag_uniqueness`
- `test_group_validity`
- `test_negation_requirements`
- `test_parser_map_coverage`
- `test_legacy_migration`
- `test_fields_match_config_fields`
- `test_complete_registry`

**New tests:**
- `test_cli_fields_completeness` — every non-empty flag field in CLI_FIELDS
- `test_flag_to_field_all_covered` — every CLI_FIELDS flag + alias + negation in FLAG_TO_FIELD
- `test_group_order_complete_permutation` — GROUP_ORDER matches all groups used
- `test_all_groups_have_fields` — every group in GROUP_ORDER has at least one field
- `test_secret_flag_correctness` — only hf_token and password fields are secret
- `test_labels_non_empty` — all CLI_FIELDS have non-empty label
- `test_tooltips_non_empty` — all CLI_FIELDS have non-empty tooltip
- `test_aliases_no_overlap` — no alias conflicts with other field flags

#### 1e. `test_presets.py` — Expand existing 6 → ~15 tests

**Existing tests (keep as-is):**
- `test_builtin_presets_independent`
- `test_unknown_preset_error`
- `test_custom_preset_roundtrip`
- `test_override_built_in`
- `test_delete_custom_preset`
- `test_builtin_presets_cannot_be_deleted`

**New tests:**
- `test_builtin_presets_cannot_be_modified` — save_custom_preset with same name replaces
- `test_custom_preset_all_150_fields` — preset with all fields populated
- `test_preset_name_special_characters` — spaces, hyphens, unicode
- `test_preset_file_permissions_600`
- `test_preset_with_none_optional_fields`
- `test_concurrent_preset_saves`
- `test_preset_empty_description`
- `test_preset_long_name` — >100 character name

---

### Module 2: Remote Connection Tests

#### 2a. `test_local_connection.py` — NEW, ~35 tests

- `test_connect_success` — llama-server exists: returns True, state=CONNECTED
- `test_connect_failure` — llama-server not found: returns False, state=ERROR
- `test_disconnect_terminates_log_process`
- `test_disconnect_preserves_state`
- `test_start_server_valid_command` — writes PID file, returns True
- `test_start_server_invalid_command` — returns False
- `test_stop_server_tracked_pid` — kills tracked PID, removes PID file
- `test_stop_server_no_tracked_pid` — returns True (no-op)
- `test_stop_server_dead_pid` — falls back gracefully
- `test_restart_server_success`
- `test_restart_server_failure`
- `test_get_server_pid_stored`
- `test_get_server_pid_pid_file`
- `test_get_server_pid_pgrep`
- `test_get_server_pid_none`
- `test_get_status_stopped` — returns STOPPED state
- `test_get_status_running` — returns CPU%, RAM, uptime
- `test_get_status_with_gpu` — parses nvidia-smi output
- `test_get_status_without_gpu` — handles missing nvidia-smi
- `test_tail_log_present`
- `test_tail_log_absent`
- `test_list_directory_valid`
- `test_list_directory_nonexistent`
- `test_upload_file_success`
- `test_upload_file_with_progress`
- `test_upload_file_failure`
- `test_download_file_success`
- `test_download_file_with_progress`
- `test_download_file_failure`
- `test_create_directory_success`
- `test_delete_file_success`
- `test_rename_file_success`
- `test_pid_file_corruption` — non-numeric content handled
- `test_concurrent_get_status` — two threads simultaneously

#### 2b. `test_ssh_connection.py` — Expand existing 4 → ~45 tests

**Existing tests (keep as-is):**
- `test_disconnect_clears_state`
- `test_shutdown_blocks_reconnect`
- `test_get_client_handles_none_transport`
- `test_thread_safe_disconnect`

**New tests:**
- `test_connect_password_auth` — connects with password
- `test_connect_key_auth` — connects with SSH key
- `test_connect_wrong_password` — returns False, state=ERROR
- `test_connect_wrong_hostname` — returns False, timeout
- `test_connect_paramiko_missing` — state=ERROR
- `test_disconnect_closes_client_and_sftp`
- `test_shutdown_blocks_all_reconnects`
- `test_execute_success` — returns True
- `test_execute_failure` — returns False
- `test_execute_and_capture_stdout` — returns stdout
- `test_execute_and_capture_stderr_on_error` — raises with stderr
- `test_upload_file_sftp_progress`
- `test_upload_file_to_nonexistent_dir` — creates parent dirs
- `test_download_file_sftp`
- `test_list_directory_valid`
- `test_list_directory_nonexistent`
- `test_list_directory_permission_denied`
- `test_create_directory_new`
- `test_create_directory_existing`
- `test_create_directory_nested`
- `test_delete_file_present`
- `test_delete_file_missing`
- `test_rename_file_success`
- `test_rename_file_failure`
- `test_get_server_pid_stored`
- `test_get_server_pid_pid_file`
- `test_get_server_pid_pgrep`
- `test_get_server_pid_none`
- `test_get_status_full_parsing` — CPU, RAM, GPU, port
- `test_start_server_pid_returned`
- `test_start_server_no_pid`
- `test_start_server_error`
- `test_stop_server_graceful_killsigkill_fallback`
- `test_stop_server_already_dead`
- `test_restart_server_cycle`
- `test_tail_log_present`
- `test_tail_log_missing`
- `test_scan_gguf_files`
- `test_scan_bin_files`
- `test_scan_safetensors_files`
- `test_model_family_detection` — llama, mistral, qwen, codellama, phi, gemma, unknown
- `test_quant_type_detection` — Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q4_1, Q3_K_M, Q3_K_S, Q2_K
- `test_concurrent_connect_disconnect`
- `test_get_sftp_dead_channel_recovery`
- `test_transport_active_none_client`
- `test_exec_quick_no_hang` — nohup'd process doesn't hang
- `test_mkdirs_nested`
- `test_connection_drop_recovery`
- `test_retry_transient_failure` — max 3 retries on ConnectionResetError
- `test_retry_auth_failure` — no retry on auth failure, 3 attempts then ERROR
- `test_retry_timeout` — retries on TimeoutError
- `test_retry_exhausted` — all 3 retries fail → state=ERROR

#### 2c. `test_server_manager.py` — NEW, ~25 tests

- `test_add_server_local_creates_local_connection`
- `test_add_server_ssh_creates_ssh_connection`
- `test_add_server_persist_true_saves_to_json`
- `test_add_server_persist_false_no_save`
- `test_remove_server_disconnects_and_removes`
- `test_get_connection_returns_correct_type`
- `test_get_or_create_connection_reuses`
- `test_get_or_create_connection_creates_new`
- `test_load_saved_servers_from_storage`
- `test_get_all_configs_returns_list`
- `test_connect_delegates`
- `test_disconnect_delegates`
- `test_disconnect_all_cleans_up`
- `test_start_server_delegates`
- `test_stop_server_delegates`
- `test_restart_server_delegates`
- `test_execute_delegates`
- `test_execute_and_capture_delegates`
- `test_execute_and_capture_unknown_raises`
- `test_upload_file_delegates`
- `test_download_file_delegates`
- `test_list_directory_delegates`
- `test_create_directory_delegates`
- `test_delete_file_delegates`
- `test_rename_file_delegates`

#### 2d. `test_ssh_key.py` — NEW, ~10 tests

- `test_key_pair_generation_files_created`
- `test_key_pair_public_matches_private`
- `test_install_to_remote_copies_to_authorized_keys`
- `test_install_verification_login_works`
- `test_existing_key_not_regenerated`
- `test_directory_created_if_missing`
- `test_private_key_permissions_600`
- `test_key_generation_failure_handled`

#### 2e. `test_storage.py` — NEW, ~10 tests

- `test_save_server_writes_json`
- `test_save_server_preserves_password`
- `test_load_servers_returns_list`
- `test_remove_server_from_storage`
- `test_storage_file_permissions_600`
- `test_corrupt_json_handling`
- `test_empty_storage_returns_empty`
- `test_multiple_servers_crud`

---

### Module 3: GPU, Estimation & Models

#### 3a. `test_gpu_detector.py` — NEW, ~15 tests

- `test_cuda_detection_success` — nvidia-smi returns GPU info
- `test_cuda_detection_not_found` → CPU fallback
- `test_metal_detection` — macOS environment
- `test_vulkan_detection` — vulkaninfo available
- `test_multi_gpu` — gpu_infos populated
- `test_no_gpu_cpu_fallback` — supported=False
- `test_invalid_nvidia_smi_output`
- `test_gpu_zero_vram`
- `test_driver_version_parsing`

#### 3b. `test_estimator.py` — Expand existing 5 → ~25 tests

**Existing tests (keep as-is):**
- `test_metadata_cache_reuse`
- `test_metadata_cache_invalidation`
- `test_metadata_cache_missing_file`
- `test_vram_no_double_counting`
- `test_vram_flash_workspace`
- `test_vram_remaining`

**New tests:**
- `test_multi_gpu_distribution` — tensor_split across GPUs
- `test_kv_cache_calculation` — various context sizes
- `test_draft_model_vram_overhead`
- `test_mmproj_vram_overhead`
- `test_safety_margin_7_percent_applied`
- `test_quantization_effect_f16_vs_q4_0_vs_q8_0`
- `test_speculative_decoding_vram`
- `test_context_256_edge`
- `test_context_65536_edge`
- `test_gpu_layers_zero`
- `test_gpu_layers_negative_one`
- `test_invalid_gguf_missing_metadata`
- `test_gguf_unexpected_keys`

#### 3c. `test_model_scanner.py` — NEW, ~10 tests

- `test_gguf_files_detected`
- `test_bin_files_detected`
- `test_safetensors_files_detected`
- `test_non_model_files_ignored`
- `test_recursive_subdirectory_scanning`
- `test_empty_directory`
- `test_large_file_size_mb_calculation`
- `test_family_detection_all_types` — llama, mistral, qwen, codellama, phi, gemma, unknown
- `test_quant_type_detection_all_patterns` — Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q4_1, Q3_K_M, Q3_K_S, Q2_K
- `test_missing_file_handling`

#### 3d. `test_model_storage.py` — NEW, ~5 tests

- `test_save_scan_results`
- `test_load_scan_results`
- `test_per_server_cache_isolation`
- `test_cache_invalidation`
- `test_corrupt_cache_handling`

---

### Module 4: Benchmark, CLI & Preview Tab

#### 4a. `test_benchmark_parser.py` — Expand existing 7 → ~15 tests

**Existing tests (keep as-is):**
- `test_server_log_parsing`
- `test_old_cli_format_parsing`
- `test_aggregation`
- `test_unrelated_line_filtering`
- `test_storage_roundtrip`
- `test_fingerprint_stability`
- `test_latest_session_lookup`

**New tests:**
- `test_multiple_slots_aggregation`
- `test_empty_log`
- `test_malformed_timing_lines`
- `test_zero_tokens`
- `test_negative_times`
- `test_extremely_high_tps`
- `test_partial_timing_line`

#### 4b. `test_benchmark_fingerprint.py` — NEW, ~5 tests

- `test_same_config_same_fingerprint`
- `test_different_context_different_fingerprint`
- `test_different_gpu_layers_different_fingerprint`
- `test_full_150_field_config_stable`
- `test_empty_config_fingerprint`

#### 4c. `test_benchmark_storage.py` — Expand existing → ~10 tests

**New/expanded tests:**
- `test_save_session`
- `test_load_sessions_sorted_by_date`
- `test_max_500_cap_oldest_removed`
- `test_latest_session_returns_most_recent`
- `test_delete_session`
- `test_corrupt_file_handling`
- `test_empty_storage`
- `test_session_sort_order_descending`
- `test_session_missing_optional_fields`

#### 4d. `test_cli.py` — NEW, ~15 tests

- `test_detect_subcommand_prints_gpu_info`
- `test_scan_subcommand_lists_models`
- `test_launch_dry_run_with_model`
- `test_launch_dry_run_with_preset`
- `test_launch_dry_run_with_all_cli_args`
- `test_launch_no_dry_run_calls_subprocess`
- `test_launch_invalid_preset_error_exit_1`
- `test_launch_missing_model_validation_error`
- `test_no_subcommand_help_and_exit_1`
- `test_unknown_flag_argparse_error`
- `test_kv_cache_all_9_types`
- `test_backend_all_6_types`
- `test_logging_setup_verified`
- `test_exit_code_0_on_success`
- `test_exit_code_1_on_error`

#### 4e. `test_preview_tab.py` — NEW, ~10 tests

- `test_command_text_is_editable` — setReadOnly is False
- `test_generated_command_displayed`
- `test_user_edit_persists_across_config_changes`
- `test_regenerate_overwrites_user_edits`
- `test_launch_uses_text_as_is` — edited or generated
- `test_copy_command_works`
- `test_server_output_appended_to_log`
- `test_running_state_ui_update`
- `test_edit_note_shown_when_user_modifies`
- `test_edit_note_cleared_after_regenerate`

---

### Module 5: Integration Tests — Local

#### 5a. `test_integration_local.py` — NEW, ~30 tests

- `test_full_local_workflow` — config → build → start → poll → stop
- `test_validation_blocks_invalid_launch`
- `test_build_parse_build_roundtrip`
- `test_multiple_restart_cycles`
- `test_log_file_creation`
- `test_pid_file_lifecycle` — create, read, remove
- `test_concurrent_status_polling`
- `test_server_non_default_port`
- `test_server_embeddings_enabled`
- `test_command_with_extra_flags`
- `test_server_binary_not_found_error`
- `test_pid_file_corruption_recovery`
- `test_stop_server_no_process` — no crash
- `test_start_server_timeout` — handled gracefully
- `test_get_status_during_startup`
- `test_command_preview_shows_validation_error`
- `test_command_preview_shows_valid_command`
- `test_resource_estimate_updated`
- `test_local_connection_error_handling`
- `test_manager_multiple_local_servers`

---

### Module 6: Integration Tests — Remote (192.168.10.41)

#### 6a. `test_integration_remote.py` — NEW, ~35 tests

Remote host: 192.168.10.41, user: jaw, password: [redacted]

- `test_ssh_password_auth_connects`
- `test_ssh_key_auth_connects` — after key install
- `test_full_remote_workflow` — connect → config → build → start → poll → stop → disconnect
- `test_file_upload_download_cycle` — SFTP
- `test_remote_model_directory_scan`
- `test_remote_status_with_gpu_parsing`
- `test_remote_log_streaming`
- `test_remote_server_restart`
- `test_connection_drop_recovery` — 3 retries max
- `test_local_and_remote_servers_simultaneously`
- `test_server_switching_no_leaks`
- `test_ssh_key_generation_and_install`
- `test_working_directory_verification`
- `test_multiple_sequential_ssh_connections`
- `test_remote_execute_and_capture_error`
- `test_remote_list_directory_various_paths`
- `test_remote_create_delete_rename_file`
- `test_remote_start_invalid_command`
- `test_remote_stop_already_dead`
- `test_remote_tail_log_missing_file`
- `test_concurrent_remote_status_polling`
- `test_retry_on_transient_failure`
- `test_retry_auth_failure_no_retry`
- `test_retry_timeout`
- `test_retry_all_3_failures`
- `test_remote_sftp_channel_closed_mid_transfer`
- `test_remote_execute_command_success`
- `test_remote_execute_command_failure`
- `test_server_config_updates_reflected`
- `test_manager_disconnect_all_remotes`
- `test_remote_connection_state_transitions`
- `test_remote_pid_tracking_lifecycle`
- `test_remote_log_file_creation`
- `test_remote_server_on_different_port`
- `test_remote_model_scan_empty_directory`

---

### Module 7: Edge Cases & Error Handling

#### 7a. `test_edge_cases.py` — NEW, ~25 tests

- `test_paths_with_spaces`
- `test_paths_with_brackets`
- `test_paths_with_unicode`
- `test_paths_with_dollar_home`
- `test_paths_with_semicolons`
- `test_context_1048576`
- `test_predict_65536`
- `test_batch_65536`
- `test_threads_zero`
- `test_batch_size_1`
- `test_predict_zero`
- `test_gpu_layers_zero`
- `test_gpu_layers_negative_one`
- `test_predict_negative_one`
- `test_threads_batch_negative_one`
- `test_dry_penalty_last_n_negative_one`
- `test_seed_negative_one`
- `test_parallel_negative_one`
- `test_all_booleans_true_simultaneously`
- `test_all_choices_in_single_config`
- `test_empty_string_vs_none_vs_default`
- `test_float_precision_values` — 0.05, 1.75, -1.0, 100.0
- `test_concurrent_builder_calls`
- `test_reentrant_validator_calls`
- `test_config_to_dict_roundtrip`

#### 7b. `test_error_handling.py` — NEW, ~15 tests

- `test_validator_wrong_model_extension`
- `test_builder_extra_flags_with_none`
- `test_parser_coerce_failure_logged`
- `test_ssh_connection_reset_error_retry`
- `test_ssh_timeout_after_retries`
- `test_local_start_server_timeout`
- `test_presets_disk_full_graceful`
- `test_presets_corrupt_json_warning`
- `test_estimator_missing_gguf_metadata`
- `test_sftp_channel_closed_mid_transfer`
- `test_console_log_file_deleted_while_streaming`
- `test_storage_permission_denied`
- `test_migrate_config_dict_unknown_keys_pass_through`
- `test_cli_llama_server_not_found`
- `test_parser_unrecognized_and_duplicates_both_reported`

---

### Module 8: Program Change Verification Tests

#### 8a. `test_parser_3tuple.py` — NEW, ~10 tests

- `test_three_tuple_return`
- `test_duplicate_same_field_in_duplicates_list`
- `test_unrecognized_flag_in_unrecognized_list`
- `test_both_lists_populated_simultaneously`
- `test_logger_warning_called_for_duplicate`
- `test_logger_warning_called_for_unrecognized`
- `test_no_crash_on_empty_lists`
- `test_duplicate_keeps_last_value`
- `test_unrecognized_does_not_affect_params`
- `test_duplicate_same_field_twice_counts_two`

#### 8b. `test_ssh_retry.py` — NEW, ~10 tests

- `test_successful_first_try_no_retry`
- `test_retry_one_success`
- `test_retry_two_success`
- `test_retry_three_fail_then_error_state`
- `test_auth_failure_no_retry`
- `test_timeout_retry`
- `test_connection_reset_retry`
- `test_retry_success_sets_connected`
- `test_retry_exhausted_sets_error`
- `test_fixed_1_second_delay`

#### 8c. `test_validator_extension.py` — NEW, ~5 tests

- `test_gguf_extension_valid`
- `test_bin_extension_valid`
- `test_safetensors_extension_valid`
- `test_txt_extension_invalid`
- `test_no_extension_invalid`

#### 8d. `test_preview_editable.py` — NEW, ~5 tests

- `test_command_text_not_read_only`
- `test_regenerate_button_exists`
- `test_regenerate_restores_generated`
- `test_user_edit_stays_until_regenerate`
- `test_launch_uses_text_content`

---

## Test Execution Order

Each module is independently runnable:

```bash
# Phase 0: Implement code changes first
# (parser.py, import_tab.py, preview_tab.py, ssh.py, cli/main.py, validator.py)

# Then run modules in order:
pytest tests/test_parser.py -v                    # Module 1a
pytest tests/test_builder.py -v                   # Module 1b
pytest tests/test_validator.py -v                 # Module 1c
pytest tests/test_fields.py -v                    # Module 1d
pytest tests/test_presets.py -v                   # Module 1e
pytest tests/test_local_connection.py -v          # Module 2a
pytest tests/test_ssh_connection.py -v            # Module 2b
pytest tests/test_server_manager.py -v            # Module 2c
pytest tests/test_ssh_key.py -v                   # Module 2d
pytest tests/test_storage.py -v                   # Module 2e
pytest tests/test_gpu_detector.py -v              # Module 3a
pytest tests/test_estimator.py -v                 # Module 3b
pytest tests/test_model_scanner.py -v             # Module 3c
pytest tests/test_model_storage.py -v             # Module 3d
pytest tests/test_benchmark_parser.py -v          # Module 4a
pytest tests/test_benchmark_fingerprint.py -v     # Module 4b
pytest tests/test_benchmark_storage.py -v         # Module 4c
pytest tests/test_cli.py -v                       # Module 4d
pytest tests/test_preview_tab.py -v               # Module 4e
pytest tests/test_integration_local.py -v         # Module 5
pytest tests/test_integration_remote.py -v        # Module 6
pytest tests/test_edge_cases.py -v                # Module 7a
pytest tests/test_error_handling.py -v            # Module 7b
pytest tests/test_parser_3tuple.py -v             # Module 8a
pytest tests/test_ssh_retry.py -v                 # Module 8b
pytest tests/test_validator_extension.py -v       # Module 8c
pytest tests/test_preview_editable.py -v          # Module 8d
```

## Summary

| Module | Existing | New | Total |
|--------|----------|-----|-------|
| 1a. Parser | 8 | ~32 | ~40 |
| 1b. Builder | 9 | ~41 | ~50 |
| 1c. Validator | 10 | ~30 | ~40 |
| 1d. Fields | 8 | ~7 | ~15 |
| 1e. Presets | 6 | ~9 | ~15 |
| 2a. Local Connection | 0 | 35 | 35 |
| 2b. SSH Connection | 4 | ~41 | ~45 |
| 2c. Server Manager | 0 | 25 | 25 |
| 2d. SSH Key | 0 | 10 | 10 |
| 2e. Storage | 0 | 10 | 10 |
| 3a. GPU Detector | 0 | 15 | 15 |
| 3b. Estimator | 5 | ~20 | ~25 |
| 3c. Model Scanner | 0 | 10 | 10 |
| 3d. Model Storage | 0 | 5 | 5 |
| 4a. Benchmark Parser | 7 | ~8 | ~15 |
| 4b. Benchmark Fingerprint | 0 | 5 | 5 |
| 4c. Benchmark Storage | 7 | ~3 | ~10 |
| 4d. CLI | 0 | 15 | 15 |
| 4e. Preview Tab | 0 | 10 | 10 |
| 5. Integration Local | 0 | 30 | 30 |
| 6. Integration Remote | 0 | 35 | 35 |
| 7a. Edge Cases | 0 | 25 | 25 |
| 7b. Error Handling | 0 | 15 | 15 |
| 8a. Parser 3-tuple | 0 | 10 | 10 |
| 8b. SSH Retry | 0 | 10 | 10 |
| 8c. Validator Extension | 0 | 5 | 5 |
| 8d. Preview Editable | 0 | 5 | 5 |
| **Total** | **58** | **~456** | **~514** |
