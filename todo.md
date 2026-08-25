# Memory Estimation Validation - Findings & Next Steps

## Environment

- **Remote Host**: 192.168.10.41 (RTX 4070 12GB, i9-12900K, 31GB RAM)
- **Model**: Qwen3.5-9B-Q4_K_M (5.3GB, 32 layers, hybrid SSM/attention, full attention every 4th layer)
- **llama.cpp**: af1127d3c (CUDA 8.9)
- **Test Date**: 2026-07-25

## Test Matrix (18 configs)

- GPU layers: 0, 8, 16, 24, 32
- Context size: 512, 1024, 2048, 4096, 8192, 16384
- KV offload: True/False
- Cache type: f16, q8_0, q4_0
- Ubatch: 128, 256, 512, 1024

## Results Summary

| Parameter Group | Best Accuracy | Worst Accuracy | Typical Error |
|-----------------|--------------|----------------|---------------|
| Context Size | -4.2% (ctx=512) | -11.9% (ctx=16384) | ±5% |
| KV Offload | -20.9% | -20.9% | -21% |
| Cache Type | -35% (q4_0) | -48% (q8_0) | -42% |
| Ubatch | -45.7% (1024) | -47.4% (128) | -47% |
| GPU Layers | -47% (32) | -85% (0) | -60% |

## Detailed Findings

### 1. Context Size - ACCURATE (±5% typical)
- Estimator KV cache formula is correct for Qwen3.5's hybrid attention
- Minor underestimation at higher contexts (8192, 16384) is acceptable

### 2. GPU Layers - SEVERE UNDERESTIMATION
- gpu_layers=0: -85% (all-CPU mode, estimator reports near-zero VRAM but llama.cpp still allocates CUDA driver overhead)
- gpu_layers=32: -47% (all layers on GPU)
- The estimator doesn't account for:
  - CUDA context overhead (currently fixed 300MB per GPU, but actual appears higher)
  - Memory fragmentation / driver reservation
  - Qwen3.5 hybrid architecture: KV cache only on 8/32 layers, but weight buffers may differ

### 3. Cache Type - UNDERESTIMATION
- q4_0: -35%, q8_0: -48%
- Estimator assumes cache is fully GPU-resident when kv_offload=True
- Actual behavior: llama.cpp may allocate additional internal buffers for quantized KV types

### 4. Ubatch - CONSISTENTLY -47% UNDERESTIMATE
- All ubatch values (128-1024) show ~47% error
- Estimator formula: `hidden * ubatch * 3 * 2 * layers / 1024^2 * 1.25`
- This is the **inference buffer** component. The consistent error suggests the 25% headroom multiplier (`1.25`) is far too low for Qwen3.5
- **Key insight**: ubatch size has no effect on actual VRAM usage (all configs cluster around 10107-10267 MiB), suggesting llama.cpp's internal buffer sizing is not linearly proportional to ubatch

### 5. Context 4096 Anomaly
- ctx=4096 shows 10107 MiB VRAM (same as all-GPU layers), identical to ctx=1024 gpu_layers=32
- This is likely the baseline "model load" VRAM. Other ctx values at gpu_layers=32 should be checked separately.

## Root Cause Analysis

The estimator's VRAM formula is missing or underestimating:

1. **CUDA driver/runtime overhead**: Fixed 300MB assumption is too low. Actual appears to be ~1500-2000MB.
2. **Inference buffer multiplier**: The 1.25x headroom factor is insufficient. Actual buffers are ~2.3-2.5x larger than the base formula.
3. **qwen35 hybrid attention**: The KV cache calculation accounts for `attn_interval=4` (8 KV layers), but weight distribution and compute buffers may differ from standard llama.
4. **Multi-sequence / unified KV cache**: llama.cpp uses `kv_unified=true` by default with `n_parallel=4`, which allocates KV cache slots upfront regardless of actual concurrent requests.

## Next Steps (Priority Order)

### [HIGH] Fix inference buffer multiplier
- Current: `1.25x` headroom
- Target: ~2.5x based on ubatch=1024 data point
- Affects ALL configurations that use GPU layers
- File: `estimator/estimator.py` line ~1229: `_compute_inference_buffers` returns `total_mb * 1.25`
- **Action**: Increase multiplier from 1.25 to ~2.5, or make it data-driven

### [HIGH] Recalibrate CUDA runtime overhead
- Current: 300MB fixed per GPU
- Actual measured overhead at gpu_layers=0: ~2035 MiB (includes CUDA init, driver, model loading overhead even without GPU layers)
- At gpu_layers=32: model weights ~5099 MiB, but actual is ~10107 MiB
- Implied overhead: ~10107 - 5099 = ~5008 MiB for context + buffers + CUDA runtime
- **Action**: Separate CUDA runtime overhead from model-dependent overhead. CUDA init alone appears to be ~1500-2000MiB.

### [MEDIUM] Validate cache type overhead
- Test with actual llama-server to understand why q8_0/q4_0 cache types show 35-48% underestimation
- May need architecture-specific cache buffer adjustments

### [MEDIUM] Architecture-specific adjustments
- Qwen3.5 hybrid SSM/attention differs from standard llama
- Test other architectures (standard llama, mistral, phi, gemma) to see if underestimation is universal or Qwen3.5-specific
- Consider per-architecture multiplier adjustments

### [LOW] Add confidence intervals
- Instead of a single point estimate, show "likely range" based on historical measurement data
- Could track past measurements and update multipliers over time

### [LOW] Add real-measurement feedback loop
- After a server runs, record actual VRAM usage
- Compare to prediction and log deltas
- Use deltas to suggest parameter adjustments

## Data Files

- `/tmp/test_outputs/mem_validation_YYYYMMDD_HHMM.json` - Raw measurement data
- Remote SSH config: user=jaw, host=192.168.10.41

## Key Code Locations

- `estimator/estimator.py:1191-1229` - `_compute_inference_buffers` (inference buffer formula + 1.25x headroom)
- `estimator/estimator.py:1156-1189` - `_compute_kv_cache_per_token` (KV cache formula - seems accurate)
- `estimator/estimator.py:908-919` - Backend overhead constants (CUDA=300MB, HIP=250MB, etc.)
- `estimator/estimator.py:936-954` - GPU/CPU weight split logic
