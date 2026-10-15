# How the VRAM / RAM estimate works

The *Resource Usage* panel predicts what `llama-server` will allocate before
you launch it. The model is deliberately simple and is calibrated against the
sizes llama-server itself prints (`load_tensors`, `llama_kv_cache`,
`llama_memory_recurrent`, `sched_reserve` with `-lv 4`) plus the process's
`nvidia-smi` footprint.

## Terms

| Term | Formula | Where it lives |
|---|---|---|
| Model weights | Sum of the GGUF tensor sizes (block-exact, same table as ggml) | The output layer is offloaded first, then the last `N-1` repeating layers (`--gpu-layers N`); the token embedding always stays in host memory. `--cpu-moe` / `--n-cpu-moe` keep expert tensors on the host. |
| KV cache | `ctx × kv_heads × (key_len + value_len) × bytes(cache type)` per attention layer. Sliding-window layers are capped at `window + ubatch`; SSM / linear-attention layers have no KV cache. The context is shared by all slots. | Follows the layer (GPU if the layer is offloaded and `--kv-offload` is on). |
| Recurrent state | `slots × ((conv_kernel-1) × (inner + 2 × groups × state) + inner × state) × 4` per recurrent layer | Follows the layer. |
| Compute buffer | `ubatch × (3 × n_ff + 2 × n_embd) × 4` + attention mask `ubatch × ctx × 2`; without flash attention the full score matrix `ubatch × ctx × heads × 4` instead. MoE: `n_ff` is `2 × used experts × expert_ff + shared_ff`, plus ~370 MiB when experts run on the CPU. | GPU whenever a CUDA device exists (even with 0 layers offloaded). |
| Runtime | 570 MiB per CUDA GPU (measured on an RTX 5090, CUDA 13; older stacks a little less) | GPU |
| Draft model | Its own weights + its own KV cache at the same context | GPU unless `--spec-draft-ngl 0` |
| Vision projector | Its weights + ~256 MiB image-encoding buffers | GPU unless `--no-mmproj-offload` |
| RAM | Host-side weights (page cache when mmap'd) + CPU-side KV/state + host compute buffers + ~400 MB process overhead | |

## Calibration (llama.cpp build 9741, RTX 5090 32 GB)

| Run | Measured MiB | Estimated MiB | Error |
|---|---|---|---|
| Qwen2.5-VL-3B Q4_K_M, ctx 8k | 2780 | 2773 | -0.2% |
| same, ctx 32k, flash attention off | 4666 | 4685 | +0.4% |
| same, ctx 32k, flash attention on | 3668 | 3661 | -0.2% |
| same, ctx 32k, ubatch 1024 | 3776 | 3766 | -0.3% |
| same, ctx 32k, ubatch 128 | 3586 | 3583 | -0.1% |
| same, ctx 8k, 10/37 layers | 1400 | 1378 | -1.6% |
| same, ctx 8k, 0 layers | 602 | 650 | +8.1% |
| same, ctx 32k, q8_0 KV cache | 3132 | 3121 | -0.3% |
| Qwen3.8-27B Q6_K (hybrid SSM), ctx 8k | 22638 | 22642 | +0.0% |
| same, ctx 32k | 24198 | 24202 | +0.0% |
| same, ctx 32k, 1 slot | 23748 | 23744 | -0.0% |
| same, ctx 8k, 30/66 layers | 11200 | 11060 | -1.3% |
| same, ctx 8k + BF16 mmproj | 23776 | 23786 | +0.0% |
| Qwen3.6-35B-A3B Q4_K_M (MoE), ctx 8k | 20966 | 20951 | -0.1% |
| same, `--n-cpu-moe 10` | 16618 | 16605 | -0.1% |
| same, `--cpu-moe` | 2734 | 2721 | -0.5% |

Mean absolute error 0.8%. The remaining uncertainty is the per-GPU runtime
figure, which depends on the driver, and the compute buffer when layers are
split between CPU and GPU (extra staging buffers, ~1-2%).

## Reproducing

```bash
llama-server -m model.gguf -c 8192 -ngl 99 -fit off -lv 4 2>&1 | \
  grep -E "model buffer size|KV buffer size|RS buffer size|compute buffer size"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

`-fit off` matters: with the default `--fit on`, llama-server silently trims
`--gpu-layers` and the context to what fits, so the numbers would not
correspond to the requested configuration.
