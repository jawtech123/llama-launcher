# llama-launcher

A production-quality desktop + CLI application that generates and manages `llama-server` launch configurations.

## Features

- **Config-First Architecture**: All UI (CLI + GUI) manipulates a shared `LlamaConfig` object
- **GPU Auto-Detection**: Detects CUDA (NVIDIA), Metal (Apple), and Vulkan backends
- **VRAM-Based Optimizer**: Suggests optimal settings based on model size and GPU VRAM
- **Preset System**: Built-in presets (basic, fast, quality, advanced) with JSON export/import
- **Model Scanner**: Scans filesystem for `.gguf`, `.bin`, `.safetensors` files
- **Quantization Recommender**: Recommends Q2/Q3/Q4/Q5/Q8 based on VRAM and target speed/quality
- **Benchmark System**: Throughput tests, latency tests, and VRAM saturation testing
- **GUI & CLI Interfaces**: PySide6 desktop app and command-line tool

## Installation

```bash
pip install .
# or
pip install PySide6
```

## CLI Usage

```bash
# Detect GPU
llama-launcher detect

# Scan models in directory
llama-launcher scan /path/to/models

# Optimize config for VRAM
llama-launcher optimize --model path/to/model.gguf --context 4096

# Recommend quantization
llama-launcher quant --family llama --vram 12288 --target balanced

# Launch server with preset
llama-launcher launch --preset fast --model model.gguf --dry-run

# Launch server with custom config
llama-launcher launch --model model.gguf --context 8192 --gpu-layers 99 --threads 8

# Export/import profiles
llama-launcher profile export --model model.gguf --output profile.json --name my-profile
llama-launcher profile import --input profile.json
```

## GUI Usage

```bash
python -m gui.main_window
```

The GUI includes:
- Model Browser (filesystem scanner)
- GPU Info Panel
- Preset Selector
- Advanced Config Editor
- Benchmark Tab
- Command Preview & Launch Button

## Architecture

```
llama-launcher/
├── core/          # Config schema, validator, builder, presets
├── gpu/           # GPU auto-detection (CUDA, Metal, Vulkan)
├── optimizer/     # VRAM-based configuration optimizer
├── benchmarks/    # Benchmark execution and results
├── quant/         # Quantization recommendation engine
├── models/        # Model filesystem scanner
├── profiles/      # Profile export/import (JSON)
├── cli/           # CLI entry point
├── gui/           # PySide6 GUI application
└── utils/         # Utility functions
```

## Configuration Flow

```
UI → Config → Validator → Optimizer → Builder → Command
```

All commands are generated via `build_command(config: LlamaConfig) -> str`.
