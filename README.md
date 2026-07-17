# llama-launcher

A desktop + CLI application that generates, saves, and launches `llama-server` (llama.cpp) configurations.

## Features

- **GUI (PySide6)**: Basic + Advanced configuration tabs, GPU info, model browser,
  live command preview & launch, server management (local + SSH), and launch-command import.
- **Config-First Architecture**: every UI action manipulates a shared `LlamaConfig`;
  commands are produced by a single `build_command(config)` and parsed back by `core/parser.py`.
- **GPU Auto-Detection**: CUDA (NVIDIA), Metal (Apple), Vulkan, with CPU fallback.
- **Resource Estimation**: VRAM/RAM usage estimated from GGUF metadata before launch.
- **Preset System**: built-in presets (Small 8GB, Medium 12GB, Large 32GB) plus custom presets.
- **Remote Servers**: manage llama-server on remote hosts over SSH (status, logs, file browser, key install).
- **Structured Logging**: console + rotating log file; no silent failures.

## Installation

```bash
pip install .
pip install ".[dev]"   # with test tooling
```

A convenience script is provided for running from a source checkout:

```bash
./start.sh
```

## Usage

### GUI

```bash
llama-launcher-gui          # installed entry point
python -m llama_launcher    # from the source checkout
```

### CLI

```bash
# Detect GPU
llama-launcher detect

# Scan models in a directory
llama-launcher scan /path/to/models

# Show the launch command for a config (dry run)
llama-launcher launch --model model.gguf --context 8192 --gpu-layers 99 --threads 8 --dry-run

# Launch using a saved preset
llama-launcher launch --preset "Small 8GB" --model model.gguf
```

## Tests

```bash
python -m pytest
```

## Configuration & Data

Runtime state lives in `~/.config/llama-launcher/` (honors `XDG_CONFIG_HOME`):

- `presets.json` — custom presets (chmod 600, may contain HF tokens)
- `servers.json` — server definitions (chmod 600, contains credentials)
- `models.json` — per-server model scan cache
- `folders.json` — model browser source folders
- `logs/` — rotating application logs

Legacy `~/.llama_launcher_*.json` files are migrated automatically on first run.

## Architecture

```
llama_launcher/
├── core/          # LlamaConfig schema, validator, command builder, command parser, presets
├── gpu/           # GPU auto-detection (CUDA, Metal, Vulkan)
├── estimator/     # GGUF metadata reader + VRAM/RAM estimation
├── models/        # Model filesystem scanner + per-server scan cache
├── remote/        # ServerManager: local (subprocess) and SSH (paramiko) connections
├── gui/           # PySide6 main window, server management, resource usage panel
├── cli/           # CLI entry point
├── utils/         # Config dir management, logging setup, helpers
├── app.py         # GUI bootstrap
└── __main__.py    # python -m llama_launcher
```

Retired subsystems (benchmarks, quant recommender, VRAM optimizer, legacy profiles)
are preserved under `_attic/` and are not part of the package.

## Configuration Flow

```
UI → LlamaConfig → Validator → Builder → Command
Command → Parser → LlamaConfig (import)
```

All commands are generated via `build_command(config: LlamaConfig) -> str` and
parsed via `parse_launch_command(command: str) -> (params, unrecognized)`.
