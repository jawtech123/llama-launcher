"""CLI Main Entry Point."""
import argparse
import logging
import subprocess
import sys

from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.validator import validate_config
from llama_launcher.core.builder import build_command
from llama_launcher.core.presets import PresetManager
from llama_launcher.gpu.detector import detect_gpu
from llama_launcher.models.scanner import scan_models
from llama_launcher.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a directory for model files."""
    models = scan_models(args.directory)
    print(f"Found {len(models)} model(s):")
    for m in models:
        print(f"  - {m.file_name} ({m.file_size_mb:.2f}MB) [{m.format}]")
        if m.family:
            print(f"    Family: {m.family}, Quant: {m.quant_type or 'Unknown'}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """Detect GPU backend and VRAM."""
    gpu_info = detect_gpu()
    print("GPU Detected:")
    print(f"  Backend: {gpu_info.backend}")
    print(f"  Device: {gpu_info.device_name}")
    print(f"  VRAM: {gpu_info.vram_mb} MB")
    print(f"  Supported: {gpu_info.supported}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    """Build (and optionally run) a llama-server command."""
    if args.preset:
        try:
            config = PresetManager.get_preset(args.preset)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        config = LlamaConfig(
            model_path=args.model or "",
            context_size=args.context,
            gpu_layers=args.gpu_layers,
            threads=args.threads,
            port=args.port,
            host=args.host,
            backend=args.backend,
            use_mmap=True,
            kv_cache_type=args.kv_cache,
            batch_size=args.batch,
        )

    if args.model:
        config.model_path = args.model

    try:
        validate_config(config)
        cmd = build_command(config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Launch Command:")
    print(cmd)

    if not args.dry_run:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("llama-server exited with code %s", e.returncode)
            return e.returncode or 1
    return 0


def main() -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="llama-launcher - Manage llama-server configurations"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    scan_parser = subparsers.add_parser("scan", help="Scan for models")
    scan_parser.add_argument("directory", help="Directory to scan")
    scan_parser.set_defaults(func=cmd_scan)

    detect_parser = subparsers.add_parser("detect", help="Detect GPU")
    detect_parser.set_defaults(func=cmd_detect)

    launch_parser = subparsers.add_parser("launch", help="Launch llama-server")
    launch_parser.add_argument("--model", help="Model path")
    launch_parser.add_argument("--preset", help="Preset name (see GUI preset manager)")
    launch_parser.add_argument("--context", type=int, default=4096)
    launch_parser.add_argument("--gpu-layers", type=int, default=-1)
    launch_parser.add_argument("--threads", type=int, default=0)
    launch_parser.add_argument("--port", type=int, default=8080)
    launch_parser.add_argument("--host", default="127.0.0.1")
    launch_parser.add_argument("--backend", default="auto")
    launch_parser.add_argument("--kv-cache", default="f16")
    launch_parser.add_argument("--batch", type=int, default=2048)
    launch_parser.add_argument("--dry-run", action="store_true", help="Only show command")
    launch_parser.set_defaults(func=cmd_launch)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
