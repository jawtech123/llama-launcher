"""CLI Main Entry Point."""
import argparse
import sys
from pathlib import Path

from core.config import LlamaConfig
from core.validator import validate_config
from core.builder import build_command
from core.presets import PresetManager
from gpu.detector import detect_gpu
from optimizer.vram_optimizer import optimize_for_vram
from models.scanner import scan_models
from quant.recommender import recommend_quantization
from profiles.manager import export_profile, import_profile


def cmd_scan(args):
    """Scan models in directory."""
    models = scan_models(args.directory)
    print(f"Found {len(models)} model(s):")
    for m in models:
        print(f"  - {m.file_name} ({m.file_size_mb:.2f}MB) [{m.format}]")
        if m.family:
            print(f"    Family: {m.family}, Quant: {m.quant_type or 'Unknown'}")


def cmd_detect(args):
    """Detect GPU."""
    gpu_info = detect_gpu()
    print(f"GPU Detected:")
    print(f"  Backend: {gpu_info.backend}")
    print(f"  Device: {gpu_info.device_name}")
    print(f"  VRAM: {gpu_info.vram_mb} MB")
    print(f"  Supported: {gpu_info.supported}")


def cmd_optimize(args):
    """Optimize config for VRAM."""
    gpu_info = detect_gpu()
    
    # Get model info
    model_size_mb = 0
    if args.model and Path(args.model).exists():
        from utils.helpers import get_file_size_mb
        model_size_mb = get_file_size_mb(args.model)
    
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
        batch_size=args.batch
    )
    
    optimized = optimize_for_vram(config, gpu_info, model_size_mb)
    print("Optimized Configuration:")
    print(f"  Model: {optimized.model_path}")
    print(f"  Context: {optimized.context_size}")
    print(f"  GPU Layers: {optimized.gpu_layers}")
    print(f"  Threads: {optimized.threads}")
    print(f"  KV Cache Type: {optimized.kv_cache_type}")
    print(f"  Batch Size: {optimized.batch_size}")


def cmd_quant(args):
    """Recommend quantization."""
    recommendation = recommend_quantization(args.family, args.vram, args.target)
    print("Quantization Recommendation:")
    print(f"  Recommended: {recommendation['recommended_quantization'].upper()}")
    print(f"  Quality Level: {recommendation['quality_level']}")
    print(f"  Estimated Size: {recommendation['estimated_size_mb']} MB")
    print(f"  Explanation: {recommendation['explanation']}")
    print(f"  Fallback Options: {', '.join(recommendation['fallback_options'])}")


def cmd_launch(args):
    """Launch llama-server with config."""
    # Load preset or create config
    if args.preset:
        config = PresetManager.get_preset(args.preset)
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
            batch_size=args.batch
        )
    
    if args.model:
        config.model_path = args.model
    
    # Validate and build command
    try:
        validate_config(config)
        cmd = build_command(config)
        print("Launch Command:")
        print(cmd)
        
        if not args.dry_run:
            import subprocess
            subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_profile(args):
    """Profile management (export/import)."""
    if args.action == 'export':
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
            name=args.name
        )
        export_profile(config, args.output)
        print(f"Profile exported to {args.output}")
    elif args.action == 'import':
        config = import_profile(args.input)
        print(f"Imported profile: {config.name or 'unnamed'}")
        print(f"Model: {config.model_path}")
        print(f"Context: {config.context_size}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="llama-launcher - Manage llama-server configurations")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for models')
    scan_parser.add_argument('directory', help='Directory to scan')
    scan_parser.set_defaults(func=cmd_scan)
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect GPU')
    detect_parser.set_defaults(func=cmd_detect)
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize config for VRAM')
    opt_parser.add_argument('--model', help='Model path')
    opt_parser.add_argument('--context', type=int, default=4096)
    opt_parser.add_argument('--gpu-layers', type=int, default=-1)
    opt_parser.add_argument('--threads', type=int, default=0)
    opt_parser.add_argument('--port', type=int, default=8080)
    opt_parser.add_argument('--host', default='127.0.0.1')
    opt_parser.add_argument('--backend', default='auto')
    opt_parser.add_argument('--kv-cache', default='f16')
    opt_parser.add_argument('--batch', type=int, default=512)
    opt_parser.set_defaults(func=cmd_optimize)
    
    # Quant command
    quant_parser = subparsers.add_parser('quant', help='Recommend quantization')
    quant_parser.add_argument('--family', default='llama')
    quant_parser.add_argument('--vram', type=int, default=8192)
    quant_parser.add_argument('--target', choices=['speed', 'quality', 'balanced'], default='balanced')
    quant_parser.set_defaults(func=cmd_quant)
    
    # Launch command
    launch_parser = subparsers.add_parser('launch', help='Launch llama-server')
    launch_parser.add_argument('--model', help='Model path')
    launch_parser.add_argument('--preset', choices=['basic', 'fast', 'quality', 'advanced'])
    launch_parser.add_argument('--context', type=int, default=4096)
    launch_parser.add_argument('--gpu-layers', type=int, default=-1)
    launch_parser.add_argument('--threads', type=int, default=0)
    launch_parser.add_argument('--port', type=int, default=8080)
    launch_parser.add_argument('--host', default='127.0.0.1')
    launch_parser.add_argument('--backend', default='auto')
    launch_parser.add_argument('--kv-cache', default='f16')
    launch_parser.add_argument('--batch', type=int, default=512)
    launch_parser.add_argument('--dry-run', action='store_true', help='Only show command')
    launch_parser.set_defaults(func=cmd_launch)
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Manage profiles')
    profile_sub = profile_parser.add_subparsers(dest='action')
    
    profile_export = profile_sub.add_parser('export')
    profile_export.add_argument('--model', required=True)
    profile_export.add_argument('--output', required=True)
    profile_export.add_argument('--name', default='profile')
    profile_export.add_argument('--context', type=int, default=4096)
    profile_export.add_argument('--gpu-layers', type=int, default=-1)
    profile_export.add_argument('--threads', type=int, default=0)
    profile_export.add_argument('--port', type=int, default=8080)
    profile_export.add_argument('--host', default='127.0.0.1')
    profile_export.add_argument('--backend', default='auto')
    profile_export.add_argument('--kv-cache', default='f16')
    profile_export.add_argument('--batch', type=int, default=512)
    profile_export.set_defaults(func=cmd_profile)
    
    profile_import = profile_sub.add_parser('import')
    profile_import.add_argument('--input', required=True)
    profile_import.set_defaults(func=cmd_profile)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
