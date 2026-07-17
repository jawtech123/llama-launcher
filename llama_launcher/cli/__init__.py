"""CLI Module."""

__all__ = ["main"]


def main(*args, **kwargs):
    """Lazy entry to keep ``python -m llama_launcher.cli.main`` warning-free."""
    from .main import main as _main

    return _main(*args, **kwargs)
