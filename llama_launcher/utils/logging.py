"""Structured logging setup for llama-launcher.

Call :func:`setup_logging` once at process entry (GUI or CLI). Modules then use
the standard ``logging.getLogger(__name__)`` pattern; all module loggers live
under the configured ``llama_launcher`` namespace.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

_NAMESPACE = "llama_launcher"
_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console + rotating file logging. Safe to call more than once."""
    global _configured
    root = logging.getLogger(_NAMESPACE)
    if _configured:
        return root

    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        from llama_launcher.utils.paths import config_dir

        log_dir = config_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "llama-launcher.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        root.warning("File logging unavailable", exc_info=True)

    root.propagate = False
    _configured = True
    return root
