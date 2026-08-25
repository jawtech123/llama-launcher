"""Utility Helper Functions."""
import os
import sys
from pathlib import Path


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return 0.0


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_age(seconds: float) -> str:
    """Human-readable age, for saying how stale a cached reading is."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(minutes)} min"
    hours = minutes / 60
    if hours < 36:
        return f"{int(hours)}h"
    return f"{int(hours / 24)}d"


def get_icons_dir() -> Path:
    """Return the icons directory, working in both dev and bundled modes."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "llama_launcher" / "icons"
    pkg = Path(__file__).resolve().parent.parent  # llama_launcher/
    icons = pkg / "icons"
    if icons.is_dir():
        return icons
    return pkg.parent / "llama_launcher" / "icons"


def find_app_icon() -> Path | None:
    """Locate the application icon (PNG preferred)."""
    pkg = Path(__file__).resolve().parent.parent  # llama_launcher/
    repo = pkg.parent  # repo root

    # Installed package icons (32x32 for best compatibility)
    pkg_icons = pkg / "icons"
    png = pkg_icons / "ll-32.png"
    if png.is_file():
        return png

    # Repo root ll-32.png
    root_32 = repo / "ll-32.png"
    if root_32.is_file():
        return root_32

    # Repo root ll.png (fallback)
    root_png = repo / "ll.png"
    if root_png.is_file():
        return root_png

    ico = pkg_icons / "ll.ico"
    if ico.is_file():
        return ico

    return None
