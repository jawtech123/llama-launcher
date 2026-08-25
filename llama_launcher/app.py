"""GUI application bootstrap."""
import logging
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize
from llama_launcher.utils.helpers import find_app_icon, get_icons_dir
from llama_launcher.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run() -> int:
    """Start the PySide6 GUI. Returns the process exit code."""
    setup_logging()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        logger.error("PySide6 is not installed. Install with: pip install PySide6")
        return 1

    from llama_launcher.gui.main_window import LlamaLauncherGUI
    from llama_launcher.gui.theme import apply_dark_theme

    app = QApplication(sys.argv)
    app.setApplicationName("llama-launcher")
    app.setOrganizationName("llama-launcher")
    icon_dir = get_icons_dir()
    icon = QIcon()
    for size in [16, 32, 48, 64, 128, 256]:
        icon_path = icon_dir / f"ll-{size}.png"
        if icon_path.is_file():
            icon.addFile(str(icon_path), QSize(size, size))
    if not icon.isNull():
        app.setWindowIcon(icon)
        logger.debug("Loaded multi-size icon")
    else:
        logger.warning("No application icon found")
    app.setDesktopFileName("llama-launcher")
    apply_dark_theme(app)

    window = LlamaLauncherGUI()
    window.show()
    logger.info("GUI started")
    return app.exec()


def main() -> None:
    """Console-script entry point."""
    sys.exit(run())


if __name__ == "__main__":
    main()
