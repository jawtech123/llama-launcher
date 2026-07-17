"""GUI application bootstrap."""
import logging
import sys

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

    app = QApplication(sys.argv)
    app.setApplicationName("llama-launcher")
    app.setOrganizationName("llama-launcher")

    window = LlamaLauncherGUI()
    window.show()
    logger.info("GUI started")
    return app.exec()


def main() -> None:
    """Console-script entry point."""
    sys.exit(run())


if __name__ == "__main__":
    main()
