"""Application-wide dark theme.

Applied once in :func:`llama_launcher.app.run` before the window is created.
Custom-painted widgets (resource usage panel, status indicators) import the
color constants below instead of hardcoding colors.
"""
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Base surfaces
BG_BASE = "#1e1e1e"
BG_PANEL = "#252526"
BG_INPUT = "#2d2d30"
BG_HOVER = "#3e3e42"
BORDER = "#3e3e42"

# Text
FG_TEXT = "#d4d4d4"
FG_DIM = "#9d9d9d"
FG_MUTED = "#808080"

# Accents / semantic
ACCENT = "#2196F3"
SUCCESS = "#4CAF50"
WARNING = "#ff9800"
ERROR = "#f44336"

# Warning banner backgrounds (used by the resource panel)
WARNING_BG = "#3e2723"
ERROR_BG = "#3e1010"

_MONOSPACE = "monospace"


def apply_dark_theme(app: QApplication) -> None:
    """Apply the Fusion-based dark palette and global stylesheet."""
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(BG_BASE))
    p.setColor(QPalette.ColorRole.WindowText, QColor(FG_TEXT))
    p.setColor(QPalette.ColorRole.Base, QColor(BG_INPUT))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(FG_TEXT))
    p.setColor(QPalette.ColorRole.Text, QColor(FG_TEXT))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(FG_MUTED))
    p.setColor(QPalette.ColorRole.Button, QColor(BG_PANEL))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(FG_TEXT))
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(FG_MUTED))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(FG_MUTED))
    app.setPalette(p)

    app.setStyleSheet(f"""
        QToolTip {{
            background-color: {BG_PANEL};
            color: {FG_TEXT};
            border: 1px solid {BORDER};
            padding: 4px 6px;
        }}
        QGroupBox {{
            border: 1px solid {BORDER};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: {FG_DIM};
        }}
        QPushButton {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 5px 12px;
        }}
        QPushButton:hover {{ background-color: {BG_HOVER}; }}
        QPushButton:pressed {{ background-color: {BORDER}; }}
        QPushButton:disabled {{ color: {FG_MUTED}; }}
        QPushButton[accent="true"] {{
            background-color: {SUCCESS};
            border: none;
            color: white;
            font-weight: bold;
        }}
        QPushButton[accent="true"]:hover {{ background-color: #43a047; }}
        QPushButton[danger="true"] {{
            background-color: {ERROR};
            border: none;
            color: white;
            font-weight: bold;
        }}
        QPushButton[danger="true"]:hover {{ background-color: #e53935; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget, QTreeWidget {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER};
            border-radius: 3px;
            padding: 2px 4px;
            selection-background-color: {ACCENT};
        }}
        QLineEdit[invalid="true"], QComboBox[invalid="true"] {{
            border: 1px solid {ERROR};
        }}
        QTabWidget::pane {{ border: 1px solid {BORDER}; }}
        QTabBar::tab {{
            background-color: {BG_PANEL};
            border: 1px solid {BORDER};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 12px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{ background-color: {BG_INPUT}; }}
        QTabBar::tab:hover {{ background-color: {BG_HOVER}; }}
        QSplitter::handle {{ background-color: {BORDER}; }}
        QSplitter::handle:horizontal {{ width: 3px; }}
        QSplitter::handle:vertical {{ height: 3px; }}
        QProgressBar {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER};
            border-radius: 3px;
            text-align: center;
        }}
        QProgressBar::chunk {{ background-color: {ACCENT}; }}
    """)


def monospace_style(color: str = FG_TEXT, size_px: int = 11) -> str:
    """Inline style for monospace readouts (used by the resource panel)."""
    return f"color: {color}; font-family: {_MONOSPACE}; font-size: {size_px}px;"
