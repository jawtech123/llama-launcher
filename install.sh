#!/usr/bin/env bash
# llama-launcher installer.
#
# Creates (or reuses) a Python virtual environment in ./venv, installs the
# application and its dependencies into it, and optionally registers a
# desktop entry. Nothing is installed system-wide; removing the directory
# (plus ~/.config/llama-launcher for settings) fully uninstalls.
#
# Usage:
#   ./install.sh [--dev] [--no-desktop]
#
#   --dev          also install test tooling (pytest)
#   --no-desktop   skip creating the Linux .desktop launcher entry
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
INSTALL_EXTRAS=""
CREATE_DESKTOP=1

for arg in "$@"; do
    case "$arg" in
        --dev) INSTALL_EXTRAS="[dev]" ;;
        --no-desktop) CREATE_DESKTOP=0 ;;
        -h|--help)
            grep '^#' "$0" | head -15
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

echo "==> llama-launcher installer"

# ---------------------------------------------------------------- python ---
PYTHON_BIN=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3 not found. Install Python 3.10 or newer." >&2
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required (found $PY_VERSION)." >&2
    exit 1
fi
echo "==> Using $PYTHON_BIN ($PY_VERSION)"

# ------------------------------------------------------------------ venv ---
if [ ! -x "$VENV_DIR/bin/pip" ]; then
    echo "==> Creating virtual environment in $VENV_DIR"
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "ERROR: could not create a virtual environment." >&2
        echo "       On Debian/Ubuntu install:  sudo apt install python3-venv" >&2
        exit 1
    fi
else
    echo "==> Reusing existing virtual environment in $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --quiet --upgrade pip

echo "==> Installing llama-launcher${INSTALL_EXTRAS:+ with dev extras}"
pip install ".${INSTALL_EXTRAS}"

# ------------------------------------------------------- desktop entry -----
if [ "$CREATE_DESKTOP" -eq 1 ] && [ -d "$HOME/.local/share/applications" ]; then
    DESKTOP_FILE="$HOME/.local/share/applications/llama-launcher.desktop"
    echo "==> Creating desktop entry: $DESKTOP_FILE"
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=llama-launcher
Comment=Generate and manage llama-server launch configurations
Exec=$VENV_DIR/bin/llama-launcher-gui
Icon=utilities-terminal
Terminal=false
Categories=Development;Utility;
EOF
    chmod 644 "$DESKTOP_FILE"
fi

# --------------------------------------------------------------- verify ----
echo "==> Verifying installation"
"$VENV_DIR/bin/python" -c "import llama_launcher; print('    llama_launcher', llama_launcher.__version__, 'installed OK')"

cat <<EOF

Installation complete.

Run the GUI:
    ./start.sh
  or:
    $VENV_DIR/bin/llama-launcher-gui

Run the CLI:
    $VENV_DIR/bin/llama-launcher --help

Settings and benchmark history are stored in:
    ~/.config/llama-launcher/

EOF
