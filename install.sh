#!/usr/bin/env bash
# Install llama-launcher into a local virtual environment (./venv) and,
# on Linux desktops, register a launcher entry. Nothing is installed
# system-wide; see docs/SETUP.md.
#
#   ./install.sh              # venv + package + desktop entry
#   ./install.sh --dev        # also install pytest
#   ./install.sh --no-desktop # skip the desktop entry
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DEV=0
DESKTOP=1
for arg in "$@"; do
    case "$arg" in
        --dev) DEV=1 ;;
        --no-desktop) DESKTOP=0 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# --- 1. find a usable python ------------------------------------------------
PYTHON=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
       && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON=$(command -v "$candidate")
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "error: Python 3.10 or newer is required (python3 not found or too old)." >&2
    exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"

# --- 2. virtual environment --------------------------------------------------
if [ ! -x venv/bin/python ]; then
    if ! "$PYTHON" -m venv venv 2>/tmp/llama-launcher-venv.err; then
        cat /tmp/llama-launcher-venv.err >&2
        echo "error: could not create a virtual environment." >&2
        echo "       On Debian/Ubuntu: sudo apt install python3-venv" >&2
        exit 1
    fi
    echo "Created ./venv"
else
    echo "Reusing ./venv"
fi

# --- 3. install the package ------------------------------------------------
venv/bin/python -m pip install --quiet --upgrade pip
if [ "$DEV" -eq 1 ]; then
    venv/bin/python -m pip install --quiet --upgrade -e ".[dev]"
else
    venv/bin/python -m pip install --quiet --upgrade -e .
fi
echo "Installed llama-launcher $(venv/bin/python -c 'import llama_launcher; print(llama_launcher.__version__)')"

# --- 4. desktop entry (Linux) -------------------------------------------------
if [ "$DESKTOP" -eq 1 ] && [ "$(uname -s)" = "Linux" ]; then
    APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
    mkdir -p "$APP_DIR"
    for size in 16 32 64 128 256; do
        if [ -f "llama_launcher/icons/ll-$size.png" ]; then
            mkdir -p "$ICON_DIR/${size}x${size}/apps"
            cp "llama_launcher/icons/ll-$size.png" "$ICON_DIR/${size}x${size}/apps/llama-launcher.png"
        fi
    done
    cat > "$APP_DIR/llama-launcher.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=llama-launcher
Comment=Configure, launch and manage llama.cpp servers
Exec=$(pwd)/start.sh
Icon=llama-launcher
Terminal=false
Categories=Development;Utility;
StartupWMClass=llama-launcher
DESKTOP
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APP_DIR" 2>/dev/null || true
    echo "Desktop entry written to $APP_DIR/llama-launcher.desktop"
fi

# --- 5. verify -----------------------------------------------------------
venv/bin/python -c 'import llama_launcher.core, llama_launcher.estimator, llama_launcher.remote' \
    && echo "OK. Start the GUI with ./start.sh (CLI: venv/bin/llama-launcher --help)."
if ! command -v llama-server >/dev/null 2>&1; then
    echo "note: llama-server is not on PATH; set its location in Server Management > Edit Server." >&2
fi
