"""Pytest bootstrap: make the in-repo package importable."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True, scope="session")
def isolate_app_config(tmp_path_factory):
    """Point the app's config directory at a throwaway location.

    Several tests persist through the real storage APIs (``add_server`` defaults
    to ``persist=True``), which would otherwise write into the developer's own
    ``~/.config/llama-launcher/``. ``utils.paths.config_dir()`` reads
    ``XDG_CONFIG_HOME`` on every call and caches nothing, so overriding it here
    isolates every config file (servers, presets, models, folders) for the whole
    session. Per-test ``isolated_config_dir`` fixtures simply nest inside this.
    """
    previous = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("xdg-config"))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = previous
