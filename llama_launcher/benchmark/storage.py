"""Persistence for benchmark sessions (config dir JSON store)."""
import json
import logging
import time
import uuid

from llama_launcher.utils.paths import config_file

logger = logging.getLogger(__name__)

MAX_SESSIONS = 500


def _store_file():
    return config_file("benchmarks.json")


def load_sessions() -> list:
    """Load all recorded benchmark sessions (oldest first)."""
    store = _store_file()
    if not store.exists():
        return []
    try:
        with open(store, "r") as f:
            data = json.load(f)
        sessions = data.get("sessions", [])
        return sessions if isinstance(sessions, list) else []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read benchmarks file %s: %s", store, e)
        return []


def _save_sessions(sessions: list) -> None:
    store = _store_file()
    with open(store, "w") as f:
        json.dump({"sessions": sessions[-MAX_SESSIONS:]}, f, indent=2)


def append_session(session: dict) -> None:
    """Append a finished session to the store."""
    session.setdefault("id", uuid.uuid4().hex[:12])
    session.setdefault("started", time.time())
    sessions = load_sessions()
    sessions.append(session)
    _save_sessions(sessions)


def replace_sessions_for_source(source: str, session: dict) -> None:
    """Replace all sessions imported from ``source`` with ``session``.

    Remote-log imports re-parse the whole log tail; replacing prevents
    double counting when the same server's log is imported repeatedly.
    """
    session.setdefault("id", uuid.uuid4().hex[:12])
    session.setdefault("started", time.time())
    sessions = [s for s in load_sessions() if s.get("source") != source]
    sessions.append(session)
    _save_sessions(sessions)


def clear_sessions() -> None:
    """Delete all recorded sessions."""
    _save_sessions([])


def latest_session_for_preset(preset_name: str):
    """Return the newest session recorded for ``preset_name``, or None."""
    matches = [
        s for s in load_sessions()
        if s.get("preset") and s["preset"] == preset_name
    ]
    if not matches:
        return None
    return max(matches, key=lambda s: s.get("started", 0))
