"""Tests for SSHServerConnection thread-safety and shutdown semantics."""
import threading
import time

from llama_launcher.remote.ssh import SSHServerConnection
from llama_launcher.remote.types import (
    AuthMethod,
    ConnectionState,
    ConnectionType,
    ServerConfig,
)


def _config():
    return ServerConfig(
        friendly_name="test",
        connection_type=ConnectionType.SSH,
        hostname="example.invalid",
        username="user",
        auth_method=AuthMethod.PASSWORD,
        password="pw",
    )


class _FakeTransport:
    def __init__(self):
        self._active = True
        self.closed = threading.Event()

    def is_active(self):
        return self._active

    def close(self):
        self._active = False
        self.closed.set()


class _FakeClient:
    def __init__(self):
        self.transport = _FakeTransport()

    def get_transport(self):
        return self.transport

    def close(self):
        self.transport.close()


def test_disconnect_clears_state_and_client():
    conn = SSHServerConnection(_config())
    conn._client = _FakeClient()
    conn._state = ConnectionState.CONNECTED
    conn.disconnect()
    assert conn._client is None
    assert conn.connection_state == ConnectionState.DISCONNECTED


def test_shutdown_blocks_reconnect():
    conn = SSHServerConnection(_config())
    conn.shutdown()
    assert conn._closing is True
    # connect() must refuse after shutdown, not dial out during teardown
    assert conn.connect() is False
    assert conn.connection_state != ConnectionState.CONNECTED


def test_get_client_handles_none_transport():
    """get_transport() may return None; must not raise AttributeError."""
    conn = SSHServerConnection(_config())

    class _NoTransportClient:
        def get_transport(self):
            return None

        def close(self):
            pass

    conn._client = _NoTransportClient()
    conn._closing = True  # make connect() refuse so _get_client raises
    try:
        conn._get_client()
        raise AssertionError("expected ConnectionError")
    except ConnectionError:
        pass


def test_disconnect_unblocks_and_is_race_safe():
    """disconnect() from another thread closes the transport and never
    deadlocks, even with concurrent users of the connection."""
    conn = SSHServerConnection(_config())
    fake = _FakeClient()
    conn._client = fake
    conn._state = ConnectionState.CONNECTED

    errors = []

    def user():
        for _ in range(50):
            try:
                conn.disconnect()
                conn._client = fake  # pretend a worker re-set it
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

    threads = [threading.Thread(target=user) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not errors
    assert fake.transport.closed.is_set()
