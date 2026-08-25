"""Tests for SSH key provisioning and password handling.

The point of key auth here is to stop storing passwords, so these cover both
the provisioning flow and the storage rules that go with it.
"""
from unittest.mock import patch

import pytest

from llama_launcher.remote import ssh_key
from llama_launcher.remote.ssh import SSHServerConnection
from llama_launcher.remote.storage import load_servers, save_server
from llama_launcher.remote.types import ServerConfig


def _config(**kwargs):
    defaults = dict(friendly_name="host-a", hostname="10.0.0.9", username="bob",
                    ssh_port=22)
    defaults.update(kwargs)
    return ServerConfig(**defaults)


class TestInstallCommand:
    def test_is_idempotent(self):
        cmd = ssh_key._install_command("ssh-ed25519 AAAA user@host")
        # Appends only when the exact line is absent.
        assert "grep -qxF" in cmd
        assert "||" in cmd

    def test_creates_ssh_dir_with_correct_modes(self):
        cmd = ssh_key._install_command("ssh-ed25519 AAAA")
        assert "chmod 700 ~/.ssh" in cmd
        assert "chmod 600 ~/.ssh/authorized_keys" in cmd

    def test_quotes_the_key_against_injection(self):
        cmd = ssh_key._install_command("ssh-ed25519 AAAA$(touch /tmp/pwned) c")
        # Single-quoted, so the substitution is inert text.
        assert "'ssh-ed25519 AAAA$(touch /tmp/pwned) c'" in cmd

    def test_handles_a_key_containing_quotes(self):
        cmd = ssh_key._install_command("ssh-ed25519 AAAA it's-a-key")
        assert "it's-a-key" in cmd.replace("'\"'\"'", "'")


class TestKeyDiscovery:
    """Which key gets used matters: generating a new one asks for a password on
    hosts that already trust an existing key."""

    @pytest.fixture
    def fake_home(self, tmp_path, monkeypatch):
        ssh = tmp_path / ".ssh"
        ssh.mkdir()
        monkeypatch.setattr(ssh_key.Path, "home", staticmethod(lambda: tmp_path))
        return ssh

    def _make_key(self, ssh_dir, name):
        (ssh_dir / name).write_text("PRIVATE")
        (ssh_dir / f"{name}.pub").write_text(f"ssh-ed25519 AAAA {name}\n")

    def test_finds_the_legacy_launcher_key(self, fake_home):
        # Earlier versions generated ~/.ssh/llama_launcher; hosts set up then
        # already trust it.
        self._make_key(fake_home, "llama_launcher")
        assert ssh_key.find_existing_key().name == "llama_launcher"

    def test_prefers_the_current_launcher_key_over_the_legacy_one(self, fake_home):
        self._make_key(fake_home, "llama_launcher")
        self._make_key(fake_home, "id_ed25519_llama_launcher")
        assert ssh_key.find_existing_key().name == "id_ed25519_llama_launcher"

    def test_falls_back_to_a_conventional_key(self, fake_home):
        self._make_key(fake_home, "id_rsa")
        assert ssh_key.find_existing_key().name == "id_rsa"

    def test_ignores_a_private_key_with_no_public_half(self, fake_home):
        (fake_home / "id_ed25519").write_text("PRIVATE")   # no .pub
        assert ssh_key.find_existing_key() is None

    def test_no_keys_at_all(self, fake_home):
        assert ssh_key.find_existing_key() is None

    def test_explicit_key_with_a_dotted_name(self, tmp_path):
        # with_suffix would look for "my.pub" instead of "my.key.pub".
        key = tmp_path / "my.key"
        key.write_text("PRIVATE")
        (tmp_path / "my.key.pub").write_text("ssh-ed25519 AAAA\n")
        found, message = ssh_key.ensure_key_pair(str(key))
        assert found == key, message

    def test_explicit_key_missing_its_public_half_is_reported(self, tmp_path):
        key = tmp_path / "solo.key"
        key.write_text("PRIVATE")
        found, message = ssh_key.ensure_key_pair(str(key))
        assert found is None
        assert "solo.key.pub" in message


class TestSetupKeyAuthGuards:
    def test_requires_host_and_username(self):
        ok, msg, _ = ssh_key.setup_key_auth(_config(hostname=""), "pw")
        assert ok is False and "Host and username" in msg
        ok, msg, _ = ssh_key.setup_key_auth(_config(username=""), "pw")
        assert ok is False and "Host and username" in msg

    def test_requires_a_password_for_the_one_time_install(self):
        ok, msg, _ = ssh_key.setup_key_auth(_config(), "")
        assert ok is False and "password" in msg.lower()


class _FakeConn:
    """Stands in for SSHServerConnection across the two provisioning steps."""

    instances = []

    def __init__(self, config, allow_multiplexing=True):
        self.config = config
        self.allow_multiplexing = allow_multiplexing
        self._last_stderr = ""
        self.executed = []
        _FakeConn.instances.append(self)

    def test_connection(self):
        return (self.config.auth_method != "broken", "probe")

    def execute(self, command):
        self.executed.append(command)
        return True


class TestSetupKeyAuthFlow:
    @pytest.fixture(autouse=True)
    def _reset(self):
        _FakeConn.instances = []

    def _run(self, tmp_path, **cfg_kwargs):
        key = tmp_path / "id_ed25519"
        key.write_text("PRIVATE")
        (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAAKEY user@host\n")
        with patch.object(ssh_key, "SSHServerConnection", _FakeConn, create=True), \
             patch("llama_launcher.remote.ssh.SSHServerConnection", _FakeConn):
            return ssh_key.setup_key_auth(_config(**cfg_kwargs), "secret", str(key))

    def test_succeeds_and_returns_the_key_path(self, tmp_path):
        ok, message, key_path = self._run(tmp_path)
        assert ok is True
        assert key_path.endswith("id_ed25519")
        assert "no longer needed" in message

    def test_installs_then_verifies_with_different_credentials(self, tmp_path):
        self._run(tmp_path)
        installer, verifier = _FakeConn.instances
        assert installer.config.auth_method == "password"
        assert installer.config.password == "secret"
        assert verifier.config.auth_method == "key"
        # The password must not linger in the config used to verify.
        assert verifier.config.password == ""
        assert verifier.config.key_path.endswith("id_ed25519")

    def test_both_steps_refuse_connection_reuse(self, tmp_path):
        # A shared control socket is keyed on host/user, not credentials, so
        # reuse would let a wrong password pass and void the key check.
        self._run(tmp_path)
        assert all(not c.allow_multiplexing for c in _FakeConn.instances)

    def test_the_public_key_is_what_gets_installed(self, tmp_path):
        self._run(tmp_path)
        installer = _FakeConn.instances[0]
        assert any("AAAAKEY" in cmd for cmd in installer.executed)
        assert not any("PRIVATE" in cmd for cmd in installer.executed)

    def test_missing_public_key_is_reported(self, tmp_path):
        key = tmp_path / "lonely_key"
        key.write_text("PRIVATE")           # no .pub alongside
        ok, message, _ = ssh_key.setup_key_auth(_config(), "secret", str(key))
        assert ok is False
        assert "public key" in message.lower()


class TestMultiplexingControl:
    def test_default_connection_reuses_the_master(self):
        argv = SSHServerConnection(_config())._ssh_base_cmd
        assert any("ControlMaster=auto" in a for a in argv)
        assert "ControlPath=none" not in argv

    def test_opting_out_refuses_an_existing_master(self):
        argv = SSHServerConnection(_config(), allow_multiplexing=False)._ssh_base_cmd
        assert "ControlPath=none" in argv
        assert not any("ControlMaster" in a for a in argv)


class TestPasswordPersistence:
    def test_password_is_not_written_unless_remembered(self):
        save_server(_config(friendly_name="no-save", auth_method="password",
                            password="hunter2", remember_password=False))
        stored = {s.friendly_name: s for s in load_servers()}["no-save"]
        assert stored.password == ""
        assert stored.remember_password is False

    def test_password_is_written_when_remembered(self):
        save_server(_config(friendly_name="do-save", auth_method="password",
                            password="hunter2", remember_password=True))
        stored = {s.friendly_name: s for s in load_servers()}["do-save"]
        assert stored.password == "hunter2"
        assert stored.remember_password is True

    def test_password_never_reaches_the_file_when_not_remembered(self, tmp_path):
        from llama_launcher.remote.storage import _servers_file
        save_server(_config(friendly_name="secretive", auth_method="password",
                            password="s3cr3t-value", remember_password=False))
        assert "s3cr3t-value" not in _servers_file().read_text()

    def test_existing_stored_password_keeps_working(self):
        # Entries written before the opt-in existed have a password but no flag;
        # dropping it on the next save would break those servers.
        from llama_launcher.remote.storage import _config_to_dict, _dict_to_config
        legacy = _config_to_dict(_config(auth_method="password",
                                         password="old", remember_password=True))
        del legacy["remember_password"]
        restored = _dict_to_config(legacy)
        assert restored.password == "old"
        assert restored.remember_password is True

    def test_absent_password_stays_unremembered(self):
        from llama_launcher.remote.storage import _config_to_dict, _dict_to_config
        entry = _config_to_dict(_config())
        del entry["remember_password"]
        assert _dict_to_config(entry).remember_password is False
