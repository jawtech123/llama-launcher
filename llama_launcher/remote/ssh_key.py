"""SSH Key Installation Utility."""
from pathlib import Path
from typing import Optional
import paramiko


DEFAULT_KEY_DIR = Path.home() / ".ssh"
DEFAULT_KEY_NAME = "llama_launcher"


def generate_key_pair(key_path: Optional[str] = None) -> tuple:
    """Generate an RSA key pair for SSH authentication.

    Returns:
        (private_key_path, public_key_content) tuple.
    """
    if key_path is None:
        key_path = str(DEFAULT_KEY_DIR / DEFAULT_KEY_NAME)

    key_dir = Path(key_path).parent
    key_dir.mkdir(parents=True, exist_ok=True)

    private_key = paramiko.RSAKey.generate(4096)
    private_key.write_private_key_file(key_path)
    Path(key_path).chmod(0o600)

    public_key = (
        f"{private_key.get_name()} {private_key.get_base64()} "
        f"llama-launcher@{Path(key_path).name}"
    )
    public_key_path = key_path + ".pub"
    with open(public_key_path, "w") as f:
        f.write(public_key + "\n")
    Path(public_key_path).chmod(0o644)

    return (key_path, public_key)


def read_public_key(key_path: str) -> str:
    """Read public key content from a .pub file or derive from private key."""
    pub_path = Path(key_path)
    if str(pub_path).endswith(".pub"):
        with open(pub_path) as f:
            return f.read().strip()
    pub_path = Path(str(pub_path) + ".pub")
    if pub_path.exists():
        with open(pub_path) as f:
            return f.read().strip()
    key = paramiko.RSAKey.from_private_key_file(str(pub_path.with_suffix("")))
    return (
        f"{key.get_name()} {key.get_base64()} llama-launcher@{pub_path.name}"
    )


def install_key_to_server(
    hostname: str,
    port: int,
    username: str,
    password: str,
    public_key: str,
    key_path: Optional[str] = None,
) -> bool:
    """Connect to a remote server via password and install the public key.

    Workflow:
        1. Connect via password authentication
        2. Create ~/.ssh directory if needed
        3. Append public key to ~/.ssh/authorized_keys
        4. Set correct permissions
        5. Verify key authentication works
        6. Optionally save the private key path

    Returns:
        True if key installation and verification succeeded.
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            timeout=15,
            allow_agent=False,
            look_for_keys=False,
        )

        ssh_dir_commands = [
            "mkdir -p ~/.ssh",
            "chmod 700 ~/.ssh",
        ]
        for cmd in ssh_dir_commands:
            client.exec_command(cmd, timeout=10)

        append_cmd = (
            f"grep -qF '{public_key.split()[1]}' ~/.ssh/authorized_keys "
            f"|| echo '{public_key}' >> ~/.ssh/authorized_keys"
        )
        client.exec_command(append_cmd, timeout=10)

        client.exec_command("chmod 600 ~/.ssh/authorized_keys", timeout=10)
        client.close()

        verification_key = None
        if key_path:
            key_file = Path(key_path)
            if str(key_file).endswith(".pub"):
                key_file = Path(str(key_file)[:-4])
            verification_key = str(key_file.expanduser())

        verification_success = verify_key_auth(
            hostname, port, username, verification_key
        )
        return verification_success

    except Exception:
        return False


def verify_key_auth(
    hostname: str,
    port: int,
    username: str,
    key_path: str,
) -> bool:
    """Verify that SSH key authentication works for a given server."""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            key_filename=key_path,
            timeout=15,
            allow_agent=False,
            look_for_keys=False,
        )
        _, stdout, _ = client.exec_command("echo ok", timeout=10)
        result = stdout.read().decode().strip()
        client.close()
        return result == "ok"
    except Exception:
        return False


def default_key_exists() -> bool:
    """Check if the default launcher SSH key pair exists."""
    key_dir = DEFAULT_KEY_DIR
    if not key_dir.exists():
        return False
    for suffix in ("", ".pub"):
        if not (key_dir / f"{DEFAULT_KEY_NAME}{suffix}").exists():
            return False
    return True
