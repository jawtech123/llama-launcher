# llama-launcher Setup Guide

This guide covers installation, first-run configuration, remote servers, and
benchmarking. For a feature overview see the [README](../README.md).

## 1. Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python3 --version`; on Debian/Ubuntu also install `python3-venv` |
| `llama-server` binary | from [llama.cpp](https://github.com/ggml-org/llama.cpp), on `PATH` or configured per server |
| NVIDIA drivers + `nvidia-smi` | optional; enables GPU detection and the live GPU bars |
| Qt/X11 or Wayland desktop | the GUI needs a display; the CLI does not |

## 2. Installation

### Quick install (recommended)

```bash
git clone <this-repo>
cd llama-launcher
./install.sh
```

The script:

1. locates a suitable `python3` (>= 3.10, errors with an apt hint if venv support is missing),
2. creates (or reuses) the virtual environment in `./venv`,
3. installs `llama-launcher` and its dependencies (PySide6, paramiko) into it,
4. creates a desktop entry at `~/.local/share/applications/llama-launcher.desktop`,
5. verifies the installation by importing the package.

Options:

```bash
./install.sh --dev           # also install pytest for running the test suite
./install.sh --no-desktop    # skip the desktop entry
```

Nothing is installed system-wide. Everything stays inside the repository
directory (the venv) plus the config directory (see below).

### Manual install

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install .            # or: pip install ".[dev]"
```

### Upgrading

```bash
git pull
./install.sh             # re-runs pip install into the existing venv
```

### Uninstalling

```bash
rm -rf venv
rm -f ~/.local/share/applications/llama-launcher.desktop
rm -rf ~/.config/llama-launcher   # optional: presets, servers, benchmark history
```

## 3. Running

```bash
./start.sh                    # GUI; activates ./venv automatically
venv/bin/llama-launcher-gui   # GUI, without the wrapper
venv/bin/llama-launcher       # CLI (detect / scan / launch)
```

## 4. First-run configuration

1. **GPU detection** runs automatically on startup (see the *GPU Info* tab).
2. **Model folders**: open *Model Browser*, add the directories containing your
   `.gguf` files, and click *Scan All Folders for Models*.
3. **Select a model**: clicking one in the browser fills *Main Model* on the
   *Basic Config* tab. The *Resource Usage* panel at the bottom immediately
   estimates VRAM/RAM, and the *Live Usage* bars show realtime CPU/GPU load.
4. **Pick a preset** (or adjust parameters). The *Command Preview & Launch*
   tab always shows the exact command that will run.

Configuration is stored in `~/.config/llama-launcher/` (honors
`XDG_CONFIG_HOME`). Files containing credentials (`servers.json`,
`presets.json`) are written with mode `600`.

## 5. Remote servers (SSH)

Open *Server Management* → **+** to add a server:

- **Type**: SSH, then fill in host, port, and username.
- **Auth**: *Password* initially — enter the password.
- **Paths**: set *Working Dir*, *Server Exec* (path to `llama-server` on the
  remote host), and *Model Dirs*.

### Switching to key authentication (recommended)

Select the server and click **Install SSH Key**. The launcher uses the stored
username/password **once** to install its public key
(`~/.ssh/llama_launcher`, generated automatically if missing) into the remote
`authorized_keys`, verifies key login works, then switches the server
configuration to key auth and saves it. No manual `ssh-copy-id` step needed.

Once connected you can start/stop the remote `llama-server`, watch the live
log, browse/upload/download remote files, and import the remote log into the
Benchmark tab.

## 6. Benchmarking

The **Benchmark** tab collects throughput stats from llama-server logs:

- **Current Session** — live tokens in/out, tokens/sec in/out, request count
  while a locally launched server runs.
- **Session History** — every finished session, persisted across restarts.
- **Import Remote Log** — pulls the selected server's log and imports its
  stats (re-importing the same server replaces the previous import).

Benchmarks are **linked to presets**: a session recorded while preset *X* was
active is tagged with *X* and a fingerprint of the exact parameters. The **ℹ**
button next to the preset selector shows the recorded tok/s for that preset.
**Changing any parameter resets the info** (the button then reports the data
as stale) so benchmark numbers never silently apply to a different
configuration.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `llama-server` won't start | Check the path in *Server Management → Edit Server → Server Exec*; the log pane shows the server's stderr |
| "Address already in use" on port 8080 | A previous server is still running (`ss -tlnp \| grep 8080`); stop it or pick another port |
| GPU shows as CPU | Ensure `nvidia-smi` works in a shell; Vulkan/Metal are used as fallbacks |
| GUI won't start over SSH | Use X11 forwarding (`ssh -X`) or run the CLI instead |
| SSH key install fails | Verify username/password and that the remote `~/.ssh` directory is writable; the launcher reports the failing step |

Application logs rotate in `~/.config/llama-launcher/logs/`.
