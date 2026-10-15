# llama-launcher Setup Guide

This guide covers installation, first-run configuration, remote servers, and
benchmarking. For a feature overview see the [README](../README.md).

## 1. Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python3 --version`; on Debian/Ubuntu also install `python3-venv` |
| `llama-server` binary | from [llama.cpp](https://github.com/ggml-org/llama.cpp), on `PATH` or configured per server |
| NVIDIA drivers + `nvidia-smi` | optional; enables GPU detection and the live GPU bars |
| AMD: `amdgpu` kernel driver (any distro); ROCm (`amd-smi`/`rocm-smi`) optional | detection and live stats work without ROCm via sysfs; ROCm adds the gfx target and per-process VRAM |
| OpenSSH client (`ssh`, `scp`, `ssh-keygen`) | only for remote servers |
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
3. installs `llama-launcher` and its dependency (PySide6) into it,
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
4. **Pick a preset** (or adjust parameters). The *Preview & Launch* tab shows
   the exact command that will run. The command box is editable: anything you
   type there is what Launch runs, until you press *Regenerate*.
5. **Chat template** (optional, Basic tab): some models ship a template that
   does not work with a given client (tool calling from coding agents is the
   usual case). Paste a Jinja template or a built-in name such as `chatml`,
   or press *Load Model's Template* to start from the one embedded in the
   model. *Template file* passes a path that exists on the server instead.

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

In the server dialog click **Set Up Key Auth**. The launcher uses the
password **once** to install its public key (an existing `~/.ssh/id_ed25519`
or `id_rsa`, else `~/.ssh/id_ed25519_llama_launcher` is generated) into the
remote `authorized_keys`, verifies key login works, then switches the server
configuration to key auth. No manual `ssh-copy-id` step needed. Unknown hosts
are trusted on first contact; a host whose key later changes is refused.

Once connected you can start/stop the remote `llama-server`, watch the live
log, browse/upload/download remote files, and import the remote log into the
Benchmark tab.

### AMD GPUs (ROCm / Vulkan)

The GPU Info tab detects AMD cards through `amd-smi` or `rocm-smi` when ROCm is
installed and through the `amdgpu` driver's sysfs files otherwise, so it works
on any Linux box with an AMD card, local or remote. It tells you which
llama.cpp build to use:

- ROCm installed: build with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx...` (the
  target is shown as *Compute target*). Cards ROCm does not officially support
  (RX 7600/7700, 780M/880M iGPUs, RX 6600/6700) usually run with
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3) or `10.3.0` (RDNA2); the tab names
  the value for the detected card.
- No ROCm: build with `-DGGML_VULKAN=ON`. No extra setup beyond the Vulkan
  driver (`mesa-vulkan-drivers`).

Set such variables per server in *Server Management → Edit Server →
Environment* (one `KEY=VALUE` per line); they are applied whenever the
launcher starts that server, and the status panel shows them.

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

| Estimate says it fits but llama-server runs out of memory | Something else holds VRAM (the live bars show it); the estimate counts only your own server's allocation as reclaimable |
| A flag in the generated command is rejected | Your llama.cpp build is older than the registry; remove the flag on the Preview tab (the box is editable) and rebuild llama.cpp when convenient |

Application logs rotate in `~/.config/llama-launcher/logs/`.

For how the VRAM/RAM estimate is computed and calibrated see
[ESTIMATOR.md](ESTIMATOR.md).
