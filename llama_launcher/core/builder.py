"""CLI Command Builder.

Builds a llama-server command string from a :class:`LlamaConfig`, driven by the
field registry (:mod:`llama_launcher.core.fields`). Values are shell-quoted
where necessary, so the result round-trips through ``shlex.split`` and can be
pasted directly into a terminal.

Emission rules:
- bool fields: emit ``flag`` when a default-False field is True; emit
  ``negates`` when a default-True field is False.
- int/float/choice fields: emit ``flag value`` when the value differs from the
  llama-server default.
- str fields: emit ``flag value`` when non-empty.
"""
import shlex
from typing import List

from .config import LlamaConfig
from .fields import CLI_FIELDS

#: Per-field custom emission conditions (value -> bool).
_EMIT_OVERRIDES = {
    "parallel": lambda v: v > 0,  # llama-server rejects --parallel 0/-1
}


def build_command(config: LlamaConfig) -> str:
    """Build the llama-server command for the given configuration."""
    args: List[str] = ["llama-server"]

    for field in CLI_FIELDS:
        value = getattr(config, field.name, field.default)

        if field.kind == "bool":
            if field.default is False and value:
                args.append(field.flag)
            elif field.default is True and not value and field.negates:
                args.append(field.negates)
            continue

        if field.name in _EMIT_OVERRIDES:
            if _EMIT_OVERRIDES[field.name](value):
                args.extend([field.flag, shlex.quote(str(value))])
            continue

        if field.kind in ("int", "float", "choice"):
            if value != field.default:
                args.extend([field.flag, shlex.quote(str(value))])
        elif field.kind == "str":
            # Emit when set and different from the llama-server default
            # (defaults are usually "", but host/samplers/sampler_seq differ).
            if value and value != field.default:
                args.extend([field.flag, shlex.quote(str(value))])

    # User-provided raw fragments, appended verbatim (already quoted by user).
    for extra in config.extra_flags:
        args.append(str(extra))

    return " ".join(args)
