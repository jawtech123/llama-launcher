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
- fields marked ``always_emit`` are passed explicitly regardless of value,
  because llama-server's own default for them is unknowable (model-dependent
  context size) or has changed between builds (``--jinja``).
"""
import shlex
from typing import List

from .config import LlamaConfig
from .fields import CLI_FIELDS

#: Per-field custom emission conditions (value -> bool).
_EMIT_OVERRIDES = {
    "parallel": lambda v: v > 0,  # llama-server rejects --parallel 0/-1
}


def _format_number(value) -> str:
    """Render numbers the way a person would type them (``2`` not ``2.0``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_command(config: LlamaConfig) -> str:
    """Build the llama-server command for the given configuration."""
    args: List[str] = ["llama-server"]

    for field in CLI_FIELDS:
        value = getattr(config, field.name, field.default)

        if field.kind == "bool":
            if value and (field.default is False or field.always_emit):
                args.append(field.flag)
            elif not value and field.negates and (field.default is True or field.always_emit):
                args.append(field.negates)
            continue

        if field.name in _EMIT_OVERRIDES:
            if _EMIT_OVERRIDES[field.name](value):
                args.extend([field.flag, shlex.quote(str(value))])
            continue

        if field.kind in ("int", "float", "choice"):
            if field.always_emit or value != field.default:
                args.extend([field.flag, shlex.quote(_format_number(value))])
        elif field.kind == "str":
            # Emit when set and different from the llama-server default
            # (defaults are usually "", but host/samplers/sampler_seq differ).
            if value and value != field.default:
                args.extend([field.flag, shlex.quote(str(value))])

    # User-provided raw fragments, appended verbatim (already quoted by user).
    for extra in config.extra_flags:
        args.append(str(extra))

    return " ".join(args)
