"""Parse llama-server command lines into LlamaConfig field values.

Driven by the field registry (:mod:`llama_launcher.core.fields`): the flag map
is derived from each field's ``flag``, ``aliases``, and ``negates`` entries, and
value coercion comes from the ``LlamaConfig`` dataclass field types. Qt-free so
it can be shared by the GUI importer, the CLI, and unit tests.
"""
import dataclasses
import logging
import shlex
from typing import Any, Dict, List, Tuple

from llama_launcher.core.config import LlamaConfig
from llama_launcher.core.fields import FLAG_TO_FIELD, NEGATION_FLAGS

logger = logging.getLogger(__name__)

_FIELD_TYPES: Dict[str, Any] = {f.name: f.type for f in dataclasses.fields(LlamaConfig)}

_FALSE_STRINGS = {"false", "0", "off", "no"}


def _coerce(config_key: str, value: str) -> Any:
    """Coerce a string value to the LlamaConfig field's declared type."""
    field_type = _FIELD_TYPES.get(config_key)
    try:
        if field_type is bool:
            return value.strip().lower() not in _FALSE_STRINGS
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
    except (ValueError, TypeError):
        logger.warning("Could not coerce %r for field %s; keeping raw value", value, config_key)
    return value


def _is_bool_field(config_key: str) -> bool:
    return _FIELD_TYPES.get(config_key) is bool


def _looks_like_value(token: str) -> bool:
    """A token is a value unless it looks like a flag.

    Negative numbers (e.g. ``-1`` for ``--seed``) count as values.
    """
    if not token.startswith("-"):
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_launch_command(command_str: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Parse a llama-server command line.

    Returns:
        (params, unrecognized_flags, duplicate_flags): a dict of LlamaConfig
        field values, a list of tokens that could not be mapped to a known
        field, and a list of flags that were specified more than once.
    """
    try:
        args = shlex.split(command_str)
    except ValueError as e:
        logger.warning("shlex failed (%s); falling back to naive split", e)
        args = command_str.split()

    # Drop the executable name if present (e.g. "llama-server", "./llama-server").
    if args and not args[0].startswith("-") and not args[0].endswith(
        (".gguf", ".bin", ".safetensors")
    ):
        args = args[1:]

    params: Dict[str, Any] = {}
    unrecognized_flags: List[str] = []
    duplicate_flags: List[str] = []
    seen_fields: set = set()
    i = 0

    while i < len(args):
        arg = args[i]

        # --flag=value form
        if arg.startswith("-") and "=" in arg:
            key, value = arg.split("=", 1)
            field = FLAG_TO_FIELD.get(key)
            if field:
                if field.name in seen_fields:
                    duplicate_flags.append(arg)
                    logger.warning("Duplicate flag: %s (keeping last value)", arg)
                else:
                    seen_fields.add(field.name)
                params[field.name] = _coerce(field.name, value)
            else:
                unrecognized_flags.append(arg)
                logger.warning("Unrecognized flag: %s", arg)
            i += 1
            continue

        if arg.startswith("-"):
            field = FLAG_TO_FIELD.get(arg)
            if field is None:
                unrecognized_flags.append(arg)
                logger.warning("Unrecognized flag: %s", arg)
                i += 1
                continue

            if _is_bool_field(field.name):
                if field.name in seen_fields:
                    duplicate_flags.append(arg)
                    logger.warning("Duplicate flag: %s (keeping last value)", arg)
                else:
                    seen_fields.add(field.name)
                params[field.name] = arg not in NEGATION_FLAGS
                i += 1
                continue

            # Value-taking flag: consume the next token if it is a value.
            if i + 1 < len(args) and _looks_like_value(args[i + 1]):
                if field.name in seen_fields:
                    duplicate_flags.append(arg)
                    logger.warning("Duplicate flag: %s (keeping last value)", arg)
                else:
                    seen_fields.add(field.name)
                params[field.name] = _coerce(field.name, args[i + 1])
                i += 2
            else:
                logger.warning("Flag %s missing a value; skipped", arg)
                i += 1
            continue

        # Bare token: treat as model path if we don't have one yet.
        if "model_path" not in params:
            if arg.endswith((".gguf", ".bin", ".safetensors")) or "/" in arg or "\\" in arg:
                params["model_path"] = arg
        i += 1

    return params, unrecognized_flags, duplicate_flags
