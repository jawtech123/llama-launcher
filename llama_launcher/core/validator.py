"""Configuration Validator."""
import logging

from .config import LlamaConfig
from .fields import FIELDS

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


def validate_config(config: LlamaConfig) -> None:
    """Validate a LlamaConfig object against the field registry.

    Raises:
        ValidationError: on the first invalid field found.
    """
    if not config.model_path or not isinstance(config.model_path, str):
        raise ValidationError("model_path must be a non-empty string")

    if not isinstance(config.host, str) or not config.host:
        raise ValidationError("host must be a non-empty string")

    for field in FIELDS:
        value = getattr(config, field.name, None)
        if value is None:
            continue

        if field.kind == "bool":
            if not isinstance(value, bool):
                raise ValidationError(f"{field.name} must be a boolean")

        elif field.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValidationError(f"{field.name} must be an integer")
            if not (field.minimum <= value <= field.maximum):
                raise ValidationError(
                    f"{field.name} must be between {int(field.minimum)} and {int(field.maximum)}"
                )

        elif field.kind == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError(f"{field.name} must be a number")
            if not (field.minimum <= value <= field.maximum):
                raise ValidationError(
                    f"{field.name} must be between {field.minimum} and {field.maximum}"
                )

        elif field.kind == "choice":
            if value not in field.choices:
                raise ValidationError(
                    f"{field.name} must be one of {sorted(field.choices)}"
                )
