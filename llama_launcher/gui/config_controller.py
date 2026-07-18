"""Widget <-> LlamaConfig binding controller.

The controller owns the live :class:`LlamaConfig`. Widgets register a
getter/setter pair plus their change signal; when any widget changes, the new
value is written to the config and pushed to every other widget bound to the
same field. An update guard prevents signal feedback loops, eliminating the
blockSignals juggling the old code needed.
"""
import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from llama_launcher.core.config import LlamaConfig

logger = logging.getLogger(__name__)

Getter = Callable[[], Any]
Setter = Callable[[Any], None]


class ConfigController(QObject):
    """Holds the live LlamaConfig and synchronizes all bound widgets."""

    #: Emitted with the field name after any change ("*" for bulk updates).
    field_changed = Signal(str)

    def __init__(self, config: Optional[LlamaConfig] = None, parent=None):
        super().__init__(parent)
        self.config = config or LlamaConfig()
        self._bindings: Dict[str, List[Tuple[Getter, Setter]]] = {}
        self._updating = False

    # -- Binding ---------------------------------------------------------

    def bind(self, field_name: str, getter: Getter, setter: Setter, signal) -> None:
        """Bind a widget to a config field.

        Args:
            field_name: LlamaConfig attribute name.
            getter: Callable returning the widget's current value.
            setter: Callable applying a value to the widget.
            signal: The widget's change signal (any signature).
        """
        if not hasattr(self.config, field_name):
            raise AttributeError(f"LlamaConfig has no field {field_name!r}")
        self._bindings.setdefault(field_name, []).append((getter, setter))
        signal.connect(lambda *args, g=getter, fn=field_name: self._on_widget_changed(fn, g))

    # -- Value access -----------------------------------------------------

    def get(self, field_name: str) -> Any:
        """Current config value for a field."""
        return getattr(self.config, field_name)

    def set(self, field_name: str, value: Any) -> None:
        """Set a field programmatically: update config + all bound widgets."""
        setattr(self.config, field_name, value)
        self._push(field_name, value)
        self.field_changed.emit(field_name)

    def set_config(self, config: LlamaConfig) -> None:
        """Copy all values from another config (e.g. preset load), then resync."""
        for f in dataclasses.fields(LlamaConfig):
            setattr(self.config, f.name, getattr(config, f.name))
        self.sync_ui()

    # -- Synchronization --------------------------------------------------

    def sync_ui(self) -> None:
        """Push current config values into every bound widget."""
        self._updating = True
        try:
            for name, bindings in self._bindings.items():
                value = getattr(self.config, name)
                for _, setter in bindings:
                    setter(value)
        finally:
            self._updating = False
        self.field_changed.emit("*")

    # -- Internals ---------------------------------------------------------

    def _on_widget_changed(self, field_name: str, getter: Getter) -> None:
        if self._updating:
            return
        value = getter()
        setattr(self.config, field_name, value)
        self._push(field_name, value, skip=getter)
        self.field_changed.emit(field_name)

    def _push(self, field_name: str, value: Any, skip: Optional[Getter] = None) -> None:
        self._updating = True
        try:
            for getter, setter in self._bindings.get(field_name, []):
                if getter is skip:
                    continue
                setter(value)
        finally:
            self._updating = False
