"""Advanced Config tab, generated from the field registry.

Every editor row is created from :class:`~llama_launcher.core.fields.Field`
definitions and bound to the shared :class:`ConfigController`, so widget
construction, config sync, and tooltips never diverge from the command
builder/parser.
"""
import logging
from typing import Any, Callable, Tuple

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from llama_launcher.core.fields import GROUP_ORDER, Field, fields_in_group
from llama_launcher.gui.config_controller import ConfigController
from llama_launcher.gui.widgets import info_label

logger = logging.getLogger(__name__)


def create_field_widget(field: Field, parent: QWidget = None) -> Tuple[QWidget, Callable, Callable, Any]:
    """Create the editor widget for a field.

    Returns:
        (widget, getter, setter, change_signal) suitable for controller.bind().
    """
    if field.kind == "int":
        w = QSpinBox(parent)
        w.setRange(int(field.minimum), int(field.maximum))
        if field.special_text:
            w.setSpecialValueText(field.special_text)
        return w, w.value, w.setValue, w.valueChanged

    if field.kind == "float":
        w = QDoubleSpinBox(parent)
        w.setRange(field.minimum, field.maximum)
        w.setSingleStep(field.step)
        w.setDecimals(field.decimals)
        return w, w.value, w.setValue, w.valueChanged

    if field.kind == "choice":
        w = QComboBox(parent)
        w.addItems(list(field.choices))
        return w, w.currentText, w.setCurrentText, w.currentTextChanged

    if field.kind == "bool":
        w = QCheckBox(parent)
        return w, w.isChecked, w.setChecked, w.stateChanged

    # kind == "str"
    w = QLineEdit(parent)
    if field.placeholder:
        w.setPlaceholderText(field.placeholder)
    if field.secret:
        w.setEchoMode(QLineEdit.EchoMode.Password)
    return w, w.text, w.setText, w.textChanged


class AdvancedConfigTab(QWidget):
    """Scrollable advanced configuration editor generated from the registry."""

    def __init__(
        self,
        controller: ConfigController,
        preset_names=None,
        on_preset_selected=None,
        on_save_preset=None,
        on_manage_presets=None,
        parent=None,
    ):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)

        # Preset selector bar (mirrors the Basic tab)
        if preset_names is not None:
            bar = QHBoxLayout()
            bar.addWidget(QLabel("Preset:"))
            self.preset_combo = QComboBox()
            self.preset_combo.addItems(preset_names)
            if on_preset_selected:
                self.preset_combo.currentTextChanged.connect(on_preset_selected)
            bar.addWidget(self.preset_combo)

            self.save_preset_btn = QPushButton("Save Preset")
            if on_save_preset:
                self.save_preset_btn.clicked.connect(on_save_preset)
            bar.addWidget(self.save_preset_btn)
            bar.addStretch()

            self.manage_presets_btn = QPushButton("Manage Presets...")
            if on_manage_presets:
                self.manage_presets_btn.clicked.connect(on_manage_presets)
            bar.addWidget(self.manage_presets_btn)
            outer.addLayout(bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)

        for group in GROUP_ORDER:
            fields = fields_in_group(group)
            if not fields:
                continue
            box = QGroupBox(group)
            form = QFormLayout(box)
            for field in fields:
                widget, getter, setter, signal = create_field_widget(field, container)
                widget.setToolTip(field.tooltip)
                form.addRow(info_label(field.label, field.tooltip), widget)
                controller.bind(field.name, getter, setter, signal)
            layout.addWidget(box)
        layout.addStretch()
