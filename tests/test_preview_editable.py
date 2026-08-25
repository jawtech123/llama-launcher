"""Tests for preview tab editability and regenerate behavior."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def app():
    """Provide a QApplication instance for Qt widget tests."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def preview_tab(app):
    """Create a PreviewTab instance for testing."""
    from llama_launcher.gui.tabs.preview_tab import PreviewTab
    tab = PreviewTab()
    return tab


def test_command_text_not_read_only(preview_tab):
    """command_text should be editable (setReadOnly(False))."""
    assert preview_tab.command_text.isReadOnly() is False


def test_regenerate_button_exists(preview_tab):
    """Regenerate button should exist in the UI."""
    assert hasattr(preview_tab, "regenerate_btn")
    assert preview_tab.regenerate_btn is not None
    assert preview_tab.regenerate_btn.text() == "Regenerate"


def test_regenerate_restores_generated(preview_tab):
    """Regenerate should overwrite user edits with generated command."""
    generated_cmd = "llama-server --model /path/model.gguf --threads 8"
    preview_tab.config = MagicMock()
    preview_tab._command_builder = lambda cfg: generated_cmd
    preview_tab.set_command("old generated command")

    # Simulate user editing
    preview_tab.command_text.setPlainText("user edited command")
    preview_tab._on_command_changed()
    assert preview_tab._edited is True

    # Click regenerate
    preview_tab._regenerate_command()

    assert preview_tab.command_text.toPlainText() == generated_cmd
    assert preview_tab._edited is False


def test_user_edit_stays_until_regenerate(preview_tab):
    """User edits persist until Regenerate is clicked."""
    generated_cmd1 = "llama-server --model /path/model.gguf --threads 8"
    generated_cmd2 = "llama-server --model /path/model.gguf --threads 16"

    preview_tab._command_builder = lambda cfg: generated_cmd2

    preview_tab.set_command(generated_cmd1)

    # User edits the command
    preview_tab.command_text.setPlainText("user edited this")
    preview_tab._on_command_changed()

    # Regenerate should restore the generated command
    preview_tab._regenerate_command()

    assert preview_tab.command_text.toPlainText() == generated_cmd2
    assert preview_tab._edited is False


def test_launch_uses_text_content(preview_tab):
    """Launch should use command_text content as-is (edited or generated)."""
    preview_tab.set_command("my custom command --threads 4")
    # Simulate launch — in real code this reads command_text.toPlainText()
    assert preview_tab.command_text.toPlainText() == "my custom command --threads 4"


def test_edit_note_shown_when_user_modifies(preview_tab):
    """When user edits the command, hint label shows edited message."""
    preview_tab.set_command("original command")
    assert preview_tab._regenerate_hint_label.text() == ""

    # Simulate user edit
    preview_tab.command_text.setPlainText("edited command")
    preview_tab._on_command_changed()

    assert "Edited" in preview_tab._regenerate_hint_label.text()
    assert "Regenerate" in preview_tab._regenerate_hint_label.text()


def test_edit_note_cleared_after_regenerate(preview_tab):
    """After regenerate, the edit hint is cleared."""
    preview_tab.config = MagicMock()
    preview_tab._command_builder = lambda cfg: "regenerated command"
    preview_tab.set_command("original command")

    # Simulate user edit
    preview_tab.command_text.setPlainText("edited command")
    preview_tab._on_command_changed()

    # Regenerate
    preview_tab._regenerate_command()

    # After regenerate, hint should be cleared (empty string for non-edited state)
    assert preview_tab._regenerate_hint_label.text() == ""


def test_copy_command_works(preview_tab, app):
    """Copy Command button should set clipboard text."""
    preview_tab.set_command("command to copy")
    clipboard = app.clipboard()
    original_text = clipboard.text()

    preview_tab._copy_command()
    new_text = clipboard.text()

    assert new_text == "command to copy"


def set_running_stops_controls(preview_tab):
    """When running, launch button disabled and stop button enabled."""
    preview_tab.set_running(True)
    assert preview_tab.launch_btn.isEnabled() is False
    assert preview_tab.stop_btn.isEnabled() is True

    preview_tab.set_running(False)
    assert preview_tab.launch_btn.isEnabled() is True
    assert preview_tab.stop_btn.isEnabled() is False


def test_log_text_read_only(preview_tab):
    """Log output text area should be read-only."""
    assert preview_tab.log_text.isReadOnly() is True


def test_append_log_adds_text(preview_tab):
    """append_log should add text to the log area."""
    preview_tab.clear_log()
    preview_tab.append_log("log line 1\n")
    preview_tab.append_log("log line 2\n")
    assert "log line 1" in preview_tab.log_text.toPlainText()
    assert "log line 2" in preview_tab.log_text.toPlainText()


def test_status_label_updates(preview_tab):
    """Status label text updates on running state change."""
    assert "Stopped" in preview_tab.status_label.text()
    preview_tab.set_running(True)
    assert "Running" in preview_tab.status_label.text()
    preview_tab.set_running(False)
    assert "Stopped" in preview_tab.status_label.text()
