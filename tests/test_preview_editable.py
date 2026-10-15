"""Tests for the editable command box on the Preview tab."""
import pytest


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def preview_tab(app):
    from llama_launcher.gui.tabs.preview_tab import PreviewTab
    return PreviewTab()


def test_command_text_is_editable(preview_tab):
    assert preview_tab.command_text.isReadOnly() is False
    assert preview_tab.regenerate_btn.text() == "Regenerate"


def test_generated_command_shown_and_not_marked_edited(preview_tab):
    preview_tab.set_command("llama-server --model /m.gguf")
    assert preview_tab.command() == "llama-server --model /m.gguf"
    assert not preview_tab.is_edited()


def test_user_edit_survives_config_changes_until_regenerate(preview_tab):
    preview_tab.set_command("llama-server --model /m.gguf --threads 8")
    preview_tab.command_text.setPlainText("llama-server --model /m.gguf --threads 8 --my-flag")
    assert preview_tab.is_edited()

    # A config change elsewhere produces a new command: the edit is kept.
    preview_tab.set_command("llama-server --model /m.gguf --threads 16")
    assert preview_tab.command().endswith("--my-flag")
    assert "Regenerate" in preview_tab._hint_label.text()

    # Regenerate (no owner wired) restores the latest generated text.
    preview_tab._regenerate_command()
    assert preview_tab.command() == "llama-server --model /m.gguf --threads 16"
    assert not preview_tab.is_edited()


def test_regenerate_emits_signal_for_owner(preview_tab):
    seen = []
    preview_tab.regenerate_requested.connect(lambda: seen.append(True))
    preview_tab.set_command("a")
    preview_tab.command_text.setPlainText("b")
    preview_tab._regenerate_command()
    assert seen == [True]


def test_forced_set_overrides_edit(preview_tab):
    preview_tab.set_command("a")
    preview_tab.command_text.setPlainText("edited")
    preview_tab.set_command("preset command", force=True)
    assert preview_tab.command() == "preset command"
    assert not preview_tab.is_edited()


def test_reverting_edit_by_hand_clears_edited_state(preview_tab):
    preview_tab.set_command("same")
    preview_tab.command_text.setPlainText("other")
    assert preview_tab.is_edited()
    preview_tab.command_text.setPlainText("same")
    assert not preview_tab.is_edited()
