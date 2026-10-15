"""Tests for GitHub Copilot setup and unload."""

from __future__ import annotations

from pathlib import Path
import tempfile

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .conftest import FakeCopilotClient


async def test_setup_and_unload(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """The runtime starts isolated and is stopped on unload."""
    client = setup_integration
    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert client.started
    assert client.kwargs["mode"] == "empty"
    assert "ha-github-copilot-" in client.kwargs["base_directory"]
    assert client.kwargs["working_directory"] != client.kwargs["base_directory"]
    assert client.kwargs["github_token"] == "github_pat_test"
    assert hass.states.get("conversation.github_copilot_conversation")
    assert hass.states.get("ai_task.github_copilot_ai_task")

    assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED
    assert client.stopped


async def test_setup_auth_failure_starts_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """A rejected token puts the entry into reauth."""

    def configure(client: FakeCopilotClient) -> None:
        client.authenticated = False

    mock_copilot.configure = configure
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    flows = hass.config_entries.flow.async_progress()
    assert [flow["context"]["source"] for flow in flows] == ["reauth"]
    assert mock_copilot.instances[0].stopped


async def test_setup_runtime_failure_retries(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """A runtime that can't start is retried later."""

    def configure(client: FakeCopilotClient) -> None:
        client.start_error = RuntimeError("download failed")

    mock_copilot.configure = configure
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_runtime_stops_on_shutdown(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """HA doesn't unload entries at shutdown, so the runtime stops on STOP."""
    root = Path(mock_config_entry.runtime_data._root_dir)
    assert root.name.startswith(f"ha-github-copilot-{mock_config_entry.entry_id}-")
    assert root.is_dir()

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    assert setup_integration.stopped
    assert not root.exists()


async def test_stale_scratch_dirs_are_wiped(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """Leftovers from a crash are removed before the runtime starts."""
    tmp = Path(tempfile.gettempdir())
    stale = tmp / f"ha-github-copilot-{mock_config_entry.entry_id}-deadbeef"
    (stale / "home" / "session-state").mkdir(parents=True)
    other = tmp / "ha-github-copilot-other-entry-cafebabe"
    other.mkdir()

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert not stale.exists()
    assert other.exists()  # other entries' directories are left alone
    root = Path(mock_config_entry.runtime_data._root_dir)
    assert (root / "home").is_dir()
    assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
    assert not root.exists()
    other.rmdir()
