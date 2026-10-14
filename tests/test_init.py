"""Tests for GitHub Copilot setup and unload."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntryState
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
