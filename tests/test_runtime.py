"""Tests for the runtime lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

from homeassistant.core import HomeAssistant
import pytest

from custom_components.github_copilot.runtime import (
    CopilotAuthError,
    CopilotConnectionError,
    CopilotRuntime,
)

from .conftest import FakeCopilotClient


async def test_closed_runtime_never_restarts(
    hass: HomeAssistant, mock_copilot: type[FakeCopilotClient]
) -> None:
    """A request racing an unload can't resurrect the runtime."""
    runtime = CopilotRuntime(hass, "token", name="closed")
    await runtime.async_start()
    client = runtime.client
    await runtime.async_close()
    assert client.stopped

    with pytest.raises(CopilotConnectionError):
        await runtime.async_restart(None)
    assert runtime.client is None
    assert len(mock_copilot.instances) == 1


async def test_close_waits_for_in_flight_restart(
    hass: HomeAssistant, mock_copilot: type[FakeCopilotClient]
) -> None:
    """Close after a concurrent restart still leaves nothing running."""
    runtime = CopilotRuntime(hass, "token", name="race")
    await runtime.async_start()
    gate = asyncio.Event()

    def configure(client: FakeCopilotClient) -> None:
        client.start_gate = gate

    mock_copilot.configure = configure
    restart = asyncio.create_task(runtime.async_restart(runtime.client))
    await asyncio.sleep(0)
    close = asyncio.create_task(runtime.async_close())
    await asyncio.sleep(0)
    gate.set()
    await restart
    await close
    assert runtime.client is None
    assert all(c.stopped for c in mock_copilot.instances if c.started)


async def test_cancelled_start_cleans_up(
    hass: HomeAssistant, mock_copilot: type[FakeCopilotClient]
) -> None:
    """Cancelling startup (e.g. closing the config flow) leaves nothing behind."""

    def configure(client: FakeCopilotClient) -> None:
        client.start_gate = asyncio.Event()

    mock_copilot.configure = configure
    runtime = CopilotRuntime(hass, "token")
    task = asyncio.create_task(runtime.async_start())
    await asyncio.sleep(0.05)
    root = Path(mock_copilot.instances[0].kwargs["base_directory"]).parent
    assert root.is_dir()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await hass.async_block_till_done()
    assert runtime.client is None
    assert mock_copilot.instances[0].stopped
    assert not root.exists()


async def test_hung_stop_is_forced(
    hass: HomeAssistant,
    mock_copilot: type[FakeCopilotClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged runtime can't hang unload."""
    monkeypatch.setattr("custom_components.github_copilot.runtime.STOP_TIMEOUT", 0.01)
    runtime = CopilotRuntime(hass, "token")
    await runtime.async_start()
    client = runtime.client
    client.stop_hangs = True
    await runtime.async_close()
    assert client.force_stopped


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("HTTP 403 Forbidden"), CopilotAuthError),
        (ConnectionError("runtime exited"), CopilotConnectionError),
    ],
)
async def test_list_models_error_classification(
    hass: HomeAssistant,
    mock_copilot: type[FakeCopilotClient],
    error: Exception,
    expected: type[Exception],
) -> None:
    """Only auth-looking model-list failures are treated as bad tokens."""

    def configure(client: FakeCopilotClient) -> None:
        client.list_models_error = error

    mock_copilot.configure = configure
    runtime = CopilotRuntime(hass, "token")
    with pytest.raises(expected):
        await runtime.async_start()
    assert runtime.client is None
    assert mock_copilot.instances[0].stopped


async def test_close_drains_cleanup_before_stopping(
    hass: HomeAssistant, mock_copilot: type[FakeCopilotClient]
) -> None:
    """In-flight session cleanup finishes before the runtime process stops."""
    runtime = CopilotRuntime(hass, "token")
    await runtime.async_start()
    client = runtime.client
    seen: list[bool] = []

    async def cleanup() -> None:
        await asyncio.sleep(0.01)
        seen.append(client.stopped)

    runtime.async_schedule_cleanup(cleanup())
    await runtime.async_close()
    assert seen == [False]
    assert client.stopped


async def test_stop_notifies_listeners(
    hass: HomeAssistant, mock_copilot: type[FakeCopilotClient]
) -> None:
    """Listeners fire once on close and unsubscribed ones don't."""
    runtime = CopilotRuntime(hass, "token")
    await runtime.async_start()
    calls: list[str] = []
    runtime.async_on_stop(lambda: calls.append("a"))
    unsubscribe = runtime.async_on_stop(lambda: calls.append("b"))
    unsubscribe()
    await runtime.async_close()
    assert calls == ["a"]
