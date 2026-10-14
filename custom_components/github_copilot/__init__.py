"""The GitHub Copilot integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ACCESS_TOKEN, EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN
from .runtime import CopilotAuthError, CopilotConnectionError, CopilotRuntime

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type GitHubCopilotConfigEntry = ConfigEntry[CopilotRuntime]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration."""
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: GitHubCopilotConfigEntry
) -> bool:
    """Start a Copilot runtime for this account."""
    runtime = CopilotRuntime(hass, entry.data[CONF_ACCESS_TOKEN], name=entry.entry_id)
    try:
        await runtime.async_start()
    except CopilotAuthError as err:
        raise ConfigEntryAuthFailed(str(err)) from err
    except CopilotConnectionError as err:
        raise ConfigEntryNotReady(str(err)) from err

    entry.runtime_data = runtime
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    async def _async_stop_runtime(event: Event) -> None:
        # Entries aren't unloaded at shutdown; stop the subprocess explicitly.
        await runtime.async_close()

    entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _async_stop_runtime)
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: GitHubCopilotConfigEntry
) -> bool:
    """Stop the runtime."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    await entry.runtime_data.async_close()
    return True


async def _async_update_listener(
    hass: HomeAssistant, entry: GitHubCopilotConfigEntry
) -> None:
    """Reload when subentries or data change."""
    await hass.config_entries.async_reload(entry.entry_id)
