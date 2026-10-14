"""Lifecycle wrapper around the GitHub Copilot SDK runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import partial
import os
import shutil
import tempfile

from copilot import CopilotClient
from copilot.client import ModelInfo
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import FALLBACK_CHAT_MODEL, LOGGER, PREFERRED_CHAT_MODELS


class CopilotError(HomeAssistantError):
    """Base error for the GitHub Copilot integration."""


class CopilotAuthError(CopilotError):
    """The token was rejected or has no Copilot access."""


class CopilotConnectionError(CopilotError):
    """The Copilot runtime could not be provisioned or started."""


@dataclass(frozen=True, slots=True)
class CopilotModel:
    """Subset of SDK model metadata the integration cares about."""

    id: str
    name: str
    vision: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    supported_media_types: tuple[str, ...] = ()
    max_images: int | None = None
    max_image_size: int | None = None

    @classmethod
    def from_sdk(cls, info: ModelInfo) -> CopilotModel:
        """Build from the SDK's ModelInfo."""
        vision_limits = info.capabilities.limits.vision
        return cls(
            id=info.id,
            name=info.name or info.id,
            vision=info.capabilities.supports.vision,
            reasoning_efforts=tuple(info.supported_reasoning_efforts or ()),
            supported_media_types=tuple(
                (vision_limits.supported_media_types or []) if vision_limits else ()
            ),
            max_images=vision_limits.max_prompt_images if vision_limits else None,
            max_image_size=vision_limits.max_prompt_image_size
            if vision_limits
            else None,
        )


def _model_enabled(info: ModelInfo) -> bool:
    return info.policy is None or info.policy.state != "disabled"


@dataclass
class CopilotRuntime:
    """One SDK client (and runtime subprocess) per config entry."""

    hass: HomeAssistant
    token: str
    client: CopilotClient | None = None
    models: dict[str, CopilotModel] = field(default_factory=dict)
    _root_dir: str | None = None
    _restart_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def async_restart(self, failed_client: CopilotClient | None) -> None:
        """Replace a runtime that has died; concurrent callers restart once."""
        async with self._restart_lock:
            if self.client is not None and self.client is not failed_client:
                return
            LOGGER.warning("Restarting the GitHub Copilot runtime")
            await self.async_stop()
            await self.async_start()

    async def async_start(self) -> None:
        """Provision the runtime, start it and verify the token."""
        try:
            self._root_dir, home_dir, work_dir = await self.hass.async_add_executor_job(
                _make_dirs
            )
            # Constructing the client resolves (and on first use downloads) the
            # runtime bundle with blocking I/O, so keep it off the event loop.
            self.client = await self.hass.async_add_executor_job(
                partial(
                    CopilotClient,
                    mode="empty",
                    base_directory=home_dir,
                    working_directory=work_dir,
                    github_token=self.token,
                    use_logged_in_user=False,
                    log_level="error",
                    client_info={"application_name": "home-assistant"},
                )
            )
            await self.client.start()
        except Exception as err:
            await self.async_stop()
            raise CopilotConnectionError(
                f"Unable to start the GitHub Copilot runtime: {err}"
            ) from err

        try:
            status = await self.client.get_auth_status()
        except Exception as err:
            await self.async_stop()
            raise CopilotConnectionError(
                f"Unable to query GitHub Copilot auth status: {err}"
            ) from err
        if not status.isAuthenticated:
            await self.async_stop()
            raise CopilotAuthError(status.statusMessage or "Not authenticated")

        try:
            await self.async_refresh_models()
        except Exception as err:
            await self.async_stop()
            # Listing models is the first call that needs Copilot entitlement.
            raise CopilotAuthError(
                f"Token cannot access GitHub Copilot models: {err}"
            ) from err

    async def async_refresh_models(self) -> None:
        """Refresh the cached model list."""
        assert self.client is not None
        infos = await self.client.list_models()
        self.models = {
            info.id: CopilotModel.from_sdk(info)
            for info in infos
            if _model_enabled(info)
        }

    def default_model(self) -> str:
        """Pick a sensible default model for new subentries."""
        for model_id in PREFERRED_CHAT_MODELS:
            if model_id in self.models:
                return model_id
        return next(iter(self.models), FALLBACK_CHAT_MODEL)

    async def async_stop(self) -> None:
        """Stop the runtime subprocess and remove its scratch directories."""
        client, self.client = self.client, None
        if client is not None:
            try:
                await client.stop()
            except Exception:
                LOGGER.debug("Error stopping Copilot client", exc_info=True)
        root_dir, self._root_dir = self._root_dir, None
        if root_dir is not None:
            await self.hass.async_add_executor_job(
                partial(shutil.rmtree, root_dir, ignore_errors=True)
            )


def _make_dirs() -> tuple[str, str, str]:
    """Create an isolated Copilot home and an empty working directory.

    Keeping these out of the Home Assistant config directory means no user
    skills, MCP servers or instruction files are ever discovered, and nothing
    the runtime writes ends up in backups.
    """
    root = tempfile.mkdtemp(prefix="ha-github-copilot-")
    home_dir = os.path.join(root, "home")
    work_dir = os.path.join(root, "work")
    os.makedirs(home_dir)
    os.makedirs(work_dir)
    return root, home_dir, work_dir
