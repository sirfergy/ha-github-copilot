"""Lifecycle wrapper around the GitHub Copilot SDK runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from functools import partial
import os
import shutil
import tempfile
from typing import Any
import uuid

from copilot import CopilotClient
from copilot.client import ModelInfo
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN, FALLBACK_CHAT_MODEL, LOGGER, PREFERRED_CHAT_MODELS

# Bound how long a hung runtime can delay unload, restart or shutdown.
STOP_TIMEOUT = 10


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


def _is_auth_failure(err: BaseException) -> bool:
    text = str(err).lower()
    return any(marker in text for marker in ("401", "403", "unauthorized", "forbidden"))


@dataclass
class CopilotRuntime:
    """One SDK client (and runtime subprocess) per config entry.

    Start, restart and close are serialized by one lock, and a closed runtime
    never starts again, so a request that races an unload can't resurrect it.
    """

    hass: HomeAssistant
    token: str
    # Stable name (the config entry id) for the scratch directory; None for
    # throwaway runtimes such as the one used to validate a token.
    name: str | None = None
    client: CopilotClient | None = None
    models: dict[str, CopilotModel] = field(default_factory=dict)
    _root_dir: str | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _closed: bool = False
    _cleanup_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _stop_listeners: set[Callable[[], None]] = field(default_factory=set)

    @callback
    def async_on_stop(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` if the runtime stops; returns an unsubscribe.

        The SDK emits nothing to in-flight sessions when its process exits, so
        requests register here to fail fast instead of waiting for a timeout.
        """
        self._stop_listeners.add(listener)
        return partial(self._stop_listeners.discard, listener)

    @callback
    def _notify_stop(self) -> None:
        listeners, self._stop_listeners = self._stop_listeners, set()
        for listener in listeners:
            listener()

    @callback
    def async_schedule_cleanup(self, target: Coroutine[Any, Any, None]) -> None:
        """Run per-request cleanup in the background, tracked for close()."""
        task = self.hass.async_create_background_task(target, f"{DOMAIN} cleanup")
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def async_start(self) -> None:
        """Provision the runtime, start it and verify the token."""
        async with self._lock:
            self._raise_if_closed()
            await self._async_start()

    async def async_restart(self, failed_client: CopilotClient | None) -> None:
        """Replace a runtime that has died; concurrent callers restart once."""
        async with self._lock:
            self._raise_if_closed()
            if self.client is not None and self.client is not failed_client:
                return
            LOGGER.warning("Restarting the GitHub Copilot runtime")
            await self._async_stop()
            await self._async_start()

    async def async_close(self) -> None:
        """Stop for good (entry unload or Home Assistant shutdown)."""
        self._closed = True
        self._notify_stop()
        # Let in-flight session cleanup finish (each step is bounded) so no
        # RPC is cancelled mid-flight while the runtime process exits.
        if self._cleanup_tasks:
            await asyncio.wait(set(self._cleanup_tasks), timeout=STOP_TIMEOUT)
        async with self._lock:
            await self._async_stop_shielded()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise CopilotConnectionError("The GitHub Copilot runtime was stopped")

    async def _async_start(self) -> None:
        try:
            await self._async_launch()
        except BaseException:
            # Includes cancellation (e.g. the config flow was closed mid-download).
            await self._async_stop_shielded()
            raise

    async def _async_launch(self) -> None:
        self._root_dir = os.path.join(
            tempfile.gettempdir(),
            f"ha-github-copilot-{self.name or uuid.uuid4().hex}",
        )
        try:
            home_dir, work_dir = await self.hass.async_add_executor_job(
                _make_dirs, self._root_dir
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
            status = await self.client.get_auth_status()
        except Exception as err:
            raise CopilotConnectionError(
                f"Unable to start the GitHub Copilot runtime: {err}"
            ) from err
        if not status.isAuthenticated:
            raise CopilotAuthError(status.statusMessage or "Not authenticated")

        try:
            await self.async_refresh_models()
        except Exception as err:
            # Listing models is the first call that needs Copilot entitlement.
            if _is_auth_failure(err):
                raise CopilotAuthError(
                    f"Token cannot access GitHub Copilot: {err}"
                ) from err
            raise CopilotConnectionError(
                f"Unable to list GitHub Copilot models: {err}"
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

    async def _async_stop_shielded(self) -> None:
        """Stop even if the caller is cancelled while waiting."""
        # A tracked task, so Home Assistant waits for it during shutdown.
        await asyncio.shield(
            self.hass.async_create_task(self._async_stop(), f"{DOMAIN} stop runtime")
        )

    async def _async_stop(self) -> None:
        """Stop the runtime subprocess and remove its scratch directories."""
        self._notify_stop()
        client, self.client = self.client, None
        root_dir, self._root_dir = self._root_dir, None
        try:
            if client is not None:
                try:
                    async with asyncio.timeout(STOP_TIMEOUT):
                        await client.stop()
                except Exception:
                    LOGGER.debug("Graceful runtime stop failed", exc_info=True)
                    try:
                        await client.force_stop()
                    except Exception:
                        LOGGER.debug("Forced runtime stop failed", exc_info=True)
        finally:
            if root_dir is not None:
                await self.hass.async_add_executor_job(
                    partial(shutil.rmtree, root_dir, ignore_errors=True)
                )


def _make_dirs(root: str) -> tuple[str, str]:
    """Create an isolated Copilot home and an empty working directory.

    Keeping these out of the Home Assistant config directory means no user
    skills, MCP servers or instruction files are ever discovered, and nothing
    the runtime writes ends up in backups. The root is wiped first so anything
    left behind by a crash or unclean shutdown is removed.
    """
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, mode=0o700)
    home_dir = os.path.join(root, "home")
    work_dir = os.path.join(root, "work")
    os.makedirs(home_dir)
    os.makedirs(work_dir)
    return home_dir, work_dir
