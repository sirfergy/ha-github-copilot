"""Config flow for the GitHub Copilot integration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import aiohttp
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
import probatio

from .const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_CONVERSATION_OPTIONS,
    DOMAIN,
    LOGGER,
    REASONING_EFFORTS,
    RECOMMENDED_REASONING_EFFORT,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
)
from .runtime import CopilotAuthError, CopilotConnectionError, CopilotRuntime

GITHUB_USER_URL = "https://api.github.com/user"

STEP_USER_SCHEMA = probatio.Schema(
    {
        probatio.Required(CONF_ACCESS_TOKEN): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        )
    }
)


class InvalidAuth(Exception):
    """The token was rejected."""


class CannotConnect(Exception):
    """GitHub or the Copilot runtime could not be reached."""


async def async_validate_token(hass: HomeAssistant, token: str) -> dict[str, Any]:
    """Validate a token against GitHub and the Copilot runtime.

    Returns the GitHub user and the default model to use for new subentries.
    """
    session = async_get_clientsession(hass)
    try:
        async with session.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            # 403 here usually means rate limiting rather than a bad token.
            if resp.status == HTTPStatus.UNAUTHORIZED:
                raise InvalidAuth
            resp.raise_for_status()
            user = await resp.json()
    except (aiohttp.ClientError, TimeoutError) as err:
        raise CannotConnect from err

    runtime = CopilotRuntime(hass, token)
    try:
        await runtime.async_start()
        default_model = runtime.default_model()
    except CopilotAuthError as err:
        LOGGER.debug("Copilot rejected token: %s", err)
        raise InvalidAuth from err
    except CopilotConnectionError as err:
        LOGGER.warning("Could not start the GitHub Copilot runtime: %s", err)
        raise CannotConnect from err
    finally:
        await runtime.async_close()

    return {
        "user_id": str(user["id"]),
        "login": user["login"],
        "default_model": default_model,
    }


class GitHubCopilotConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for GitHub Copilot."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the flow."""
        self._token: str | None = None
        self._validate_task: asyncio.Task[dict[str, Any]] | None = None
        self._errors: dict[str, str] = {}
        self._info: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a token."""
        if user_input is not None:
            self._token = user_input[CONF_ACCESS_TOKEN].strip()
            return await self.async_step_validate()

        return self.async_show_form(
            step_id="reauth_confirm" if self.source == "reauth" else "user",
            data_schema=STEP_USER_SCHEMA,
            errors=self._errors,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle an expired or revoked token."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a replacement token."""
        return await self.async_step_user(user_input)

    async def async_step_validate(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Validate the token; the first run may download the Copilot runtime."""
        if (task := self._validate_task) and task.done():
            self._validate_task = None
            self._errors = {}
            try:
                self._info = task.result()
            except InvalidAuth:
                self._errors["base"] = "invalid_auth"
            except CannotConnect:
                self._errors["base"] = "cannot_connect"
            except Exception:
                LOGGER.exception("Unexpected error validating GitHub Copilot token")
                self._errors["base"] = "unknown"
            return self.async_show_progress_done(
                next_step_id="reauth_confirm"
                if self._errors and self.source == "reauth"
                else "user"
                if self._errors
                else "finish"
            )

        if self._validate_task is None:
            assert self._token is not None
            self._validate_task = self.hass.async_create_task(
                async_validate_token(self.hass, self._token)
            )
        return self.async_show_progress(
            step_id="validate",
            progress_action="validate",
            progress_task=self._validate_task,
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Create or update the entry."""
        assert self._token is not None
        await self.async_set_unique_id(self._info["user_id"])
        if self.source == "reauth":
            self._abort_if_unique_id_mismatch(reason="wrong_account")
            return self.async_update_reload_and_abort(
                self._get_reauth_entry(),
                data_updates={CONF_ACCESS_TOKEN: self._token},
            )
        self._abort_if_unique_id_configured()

        model = {CONF_CHAT_MODEL: self._info["default_model"]}
        return self.async_create_entry(
            title=f"GitHub Copilot ({self._info['login']})",
            data={CONF_ACCESS_TOKEN: self._token},
            subentries=[
                {
                    "subentry_type": SUBENTRY_CONVERSATION,
                    "data": {**DEFAULT_CONVERSATION_OPTIONS, **model},
                    "title": DEFAULT_CONVERSATION_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": SUBENTRY_AI_TASK,
                    "data": {**DEFAULT_AI_TASK_OPTIONS, **model},
                    "title": DEFAULT_AI_TASK_NAME,
                    "unique_id": None,
                },
            ],
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            SUBENTRY_CONVERSATION: LLMSubentryFlowHandler,
            SUBENTRY_AI_TASK: LLMSubentryFlowHandler,
        }


class LLMSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure a conversation agent or AI task."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        if self._subentry_type == SUBENTRY_AI_TASK:
            self.options = dict(DEFAULT_AI_TASK_OPTIONS)
        else:
            self.options = dict(DEFAULT_CONVERSATION_OPTIONS)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show the settings form."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        runtime: CopilotRuntime = entry.runtime_data

        errors: dict[str, str] = {}
        if user_input is not None and user_input[CONF_CHAT_MODEL] not in runtime.models:
            # Copilot silently substitutes another model for unknown ids.
            errors[CONF_CHAT_MODEL] = "model_not_found"
            self.options = {**self.options, **user_input}
            user_input = None

        if user_input is not None:
            options = {**self.options, **user_input}
            if self._subentry_type == SUBENTRY_CONVERSATION:
                # Optional fields that were cleared in the form are omitted.
                for key in (CONF_PROMPT, CONF_LLM_HASS_API):
                    if not user_input.get(key):
                        options.pop(key, None)
            if self._is_new:
                title = options.pop(CONF_NAME)
                return self.async_create_entry(title=title, data=options)
            return self.async_update_and_abort(
                entry, self._get_reconfigure_subentry(), data=options
            )

        known_apis = {api.id: api.name for api in llm.async_get_apis(self.hass)}
        if suggested := self.options.get(CONF_LLM_HASS_API):
            if isinstance(suggested, str):
                suggested = [suggested]
            self.options[CONF_LLM_HASS_API] = [a for a in suggested if a in known_apis]

        schema: dict[Any, Any] = {}
        if self._is_new:
            default_name = (
                DEFAULT_AI_TASK_NAME
                if self._subentry_type == SUBENTRY_AI_TASK
                else DEFAULT_CONVERSATION_NAME
            )
            schema[
                probatio.Required(
                    CONF_NAME, default=self.options.get(CONF_NAME, default_name)
                )
            ] = str
        if self._subentry_type == SUBENTRY_CONVERSATION:
            schema[
                probatio.Optional(
                    CONF_PROMPT,
                    description={"suggested_value": self.options.get(CONF_PROMPT)},
                )
            ] = TemplateSelector()
            schema[
                probatio.Optional(
                    CONF_LLM_HASS_API,
                    description={
                        "suggested_value": self.options.get(CONF_LLM_HASS_API)
                    },
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=name, value=api_id)
                        for api_id, name in known_apis.items()
                    ],
                    multiple=True,
                )
            )
        schema[
            probatio.Required(
                CONF_CHAT_MODEL,
                default=self.options.get(CONF_CHAT_MODEL) or runtime.default_model(),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(label=f"{model.name} ({model.id})", value=model.id)
                    for model in runtime.models.values()
                ],
                mode=SelectSelectorMode.DROPDOWN,
                custom_value=True,
                sort=True,
            )
        )
        schema[
            probatio.Required(
                CONF_REASONING_EFFORT,
                default=self.options.get(
                    CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                ),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=list(REASONING_EFFORTS),
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_REASONING_EFFORT,
            )
        )

        return self.async_show_form(
            step_id="init", data_schema=probatio.Schema(schema), errors=errors
        )
