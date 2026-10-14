"""Tests for the GitHub Copilot config flow."""

from __future__ import annotations

from http import HTTPStatus

from homeassistant import config_entries
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.github_copilot.config_flow import GITHUB_USER_URL
from custom_components.github_copilot.const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    DOMAIN,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
)

from .conftest import TOKEN, FakeCopilotClient


async def _submit_token(hass: HomeAssistant, flow_id: str, token: str = TOKEN):
    result = await hass.config_entries.flow.async_configure(
        flow_id, {CONF_ACCESS_TOKEN: token}
    )
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    await hass.async_block_till_done()
    return await hass.config_entries.flow.async_configure(flow_id)


async def test_user_flow(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """A valid token creates the entry with default subentries."""
    aioclient_mock.get(GITHUB_USER_URL, json={"id": 42, "login": "octocat"})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await _submit_token(hass, result["flow_id"], f"  {TOKEN}  ")
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "GitHub Copilot (octocat)"
    assert result["data"] == {CONF_ACCESS_TOKEN: TOKEN}
    assert result["result"].unique_id == "42"
    (_method, _url, _data, headers) = aioclient_mock.mock_calls[0]
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    subentries = list(result["result"].subentries.values())
    assert [s.subentry_type for s in subentries] == [
        SUBENTRY_CONVERSATION,
        SUBENTRY_AI_TASK,
    ]
    assert all(s.data[CONF_CHAT_MODEL] == "gpt-5-mini" for s in subentries)
    assert subentries[0].data[CONF_LLM_HASS_API] == ["assist"]

    # The validation runtime is isolated and torn down afterwards.
    validator = mock_copilot.instances[0]
    assert validator.kwargs["mode"] == "empty"
    assert validator.kwargs["github_token"] == TOKEN
    assert validator.kwargs["use_logged_in_user"] is False
    assert validator.stopped


async def test_default_model_falls_back(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """Without preferred models the first available model is used."""
    from .conftest import _model

    def configure(client: FakeCopilotClient) -> None:
        client.models = [_model("only-model")]

    mock_copilot.configure = configure
    aioclient_mock.get(GITHUB_USER_URL, json={"id": 42, "login": "octocat"})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await _submit_token(hass, result["flow_id"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    for subentry in result["result"].subentries.values():
        assert subentry.data[CONF_CHAT_MODEL] == "only-model"


@pytest.mark.parametrize(
    ("status", "authenticated", "start_error", "error"),
    [
        (HTTPStatus.UNAUTHORIZED, True, None, "invalid_auth"),
        (HTTPStatus.FORBIDDEN, True, None, "cannot_connect"),
        (HTTPStatus.INTERNAL_SERVER_ERROR, True, None, "cannot_connect"),
        (HTTPStatus.OK, False, None, "invalid_auth"),
        (HTTPStatus.OK, True, RuntimeError("no runtime"), "cannot_connect"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
    status: HTTPStatus,
    authenticated: bool,
    start_error: Exception | None,
    error: str,
) -> None:
    """Errors are reported and the user can retry."""

    def configure(client: FakeCopilotClient) -> None:
        client.authenticated = authenticated
        client.start_error = start_error

    mock_copilot.configure = configure
    aioclient_mock.get(
        GITHUB_USER_URL, status=status, json={"id": 42, "login": "octocat"}
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await _submit_token(hass, result["flow_id"])
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": error}

    mock_copilot.configure = None
    aioclient_mock.clear_requests()
    aioclient_mock.get(GITHUB_USER_URL, json={"id": 42, "login": "octocat"})
    result = await _submit_token(hass, result["flow_id"])
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_already_configured(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """The same GitHub account can only be added once."""
    MockConfigEntry(domain=DOMAIN, unique_id="42", data={}).add_to_hass(hass)
    aioclient_mock.get(GITHUB_USER_URL, json={"id": 42, "login": "octocat"})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await _submit_token(hass, result["flow_id"])
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
    mock_config_entry: MockConfigEntry,
) -> None:
    """Reauth stores the new token for the same account."""
    aioclient_mock.get(GITHUB_USER_URL, json={"id": 1, "login": "octocat"})
    result = await mock_config_entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    result = await _submit_token(hass, result["flow_id"], "github_pat_new")
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_config_entry.data[CONF_ACCESS_TOKEN] == "github_pat_new"


async def test_reauth_wrong_account(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
    mock_config_entry: MockConfigEntry,
) -> None:
    """Reauth with a different account is refused."""
    aioclient_mock.get(GITHUB_USER_URL, json={"id": 99, "login": "someoneelse"})
    result = await mock_config_entry.start_reauth_flow(hass)
    result = await _submit_token(hass, result["flow_id"], "github_pat_new")
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "wrong_account"
    assert mock_config_entry.data[CONF_ACCESS_TOKEN] == TOKEN


async def test_reauth_invalid_token(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_copilot: type[FakeCopilotClient],
    mock_config_entry: MockConfigEntry,
) -> None:
    """A rejected token during reauth returns to the reauth form."""
    aioclient_mock.get(GITHUB_USER_URL, status=HTTPStatus.UNAUTHORIZED)
    result = await mock_config_entry.start_reauth_flow(hass)
    result = await _submit_token(hass, result["flow_id"], "github_pat_bad")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_add_conversation_subentry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A second conversation agent can be added with its own model."""
    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, SUBENTRY_CONVERSATION),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    schema_keys = {str(key) for key in result["data_schema"].schema}
    assert {CONF_NAME, CONF_PROMPT, CONF_LLM_HASS_API, CONF_CHAT_MODEL} <= schema_keys
    model_selector = next(
        value
        for key, value in result["data_schema"].schema.items()
        if str(key) == CONF_CHAT_MODEL
    )
    # Disabled models are not offered.
    assert {o["value"] for o in model_selector.config["options"]} == {
        "gpt-5-mini",
        "claude-haiku-4.5",
        "text-only",
    }

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Claude",
            CONF_CHAT_MODEL: "claude-haiku-4.5",
            CONF_REASONING_EFFORT: "default",
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Claude"
    assert result["data"] == {
        CONF_CHAT_MODEL: "claude-haiku-4.5",
        CONF_REASONING_EFFORT: "default",
    }


async def test_reconfigure_ai_task_subentry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """The AI task subentry has no prompt/LLM API fields and can be updated."""
    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.subentry_type == SUBENTRY_AI_TASK
    )
    result = await mock_config_entry.start_subentry_reconfigure_flow(
        hass, subentry.subentry_id
    )
    assert result["type"] is FlowResultType.FORM
    schema_keys = {str(key) for key in result["data_schema"].schema}
    assert schema_keys == {CONF_CHAT_MODEL, CONF_REASONING_EFFORT}

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_CHAT_MODEL: "claude-haiku-4.5", CONF_REASONING_EFFORT: "high"},
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert mock_config_entry.subentries[subentry.subentry_id].data == {
        CONF_CHAT_MODEL: "claude-haiku-4.5",
        CONF_REASONING_EFFORT: "high",
    }


async def test_subentry_requires_loaded_entry(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Subentries can't be configured while the entry isn't loaded."""
    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, SUBENTRY_AI_TASK),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"
