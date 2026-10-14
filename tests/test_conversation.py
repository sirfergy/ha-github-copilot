"""Tests for the GitHub Copilot conversation agent."""

from __future__ import annotations

from typing import Any

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import chat_session, intent, llm
import probatio
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.github_copilot.const import (
    CONF_CHAT_MODEL,
    CONF_REASONING_EFFORT,
    SUBENTRY_CONVERSATION,
)

from .conftest import FakeCopilotClient, FakeSession

AGENT_ID = "conversation.github_copilot_conversation"


class LightTool(llm.Tool):
    """A fake tool that records calls."""

    name = "TurnOnLight"
    description = "Turns on a light"
    parameters = probatio.Schema({probatio.Required("name"): str})
    integration = "test"

    def __init__(self) -> None:
        """Initialize."""
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        """Record the call."""
        self.calls.append(tool_input.tool_args)
        if tool_input.tool_args["name"] == "broken":
            raise intent.IntentHandleError("No such light")
        return llm.ToolResult(data={"success": True})


@pytest.fixture
def light_tool(hass: HomeAssistant) -> LightTool:
    """Register a test LLM API exposing one tool."""
    tool = LightTool()

    class TestAPI(llm.API):
        async def async_get_api_instance(
            self, llm_context: llm.LLMContext
        ) -> llm.APIInstance:
            return llm.APIInstance(
                api=self,
                api_prompt="You can turn on lights.",
                llm_context=llm_context,
                tools=[tool],
            )

    llm.async_register_api(hass, TestAPI(hass=hass, id="test-api", name="Test API"))
    return tool


def _conversation_subentry(entry: MockConfigEntry):
    return next(
        s for s in entry.subentries.values() if s.subentry_type == SUBENTRY_CONVERSATION
    )


async def _set_options(
    hass: HomeAssistant, entry: MockConfigEntry, **options: Any
) -> None:
    subentry = _conversation_subentry(entry)
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, **options}
    )
    await hass.async_block_till_done()


async def _ask(
    hass: HomeAssistant, text: str, conversation_id: str | None = None
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass, text, conversation_id, Context(), agent_id=AGENT_ID
    )


async def test_simple_reply(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A plain answer streams back and the session is configured safely."""
    await _set_options(hass, mock_config_entry, **{CONF_LLM_HASS_API: []})
    client = mock_copilot_client(hass)
    result = await _ask(hass, "hello")
    await hass.async_block_till_done()

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Hello from Copilot"

    session = client.sessions[-1]
    assert session.options["model"] == "gpt-5-mini"
    assert session.options["reasoning_effort"] == "low"
    assert session.options["tools"] == []
    assert session.options["available_tools"] == []
    assert session.options["system_message"]["mode"] == "replace"
    assert "voice assistant" in session.options["system_message"]["content"]
    assert session.options["infinite_sessions"] == {"enabled": False}
    assert session.sent[0]["prompt"] == "hello"
    assert session.sent[0]["response_schema"] is None
    # Sessions are torn down after each request.
    assert session.disconnected
    assert client.deleted == [session.session_id]


def mock_copilot_client(hass: HomeAssistant) -> FakeCopilotClient:
    """Return the client of the loaded entry (it is recreated on reload)."""
    return FakeCopilotClient.instances[-1]


async def test_reasoning_effort_only_when_supported(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """Reasoning effort is omitted for models that don't support it."""
    await _set_options(
        hass,
        mock_config_entry,
        **{CONF_CHAT_MODEL: "claude-haiku-4.5", CONF_REASONING_EFFORT: "high"},
    )
    client = mock_copilot_client(hass)
    await _ask(hass, "hello")
    assert "reasoning_effort" not in client.sessions[-1].options


async def test_tool_calls(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    light_tool: LightTool,
) -> None:
    """Copilot-driven tool calls run HA tools and are logged as external."""
    await _set_options(hass, mock_config_entry, **{CONF_LLM_HASS_API: ["test-api"]})
    client = mock_copilot_client(hass)
    results: list[Any] = []

    async def script(session: FakeSession) -> None:
        session.say("Turning it on.")
        results.append(await session.call_tool("TurnOnLight", {"name": "kitchen"}))
        results.append(await session.call_tool("TurnOnLight", {"name": "broken"}))
        session.say("Kitchen is on; the other light doesn't exist.")
        session.idle()

    client.script = script
    result = await _ask(hass, "turn on the kitchen light")

    assert result.response.speech["plain"]["speech"] == (
        "Kitchen is on; the other light doesn't exist."
    )
    assert light_tool.calls == [{"name": "kitchen"}, {"name": "broken"}]
    assert results[0].result_type == "success"
    assert results[0].text_result_for_llm == '{"success": true}'
    assert results[1].result_type == "failure"
    assert "No such light" in results[1].text_result_for_llm

    session = client.sessions[-1]
    (tool,) = session.options["tools"]
    assert tool.name == "TurnOnLight"
    assert tool.skip_permission is True
    assert tool.parameters == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    assert session.options["available_tools"] == ["custom:*"]
    assert "You can turn on lights." in session.options["system_message"]["content"]

    chat_log_content = _chat_log_content(hass, result.conversation_id)
    roles = [(c.role, getattr(c, "content", None)) for c in chat_log_content[1:]]
    assert roles == [
        ("user", "turn on the kitchen light"),
        ("assistant", "Turning it on."),
        ("tool_result", None),
        ("assistant", None),
        ("tool_result", None),
        ("assistant", "Kitchen is on; the other light doesn't exist."),
    ]
    first_call = chat_log_content[2].tool_calls[0]
    assert first_call.external is True
    assert first_call.id == "call_TurnOnLight"
    assert chat_log_content[3].result == llm.ToolResult(data={"success": True})
    assert chat_log_content[5].result.error is True


def _chat_log_content(hass: HomeAssistant, conversation_id: str):
    with (
        chat_session.async_get_chat_session(hass, conversation_id) as session,
        conversation.async_get_chat_log(hass, session) as chat_log,
    ):
        return list(chat_log.content)


async def test_history_is_replayed(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """Follow-up turns include earlier messages as context."""
    await _set_options(hass, mock_config_entry, **{CONF_LLM_HASS_API: []})
    client = mock_copilot_client(hass)
    first = await _ask(hass, "my name is Sean")
    await _ask(hass, "what is my name?", first.conversation_id)

    prompt = client.sessions[-1].sent[0]["prompt"]
    assert "User: my name is Sean" in prompt
    assert "Assistant: Hello from Copilot" in prompt
    assert prompt.endswith("Current user message:\nwhat is my name?")


async def test_error_event(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """Copilot session errors surface as a conversation error."""
    client = mock_copilot_client(hass)

    async def script(session: FakeSession) -> None:
        session.error("quota exceeded")
        session.idle()

    client.script = script
    result = await _ask(hass, "hello")
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert "quota exceeded" in result.response.speech["plain"]["speech"]
    await hass.async_block_till_done()
    assert client.sessions[-1].aborted


async def test_timeout(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck request times out and is aborted."""
    monkeypatch.setattr(
        "custom_components.github_copilot.conversation.CONVERSATION_TIMEOUT", 0.05
    )
    client = mock_copilot_client(hass)

    async def script(session: FakeSession) -> None:
        session.say("thinking")

    client.script = script
    result = await _ask(hass, "hello")
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert "did not respond" in result.response.speech["plain"]["speech"]
    await hass.async_block_till_done()
    assert client.sessions[-1].aborted


async def test_sub_agent_events_ignored(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """Events tagged with an agent id are not mixed into the reply."""
    from copilot.session_events import AssistantMessageData

    client = mock_copilot_client(hass)

    async def script(session: FakeSession) -> None:
        session.emit(
            AssistantMessageData(content="sub agent", message_id="x"), agent_id="sub"
        )
        session.say("root")
        session.idle()

    client.script = script
    result = await _ask(hass, "hello")
    assert result.response.speech["plain"]["speech"] == "root"


async def test_runtime_restarts_when_dead(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A dead runtime is replaced once and the request still succeeds."""
    await _set_options(hass, mock_config_entry, **{CONF_LLM_HASS_API: []})
    dead = mock_copilot_client(hass)
    dead.create_session_error = RuntimeError("process exited")
    dead.alive = False

    result = await _ask(hass, "hello")

    assert result.response.speech["plain"]["speech"] == "Hello from Copilot"
    assert dead.stopped
    replacement = mock_copilot_client(hass)
    assert replacement is not dead
    assert len(replacement.sessions) == 1


async def test_runtime_restart_failure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    mock_copilot: type[FakeCopilotClient],
) -> None:
    """If the runtime can't come back the user gets a clear error."""
    dead = mock_copilot_client(hass)
    dead.create_session_error = RuntimeError("process exited")
    dead.alive = False

    def configure(client: FakeCopilotClient) -> None:
        client.start_error = RuntimeError("download failed")

    mock_copilot.configure = configure
    result = await _ask(hass, "hello")
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert "not available" in result.response.speech["plain"]["speech"]


async def test_unauthorized_starts_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A 401 from Copilot mid-request asks the user to re-authenticate."""
    from copilot.session_events import SessionErrorData

    client = mock_copilot_client(hass)

    async def script(session: FakeSession) -> None:
        session.emit(
            SessionErrorData(
                error_type="authentication", message="Bad credentials", status_code=401
            )
        )
        session.idle()

    client.script = script
    result = await _ask(hass, "hello")
    assert result.response.response_type is intent.IntentResponseType.ERROR
    flows = hass.config_entries.flow.async_progress()
    assert [flow["context"]["source"] for flow in flows] == ["reauth"]


async def test_request_error_does_not_restart_runtime(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A bad request (e.g. unknown model) fails alone; the runtime survives."""
    client = mock_copilot_client(hass)
    client.create_session_error = ValueError("unknown model")

    result = await _ask(hass, "hello")

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert "unknown model" in result.response.speech["plain"]["speech"]
    assert not client.stopped
    assert mock_copilot_client(hass) is client


async def test_idle_before_message_is_ignored(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A stray idle before our message starts doesn't end the request."""
    client = mock_copilot_client(hass)

    async def script(session: FakeSession) -> None:
        session.idle()
        session.say("real answer")
        session.idle()

    client.script = script
    result = await _ask(hass, "hello")
    assert result.response.speech["plain"]["speech"] == "real answer"


async def test_unexpected_tool_error_is_not_sent_to_model(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
    light_tool: LightTool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected exception text stays local."""
    await _set_options(hass, mock_config_entry, **{CONF_LLM_HASS_API: ["test-api"]})
    client = mock_copilot_client(hass)

    async def boom(*args: Any) -> llm.ToolResult:
        raise RuntimeError("secret https://user:pass@example.com")

    monkeypatch.setattr(light_tool, "async_call", boom)
    results: list[Any] = []

    async def script(session: FakeSession) -> None:
        results.append(await session.call_tool("TurnOnLight", {"name": "x"}))
        session.say("failed")
        session.idle()

    client.script = script
    await _ask(hass, "turn on x")
    assert results[0].result_type == "failure"
    assert "secret" not in results[0].text_result_for_llm
    assert "failed unexpectedly" in results[0].text_result_for_llm


async def test_session_teardown_survives_abort_failure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: FakeCopilotClient,
) -> None:
    """A failing abort still disconnects and deletes the session."""
    client = mock_copilot_client(hass)

    async def script(session: FakeSession) -> None:
        session.abort_error = RuntimeError("abort failed")
        session.error("boom")

    client.script = script
    await _ask(hass, "hello")
    await hass.async_block_till_done()
    session = client.sessions[-1]
    assert session.aborted
    assert session.disconnected
    assert client.deleted == [session.session_id]
