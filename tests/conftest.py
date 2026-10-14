"""Fixtures for GitHub Copilot tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, field
from typing import Any, ClassVar
from unittest.mock import patch
import uuid

from copilot.client import (
    GetAuthStatusResponse,
    ModelCapabilities,
    ModelInfo,
    ModelLimits,
    ModelPolicy,
    ModelSupports,
    ModelVisionLimits,
)
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantReasoningDeltaData,
    AssistantTurnStartData,
    SessionErrorData,
    SessionEvent,
    SessionIdleData,
)
from copilot.tools import ToolInvocation
from homeassistant.const import CONF_ACCESS_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.github_copilot.const import (
    CONF_CHAT_MODEL,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_CONVERSATION_OPTIONS,
    DOMAIN,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
)

TOKEN = "github_pat_test"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading custom integrations in all tests."""


@pytest.fixture(autouse=True)
async def setup_homeassistant(hass: HomeAssistant) -> None:
    """Conversation needs the exposed-entities store from `homeassistant`."""
    assert await async_setup_component(hass, "homeassistant", {})


def _model(
    model_id: str,
    *,
    vision: bool = True,
    efforts: list[str] | None = None,
    enabled: bool = True,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=model_id.upper(),
        capabilities=ModelCapabilities(
            supports=ModelSupports(vision=vision, reasoning_effort=bool(efforts)),
            limits=ModelLimits(
                vision=ModelVisionLimits(
                    supported_media_types=["image/jpeg", "image/png"],
                    max_prompt_images=2,
                    max_prompt_image_size=1024,
                )
                if vision
                else None
            ),
        ),
        policy=ModelPolicy(state="enabled" if enabled else "disabled", terms=""),
        supported_reasoning_efforts=efforts,
    )


MODELS = [
    _model("gpt-5-mini", efforts=["low", "medium", "high"]),
    _model("claude-haiku-4.5"),
    _model("text-only", vision=False),
    _model("disabled-model", enabled=False),
]


def _event(data: Any, agent_id: str | None = None) -> SessionEvent:
    return SessionEvent(
        data=data,
        id=uuid.uuid4(),
        timestamp=None,  # type: ignore[arg-type]
        type=None,  # type: ignore[arg-type]
        agent_id=agent_id,
    )


type Script = Callable[["FakeSession"], Awaitable[None]]


@dataclass
class FakeSession:
    """Stands in for a CopilotSession and replays a scripted turn."""

    client: FakeCopilotClient
    options: dict[str, Any]
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sent: list[dict[str, Any]] = field(default_factory=list)
    aborted: bool = False
    disconnected: bool = False

    def emit(self, data: Any, agent_id: str | None = None) -> None:
        self.options["on_event"](_event(data, agent_id))

    def say(self, text: str, *, chunks: int = 2) -> None:
        """Emit one streamed assistant message."""
        message_id = str(uuid.uuid4())
        self.emit(AssistantTurnStartData(turn_id=str(uuid.uuid4())))
        size = max(1, len(text) // chunks)
        for start in range(0, len(text), size):
            self.emit(
                AssistantMessageDeltaData(
                    delta_content=text[start : start + size], message_id=message_id
                )
            )
        self.emit(AssistantMessageData(content=text, message_id=message_id))

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a registered tool handler like the runtime would."""
        tool = next(t for t in self.options["tools"] if t.name == name)
        return await tool.handler(
            ToolInvocation(
                session_id=self.session_id,
                tool_call_id=f"call_{name}",
                tool_name=name,
                arguments=args,
            )
        )

    def idle(self) -> None:
        self.emit(SessionIdleData())

    def error(self, message: str) -> None:
        self.emit(SessionErrorData(error_type="query", message=message))

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        self.sent.append(
            {
                "prompt": prompt,
                "attachments": attachments,
                "response_schema": response_schema,
            }
        )
        self.client.tasks.append(asyncio.create_task(self.client.script(self)))
        return "message-id"

    async def abort(self) -> None:
        self.aborted = True

    async def disconnect(self) -> None:
        self.disconnected = True


async def _default_script(session: FakeSession) -> None:
    session.say("Hello from Copilot")
    session.idle()


class FakeCopilotClient:
    """Stands in for copilot.CopilotClient."""

    instances: ClassVar[list[FakeCopilotClient]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.authenticated = True
        self.models = list(MODELS)
        self.start_error: Exception | None = None
        self.create_session_error: Exception | None = None
        self.sessions: list[FakeSession] = []
        self.deleted: list[str] = []
        self.tasks: list[asyncio.Task] = []
        self.script: Script = _default_script
        FakeCopilotClient.instances.append(self)
        if hook := FakeCopilotClient.configure:
            hook(self)

    configure: ClassVar[Callable[[FakeCopilotClient], None] | None] = None

    async def start(self) -> None:
        if self.start_error:
            raise self.start_error
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def get_auth_status(self) -> GetAuthStatusResponse:
        return GetAuthStatusResponse(
            isAuthenticated=self.authenticated, statusMessage="status"
        )

    async def list_models(self) -> list[ModelInfo]:
        return self.models

    async def create_session(self, **options: Any) -> FakeSession:
        if self.create_session_error:
            raise self.create_session_error
        session = FakeSession(self, options)
        self.sessions.append(session)
        return session

    async def delete_session(self, session_id: str) -> None:
        self.deleted.append(session_id)


@pytest.fixture
def mock_copilot() -> Generator[type[FakeCopilotClient]]:
    """Patch the SDK client everywhere it is constructed."""
    FakeCopilotClient.instances = []
    FakeCopilotClient.configure = None
    with patch(
        "custom_components.github_copilot.runtime.CopilotClient", FakeCopilotClient
    ):
        yield FakeCopilotClient


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry with one conversation and one AI task subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="GitHub Copilot (octocat)",
        unique_id="1",
        data={CONF_ACCESS_TOKEN: TOKEN},
        subentries_data=[
            {
                "subentry_type": SUBENTRY_CONVERSATION,
                "data": {
                    **DEFAULT_CONVERSATION_OPTIONS,
                    CONF_CHAT_MODEL: "gpt-5-mini",
                },
                "title": DEFAULT_CONVERSATION_NAME,
                "unique_id": None,
            },
            {
                "subentry_type": SUBENTRY_AI_TASK,
                "data": {**DEFAULT_AI_TASK_OPTIONS, CONF_CHAT_MODEL: "gpt-5-mini"},
                "title": DEFAULT_AI_TASK_NAME,
                "unique_id": None,
            },
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def setup_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_copilot: type[FakeCopilotClient],
) -> FakeCopilotClient:
    """Set up the integration and return the fake client."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_copilot.instances[-1]


__all__ = [
    "TOKEN",
    "AssistantReasoningDeltaData",
    "FakeCopilotClient",
    "FakeSession",
]
