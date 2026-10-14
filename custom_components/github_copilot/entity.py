"""Base entity that runs Home Assistant chat logs through GitHub Copilot."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
import json
import mimetypes
from typing import TYPE_CHECKING, Any

from copilot import CopilotClient
from copilot.rpc import PermissionDecisionReject
from copilot.session import CopilotSession
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantReasoningDeltaData,
    AssistantTurnStartData,
    SessionErrorData,
    SessionEvent,
    SessionIdleData,
    UserMessageData,
)
from copilot.tools import Tool, ToolInvocation, ToolResult
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.util.ulid import ulid_now
import probatio

from .const import (
    CONF_CHAT_MODEL,
    CONF_REASONING_EFFORT,
    DOMAIN,
    LOGGER,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    MAX_HISTORY_TOOL_RESULT_CHARS,
    PING_TIMEOUT,
    REASONING_EFFORT_DEFAULT,
    RECOMMENDED_REASONING_EFFORT,
    SESSION_CLOSE_TIMEOUT,
)
from .runtime import CopilotAuthError, CopilotError, CopilotModel, CopilotRuntime
from .schema import strict_output_schema, tool_parameters_schema

if TYPE_CHECKING:
    from . import GitHubCopilotConfigEntry

type _Delta = (
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
)

_DONE = object()


class CopilotRequestError(HomeAssistantError):
    """A Copilot session reported an error."""


class CopilotAuthRequestError(CopilotRequestError):
    """Copilot rejected the token mid-request (expired or revoked)."""


@dataclass
class _Run:
    """Bridges one Copilot session's events into a chat log delta stream."""

    queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    streamed_message_ids: set[str] = field(default_factory=set)
    # Idle only means "done" once the runtime has picked up our message.
    started: bool = False

    def emit(self, item: _Delta | Exception | object) -> None:
        self.queue.put_nowait(item)

    def on_event(self, event: SessionEvent) -> None:
        if event.agent_id:
            # Sub-agent chatter; we never enable sub-agents, but be safe.
            return
        match event.data:
            case UserMessageData():
                self.started = True
            case AssistantTurnStartData():
                self.started = True
                self.emit({"role": "assistant"})
            case AssistantMessageDeltaData(delta_content=text, message_id=message_id):
                if text:
                    self.streamed_message_ids.add(message_id)
                    self.emit({"content": text})
            case AssistantReasoningDeltaData(delta_content=text):
                if text:
                    self.emit({"thinking_content": text})
            case AssistantMessageData(content=text, message_id=message_id):
                if text and message_id not in self.streamed_message_ids:
                    self.emit({"content": text})
            case SessionErrorData(message=message, status_code=status_code):
                error_cls = (
                    CopilotAuthRequestError
                    if status_code == HTTPStatus.UNAUTHORIZED
                    else CopilotRequestError
                )
                self.emit(error_cls(f"GitHub Copilot error: {message}"))
            case SessionIdleData(aborted=aborted) if self.started:
                if aborted:
                    self.emit(CopilotRequestError("GitHub Copilot request was aborted"))
                self.emit(_DONE)

    async def deltas(self) -> AsyncIterator[_Delta]:
        while True:
            item = await self.queue.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item


class GitHubCopilotBaseLLMEntity(Entity):
    """Shared logic for conversation and AI task entities."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: GitHubCopilotConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        model_id = self._model_id
        model = entry.runtime_data.models.get(model_id)
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="GitHub",
            model=model.name if model else model_id,
            model_id=model_id,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _runtime(self) -> CopilotRuntime:
        return self.entry.runtime_data

    @property
    def _model_id(self) -> str:
        return self.subentry.data.get(CONF_CHAT_MODEL) or self._runtime.default_model()

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        *,
        request_timeout: float,
        structure: probatio.Schema | None = None,
    ) -> None:
        """Send the chat log to Copilot and stream the reply back into it."""
        system = chat_log.content[0]
        if not isinstance(system, conversation.SystemContent):
            raise HomeAssistantError("Chat log is missing its system prompt")
        user_index = _last_user_index(chat_log.content)
        user_content = chat_log.content[user_index]
        assert isinstance(user_content, conversation.UserContent)

        model_id = self._model_id
        model = await self._async_get_model(model_id)

        prompt = _build_prompt(chat_log.content[1:user_index], user_content.content)
        response_schema: dict[str, Any] | None = None
        if structure is not None:
            response_schema = strict_output_schema(
                probatio.to_openapi(
                    structure,
                    custom_serializer=(
                        chat_log.llm_api.custom_serializer
                        if chat_log.llm_api
                        else llm.selector_serializer
                    ),
                    openapi_version="3.1.0",
                )
            )
            # Belt and braces for models that ignore native structured output.
            prompt += (
                "\n\nRespond with only a JSON object, without code fences or any "
                "other text, that conforms to this JSON schema:\n"
                + json.dumps(response_schema)
            )

        attachments = await _async_prepare_attachments(
            self.hass, model, user_content.attachments
        )

        run = _Run()
        tools = _build_tools(chat_log, run.emit) if chat_log.llm_api else []

        session_options: dict[str, Any] = {
            "model": model_id,
            "tools": tools,
            # Mode "empty" exposes nothing unless allow-listed; only our tools.
            "available_tools": ["custom:*"] if tools else [],
            "system_message": {"mode": "replace", "content": system.content},
            "streaming": True,
            "infinite_sessions": {"enabled": False},
            "on_permission_request": _reject_permission,
            "on_event": run.on_event,
        }
        effort = self.subentry.data.get(
            CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
        )
        if (
            effort != REASONING_EFFORT_DEFAULT
            and model is not None
            and effort in model.reasoning_efforts
        ):
            session_options["reasoning_effort"] = effort

        session, session_client = await self._async_create_session(session_options)
        unsubscribe_stop = self._runtime.async_on_stop(
            partial(
                run.emit,
                CopilotRequestError(
                    "The GitHub Copilot runtime stopped (the integration was "
                    "reloaded or Home Assistant is shutting down)"
                ),
            )
        )

        completed = False
        try:
            async with asyncio.timeout(request_timeout):
                await session.send(
                    prompt,
                    attachments=attachments or None,
                    response_schema=response_schema,
                )
                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id, run.deltas()
                ):
                    pass
            completed = True
        except TimeoutError as err:
            raise HomeAssistantError(
                f"GitHub Copilot did not respond within {request_timeout} seconds"
            ) from err
        except CopilotAuthRequestError:
            self.entry.async_start_reauth(self.hass)
            raise
        except HomeAssistantError:
            raise
        except Exception as err:
            raise HomeAssistantError(f"GitHub Copilot request failed: {err}") from err
        finally:
            unsubscribe_stop()
            self._runtime.async_schedule_cleanup(
                _async_close_session(session_client, session, abort=not completed)
            )

    async def _async_get_model(self, model_id: str) -> CopilotModel:
        """Return metadata for the configured model, refusing unknown ids.

        Copilot silently substitutes another (possibly premium) model for an
        unknown id, so a typo must fail loudly instead.
        """
        runtime = self._runtime
        if (model := runtime.models.get(model_id)) is None and runtime.client:
            try:
                await runtime.async_refresh_models()
            except Exception:
                LOGGER.debug("Unable to refresh the Copilot model list", exc_info=True)
            model = runtime.models.get(model_id)
        if model is None:
            raise HomeAssistantError(
                f"Model '{model_id}' isn't available to this GitHub account; "
                "reconfigure the GitHub Copilot integration"
            )
        return model

    async def _async_create_session(
        self, options: dict[str, Any]
    ) -> tuple[CopilotSession, CopilotClient]:
        """Create a session, restarting the runtime once if it has died."""
        runtime = self._runtime
        client = runtime.client
        if client is not None:
            try:
                return await client.create_session(**options), client
            except Exception as err:
                # Only a dead runtime warrants a restart; request errors such
                # as an unknown model must not disturb other sessions.
                if await _async_is_alive(client):
                    raise HomeAssistantError(
                        f"Unable to start a GitHub Copilot session: {err}"
                    ) from err
                LOGGER.warning("The GitHub Copilot runtime stopped responding: %s", err)
        try:
            await runtime.async_restart(client)
        except CopilotAuthError as err:
            self.entry.async_start_reauth(self.hass)
            raise HomeAssistantError(
                "GitHub Copilot rejected the token; re-authenticate the integration"
            ) from err
        except CopilotError as err:
            raise HomeAssistantError(
                f"The GitHub Copilot runtime is not available: {err}"
            ) from err
        client = runtime.client
        assert client is not None
        try:
            return await client.create_session(**options), client
        except Exception as err:
            raise HomeAssistantError(
                f"Unable to start a GitHub Copilot session: {err}"
            ) from err


def _last_user_index(content: list[conversation.Content]) -> int:
    for index in range(len(content) - 1, 0, -1):
        if isinstance(content[index], conversation.UserContent):
            return index
    raise HomeAssistantError("Chat log has no user message")


def _build_prompt(history: list[conversation.Content], user_text: str) -> str:
    """Flatten earlier turns into a transcript.

    Each request runs in a fresh Copilot session, so prior turns (including
    tool calls the model made) are replayed as context.
    """
    lines: list[str] = []
    for content in history:
        if isinstance(content, conversation.UserContent):
            lines.append(f"User: {content.content}")
        elif isinstance(content, conversation.AssistantContent):
            for call in content.tool_calls or ():
                lines.append(
                    f"Assistant called tool {call.tool_name} with "
                    f"{json.dumps(call.tool_args, default=str, ensure_ascii=False)}"
                )
            if content.content:
                lines.append(f"Assistant: {content.content}")
        elif isinstance(content, conversation.ToolResultContent):
            result = json.dumps(content.result.data, default=str, ensure_ascii=False)
            if len(result) > MAX_HISTORY_TOOL_RESULT_CHARS:
                result = result[:MAX_HISTORY_TOOL_RESULT_CHARS] + "…"
            lines.append(f"Tool {content.tool_name} returned: {result}")
    if not lines:
        return user_text
    return (
        "Earlier in this conversation (context only; these actions already "
        "happened, do not repeat them unless asked):\n"
        + "\n".join(lines)
        + "\n\nCurrent user message:\n"
        + user_text
    )


def _build_tools(
    chat_log: conversation.ChatLog, emit: Callable[[Any], None]
) -> list[Tool]:
    llm_api = chat_log.llm_api
    assert llm_api is not None
    return [
        Tool(
            name=tool.name,
            description=tool.description or tool.name,
            parameters=tool_parameters_schema(
                probatio.to_openapi(
                    tool.parameters,
                    custom_serializer=llm_api.custom_serializer,
                    openapi_version="3.1.0",
                )
            ),
            handler=partial(_async_call_tool, llm_api, emit, tool.name),
            # Permission is governed by which entities are exposed to Assist.
            skip_permission=True,
        )
        for tool in llm_api.tools
    ]


async def _async_call_tool(
    llm_api: llm.APIInstance,
    emit: Callable[[Any], None],
    tool_name: str,
    invocation: ToolInvocation,
) -> ToolResult:
    """Run a Home Assistant LLM tool on Copilot's behalf and log it."""
    args = invocation.arguments if isinstance(invocation.arguments, dict) else {}
    tool_input = llm.ToolInput(
        tool_name=tool_name,
        tool_args=args,
        id=invocation.tool_call_id or ulid_now(),
        # Copilot drives the tool loop, so the chat log must not execute it again.
        external=True,
    )
    emit({"tool_calls": [tool_input]})
    LOGGER.debug("Tool call: %s(%s)", tool_name, args)
    try:
        result = await llm_api.async_call_tool(tool_input)
    except (HomeAssistantError, probatio.Invalid) as err:
        data: dict[str, Any] = {"error": type(err).__name__}
        if str(err):
            data["error_text"] = str(err)
        result = llm.ToolResult(data=data, error=True)
    except Exception:
        # Details may be sensitive; keep them in the local log only.
        LOGGER.exception("Unexpected error calling tool %s", tool_name)
        result = llm.ToolResult(
            data={
                "error": "UnexpectedError",
                "error_text": "The tool failed unexpectedly",
            },
            error=True,
        )
    LOGGER.debug("Tool response: %s", result)
    emit(
        {
            "role": "tool_result",
            "tool_call_id": tool_input.id,
            "tool_name": tool_name,
            "result": result,
        }
    )
    return ToolResult(
        text_result_for_llm=json.dumps(result.data, default=str, ensure_ascii=False),
        result_type="failure" if result.error else "success",
    )


def _reject_permission(request: Any, invocation: Any) -> PermissionDecisionReject:
    """Deny anything that isn't one of our own (pre-approved) tools."""
    LOGGER.warning("Denied unexpected GitHub Copilot permission request: %s", request)
    return PermissionDecisionReject(feedback="Not permitted in Home Assistant")


async def _async_prepare_attachments(
    hass: Any,
    model: CopilotModel | None,
    attachments: list[conversation.Attachment] | None,
) -> list[dict[str, Any]]:
    """Read attachments into inline base64 blobs, validating model limits."""
    if not attachments:
        return []
    if model is not None and not model.vision:
        raise HomeAssistantError(f"Model {model.id} does not support image attachments")
    # Hard caps apply even to custom model ids without known limits.
    max_count = min((model and model.max_images) or MAX_ATTACHMENTS, MAX_ATTACHMENTS)
    if len(attachments) > max_count:
        raise HomeAssistantError(f"At most {max_count} attachment(s) are supported")
    max_size = min(
        (model and model.max_image_size) or MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_BYTES
    )

    def _read() -> list[dict[str, Any]]:
        blobs = []
        for attachment in attachments:
            mime_type = attachment.mime_type or mimetypes.guess_type(attachment.path)[0]
            if not mime_type or not (
                mime_type.startswith("image/") or mime_type == "application/pdf"
            ):
                raise HomeAssistantError(
                    f"Unsupported attachment type: {mime_type or attachment.path.name}"
                )
            if (
                model is not None
                and model.supported_media_types
                and mime_type not in model.supported_media_types
            ):
                raise HomeAssistantError(
                    f"Model {model.id} does not support {mime_type} attachments"
                )
            if not attachment.path.is_file():
                raise HomeAssistantError(f"Attachment {attachment.path} does not exist")
            # Check the size before reading so huge files never reach memory,
            # and again afterwards in case the file grew in between.
            too_large = HomeAssistantError(
                f"Attachment {attachment.path.name} is larger than {max_size} bytes"
            )
            if attachment.path.stat().st_size > max_size:
                raise too_large
            data = attachment.path.read_bytes()
            if len(data) > max_size:
                raise too_large
            blobs.append(
                {
                    "type": "blob",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": mime_type,
                    "displayName": attachment.path.name,
                }
            )
        return blobs

    return await hass.async_add_executor_job(_read)


async def _async_close_session(
    client: CopilotClient, session: CopilotSession, *, abort: bool
) -> None:
    """Tear a per-request session down without delaying the response.

    Each step is bounded and attempted independently, so a wedged runtime
    can't hang this task or skip the remaining cleanup.
    """
    if abort:
        await _async_best_effort("abort", session.abort())
    await _async_best_effort("disconnect", session.disconnect())
    await _async_best_effort("delete", client.delete_session(session.session_id))


async def _async_best_effort(step: str, operation: Awaitable[Any]) -> None:
    try:
        async with asyncio.timeout(SESSION_CLOSE_TIMEOUT):
            await operation
    except Exception:
        LOGGER.debug("Error during Copilot session %s", step, exc_info=True)


async def _async_is_alive(client: CopilotClient) -> bool:
    """Return whether the runtime still answers."""
    try:
        async with asyncio.timeout(PING_TIMEOUT):
            await client.ping()
    except Exception:
        return False
    return True
