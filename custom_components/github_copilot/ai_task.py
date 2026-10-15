"""AI Task entity backed by GitHub Copilot."""

from __future__ import annotations

import json
import re
import struct
from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import GitHubCopilotConfigEntry
from .const import AI_TASK_TIMEOUT, IMAGE_TIMEOUT, LOGGER, SUBENTRY_AI_TASK
from .entity import GitHubCopilotBaseLLMEntity

_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GitHubCopilotConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up one AI task entity per AI task subentry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_AI_TASK:
            continue
        async_add_entities(
            [GitHubCopilotAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GitHubCopilotAITaskEntity(ai_task.AITaskEntity, GitHubCopilotBaseLLMEntity):
    """GitHub Copilot AI Task entity."""

    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    def __init__(
        self, entry: GitHubCopilotConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        if entry.runtime_data.image_generation_available:
            self._attr_supported_features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(
            chat_log, request_timeout=AI_TASK_TIMEOUT, structure=task.structure
        )

        last = chat_log.content[-1]
        if not isinstance(last, conversation.AssistantContent) or not last.content:
            raise HomeAssistantError("GitHub Copilot returned no response")
        text = last.content

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id, data=text
            )

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id, data=_parse_json(text)
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        (
            data,
            mime_type,
            revised_prompt,
        ) = await self._async_generate_image_from_chat_log(
            chat_log, request_timeout=IMAGE_TIMEOUT
        )
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=revised_prompt or "Generated an image.",
            )
        )
        width, height = _image_size(data)
        return ai_task.GenImageTaskResult(
            image_data=data,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            width=width,
            height=height,
            revised_prompt=revised_prompt,
        )


def _image_size(data: bytes) -> tuple[int | None, int | None]:
    """Read the dimensions from a PNG header, if it is one."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    return None, None


def _parse_json(text: str) -> Any:
    """Parse model JSON, tolerating code fences or stray prose around it."""
    candidates = [text.strip()]
    if match := _CODE_FENCE.match(candidates[0]):
        candidates.append(match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    LOGGER.debug("Invalid JSON from GitHub Copilot (truncated): %.500s", text)
    raise HomeAssistantError("GitHub Copilot returned an invalid JSON response")
